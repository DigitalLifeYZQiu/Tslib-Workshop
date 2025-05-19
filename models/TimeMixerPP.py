import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.StandardNorm import Normalize


def FFT_for_Period(x, k=2):
	# [B, T, C]
	xf = torch.fft.rfft(x, dim=1)
	# find period by amplitudes
	frequency_list = abs(xf).mean(0).mean(-1)
	frequency_list[0] = 0
	_, top_list = torch.topk(frequency_list, k)
	top_list = top_list.detach().cpu().numpy()
	period = x.shape[1] // top_list
	return period, abs(xf).mean(-1)[:, top_list]


class DownsamplingModule(nn.Module):
	def __init__(self, input_channels, num_scales):
		"""
		Args:
			input_channels (int): 输入时间序列的通道数（变量数）。
			num_scales (int): 降采样的尺度数量。
		"""
		super(DownsamplingModule, self).__init__()
		self.num_scales = num_scales
		self.conv_layers = nn.ModuleList()
		
		# 创建卷积层用于降采样
		for i in range(num_scales):
			# 每个卷积层的步幅为2，实现降采样
			conv_layer = nn.Conv1d(in_channels=input_channels,
			                       out_channels=input_channels,
			                       kernel_size=3, stride=2, padding=1)
			self.conv_layers.append(conv_layer)
	
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): 输入的时间序列，形状为 (batch_size, sequence_length, num_channels)

		Returns:
			list: 包含多尺度表示的列表，每个元素的形状为 (batch_size, sequence_length / 2^i, num_channels)
		"""
		# 将输入从 (batch_size, sequence_length, num_channels) 转换为 (batch_size, num_channels, sequence_length)
		x = x.permute(0, 2, 1)
		
		multi_scale_outputs = [x]  # 初始尺度为原始输入
		for i in range(self.num_scales):
			x = self.conv_layers[i](x)  # 应用卷积层进行降采样
			multi_scale_outputs.append(x)
		
		# 将每个尺度的输出转换回 (batch_size, sequence_length, num_channels)
		multi_scale_outputs = [output.permute(0, 2, 1) for output in multi_scale_outputs]
		
		return multi_scale_outputs


class InputProjection(nn.Module):
	def __init__(self, input_len, num_scales, input_channels, d_model):
		"""
		Args:
			input_channels (int): 输入时间序列的通道数（变量数）。
			d_model (int): 模型的隐藏层维度。
		"""
		super(InputProjection, self).__init__()
		self.num_scales = num_scales
		
		self.channel_attns = nn.ModuleList()
		for i in range(num_scales + 1):
			# 对每个尺度定义一个dim为input_len
			self.channel_attns.append(nn.MultiheadAttention(embed_dim=input_len, num_heads=1, batch_first=True))
			input_len = -(-input_len // 2)
		
		self.input_projection = nn.Linear(input_channels, d_model)
	
	def forward(self, x, x_mark):
		"""
		Args:
			x (list): 包含多尺度输入张量的列表，每个张量的形状为 (batch_size, sequence_length, num_channels)

		Returns:
			list: 包含多尺度表示的列表，每个元素的形状为 (batch_size, sequence_length, d_model)
		"""
		multi_scale_outputs = []
		for i in range(self.num_scales + 1):
			# 对每个尺度的输入进行注意力
			x_input = x[i].permute(0, 2, 1)
			attn_output, _ = self.channel_attns[i](x_input, x_input, x_input)
			attn_output = attn_output.permute(0, 2, 1)
			attn_output = self.input_projection(attn_output)
			multi_scale_outputs.append(attn_output)
		return multi_scale_outputs


class MixerBlock(nn.Module):
	def __init__(self, input_len, num_scales, num_resolutions, d_model):
		"""
		Args:
			input_channels (int): 输入时间序列的通道数（变量数）。
			num_scales (int): 降采样的尺度数量。
			num_resolutions (int): 多分辨率时间成像的分辨率数量。
			d_model (int): 模型的隐藏层维度。
		"""
		super(MixerBlock, self).__init__()
		self.input_len = input_len
		self.num_scales = num_scales
		self.num_resolutions = num_resolutions
		self.d_model = d_model
		
		# 多分辨率时间成像 (MRTI)
		
		# 时间图像分解 (TID)
		self.col_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
		self.row_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
		
		self.col_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
		self.row_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
		
		# 多尺度混合 (MCM)
		self.mcm_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, stride=(2, 1), kernel_size=3, padding=1)
		self.mcm_transconv = nn.ConvTranspose2d(in_channels=d_model, out_channels=d_model, stride=(2, 1),
		                                        kernel_size=(2, 1), padding=0)
		
		# 多分辨率混合 (MRM)
		self.mrm_softmax = nn.Softmax(dim=-1)
	
	def forward(self, multi_scale_inputs):
		"""
		Args:
			multi_scale_inputs (list): 多尺度输入，每个元素的形状为 (batch_size, sequence_length, num_channels)

		Returns:
			list: 包含多尺度表示的列表，每个元素的形状为 (batch_size, sequence_length, num_channels)
		"""
		# 多分辨率时间成像 (MRTI)
		multi_res_images = []
		period_list, period_weight = FFT_for_Period(multi_scale_inputs[-1], self.num_resolutions)
		for scale_idx, x in enumerate(multi_scale_inputs):
			B, T, D = x.size()
			res = []
			for i in range(self.num_resolutions):
				period = period_list[i]
				# padding
				if T % period != 0:
					length = ((T // period) + 1) * period
					padding = torch.zeros([x.shape[0], (length - T), x.shape[2]]).to(
						x.device)
					out = torch.cat([x, padding], dim=1)
				else:
					length = T
					out = x
				# reshape
				out = out.reshape(B, length // period, period, D).permute(0, 3, 1, 2).contiguous()
				res.append(out)
			multi_res_images.append(res)
		
		# 时间图像分解 (TID)
		seasonal_images = []
		trend_images = []
		for scale_idx in range(self.num_scales + 1):
			seasonal_images_output = []
			trend_images_output = []
			for res_idx in range(self.num_resolutions):
				# 对每个分辨率的输出进行注意力
				B, D, F, P = multi_res_images[scale_idx][res_idx].size()
				x_col = self.row_conv(multi_res_images[scale_idx][res_idx]).permute(0, 2, 3, 1).contiguous().reshape(
					B * F, P, D)
				x_row = self.col_conv(multi_res_images[scale_idx][res_idx]).permute(0, 3, 2, 1).contiguous().reshape(
					B * P, F, D)
				seasonal_image = self.col_attention(x_col, x_col, x_col)[0].reshape(B, F, P, D).permute(0, 3, 1,
				                                                                                        2).contiguous()
				trend_image = self.row_attention(x_row, x_row, x_row)[0].reshape(B, P, F, D).permute(0, 3, 2,
				                                                                                     1).contiguous()
				seasonal_images_output.append(seasonal_image)
				trend_images_output.append(trend_image)
			
			seasonal_images.append(seasonal_images_output)
			trend_images.append(trend_images_output)
		
		# 多尺度混合 (MCM)
		for res_idx in range(self.num_resolutions):
			for scale_idx in range(self.num_scales):
				# 对季节性和趋势成分进行混合
				seasonal_images[scale_idx + 1][res_idx] = seasonal_images[scale_idx + 1][res_idx] + self.mcm_conv(
					seasonal_images[scale_idx][res_idx])
				trend_images[self.num_scales - scale_idx - 1][res_idx] = (
						trend_images[self.num_scales - scale_idx - 1][res_idx] +
						self.mcm_transconv(trend_images[self.num_scales - scale_idx][res_idx])[:, :,
						:trend_images[self.num_scales - scale_idx - 1][res_idx].shape[-2], :])
		
		input_len = self.input_len
		hidden_states = []
		for scale_idx in range(self.num_scales + 1):
			hidden_states_output = []
			for res_idx in range(self.num_resolutions):
				# 2D -> 1D
				B, D, F, P = seasonal_images[scale_idx][res_idx].size()
				image = (seasonal_images[scale_idx][res_idx].reshape(B, D, -1).permute(0, 2, 1).contiguous()
				         + trend_images[scale_idx][res_idx].reshape(B, D, -1).permute(0, 2, 1).contiguous())
				hidden_states_output.append(image[:, :input_len, :])  # [B, L, D]
			hidden_states.append(torch.stack(hidden_states_output, dim=1))  # [B, num_resolutions, L, D]
			input_len = -(-input_len // 2)
		
		# 多分辨率混合 (MRM)
		final_outputs = []
		period_weight = self.mrm_softmax(period_weight)  # [B, num_resolutions]
		for scale_idx in range(self.num_scales + 1):
			# 对每个尺度的季节性和趋势成分进行加权求和，根据period_weight做softmax，在num_resolutions维度上加权平均
			final_output = torch.sum(hidden_states[scale_idx] * period_weight.unsqueeze(-1).unsqueeze(-1), dim=1)
			final_outputs.append(final_output)
		
		return final_outputs


class Model(nn.Module):
	
	def __init__(self, configs):
		super(Model, self).__init__()
		self.configs = configs
		self.task_name = configs.task_name
		self.seq_len = configs.seq_len
		self.label_len = configs.label_len
		self.pred_len = configs.pred_len
		self.channel_independence = 1
		self.layer = configs.e_layers
		
		self.num_scales = 3
		self.num_resolutions = configs.top_k
		
		self.normalize_layers = torch.nn.ModuleList(
			[
				Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
				for i in range(self.num_scales + 1)
			]
		)
		
		# 初始化降采样模块
		padding = 1 if torch.__version__ >= '1.5.0' else 2
		self.down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
		                           kernel_size=3, padding=padding,
		                           stride=2,
		                           padding_mode='circular',
		                           bias=False)
		# 初始化输入投影模块
		if self.channel_independence:
			self.enc_embedding = InputProjection(input_len=configs.seq_len, num_scales=self.num_scales,
			                                     input_channels=1,
			                                     d_model=configs.d_model)
		else:
			self.enc_embedding = InputProjection(input_len=configs.seq_len, num_scales=self.num_scales,
			                                     input_channels=configs.enc_in,
			                                     d_model=configs.d_model)
		# # 初始化MixerBlock
		# self.mixer_block = MixerBlock(input_len=configs.seq_len, num_scales=num_scales, num_resolutions=num_resolutions,
		#                          d_model=configs.d_model)
		self.pdm_blocks = nn.ModuleList(
			[MixerBlock(input_len=configs.seq_len, num_scales=self.num_scales, num_resolutions=self.num_resolutions,
			            d_model=configs.d_model) for _ in range(configs.e_layers)])
		
		if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
			self.predict_layers = torch.nn.ModuleList(
				[
					torch.nn.Linear(
						-(-configs.seq_len // (2 ** i)),
						configs.pred_len,
					)
					for i in range(self.num_scales + 1)
				]
			)
			
			if self.channel_independence:
				self.projection_layer = nn.Linear(
					configs.d_model, 1, bias=True)
			else:
				self.projection_layer = nn.Linear(
					configs.d_model, configs.c_out, bias=True)
				
				self.out_res_layers = torch.nn.ModuleList([
					torch.nn.Linear(
						configs.seq_len // (2 ** i),
						configs.seq_len // (2 ** i),
					)
					for i in range(self.num_scales + 1)
				])
				
				self.regression_layers = torch.nn.ModuleList(
					[
						torch.nn.Linear(
							configs.seq_len // (2 ** i),
							configs.pred_len,
						)
						for i in range(self.num_scales + 1)
					]
				)
		
		if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
			if self.channel_independence:
				self.projection_layer = nn.Linear(
					configs.d_model, 1, bias=True)
			else:
				self.projection_layer = nn.Linear(
					configs.d_model, configs.c_out, bias=True)
		if self.task_name == 'classification':
			self.act = F.gelu
			self.dropout = nn.Dropout(configs.dropout)
			self.projection = nn.Linear(
				configs.d_model * configs.seq_len, configs.num_class)
	
	def out_projection(self, dec_out, i, out_res):
		dec_out = self.projection_layer(dec_out)
		out_res = out_res.permute(0, 2, 1)
		out_res = self.out_res_layers[i](out_res)
		out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
		dec_out = dec_out + out_res
		return dec_out
	
	def pre_enc(self, x_list):
		if self.channel_independence:
			return (x_list, None)
		else:
			out1_list = []
			out2_list = []
			for x in x_list:
				x_1, x_2 = self.preprocess(x)
				out1_list.append(x_1)
				out2_list.append(x_2)
			return (out1_list, out2_list)
	
	def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
		
		# B,T,C -> B,C,T
		x_enc = x_enc.permute(0, 2, 1)
		
		x_enc_ori = x_enc
		x_mark_enc_mark_ori = x_mark_enc
		
		x_enc_sampling_list = []
		x_mark_sampling_list = []
		x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
		x_mark_sampling_list.append(x_mark_enc)
		
		for i in range(self.num_scales):
			x_enc_sampling = self.down_pool(x_enc_ori)
			
			x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
			x_enc_ori = x_enc_sampling
			
			if x_mark_enc is not None:
				x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::2, :])
				x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::2, :]
		
		x_enc = x_enc_sampling_list
		x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None
		
		return x_enc, x_mark_enc
	
	def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
		
		x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
		
		x_list = []
		for i, x in zip(range(len(x_enc)), x_enc, ):
			B, T, N = x.size()
			x = self.normalize_layers[i](x, 'norm')
			if self.channel_independence:
				x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
			x_list.append(x)
		
		# embedding
		enc_out_list = self.enc_embedding(x_list, x_mark_enc)
		
		# Past Decomposable Mixing as encoder for past
		for i in range(self.layer):
			enc_out_list = self.pdm_blocks[i](enc_out_list)
		
		# Future Multipredictor Mixing as decoder for future
		dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
		
		dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
		dec_out = self.normalize_layers[0](dec_out, 'denorm')
		return dec_out
	
	def future_multi_mixing(self, B, enc_out_list, x_list):
		dec_out_list = []
		if self.channel_independence:
			x_list = x_list[0]
			for i, enc_out in zip(range(len(x_list)), enc_out_list):
				dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
					0, 2, 1)  # align temporal dimension
				dec_out = self.projection_layer(dec_out)
				dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
				dec_out_list.append(dec_out)
		
		else:
			for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
				dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
					0, 2, 1)  # align temporal dimension
				dec_out = self.out_projection(dec_out, i, out_res)
				dec_out_list.append(dec_out)
		
		return dec_out_list
	
	def classification(self, x_enc, x_mark_enc):
		x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
		x_list = x_enc
		
		# embedding
		enc_out_list = self.enc_embedding(x_list, x_mark_enc)
		
		# MultiScale-CrissCrossAttention  as encoder for past
		for i in range(self.layer):
			enc_out_list = self.pdm_blocks[i](enc_out_list)
		
		enc_out = enc_out_list[0]
		# Output
		# the output transformer encoder/decoder embeddings don't include non-linearity
		output = self.act(enc_out)
		output = self.dropout(output)
		# zero-out padding embeddings
		output = output * x_mark_enc.unsqueeze(-1)
		# (batch_size, seq_length * d_model)
		output = output.reshape(output.shape[0], -1)
		output = self.projection(output)  # (batch_size, num_classes)
		return output
	
	def anomaly_detection(self, x_enc):
		B, T, N = x_enc.size()
		x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
		
		x_list = []
		
		for i, x in zip(range(len(x_enc)), x_enc, ):
			B, T, N = x.size()
			x = self.normalize_layers[i](x, 'norm')
			if self.channel_independence:
				x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
			x_list.append(x)
		
		# embedding
		enc_out_list = self.enc_embedding(x_list, None)
		# MultiScale-CrissCrossAttention  as encoder for past
		for i in range(self.layer):
			enc_out_list = self.pdm_blocks[i](enc_out_list)
		
		dec_out = self.projection_layer(enc_out_list[0])
		dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
		
		dec_out = self.normalize_layers[0](dec_out, 'denorm')
		return dec_out
	
	def imputation(self, x_enc, x_mark_enc, mask):
		means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
		means = means.unsqueeze(1).detach()
		x_enc = x_enc - means
		x_enc = x_enc.masked_fill(mask == 0, 0)
		stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
		                   torch.sum(mask == 1, dim=1) + 1e-5)
		stdev = stdev.unsqueeze(1).detach()
		x_enc /= stdev
		
		B, T, N = x_enc.size()
		x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
		
		x_list = []
		x_mark_list = []
		for i, x in zip(range(len(x_enc)), x_enc, ):
			B, T, N = x.size()
			if self.channel_independence:
				x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
			x_list.append(x)
		
		# embedding
		enc_out_list = self.enc_embedding(x_list, x_mark_enc)
		
		# MultiScale-CrissCrossAttention  as encoder for past
		for i in range(self.layer):
			enc_out_list = self.pdm_blocks[i](enc_out_list)
		
		dec_out = self.projection_layer(enc_out_list[0])
		dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
		
		dec_out = dec_out * \
		          (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
		dec_out = dec_out + \
		          (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
		return dec_out
	
	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
			dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
			return dec_out
		if self.task_name == 'imputation':
			dec_out = self.imputation(x_enc, x_mark_enc, mask)
			return dec_out  # [B, L, D]
		if self.task_name == 'anomaly_detection':
			dec_out = self.anomaly_detection(x_enc)
			return dec_out  # [B, L, D]
		if self.task_name == 'classification':
			dec_out = self.classification(x_enc, x_mark_enc)
			return dec_out  # [B, N]
		else:
			raise ValueError('Other tasks implemented yet')