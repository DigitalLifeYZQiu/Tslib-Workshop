import torch
import torch.nn as nn
import torch.nn.functional as F
import darts
from darts.models import AutoARIMA
from darts import TimeSeries
from data_provider.data_factory import data_provider
import datetime


class Model(nn.Module):
	"""
	Autoformer is the first method to achieve the series-wise connection,
	with inherent O(LlogL) complexity
	Paper link: https://openreview.net/pdf?id=I55UqU-M11y
	"""
	
	def __init__(self, configs):
		super(Model, self).__init__()
		self.task_name = configs.task_name
		self.seq_len = configs.seq_len
		self.label_len = configs.label_len
		self.pred_len = configs.pred_len
		self.backbone = AutoARIMA()
		# self.fit(configs)
	
	def fit(self, configs):
		self.train_data, self.train_loader = data_provider(configs, flag='train')
		trainset = self.train_data.data_x.squeeze()
		print(f"{configs.model} Fitting Training Set")
		self.backbone.fit(TimeSeries.from_values(trainset))
	
	def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
		x_enc = x_enc.to('cpu').detach().numpy().squeeze()
		x_enc = TimeSeries.from_values(x_enc)
		self.backbone.fit(x_enc)
		prediction = self.backbone.predict(self.pred_len).values()
		return torch.Tensor(prediction).reshape((1, self.pred_len, 1))
	
	def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
		raise NotImplementedError
	
	def anomaly_detection(self, x_enc):
		raise NotImplementedError
	
	def classification(self, x_enc, x_mark_enc):
		raise NotImplementedError
	
	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
			# start = datetime.datetime.now()
			dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
			# end = datetime.datetime.now()
			# print(f'totally time is {end - start}')
			return dec_out[:, -self.pred_len:, :]  # [B, L, D]
		if self.task_name == 'imputation':
			raise NotImplementedError
		if self.task_name == 'anomaly_detection':
			raise NotImplementedError
		if self.task_name == 'classification':
			raise NotImplementedError
		return None