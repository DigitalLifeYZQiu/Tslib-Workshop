import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Dict
from data_provider.data_factory import data_provider
from layers.Autoformer_EncDec import series_decomp
from models.DLinear import Model as DLinear
from models.DeepARIMA import Model as ARIMA_Scratch
# from models.ARMD.Models.autoregressive_diffusion.armd import ARMD


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
		self.device = configs.device
		self.backbone_ARIMA = ARIMA_Scratch(configs)
		self.backbone_DLinear = DLinear(configs).float()
		# self.backbond_ARMD = ARMD(configs).float()
		self.fit(configs)
	
	def fit(self, configs):
		self.train_data, self.train_loader = data_provider(configs, flag='train')
		trainset = self.train_data.data_x.squeeze()
		print(f"{configs.model} Fitting Training Set")
		# import pdb; pdb.set_trace()
		if len(trainset.shape)==1:
			self.backbone_ARIMA.fit(np.array(trainset))
		elif len(trainset.shape)==2:
			self.backbone_ARIMA.fit(np.array(trainset)[:,-1])
		else:
			self.backbone_ARIMA.fit(np.array(trainset))
	
	def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
		prediction_DLinear = self.backbone_DLinear(x_enc, x_mark_enc, x_dec, x_mark_dec)
		
		import pdb; pdb.set_trace()
		
		
		x_enc = x_enc.to('cpu').detach().numpy().squeeze()
		
		if len(x_enc.shape)==1:
			# import pdb; pdb.set_trace()
			prediction_ARIMA = torch.Tensor(self.backbone_ARIMA.predict(self.pred_len, x_enc).reshape(1, -1, 1)).to(self.device)
		elif len(x_enc.shape)==2:
			B,S = x_enc.shape
			prediction_ARIMA = []
			for i in range(B):
				prediction_ARIMA.append(torch.Tensor(self.backbone_ARIMA.predict(self.pred_len, x_enc[i]).reshape(1,-1, 1)))
			prediction_ARIMA = torch.cat(prediction_ARIMA, dim=0).to(self.device)
		elif len(x_enc.shape)==3:
			B,S,C = x_enc.shape
			prediction_ARIMA = []
			for i in range(B):
				prediction_ARIMA_C = []
				for j in range(C):
					prediction_ARIMA_C.append(torch.Tensor(self.backbone_ARIMA.predict(self.pred_len, x_enc[i,:,j]).reshape(1, -1, 1)))
				prediction_ARIMA.append(torch.cat(prediction_ARIMA_C, dim=2))
			prediction_ARIMA = torch.cat(prediction_ARIMA, dim=0).to(self.device)
		else:
			prediction_ARIMA =torch.seros_like(prediction_DLinear).to(self.device)
		# import pdb; pdb.set_trace()
		prediction = (prediction_ARIMA + prediction_DLinear) / 2
		return prediction
	
	def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
		raise NotImplementedError
	
	def anomaly_detection(self, x_enc):
		raise NotImplementedError
	
	def classification(self, x_enc, x_mark_enc):
		raise NotImplementedError
	
	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
			dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
			return dec_out[:, -self.pred_len:, :]  # [B, L, D]
		if self.task_name == 'imputation':
			raise NotImplementedError
		if self.task_name == 'anomaly_detection':
			raise NotImplementedError
		if self.task_name == 'classification':
			raise NotImplementedError
		return None


# # Example Usage
# # Generate synthetic data
# n = 200
# phi = 0.6
# theta = 0.4
# d = 1
# ts = generate_arima_data(n, phi, theta, d)
#
# # Initialize and fit the model
# model = ARIMA_Scratch(p=1, d=1, q=1)
# model.fit(ts)
#
# # Forecast the next 20 steps
# forecast_steps = 20
# forecast = model.predict(ts, steps=forecast_steps)
# print("Forecasted Values:\n", forecast)
#
# # Plot the forecast
# plt.figure(figsize=(12, 6))
# plt.plot(ts, label='Original Series')
# plt.plot(range(len(ts), len(ts) + forecast_steps), forecast, label='Forecast', color='red', marker='o')
# plt.title('ARIMA(1,1,1) Forecast')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()