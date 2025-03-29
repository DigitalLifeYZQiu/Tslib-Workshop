import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Dict
from data_provider.data_factory import data_provider
import datetime


class ARIMA_Scratch:
	def __init__(self, p: int = 12, d: int = 1, q: int = 1):
		"""
		Initialize the ARIMA model.

		Args:
			p (int): Autoregressive order.
			d (int): Differencing order.
			q (int): Moving average order.
		"""
		self.p = p
		self.d = d
		self.q = q
		self.params = None
	
	def arima_log_likelihood(self, params: np.ndarray, y: np.ndarray, p: int, q: int) -> float:
		"""
		Compute the negative log-likelihood for ARIMA(p, d, q).

		Args:
			params (np.ndarray): Array containing AR and MA parameters followed by sigma^2.
			y (np.ndarray): Differenced time series data.
			p (int): AR order.
			q (int): MA order.

		Returns:
			float: Negative log-likelihood.
		"""
		phi = params[:p]
		theta = params[p:p + q]
		sigma2 = params[-1]
		
		n = len(y)
		e = np.zeros(n)
		
		for t in range(n):
			ar_term = 0
			for i in range(p):
				if t - i - 1 >= 0:
					ar_term += phi[i] * y[t - i - 1]
			
			ma_term = 0
			for j in range(q):
				if t - j - 1 >= 0:
					ma_term += theta[j] * e[t - j - 1]
			
			e[t] = y[t] - ar_term - ma_term
		# # Compute log-likelihood
		# ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(e ** 2) / sigma2
		# return -ll  # Negative log-likelihood for minimization
		
		# Compute MSE
		return np.mean(e ** 2)
	
	def difference(self, series: np.ndarray) -> np.ndarray:
		"""
		Apply differencing to a time series to make it stationary.

		Args:
			series (np.ndarray): Original time series data.
			order (int): Order of differencing.

		Returns:
			np.ndarray: Differenced time series.
		"""
		diff_series = series.copy()
		for _ in range(self.d):
			diff_series = np.diff(diff_series)
		return diff_series
	
	def estimate_arima_parameters(self, y: np.ndarray, p: int, q: int) -> Dict[str, np.ndarray]:
		"""
		Estimate ARIMA(p, d, q) parameters using MLE.

		Args:
			y (np.ndarray): Differenced time series data.
			p (int): AR order.
			q (int): MA order.

		Returns:
			Dict[str, np.ndarray]: Estimated parameters including AR, MA, and sigma^2.
		"""
		# Initial guesses: AR and MA coefficients as 0.5, sigma^2 as variance of y
		initial_params = np.r_[0.5 * np.ones(p), 0.5 * np.ones(q), np.var(y)]
		
		# Bounds: AR and MA parameters between -1 and 1, sigma2 > 0
		bounds = [(-1, 1)] * (p + q) + [(1e-5, None)]
		
		# Minimize negative log-likelihood
		result = minimize(self.arima_log_likelihood, initial_params, args=(y, p, q),
		                  bounds=bounds, method='L-BFGS-B')
		
		if not result.success:
			raise ValueError("Optimization failed: " + result.message)
		
		estimated_params = result.x
		params = {
			'ar': estimated_params[:p],
			'ma': estimated_params[p:p + q],
			'sigma2': estimated_params[-1]
		}
		return params
	
	def forecast_arima(self, y: np.ndarray, params: Dict[str, np.ndarray], p: int, q: int,
	                   steps: int = 10) -> np.ndarray:
		"""
		Forecast future values using ARIMA(p, d, q) parameters.

		Args:
			y (np.ndarray): Differenced time series data.
			params (Dict[str, np.ndarray]): Estimated parameters including 'ar', 'ma', and 'sigma2'.
			p (int): AR order.
			q (int): MA order.
			steps (int): Number of steps to forecast.

		Returns:
			np.ndarray: Forecasted differenced values.
		"""
		phi = params['ar']
		theta = params['ma']
		
		e = np.zeros(len(y) + steps)
		y_full = np.concatenate([y, np.zeros(steps)])
		
		for t in range(len(y), len(y) + steps):
			ar_term = 0
			for i in range(p):
				if t - i - 1 >= 0:
					ar_term += phi[i] * y_full[t - i - 1]
			
			ma_term = 0
			for j in range(q):
				if t - j - 1 >= 0:
					ma_term += theta[j] * e[t - j - 1]
			
			# For forecasting, assume future errors are zero (deterministic forecast)
			y_full[t] = ar_term + ma_term
		# Alternatively, to incorporate uncertainty, you could add random noise:
		# e[t] = np.random.normal(0, np.sqrt(params['sigma2']))
		
		return y_full[len(y):]
	
	def reconstruct_original_series(self, series: np.ndarray, forecast_diff: np.ndarray) -> np.ndarray:
		"""
		Reconstruct the forecasted original series from differenced forecasts.

		Args:
			series (np.ndarray): Original time series data.
			forecast_diff (np.ndarray): Forecasted differenced values.

		Returns:
			np.ndarray: Forecasted original series values.
		"""
		last_value = series[-1]
		forecast_original = np.concatenate([series, last_value + np.cumsum(forecast_diff)])
		return forecast_original
	
	def fit(self, series: np.ndarray):
		"""
		Fit the ARIMA model to the data.

		Args:
			series (np.ndarray): Original time series data.
		"""
		# Apply differencing
		y_diff = self.difference(series)
		
		# Estimate parameters
		self.params = self.estimate_arima_parameters(y_diff, self.p, self.q)
		print("Fitted Parameters:", self.params)
	
	def predict(self, steps: int = 10, series: np.ndarray = None) -> np.ndarray:
		"""
		Forecast future values using the fitted ARIMA model.

		Args:
			series (np.ndarray): Original time series data.
			steps (int): Number of steps to forecast.

		Returns:
			np.ndarray: Forecasted original series values.
		"""
		if self.params is None:
			raise ValueError("Model has not been fitted yet.")
		
		if series is None:
			raise ValueError("Input series has not been given yet.")
		
		# Apply differencing
		y_diff = self.difference(series)
		
		# Forecast differenced series
		forecast_diff = self.forecast_arima(y_diff, self.params, self.p, self.q, steps)
		
		# Reconstruct the original scale
		forecast_original = self.reconstruct_original_series(series, forecast_diff)
		
		return forecast_original[-steps:]


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
		self.backbone = ARIMA_Scratch()
		self.fit(configs)
	
	def fit(self, configs):
		self.train_data, self.train_loader = data_provider(configs, flag='train')
		trainset = self.train_data.data_x.squeeze()
		print(f"{configs.model} Fitting Training Set")
		start = datetime.datetime.now()
		self.backbone.fit(np.array(trainset))
		end = datetime.datetime.now()
		print(f"{configs.model} Training Time: {end - start}")
	
	def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
		x_enc = x_enc.to('cpu').detach().numpy().squeeze()
		prediction = self.backbone.predict(self.pred_len, x_enc)
		return torch.Tensor(prediction).reshape((1, self.pred_len, 1))
	
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