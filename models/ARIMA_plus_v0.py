import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Dict
from data_provider.data_factory import data_provider
from layers.Autoformer_EncDec import series_decomp




class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(DLinear, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None


class ARIMA_Scratch:
	def __init__(self, p: int = 12, d: int = 1, q: int = 0):
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
		
		# Compute log-likelihood
		ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(e ** 2) / sigma2
		return -ll  # Negative log-likelihood for minimization
	
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
	
	def forecast_arima(self, y: np.ndarray, params: Dict[str, np.ndarray], p: int, q: int, steps: int = 10) -> np.ndarray:
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
	
	def predict(self, steps: int = 10, series: np.ndarray=None) -> np.ndarray:
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
		self.device = configs.device
		self.backbone_ARIMA = ARIMA_Scratch()
		self.backbone_DLinear = DLinear(configs).float()
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