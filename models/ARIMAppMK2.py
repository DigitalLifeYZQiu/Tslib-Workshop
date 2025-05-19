import torch
import torch.nn as nn
import torch.optim as optim
from layers.Autoformer_EncDec import series_decomp
import numpy as np
from data_provider.data_factory import data_provider
from utils.tools import TestTimeaLRAdjust, TestTimeEarlyStopping
import datetime
import os
import time

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class ARIMA(nn.Module):
    def __init__(self, configs):
        """
        p: 自回归阶数
        d: 差分阶数
        q: 移动平均阶数
        """
        super(ARIMA, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.p = min(int(self.seq_len / 2), configs.p)
        self.d = configs.d
        self.q = min(int(self.seq_len / 2), configs.q)
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        
        # 自回归部分
        self.ar_trend = nn.Linear(self.p,1, bias=True)
        self.ar_seasonal = nn.Linear(self.p,1, bias=True)
        # 移动平均部分
        # self.ma = nn.Linear(q,1, bias=False)
        if configs.features == 'S':
            self.ma = RevIN(num_features=1, eps=1e-5, affine=True)
        elif configs.features == 'MS' or configs.features == 'M':
            self.ma = RevIN(num_features=configs.enc_in, eps=1e-5, affine=True)
        else:
            self.ma = None
            raise ValueError(f"Unrecognized features {configs.features}")
    
    def difference(self, series: torch.Tensor) -> torch.Tensor:
        """
        差分操作
        """
        diff_series = series.clone()
        for _ in range(self.d):
            start_value = diff_series[:, :, :1]
            diff_series = diff_series[:, :, 1:] - diff_series[:, :, :-1]
            diff_series = torch.cat([start_value, diff_series], dim=2)
        return diff_series
    
    def inverse_difference(self, series: torch.Tensor, forecast_diff: torch.Tensor) -> torch.Tensor:
        for _ in range(self.d):
            last_value = series[:, :, -1:]
            forecast_diff = torch.cat([series, last_value + torch.cumsum(forecast_diff, dim=2)], dim=2)
        return forecast_diff
    
    
    def forecast(self, series: torch.Tensor) -> torch.Tensor:
        # moving average: Revin
        series = self.ma(series, 'norm')
        
        # Decompose: Autoformer type
        seasonal_init, trend_init = self.decomp(series)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # seasonal part processing
        seasonal_output = self.ar_seasonal(seasonal_init)
        seasonal_output = seasonal_output.permute(0, 2, 1)
        
        # trend part processing: with differentiate
        trend_diff = self.difference(trend_init)
        trend_output = self.ar_trend(trend_diff).to(series.device)
        trend_output = self.inverse_difference(trend_init, trend_output)
        trend_output = trend_output.permute(0, 2, 1)
        
        # Add trend and seasonal components together to get total output
        output = trend_output + seasonal_output
        
        output = self.ma(output, 'denorm')
        
        return output
    
    def mse_loss(self, y: torch.Tensor):
        B, S, C = y.shape
        e_list = []
        y_pred_list = []
        for t in range(S):
            if t > max(self.p, self.q):
                inp = y[:, t - self.p:t, :]
                pred = self.forecast(inp)
            else:
                pred = torch.tensor(0.).to(y.device).view(1,-1, 1)
            error = y[:, t, :] - pred[:, -1:, :]
            e_list.append(error)
            
            y_pred_list.append(torch.tensor(pred, dtype=torch.float32, device=y.device).view(1, -1))
        e = torch.cat(e_list, dim=1)
        
        # return torch.mean((y - y_pred) ** 2)
        return torch.mean(e ** 2)
    
    
    def fit(self, series: torch.Tensor, num_epochs=100, lr=0.01):
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # series = self.ma(series, 'norm')
        # y_diff = self.difference(series)
        
        for epoch in range(num_epochs):
            epoch_time = time.time()
            optimizer.zero_grad()
            loss = self.mse_loss(series)
            loss.backward()
            optimizer.step()
            # for name, param in self.named_parameters():
            #     print(f"{name}.grad: {param.grad}")
            
            # if (epoch + 1) % 1 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, cost time: {time.time() - epoch_time}')
        # print(f"AR parameters - Weight: {self.ar.weight.data}, Bias: {self.ar.bias.data if self.ar.bias is not None else 'None'}")
        # print(f"MA parameters - Weight: {self.ma.affine_weight}, Bias: {self.ma.affine_bias}")
            # print(f"MA parameters - Weight: {self.ma.affine_weight}, Bias: None")
    
    def predict(self, steps: int = 10, series: torch.Tensor = None) -> torch.Tensor:
        if series is None:
            raise ValueError("Input series has not been given yet.")
        # series = self.ma(series, 'norm')
        # y_diff = self.difference(series)
        
        B, S, C = series.shape
        y_full = torch.cat([series, torch.zeros((B, steps, C), dtype=torch.float32, device=series.device)], dim=1)
        
        for t in range(S, S + steps):
            inp = y_full[:, t - self.p:t, :]
            pred = self.forecast(inp)
            y_full[:, t, :] = pred[:, -1:, :]
        forecast_original = y_full[:, S:, :]
        
        return forecast_original[:, -steps:, :]


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.args = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.backbone = ARIMA(configs).to(configs.device)
    
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.backbone.eval()
        with torch.no_grad():
            if len(x_enc.shape) == 1:
                prediction = self.backbone.predict(self.pred_len, x_enc).view(1, -1, 1).to(x_enc.device)
            elif len(x_enc.shape) == 2:
                B, S = x_enc.shape
                pred_B = []
                for i in range(B):
                    pred_B.append(self.backbone.predict(self.pred_len, x_enc[i]).view(1, -1, 1))
                prediction = torch.cat(pred_B, dim=0).to(x_enc.device)
            elif len(x_enc.shape) == 3:
                B, S, C = x_enc.shape
                pred_B = []
                for i in range(B):
                    pred_C = []
                    for j in range(C):
                        pred_C.append(
                            self.backbone.predict(self.pred_len, x_enc[i, :, j].view(1, -1, 1)).view(1, -1, 1))
                    pred_B.append(torch.cat(pred_C, dim=2))
                prediction = torch.cat(pred_B, dim=0).to(x_enc.device)
            else:
                prediction = self.backbone.predict(self.pred_len, x_enc).view(1, -1, 1)
            # prediction = self.backbone.predict(self.pred_len, x_enc.view(1, -1, 1))
        return prediction
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError
    
    def anomaly_detection(self, x_enc):
        raise NotImplementedError
    
    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if 'forecast' in self.task_name:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            raise NotImplementedError
        return None