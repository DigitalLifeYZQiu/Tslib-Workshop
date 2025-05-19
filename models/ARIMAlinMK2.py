import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
import torch.optim as optim
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


class Model(nn.Module):
    def __init__(self, configs):
        """
        p: 自回归阶数
        d: 差分阶数
        q: 移动平均阶数
        """
        super(Model, self).__init__()
        self.p = configs.p
        self.d = configs.d
        self.q = configs.q
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        
        # 自回归部分
        self.ar_trend = nn.Linear(self.seq_len,self.pred_len, bias=True)
        self.ar_seasonal = nn.Linear(self.seq_len,self.pred_len, bias=True)
        # 移动平均部分
        # self.ma = nn.Linear(q,1, bias=False)
        if configs.features == 'S':
            self.ma = RevIN(num_features=1, eps=1e-5, affine=True)
        elif configs.features == 'MS' or configs.features == 'M':
            self.ma = RevIN(num_features=configs.enc_in, eps=1e-5, affine=False)
        else:
            self.ma = None
            raise ValueError(f"Unrecognized features type: {configs.features}")
    
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
        return forecast_diff[:, :, -self.pred_len:]
    
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if 'forecast' in self.task_name:
            dec_out = self.forecast(series=x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            raise NotImplementedError
        return None