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
import argparse
from typing import Tuple, Optional

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


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)
    
    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1).to(t.device)
        inputs = inputs.permute(0, 2, 1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = x.permute(0, 2, 1)
        
        return x


class FlowModel():
    def __init__(self, model : Optional[nn.Module] = None, num_steps : int =10):
        self.model = model
        self.N = num_steps
        self.loss_fn = nn.MSELoss()
    
    def flow_loss(self, x_0 : torch.Tensor, x_1 : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # x_1 is what you want
        '''
        x_0(B,seq_len+pred_len,C): initial state(noise and condition)
        x_1(B,seq_len+pred_len,C): target state(residual and condition)
        '''
        # x_0 = torch.randn_like(x_1) # dim = x_1
        B,S,C = x_0.shape
        t = torch.rand(B, 1, C).to(x_0.device)
        target = x_1 - x_0
        x_t = (1 - t) * x_0 + t * target# ! x1
        # breakpoint()
        model_out = self.model(x_input=x_t, t=t)
        flowloss = self.loss_fn(model_out, target)
        return model_out, flowloss
    
    def sample(self, noise : torch.Tensor) -> torch.Tensor:
        r'''
        noise(B,seq_len+pred_len,C): The initial state (input original concat random noise)
        '''
        x = noise
        dt = 1.0 / self.N
        for i in range(self.N):
            t = (torch.ones((x.shape[0], 1, x.shape[2])) * i / self.N).to(x.device)
            pred = self.model(x_input=x, t=t)
            x = x + (pred - noise) * dt# !不要减去 noise
        return x
  
class Model(nn.Module):
    def __init__(self, configs : argparse.Namespace):
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
        self.flow_model = MLP(self.seq_len + self.pred_len, hidden_num=100)
        self.ma = FlowModel(model=self.flow_model, num_steps=configs.sample_steps)
        
        # Instance Normalize
        if configs.features == 'S':
            self.revin = RevIN(num_features=1, eps=1e-5, affine=True)
        elif configs.features == 'MS' or configs.features == 'M':
            self.revin = RevIN(num_features=configs.enc_in, eps=1e-5, affine=False)
        else:
            self.revin = None
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
    
    def ar_forecast(self, series: torch.Tensor) -> torch.Tensor:
        series = self.revin(series, 'norm')
        
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
        
        output = self.revin(output, 'denorm')
        
        return output
    
    def ma_forecast(self, series: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # todo: flow（MA）输入原始数据分布，建模残差分布。（AR 建模低频特征，MA 拟合高频特征->残差）
        # * Ref: PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers
        r'''
        series: torch.Tensor (B,seq_len,C) - The input original data
        target: torch.Tensor (B,seq_len,C) - The target residual
        '''
        B,S,C = series.shape
        rand_error = torch.randn((B,self.pred_len,C)).to(series.device) # generate random noise for initial state (B,seq_len,C)
        x0 = torch.cat((series, rand_error), dim=1).to(series.device) # initial state: series(B,seq_len,C）concat noise(B,pred_len,C)
        if target is not None: # target is not None while training
            x1 = torch.cat((series, target), dim=1).to(series.device) # final state: series(B,seq_len,C) concat residual(B,pred_len,C)
            #! 只采样一次不合理，应该分别训练或联合优化
            vector, flowloss = self.ma.flow_loss(x_0=x0, x_1=x1) # `vector` is the learned vector pointing from x0 to x1; `flowloss` is the loss of vector field
            output = x0 + vector # sample once for training forward process
        else: # target is None while validating and testing
            output = self.ma.sample(x0) # sample self.ma.N times for validating and testing forward process
            flowloss = torch.tensor(0.).view(1,1,1).to(series.device) # flowloss cannot be calculated without target, set default to 0
        return output, flowloss
    
    def forecast(self, series: torch.Tensor, label: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ar_pred = self.ar_forecast(series)[:, -self.pred_len:, :]
        if label is not None:
            # breakpoint()
            residual = label[:, -self.pred_len:, :] - ar_pred
            ma_pred, flowloss = self.ma_forecast(series, target=residual)
            prediction = ar_pred + ma_pred[:, -self.pred_len:, :]
        else:
            prediction = ar_pred
            flowloss = torch.tensor(0.).view(1,1,1).to(series.device)
        return prediction, flowloss
        

    def forward(self,
                x_enc : torch.Tensor,
                x_mark_enc : Optional[torch.Tensor],
                x_dec : Optional[torch.Tensor],
                x_mark_dec : Optional[torch.Tensor],
                label : Optional[torch.Tensor] = None,
                mask : Optional[torch.Tensor] = None):
        if 'forecast' in self.task_name:
            dec_out, flowloss = self.forecast(series=x_enc, label=label)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            raise NotImplementedError
        return None