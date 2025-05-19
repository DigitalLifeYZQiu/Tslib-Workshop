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
    
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+1, hidden_num, bias=True)
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

# ! Not Rectified Flow
class RectifiedFlow():
  def __init__(self, model=None, num_steps=1000):
    self.model = model
    self.N = num_steps

  def get_train_tuple(self, z0=None, z1=None):
    # z0: the initial random noise
	# z1: the final result
	# z_t: the middle state(linear interpolate)
	# t: random timestep
	# target: the difference between initial state z0 and final state z1
    t = torch.rand((z1.shape[0], 1)).reshape(1,1,1).to(z0.device)
    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0

    return z_t, t, target

  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]

    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1)) * i / N
      pred = self.model(z, t)
      z = z.detach().clone() + pred * dt

      traj.append(z.detach().clone())

    return traj

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
        self.flow_model=MLP(self.q+1, hidden_num=100)
        self.ma = RectifiedFlow(model=self.flow_model, num_steps=100)
        
        if configs.features == 'S':
            self.revin = RevIN(num_features=1, eps=1e-5, affine=True)
        elif configs.features == 'MS' or configs.features == 'M':
            self.revin = RevIN(num_features=configs.enc_in, eps=1e-5, affine=True)
        else:
            self.revin = None
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
    
    
    def ar_forecast(self, series: torch.Tensor) -> torch.Tensor:
        # moving average: Revin
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
    
    def ma_forecast(self, series: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        # todo: flow（MA）输入原始数据分布，建模残差分布。（AR 建模低频特征，MA 拟合高频特征->残差）
        # * Ref: PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers
        rand_error = torch.randn((1,1,1)).to(series.device)
        
        z0 = torch.cat((series, rand_error), dim=1)
        if target is not None:
            z1 = torch.cat((series, target), dim=1)
            z_t, t, target = self.ma.get_train_tuple(z0=z0, z1=z1)
            output = self.ma.model(z_t, t)
        else:
            output = self.ma.model(z0, torch.tensor([[[0]]]).to(series.device))
        # breakpoint()
        return output
    
    # def ma_prediction(self, series: torch.Tensor) -> torch.Tensor:
    #     rand_error = torch.randn((1, 1, 1)).to(series.device)
    #     z0 = torch.cat((series, rand_error), dim=1)
    #     # todo: 补全FLow推理逻辑
    
    def mse_loss(self, y: torch.Tensor):
        B, S, C = y.shape
        e_list = []
        y_pred_list = []
        for t in range(S):
            if t > max(self.p, self.q):
                ar_inp = y[:, t - self.p:t, :].to(y.device)
                ar_pred = self.ar_forecast(series=ar_inp)
                # ma_inp = torch.tensor(e_list[t-self.q: t]).to(y.device)
                ma_inp = y[:, t - self.q:t, :].to(y.device)
                target_error = y[:, t, :].reshape((1,1,1)) - ar_pred[:, -1:, :]
                
                ma_pred = self.ma_forecast(series=ma_inp, target=target_error)
                pred = ar_pred + ma_pred
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
            ar_inp = y_full[:, t - self.p:t, :]
            ar_pred = self.ar_forecast(series=ar_inp)
            # ma_inp = torch.tensor(e_list[t-self.q: t]).to(y.device)
            ma_inp = y_full[:, t - self.q:t, :]
            ma_pred = self.ma_forecast(series=ma_inp)
            pred = ar_pred + ma_pred
            
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