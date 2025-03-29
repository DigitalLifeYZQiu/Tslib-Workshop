import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_provider.data_factory import data_provider
from utils.tools import TestTimeaLRAdjust, TestTimeEarlyStopping
import datetime
import os
import time


class ARIMA(nn.Module):
    def __init__(self, p, d, q, args=None):
        """
        p: 自回归阶数
        d: 差分阶数
        q: 移动平均阶数
        """
        super(ARIMA, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.args = args
        # 自回归部分
        self.ar = nn.Linear(p,1, bias=False)
        # 移动平均部分
        self.ma = nn.Linear(q,1, bias=False)
    
    def difference(self, series: torch.Tensor) -> torch.Tensor:
        """
        差分操作
        """
        diff_series = series.clone()
        for _ in range(self.d):
            diff_series = diff_series[:, 1:, :] - diff_series[:, :-1, :]
        return diff_series
    
    def inverse_difference(self, series: torch.Tensor, forecast_diff: torch.Tensor) -> torch.Tensor:
        for _ in range(self.d):
            last_value = series[:, -1, :]
            forecast_diff = torch.cat([series, last_value + torch.cumsum(forecast_diff, dim=1)], dim=1)
        return forecast_diff
    
    def mse_loss(self, y: torch.Tensor):
        B, S, C = y.shape
        e_list = []
        y_pred_list = []
        
        for t in range(S):
            if t > max(self.p, self.q):
                ar_term = self.ar(y[:, t-self.p:t, :].to(y.device).view(1, -1))
                ma_term = self.ma(torch.tensor(e_list[t-self.q:t]).to(y.device).view(1, -1))
            else:
                ar_term = 0
                ma_term = 0
            error = y[:, t, :] - ar_term - ma_term
            pred = ar_term + ma_term
            e_list.append(error)
            
            y_pred_list.append(torch.tensor(pred, dtype=torch.float32, device=y.device).view(1, -1))
        e = torch.cat(e_list, dim=1)
        
        # return torch.mean((y - y_pred) ** 2)
        return torch.mean(e ** 2)
    
    
    def fit(self, series: torch.Tensor, num_epochs=100, lr=0.01):
        y_diff = self.difference(series)
        
        # optimizer = optim.LBFGS(self.parameters(), lr=lr)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            epoch_time = time.time()
            
            # def closure():
            #     optimizer.zero_grad()
            #     loss = self.mse_loss(y_diff)
            #     loss.backward()
            #     return loss
            #
            # optimizer.step(closure)
            optimizer.zero_grad()
            loss = self.mse_loss(y_diff)
            loss.backward()
            optimizer.step()
            
            # 裁剪 self.ar 和 self.ma 的参数
            # with torch.no_grad():
            #     self.ar.weight.data.clamp_(-1, 1)
            #     if self.ar.bias is not None:
            #         self.ar.bias.data.clamp_(-1, 1)
            #     self.ma.weight.data.clamp_(-1, 1)
            #     if self.ma.bias is not None:
            #         self.ma.bias.data.clamp_(-1, 1)
            
            if (epoch + 1) % 1 == 0:
                loss = self.mse_loss(y_diff)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, cost time: {time.time() - epoch_time}')
        print(f"AR parameters - Weight: {self.ar.weight.data}, Bias: {self.ar.bias.data if self.ar.bias is not None else 'None'}")
        print(f"MA parameters - Weight: {self.ma.weight.data}, Bias: {self.ma.bias.data if self.ma.bias is not None else 'None'}")
    
    def predict(self, steps: int = 10, series: torch.Tensor = None) -> torch.Tensor:
        if series is None:
            raise ValueError("Input series has not been given yet.")
        
        y_diff = self.difference(series)
        
        B, S, C = y_diff.shape
        e = torch.zeros((B, S + steps, C), dtype=torch.float32, device=y_diff.device)
        y_full = torch.cat([y_diff, torch.zeros((B, steps, C), dtype=torch.float32, device=y_diff.device)], dim=1)
        
        for t in range(S, S + steps):
            ar_term = self.ar(y_full[:, t - self.p:t, :].view(1, -1))
            ma_term = self.ma(e[:, t - self.q:t, :].view(1, -1))
            
            y_full[:, t, :] = ar_term + ma_term
        forecast_diff = y_full[:, S:, :]
        forecast_original = self.inverse_difference(series, forecast_diff)
        
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
        self.backbone = ARIMA(configs.p, configs.d, configs.q, args=self.args).to(configs.device)
        self.trainfit(self.args)
    
    def trainfit(self, configs):
        self.train_data, self.train_loader = data_provider(configs, flag='train')
        trainset = self.train_data.data_x.squeeze()
        print(f"{configs.model} Fitting Training Set")
        start = datetime.datetime.now()
        device = torch.device('cuda:{}'.format(self.args.gpu))
        self.backbone.fit(torch.Tensor(trainset).view([1, -1, 1]).to(device), num_epochs=configs.train_epochs,
                          lr=configs.learning_rate)
        end = datetime.datetime.now()
        print(f"{configs.model} Training Time: {end - start}")
    
    def testfit(self, series: torch.Tensor):
        y_diff = self.backbone.difference(series)
        # * Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        
        # early_stopping = TestTimeEarlyStopping(patience=self.args.patience, verbose=True)
        epoch_time = time.time()
        for epoch in range(self.args.train_epochs):
            self.backbone.train()
            optimizer.zero_grad()
            loss = self.backbone.mse_loss(y_diff)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.args.train_epochs}], Loss: {loss.item():.4f}, cost time: {time.time() - epoch_time}')
            
            # early_stopping(loss)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            
            # TestTimeaLRAdjust(optimizer, epoch + 1, self.args)
            
        # * L-BFGS optimizer
        # optimizer = optim.
        
        params = {
            'ar': self.backbone.phi,
            'ma': self.backbone.theta,
        }
        print("Fitted Parameters:", params)
    
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
            # self.backbone.fit(x_enc)
            # self.testfit(x_enc)
            #
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            raise NotImplementedError
        return None