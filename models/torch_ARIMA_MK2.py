import torch
import torch.nn as nn
import torch.optim as optim
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
        # self.phi = nn.Parameter(torch.tensor(0.5 * torch.ones(p), dtype = torch.float32))
        self.ar = nn.Linear(p,1)
        # 移动平均部分
        # self.theta = nn.Parameter(torch.tensor(0.5 * torch.ones(q), dtype = torch.float32))
        self.ma = nn.Linear(q,1)
    
    def difference(self, series: torch.Tensor) -> torch.Tensor:
        """
        差分操作
        """
        diff_series = series.clone()
        for _ in range(self.d):
            diff_series = diff_series[:, 1:, :] - diff_series[:, :-1, :]
        return diff_series
    
    def forward(self, x, steps):
        # 存储预测结果
        predictions = []
        # 差分
        diff_x = self.difference(x)
        # 假设误差初始化为 0
        B,S,C = diff_x.shape
        e = torch.zeros((B, S, C), dtype=torch.float32, device=x.device)
        
        for _ in range(steps):
            # 自回归部分
            ar_input = diff_x[:, -self.p:, :]
            ar_output = self.ar(ar_input.view(1, -1)).squeeze()
            # 移动平均部分
            if self.q > 0:
                ma_input = e[:, -self.q:, :]
                ma_output = self.ma(ma_input.view(1, -1)).squeeze()
            else:
                ma_output = torch.tensor(0.0)
            # 预测值
            pred = ar_output + ma_output
            # 存储预测值
            predictions.append(pred)
            # 更新差分序列和误差序列
            diff_x = torch.cat((diff_x, pred.unsqueeze(0)), dim=1)
            # 计算新的误差
            new_error = pred - ar_output - ma_output
            e = torch.cat((e, new_error.unsqueeze(0)), dim=1)
        
        # 对预测值进行逆差分
        predictions = torch.stack(predictions)
        final_predictions = self.inverse_difference(predictions, x, self.d)[:, -steps:, :]
        return final_predictions
    
    def reconstruct_original_series(self, series: torch.Tensor, forecast_diff: torch.Tensor) -> torch.Tensor:
        B,S,C = series.shape
        
        last_value = series[:, -1, :]
        forecast_original = torch.cat([series, last_value + torch.cumsum(forecast_diff, dim=1)], dim=1)
        return forecast_original


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
        self.criterion = nn.MSELoss()
        
    
    def trainfit(self, configs):
        self.train_data, self.train_loader = data_provider(configs, flag='train')
        trainset = self.train_data.data_x.squeeze()
        print(f"{configs.model} Fitting Training Set")
        start = datetime.datetime.now()
        device = torch.device('cuda:{}'.format(self.args.gpu))
        self.backbone.fit(torch.Tensor(trainset).view([1, -1, 1]).to(device), num_epochs=configs.train_epochs, lr=configs.learning_rate)
        end = datetime.datetime.now()
        print(f"{configs.model} Training Time: {end - start}")
        
    def testfit(self, series: torch.Tensor):
        path = os.path.join(self.args.checkpoints, self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        early_stopping = TestTimeEarlyStopping(patience=self.args.patience, verbose=True)
        epoch_time = time.time()
        for epoch in range(self.args.train_epochs):
            self.backbone.train()
            optimizer.zero_grad()
            pred = self.backbone(series, self.pred_len)
            loss = self.backbone.mse_loss(y_diff)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.args.train_epochs}], Loss: {loss.item():.4f}, cost time: {time.time() - epoch_time}')
            
            early_stopping(loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # TestTimeaLRAdjust(optimizer, epoch + 1, self.args)
        
        params = {
            'ar': self.backbone.phi,
            'ma': self.backbone.theta,
        }
        # print("Fitted Parameters:", params)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.backbone.eval()
        with torch.no_grad():
            if len(x_enc.shape)==1:
                prediction = self.backbone.predict(self.pred_len, x_enc).view(1, -1, 1).to(x_enc.device)
            elif len(x_enc.shape)==2:
                B,S = x_enc.shape
                pred_B = []
                for i in range(B):
                    pred_B.append(self.backbone.predict(self.pred_len, x_enc[i]).view(1, -1, 1))
                prediction = torch.cat(pred_B, dim=0).to(x_enc.device)
            elif len(x_enc.shape)==3:
                B,S,C = x_enc.shape
                pred_B = []
                for i in range(B):
                    pred_C = []
                    for j in range(C):
                        pred_C.append(self.backbone.predict(self.pred_len, x_enc[i, :, j].view(1, -1, 1)).view(1, -1, 1))
                    pred_B.append(torch.cat(pred_C, dim=2))
                prediction = torch.cat(pred_B, dim=0).to(x_enc.device)
            else:
                prediction = self.backbone.predict(self.pred_len, x_enc)
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
            self.trainfit(self.args)
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            raise NotImplementedError
        return None