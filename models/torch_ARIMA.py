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
        # self.phi = nn.Parameter(torch.tensor(0.5 * torch.ones(p), dtype=torch.float32))
        self.phi = nn.Parameter(torch.Tensor(p,1))
        nn.init.xavier_uniform_(self.phi)
        # 移动平均部分
        # self.theta = nn.Parameter(torch.tensor(0.5 * torch.ones(q), dtype=torch.float32))
        self.theta = nn.Parameter(torch.Tensor(q,1))
        nn.init.xavier_uniform_(self.theta)
    
    def difference(self, series: torch.Tensor) -> torch.Tensor:
        """
        差分操作
        """
        diff_series = series.clone()
        for _ in range(self.d):
            diff_series = diff_series[:, 1:, :] - diff_series[:, :-1, :]
        return diff_series
    
    def mse_loss(self, y: torch.Tensor):
        # n = len(y)
        # n = y.shape[1]
        B, S, C = y.shape
        e = torch.zeros((B, S, C), dtype=torch.float32, device=y.device)
        y_pred = torch.zeros((B, S, C), dtype=torch.float32, device=y.device)
        
        for t in range(S):
            ar_term = 0
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_term += self.phi[i] * y[:, t - i - 1, :]
            
            ma_term = 0
            for j in range(self.q):
                if t - j - 1 >= 0:
                    ma_term += self.theta[j] * e[:, t - j - 1, :]
            e[:, t, :] = y[:, t, :] - ar_term - ma_term
            y_pred[:, t, :] = ar_term + ma_term
        
        return torch.mean((y - y_pred) ** 2)
    
    def estimate_arima_parameters(self, y: torch.Tensor, num_epochs=100, lr=0.01):
        # path = os.path.join(self.args.checkpoints, self.args.setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # early_stopping = TestTimeEarlyStopping(patience=self.args.patience, verbose=True)
        epoch_time = time.time()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.mse_loss(y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, cost time: {time.time() - epoch_time}')
            
            # early_stopping(loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            
            # TestTimeaLRAdjust(optimizer, epoch + 1, self.args)
        
        params = {
            'ar': self.phi,
            'ma': self.theta,
        }
        return params
    
    def forecast_arima(self, y: torch.Tensor, steps: int = 10) -> torch.Tensor:
        B, S, C = y.shape
        e = torch.zeros((B, S + steps, C), dtype=torch.float32, device=y.device)
        y_full = torch.cat([y, torch.zeros((B, steps, C), dtype=torch.float32, device=y.device)], dim=1)
        
        for t in range(S, S + steps):
            ar_term = 0
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_term += self.phi[i] * y_full[:, t - i - 1, :]
            
            ma_term = 0
            for j in range(self.q):
                if t - j - 1 >= 0:
                    ma_term += self.theta[j] * e[:, t - j - 1, :]
            
            y_full[:, t, :] = ar_term + ma_term
        
        return y_full[:, S:, :]
    
    def reconstruct_original_series(self, series: torch.Tensor, forecast_diff: torch.Tensor) -> torch.Tensor:
        B, S, C = series.shape
        
        last_value = series[:, -1, :]
        forecast_original = torch.cat([series, last_value + torch.cumsum(forecast_diff, dim=1)], dim=1)
        return forecast_original
    
    def fit(self, series: torch.Tensor, num_epochs=100, lr=0.01):
        y_diff = self.difference(series)
        params = self.estimate_arima_parameters(y_diff, num_epochs, lr)
        print("Fitted Parameters:", params)
    
    #
    # def forward(self, x):
    #     # 差分
    #     diff_x = self.difference(x)
    #     # 自回归部分
    #     ar_input = diff_x[-self.p:]
    #     ar_output = self.ar(ar_input.view(1, -1))
    #     # 移动平均部分
    #     if self.q > 0:
    #         ma_input = diff_x[-self.q:]
    #         ma_output = self.ma(ma_input.view(1, -1))
    #     else:
    #         ma_output = torch.tensor([0.0])
    #     # 预测值
    #     pred = ar_output + ma_output
    #     # 逆差分
    #     pred = self.inverse_difference(pred, x)
    #     return pred
    
    def predict(self, steps: int = 10, series: torch.Tensor = None) -> torch.Tensor:
        if series is None:
            raise ValueError("Input series has not been given yet.")
        
        y_diff = self.difference(series)
        forecast_diff = self.forecast_arima(y_diff, steps)
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
        
        # params = self.estimate_arima_parameters(y_diff, num_epochs, lr)
        
        # path = os.path.join(self.args.checkpoints, self.args.setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
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

# # 示例使用
# # 生成一些示例数据
# data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
#
# # 初始化 ARIMA 模型
# p, d, q = 2, 1, 1
# model = ARIMA(p, d, q)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # 训练模型
# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     # 预测
#     pred = model(data[:-1])
#     # 计算损失
#     loss = criterion(pred, data[-1].view(1))
#     # 反向传播和优化
#     loss.backward()
#     optimizer.step()
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 进行预测
# with torch.no_grad():
#     next_pred = model.predict(data, steps=96)
#     print(f'Next predicted value: {next_pred}')