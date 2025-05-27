import torch
import torch.nn as nn
from darts.models import NBEATSModel
from darts import TimeSeries
from data_provider.data_factory import data_provider


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.backbone = NBEATSModel(input_chunk_length=self.seq_len,
                                    output_chunk_length=self.pred_len,
                                    )
        self.fit(configs)
    
    def fit(self, configs):
        self.train_data, self.train_loader = data_provider(configs, flag='train')
        trainset = self.train_data.data_x.squeeze()
        print(f"{configs.model} Fitting Training Set")
        self.backbone.fit(TimeSeries.from_values(trainset))
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = x_enc.to('cpu').detach().numpy().squeeze()
        ts_input = TimeSeries.from_values(x_enc)
        prediction = self.backbone.predict(self.pred_len)
        return torch.Tensor(prediction.values()).reshape((1, self.pred_len, 1))
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError
    
    def anomaly_detection(self, x_enc):
        raise NotImplementedError
    
    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            raise NotImplementedError
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError
        if self.task_name == 'classification':
            raise NotImplementedError
        return None