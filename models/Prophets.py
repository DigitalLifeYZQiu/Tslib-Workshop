import torch
import torch.nn as nn
from darts.models import Prophet
from darts import TimeSeries
from data_provider.data_factory import data_provider
import pandas as pd
import numpy as np

class Model(nn.Module):
    """
    Prophet wrapper for Darts, adapted for Dataset_Custom output.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.features = getattr(configs, "features", "S")
        self.freq = getattr(configs, "freq", "h")
        self.backbone = Prophet()
        self.fit(configs)

    def fit(self, configs):
        self.train_data, self.train_loader = data_provider(configs, flag='train')
        batch = next(iter(self.train_loader))
        seq_x, _, _, _ = batch
        seq_x = seq_x[0]
        seq_len = seq_x.shape[0]

        times = pd.date_range(start='2000-01-01', periods=seq_len, freq=self.freq)

        df = pd.DataFrame({
            "ds": times,
            "y": seq_x.squeeze()
        })

        ts = TimeSeries.from_dataframe(df, time_col="ds", value_cols="y")

        print(f"{configs.model} Fitting Training Set")
        self.backbone.fit(ts)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = x_enc[0]
        seq_len = x_enc.shape[0]
        times = pd.date_range(start='2025-02-14', periods=seq_len, freq=self.freq)
        df = pd.DataFrame({
            "ds": times,
            "y": x_enc.cpu().numpy().squeeze()
        })
        ts_history = TimeSeries.from_dataframe(df, time_col="ds", value_cols="y")
        prediction = self.backbone.predict(self.pred_len).values()
        return torch.tensor(prediction, dtype=torch.float32).reshape((1, self.pred_len, 1))

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