import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
import math

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class conv_downsample(nn.Module):
    def __init__(self, configs):
        super(conv_downsample, self).__init__()
        self.configs = configs
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=scale, padding=scale // 2),
                nn.GELU()
            ) for scale in scales
        ])
    def forward(self, x):
        B,T,C = x.shape
        input_list = []
        input_list.append(x)
        while(math.floor(T/2)!=0):
            input_list.append()
# 多分辨率时间成像模块
class MultiResolutionTimeImaging(nn.Module):
    def __init__(self, configs):
        super(MultiResolutionTimeImaging, self).__init__()
        self.configs = configs
        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

    def forward(self, x):
        
        x = torch.fft.rfft(x, dim=1)  # 简单示例，进行傅里叶变换
        return x


