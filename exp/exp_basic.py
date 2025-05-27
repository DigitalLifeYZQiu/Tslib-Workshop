import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    TimeBert, ARIMA, AutoARIMA, DeepARIMA, LinearRegression, ExponentialSmoothing, Theta, KalmanFilter, RandomForest, \
    XGBoost, LightGBM, ARIMA_tta, AutoARIMA_tta, DeepARIMA_tta, ARIMA_plus, ARIMA_plus_tta, moment, TimeMixerPP, \
    torch_ARIMA, torch_ARIMA_MSE, torch_ARIMA_NLL, torch_ARIMA_BFGS, torch_ARIMA_nnModule, torch_ARI_nnModule, \
    ARIMAppMK1, NBEATS, Prophets,\
    ARIMAlinMK1, ARIMAppMK2, ARIMAlinMK2, ARIMAppMK3, ARIMAppMK4, ARIMAlinMK3


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'TimeBert': TimeBert,
            'Moment': moment,
            'TimeMixerPP': TimeMixerPP,
            'ARIMAlinMK1': ARIMAlinMK1,
            'ARIMAlinMK2': ARIMAlinMK2,
            'ARIMAlinMK3': ARIMAlinMK3,
            
        }
        self.statistical_model_dict = {
            'ARIMA': ARIMA,
            'ARIMAppMK1': ARIMAppMK1,
            'ARIMAppMK2': ARIMAppMK2,
            'ARIMAppMK3': ARIMAppMK3,
            'ARIMAppMK4': ARIMAppMK4,
            'torchARIMA': torch_ARIMA,
            'torch_ARIMA_MSE': torch_ARIMA_MSE,
            'torch_ARIMA_NLL': torch_ARIMA_NLL,
            'torch_ARIMA_BFGS': torch_ARIMA_BFGS,
            'torch_ARIMA_nnModule': torch_ARIMA_nnModule,
            'torch_ARI_nnModule': torch_ARI_nnModule,
            'AutoARIMA': AutoARIMA,
            'DeepARIMA': DeepARIMA,
            'LinearRegression': LinearRegression,
            'ExponentialSmoothing': ExponentialSmoothing,
            'Theta': Theta,
            'KalmanFilter': KalmanFilter,
            'RandomForest': RandomForest,
            'XGBoost': XGBoost,
            'LightGBM': LightGBM,
            'NBEATS': NBEATS,
            'Prophets': Prophets,
            'ARIMA_tta': ARIMA_tta,
            'AutoARIMA_tta': AutoARIMA_tta,
            'DeepARIMA_tta': DeepARIMA_tta,
            'ARIMA_plus': ARIMA_plus,
            'ARIMA_plus_tta': ARIMA_plus_tta,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        print('device:', self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
