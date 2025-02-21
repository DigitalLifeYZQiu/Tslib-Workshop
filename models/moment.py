import logging
import warnings

import math
from math import ceil
from typing import Optional
from argparse import Namespace

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class TASKS:
    PRETRAINING: str = "pre-training"
    LONG_HORIZON_FORECASTING: str = "long-horizon-forecasting"
    SHORT_HORIZON_FORECASTING: str = "short-horizon-forecasting"
    CLASSIFICATION: str = "classification"
    IMPUTATION: str = "imputation"
    ANOMALY_DETECTION: str = "anomaly-detection"
    EMBED: str = "embed"


@dataclass
class TimeseriesOutputs:
    forecast: npt.NDArray = None
    anomaly_scores: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False


class Masking:
    def __init__(
        self, mask_ratio: float = 0.3, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Indices with 0 mask are hidden, and with 1 are observed.
        """
        self.mask_ratio = mask_ratio
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride

    @staticmethod
    def convert_seq_to_patch_view(
        mask: torch.Tensor, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x seq_len]
        Output
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        stride = patch_len if stride is None else stride
        mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
        # mask : [batch_size x n_patches x patch_len]
        return (mask.sum(dim=-1) == patch_len).long()

    @staticmethod
    def convert_patch_to_seq_view(
        mask: torch.Tensor,
        patch_len: int = 8,
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        return mask.repeat_interleave(patch_len, dim=-1)

    def generate_mask(self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_len] or
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len] or
            [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        if x.ndim == 4:
            return self._mask_patch_view(x, input_mask=input_mask)
        elif x.ndim == 3:
            return self._mask_seq_view(x, input_mask=input_mask)

    def _mask_patch_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        input_mask = self.convert_seq_to_patch_view(
            input_mask, self.patch_len, self.stride
        )  # [B, N]
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True)  # batch_size x 1

        batch_size, _, n_patches, _ = x.shape
        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(
            batch_size, n_patches, device=x.device
        )  # noise in [0, 1], batch_size x n_channels x n_patches
        noise = torch.where(
            input_mask == 1, noise, torch.ones_like(noise)
        )  # only keep the noise of observed patches

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(
            ids_shuffle, dim=1
        )  # ids_restore: [batch_size x n_patches]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros(
            [batch_size, n_patches], device=x.device
        )  # mask: [batch_size x n_patches]
        for i in range(batch_size):
            mask[i, : len_keep[i]] = 1

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.long()

    def _mask_seq_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        mask = self._mask_patch_view(x, input_mask=input_mask)
        return self.convert_patch_to_seq_view(mask, self.patch_len).long()


class NamespaceWithDefaults(Namespace):
    @classmethod
    def from_namespace(cls, namespace):
        new_instance = cls()
        for attr in dir(namespace):
            if not attr.startswith("__"):
                setattr(new_instance, attr, getattr(namespace, attr))
        return new_instance

    def getattr(self, key, default=None):
        return getattr(self, key, default)


def get_huggingface_model_dimensions(model_name: str = "flan-t5-base"):
    from transformers import T5Config

    config = T5Config.from_pretrained(model_name)
    return config.d_model


def get_anomaly_criterion(anomaly_criterion: str = "mse"):
    if anomaly_criterion == "mse":
        return torch.nn.MSELoss(reduction="none")
    elif anomaly_criterion == "mae":
        return torch.nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Anomaly criterion {anomaly_criterion} not supported.")


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, model_name="MOMENT"):
        super(PositionalEmbedding, self).__init__()
        self.model_name = model_name

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if (
            self.model_name == "MOMENT"
            or self.model_name == "TimesNet"
            or self.model_name == "GPT4TS"
        ):
            return self.pe[:, : x.size(2)]  # d_model
        else:
            return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        seq_len: int = 512,
        patch_len: int = 8,
        stride: int = 8,
        dropout: int = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
    ):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        # nn.init.trunc_normal_(self.mask_embedding, mean=0.0, std=.02)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()
            # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails

        # Positional embedding
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
            x : [batch_size x n_channels x n_patches x patch_len]
            mask : [batch_size x seq_len]
        Output:
            x : [batch_size x n_channels x n_patches x d_model]
        """
        # 修改mask为[B,M,L],按照patch=24来做了mask
        mask = mask.view(mask.size(0), mask.size(1), -1, self.patch_len)

        mask = mask[:, :, :, 0]  # [B, M, N, 1]
        mask = mask.unsqueeze(-1)  # [B, M, N, 1]
        # mask = Masking.convert_seq_to_patch_view(
        #     mask, patch_len=self.patch_len
        # ).unsqueeze(-1)
        # mask : [batch_size x n_patches x 1]
        n_channels = x.shape[1]
        mask = mask.repeat_interleave(self.d_model, dim=-1)  # [B, M, N, D]
        # mask : [batch_size x n_channels x n_patches x d_model]

        # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len  # 8
        self.stride = stride  # 8
        if self.stride != self.patch_len:
            warnings.warn(
                "Stride and patch length are not equal. \
                          This may lead to unexpected behavior."
            )

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x : [batch_size x n_channels x num_patch x patch_len]
        return x


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
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

    def forward(self, x: torch.Tensor, mode: str = "norm", mask: torch.Tensor = None):
        """
        :param x: input tensor of shape (batch_size, n_channels, seq_len)
        :param mode: 'norm' or 'denorm'
        :param mask: input mask of shape (batch_size, seq_len)
        :return: RevIN transformed tensor
        """
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x, mask=None):
        """
        x    : batch_size x n_channels x seq_len
        mask : batch_size x seq_len
        """
        # 修改mask 为[B,M,L]，按照patch=24来做了mask
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]))
        # n_channels = x.shape[1]
        # mask = (
        #     mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
        # )  # [batch_size x n_channels x seq_len]
        # Set masked positions to NaN, and unmasked positions are taken from x

        # 转mask为bool
        mask = mask.to(dtype=torch.bool)

        masked_x = torch.where(mask, x, torch.nan)
        self.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
        self.stdev = nanstd(masked_x, dim=-1, keepdim=True).detach() + self.eps
        # self.stdev = torch.sqrt(
        #     torch.var(masked_x, dim=-1, keepdim=True) + self.eps).get_data().detach()
        # NOTE: By default not bessel correction

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
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


SUPPORTED_HUGGINGFACE_MODELS = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x seq_len], where seq_len = n_patches * patch_len
        """
        # x = x.transpose(2, 3)                 # [batch_size x n_channels x n_patches x d_model]
        x = self.linear(
            self.dropout(x)
        )  # [batch_size x n_channels x n_patches x patch_len]
        x = x.flatten(start_dim=2, end_dim=3)  # [batch_size x n_patches x seq_len]
        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_channels * d_model, n_classes)

    def forward(self, x, input_mask: torch.Tensor = None):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_classes]
        """
        x = x.nanmean(
            dim=-1
        ).squeeze()  # x: batch_size x n_channels x n_patches x d_model
        x = self.flatten(x)  # x: batch_size x n_channels * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: batch_size x n_classes
        return y


class ForecastingHead(nn.Module):
    def __init__(
        self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x forecast_horizon]
        """
        x = self.flatten(x)  # x: batch_size x n_channels x n_patches x d_model
        x = self.linear(x)  # x: batch_size x n_channels x n_patches*d_model
        x = self.dropout(x)  # x: batch_size x n_channels x forecast_horizon
        return x


class MOMENT(nn.Module):
    def __init__(self, configs: Namespace, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        configs = self._validate_inputs(configs)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len

        # Normalization, patching and embedding
        self.normalizer = RevIN(
            num_features=1, affine=configs.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=configs.patch_len, stride=configs.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            seq_len=configs.seq_len,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            add_positional_embedding=configs.getattr("add_positional_embedding", True),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
        )
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0))

        # Transformer backbone
        self.encoder = self._get_transformer_backbone(configs)

        # Prediction Head
        self.head = self._get_head(self.task_name)

    def _update_inputs(self, configs: Namespace, **kwargs) -> NamespaceWithDefaults:
        if isinstance(configs, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**configs, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(configs)

    def _validate_inputs(self, configs: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            configs.transformer_backbone == "PatchTST"
            and configs.transformer_type != "encoder_only"
        ):
            warnings.warn("PatchTST only supports encoder-only transformer backbones.")
            configs.transformer_type = "encoder_only"
        if (
            configs.transformer_backbone != "PatchTST"
            and configs.transformer_backbone not in SUPPORTED_HUGGINGFACE_MODELS
        ):
            raise NotImplementedError(
                f"Transformer backbone {configs.transformer_backbone} not supported."
                f"Please choose from {SUPPORTED_HUGGINGFACE_MODELS} or PatchTST."
            )
        if (
            configs.d_model is None
            and configs.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            configs.d_model = get_huggingface_model_dimensions(
                configs.transformer_backbone
            )
            logging.info("Setting d_model to {}".format(configs.d_model))
        elif configs.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone \
                             unless transformer backbone is a Huggingface model."
            )

        if configs.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of ['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if configs.patch_stride_len != configs.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return configs

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name in {
            TASKS.PRETRAINING,
            TASKS.ANOMALY_DETECTION,
            TASKS.IMPUTATION,
        } or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode == "zero-shot"
        ):
            return PretrainHead(
                self.configs.d_model,
                self.configs.patch_len,
                self.configs.getattr("dropout", 0.1),
                self.configs.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.configs.n_channels,
                self.configs.d_model,
                self.configs.num_class,
                self.configs.getattr("dropout", 0.1),
            )
        elif (task_name == TASKS.LONG_HORIZON_FORECASTING) or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode != "zero-shot"
        ):
            num_patches = (
                max(self.configs.seq_len, self.configs.patch_len)
                - self.configs.patch_len
            ) // self.configs.patch_stride_len + 1
            self.head_nf = self.configs.d_model * num_patches
            return ForecastingHead(
                self.head_nf,
                self.configs.forecast_horizon,
                self.configs.getattr("head_dropout", 0.1),
            )
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, configs):
        if configs.transformer_backbone == "PatchTST":
            return self._get_patchtst_encoder(configs)
        else:
            return self._get_huggingface_transformer(configs)

    def _get_huggingface_transformer(self, configs):
        from transformers import T5Config, T5EncoderModel, T5Model

        if configs.getattr("randomly_initialize_backbone", False):
            model_config = T5Config.from_pretrained(configs.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(
                f"Initializing randomly initialized\
                          transformer from {configs.transformer_backbone}."
            )
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                configs.transformer_backbone
            )
            logging.info(
                f"Initializing pre-trained \
                          transformer from {configs.transformer_backbone}."
            )

        if configs.transformer_type == "encoder_only":
            transformer_backbone = transformer_backbone.get_encoder()
        elif configs.transformer_type == "decoder_only":
            transformer_backbone = transformer_backbone.get_decoder()

        if configs.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def embed(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask  : [batch_size x 1 x seq_len]
        """

        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )

        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.configs.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)
        elif reduction == "none":
            raise NotImplementedError

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata=reduction
        )

    def pretraining(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        # 修改x_enc和mask为[B,L,M],按照patch=24来做了mask
        # 先转置

        x_enc = x_enc.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        input_mask = input_mask.permute(0, 2, 1)

        batch_size, n_channels, _ = x_enc.shape

        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

        # Normalization
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # x_enc = self.normalizer(x=x_enc, mask=input_mask, mode='norm')
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        # Some time-series are too short, so masking them out results in NaNs.

        # [batch_size x n_channels x seq_len]
        x_enc = self.tokenizer(x=x_enc)
        # [batch_size x n_channels x n_patches x patch_len]

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        # Encoder
        # attention_mask = Masking.convert_seq_to_patch_view(
        #     input_mask, self.patch_len
        # ).repeat_interleave(n_channels, dim=0)
        # [B*M,N]

        # 此时input_mask为[B,M,L]，要转为[B*M,N]
        attention_mask = input_mask.reshape(
            input_mask.size(0), input_mask.size(1), -1, self.patch_len
        )  # [B,M,N,P]
        attention_mask = attention_mask[:, :, :, 0]  # [B,M,N]
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))  # [B*M,N]
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]
        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        illegal_output = (
            self._check_model_weights_for_illegal_values()
            if self.configs.debug
            else None
        )

        # 再次转置
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def initialize_soft_prompt(self, **kwargs):
        n_soft_prompt_tokens = self.configs.n_soft_prompt_tokens
        self.soft_prompt = nn.Embedding(n_soft_prompt_tokens, self.configs.d_model)
        return self.soft_prompt

    def _cat_learned_embedding_to_input(self, prompt_embeds, enc_in) -> torch.Tensor:
        prompt_embeds = prompt_embeds.repeat(enc_in.size(0), 1, 1)
        enc_in = torch.cat([prompt_embeds, enc_in], dim=1)
        return enc_in

    def _extend_attention_mask(self, attention_mask, n_tokens):
        n_batches = attention_mask.shape[0]
        extension = torch.full((n_batches, n_tokens), 1).to(self.configs.device)
        return torch.cat([extension, attention_mask], dim=1)

    def reconstruct(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        if mask is None:
            mask = torch.ones_like(input_mask)

        batch_size, n_channels, _ = x_enc.shape
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        attention_mask = (
            Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
            .repeat_interleave(n_channels, dim=0)
            .to(x_enc.device)
        )

        n_tokens = 0
        if "prompt_embeds" in kwargs:
            prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

            if isinstance(prompt_embeds, nn.Embedding):
                prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

            n_tokens = prompt_embeds.shape[1]

            enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)

            attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

        # Encoder
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out[:, n_tokens:, :]

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

    def detect_anomalies(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        anomaly_criterion: str = "mse",
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        anomaly_criterion : str
        """
        outputs = self.reconstruct(x_enc=x_enc, input_mask=input_mask)
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=outputs.reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )

    def long_forecast(
        self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, **kwargs
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        """
        batch_size, n_channels, _ = x_enc.shape

        # Normalization
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )

        # Encoder
        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x forecast_horizon]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)

    def short_forecast(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        forecast_horizon: int = 1,
        **kwargs,
    ):
        # mask would be mask tokens which are attended to
        # and input_mask is typically unattended

        """
        x_enc : [batch_size x n_channels x seq_len]
        input_mask : [batch_size x seq_len]
        forecast_horizon : int
        """
        # Min-max scale input time-series, based on "Meta-learning
        # framework with applications to zero-shot time-series forecasting
        # scaler = torch.max(x_enc, dim=-1, keepdim=True)[0]
        # x_enc = x_enc / scaler

        batch_size, n_channels, seq_len = x_enc.shape
        frequency = kwargs["frequency"] if "frequency" in kwargs else None
        # NOTE: Add series decomposition

        num_masked_patches = ceil(forecast_horizon / self.patch_len)
        num_masked_timesteps = num_masked_patches * self.patch_len

        # Normalization
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        # Shift the time-series and mask the last few timesteps for forecasting
        x_enc = torch.roll(x_enc, shifts=-num_masked_timesteps, dims=2)
        input_mask = torch.roll(input_mask, shifts=-num_masked_timesteps, dims=1)

        # Mixed results
        # Attending to mask tokens
        input_mask[:, -num_masked_timesteps:] = 1
        mask = torch.ones_like(input_mask)
        mask[:, -num_masked_timesteps:] = 0

        # Unattending to mask tokens
        # input_mask[:, -num_masked_timesteps:] = 0
        # mask = torch.ones_like(input_mask)

        # Tokenize
        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        # Encoder
        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        end = -num_masked_timesteps + forecast_horizon
        end = None if end == 0 else end

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        forecast = dec_out[:, :, -num_masked_timesteps:end]

        # Rescale the forecast
        # forecast = forecast * scaler
        # dec_out = dec_out * scaler

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            forecast=forecast,
            metadata={"forecast_horizon": forecast_horizon},
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor = None,
        x_dec: torch.Tensor = None,
        x_mark_dec: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        if self.task_name == TASKS.PRETRAINING or self.task_name == TASKS.IMPUTATION or self.task_name==TASKS.ANOMALY_DETECTION:
            return self.pretraining(
                x_enc=x_enc, mask=mask, input_mask=torch.ones_like(mask), **kwargs
            )
        # elif (
        #     self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        #     and self.configs.finetuning_mode == "zero-shot"
        # ):
        #     return self.short_forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
        # elif self.task_name == TASKS.LONG_HORIZON_FORECASTING or (
        #     self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        #     and self.configs.finetuning_mode != "zero-shot"
        # ):
        #     return self.long_forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
        # elif self.task_name == TASKS.ANOMALY_DETECTION:
        #     return self.detect_anomalies(x_enc=x_enc, input_mask=input_mask, **kwargs)
        # else:
        #     raise NotImplementedError(f"Task {self.task_name} not implemented.")
        return

    def _check_model_weights_for_illegal_values(self):
        illegal_encoder_weights = (
            torch.stack([torch.isnan(p).any() for p in self.encoder.parameters()])
            .any()
            .item()
        )
        illegal_head_weights = (
            torch.stack([torch.isnan(p).any() for p in self.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_weights = (
            torch.stack(
                [torch.isnan(p).any() for p in self.patch_embedding.parameters()]
            )
            .any()
            .item()
        )

        return (
            illegal_encoder_weights
            or illegal_head_weights
            or illegal_patch_embedding_weights
        )


class MOMENTPipeline(MOMENT, PyTorchModelHubMixin):
    def __init__(self, config: Namespace, **kwargs: dict):
        self.new_task = kwargs.get("model_kwargs", {}).pop("task_name", "pre-training")
        super().__init__(config, **kwargs)

    def init(self) -> None:
        if self.new_task != "pre-training":
            self.head = self._get_head(self.new_task)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.model = MOMENT(configs=configs)

    def forecast(self, x_enc):
        dec_out = self.Linear(x_enc.transpose(1, 2)).transpose(1, 2)
        return dec_out

    def classification(self, x_enc):
        # x_enc = self.forecast(x_enc)
        output = x_enc.reshape(x_enc.shape[0], -1)
        output = self.projection(output)
        return output

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.model.pretraining(x_enc=x_enc, input_mask=mask, mask=None)
    
    def anomaly_detection(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.model.pretraining(x_enc=x_enc, input_mask=mask, mask=None)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
