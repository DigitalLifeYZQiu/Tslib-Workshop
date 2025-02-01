import os

import torch
from torch import nn
import math
import torch.nn.functional as F
from math import sqrt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from layers.Embed import TimeBertPatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        # '../ltsm/checkpoints/lotsa_uea_ucr_bert_d1024_l8_p16_n512_weight_p50_wo_revin_full-epoch=25.ckpt'
        # 从configs.ckpt_path的字符串中获取patch_len，规则为文件名中第一个_p后的数字
        if configs.ckpt_path != '':
            patch_len = int(configs.ckpt_path.split('_p')[1].split('_')[0])
            d_model = int(configs.ckpt_path.split('_d')[1].split('_')[0])
            layers = int(configs.ckpt_path.split('_l')[1].split('_')[0])
            d_ff = d_model * 4 if d_model == 768 or (d_model == 1024 and layers == 24) or d_model==256 or d_model==192 else d_model * 2
            n_heads = 12 if d_model == 768 else 8
            n_heads = 16 if d_model == 1024 and layers == 24 else n_heads
        else:
            patch_len = configs.patch_len
            d_model = configs.d_model
            layers = configs.e_layers
            d_ff = d_model * 4
            n_heads = 12 if d_model == 768 else 8
            n_heads = 16 if d_model == 1024 and layers == 24 else n_heads
        print("patch_len:", patch_len)
        self.use_variate_embedding = False
        # 如果configs.ckpt_path含有'_ve'字段则使用variate_embedding
        if '_ve' in configs.ckpt_path:
            self.use_variate_embedding = True
        
        stride = None
        enc_in = configs.enc_in,
        # d_model = 1024
        # d_ff = 2048
        # layers = 8
        # n_heads = 8
        dropout = 0.1
        activation = 'gelu'
        factor = 1
        position_embedding = False

        # d_model = 256
        # d_ff = 1024
        # layers = 3

        self.task_name = configs.task_name

        # self.patch_len = patch_len
        self.patch_len = configs.patch_len
        
        if stride is None:
            stride = patch_len
        self.stride = stride
        self.enc_in = enc_in
        self.d_model = d_model
        self.d_ff = d_ff
        self.layers = layers
        self.n_heads = n_heads
        self.position_embedding = position_embedding
        self.seq_len = configs.seq_len
        # 计算padding，要求将seq_len整除patch_len
        padding = (configs.seq_len + patch_len - 1) // patch_len * patch_len - configs.seq_len
        # padding = 0

        # patching and embedding
        # self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout, False)
        self.patch_embedding = TimeBertPatchEmbedding(d_model, patch_len, stride, padding, dropout, False)

        # encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),# 这里true是为了让mask生效而不是casual
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Prediction Head
        self.mask_token = nn.Parameter(torch.randn(d_model))
        self.variate_mask_token = nn.Parameter(torch.randn(d_model))
        self.cls_mask_token = nn.Parameter(torch.randn(d_model))

        # Prediction Head
        self.proj = nn.Linear(self.d_model, patch_len, bias=True)
        self.proj_dataset_cls = nn.Linear(self.d_model, 600, bias=True)
        self.proj_variate_cls = nn.Linear(self.d_model, 2, bias=True)
        
        if self.use_variate_embedding:
            self.variate_embedding = nn.Embedding(512, d_model)
        self.cls_mask_token_only = configs.cls_mask_token_only
        self.var_mask_token_only = configs.var_mask_token_only

        # patch_encoder_checkpoint = '../ltsm/checkpoints/lotsa_uea_ucr_bert_d1024_l8_p16_n512_weight_p50_wo_revin_full-epoch=25.ckpt'
        patch_encoder_checkpoint = configs.ckpt_path
        if (patch_encoder_checkpoint == '../ltsm/checkpoints/uea_ucr_bert_d1024_l8_p16_n512_weight_p50_wo_revin_full.ckpt'
                or patch_encoder_checkpoint == '../ltsm/checkpoints/lotsa_bert_d768_l12_p16_n512_weight_p50_wo_revin_full-epoch=18.ckpt'):
            self.proj_dataset_cls = nn.Linear(self.d_model, 300, bias=True)

        # patch_encoder_freeze = True
        patch_encoder_freeze = configs.freeze_patch_encoder
        
        if patch_encoder_checkpoint == 'random':
            print('loading model randomly')
        elif os.path.exists(patch_encoder_checkpoint):
            sd = torch.load(patch_encoder_checkpoint, map_location="cpu")["state_dict"]
            # 去掉sd中的前14个字符
            keys = list(sd.keys())
            for k in keys:
                if 'mask_token' == k:
                    # 在前面加上self.
                    sd[f'bert.{k}'] = sd.pop(k)
            sd = {k[5:]: v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)
        else:
            raise FileNotFoundError(f"File {patch_encoder_checkpoint} not found.")
        if patch_encoder_freeze:
            self.freeze()
            # print("Freezing All Parameters")
            # for param in self.parameters():
            #     param.requires_grad = False

        self.pos_embed = self.get_sinusoid_encoding_table(5000, self.d_model)
        if self.task_name == 'long_term_forecast':
            self.head = nn.Linear(self.d_model, configs.pred_len)
        elif self.task_name == 'classification' or self.task_name == 'classification_ablation':
            # self.head = nn.Linear(self.d_model * 31, configs.num_class)
            # if self.cls_mask_token_only:
            #     self.head = nn.Linear(self.d_model, configs.num_class)
            # else:
            #     self.head = nn.Linear(self.d_model * (((configs.seq_len + padding) // patch_len + 1) * configs.enc_in + 1), configs.num_class)
            self.head = nn.Linear(self.d_model * (((configs.seq_len + padding) // patch_len + 1) * configs.enc_in + 1),
                                  configs.num_class)
        elif self.task_name == 'imputation' or 'anomaly_detection' in self.task_name:
            self.head = nn.Linear(self.d_model * (((configs.seq_len + padding) // patch_len + 1) * configs.enc_in + 1), configs.seq_len)
            self.head1 = nn.Linear(self.d_model, patch_len, bias=True)
            # self.head2 = nn.Linear(self.d_model * ())

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze(self):
        # 将PatchEmbedding和Encoder的参数冻结
        print("Freezing Patch Embedding")
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        print("Freeze Encoder Layer")
        for param in self.encoder.parameters():
            param.requires_grad = False


    def get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc) # [B * M, N, D]
        # 前后添加self.cls_mask_token和self.variate_mask_token
        cls_mask_token = self.cls_mask_token.unsqueeze(0).unsqueeze(0).expand(B * M, 1, -1)
        variate_mask_token = self.variate_mask_token.unsqueeze(0).unsqueeze(0).expand(B * M, 1, -1)

        x = torch.cat([cls_mask_token, x, variate_mask_token], dim=1)

        _, N_, D = x.shape
        enc_out = x + self.pos_embed.to(x.device)[:, :N_, :].expand(B * M, N_, -1)
        enc_out, _ = self.encoder(enc_out) # [B * M, N_, D]
        # 取出cls_mask_token输入self.head
        dec_out = self.head(enc_out[:, 0, :]) # [B * M, 1, P]
        dec_out = dec_out.view(B, M, -1).transpose(1, 2) # [B, P, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means

        return dec_out

    def mask_reconstruction(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, M = x_enc.shape

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc) # [B * M, N, D]
        # 用where函数将mask部分替换为mask_token
        x = torch.where(mask.unsqueeze(-1), self.mask_token, x) # [B * M, N, D]
        _, N, D = x.shape

        variate_mask_token = self.variate_mask_token.unsqueeze(0).unsqueeze(0).expand(B * M, 1, -1)
        x = torch.cat([x, variate_mask_token], dim=1) # [B * M, N + 1, D]
        x = x.reshape(B, -1, D) # [B, M*(N + 1), D]
        cls_mask_token = self.cls_mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls_mask_token, x], dim=1) # [B, M*(N + 1) +1, D]

        _, N_, D = x.shape
        enc_out = x + self.pos_embed.to(x.device)[:, :N_, :].expand(B , N_, -1)
        enc_out, _ = self.encoder(enc_out) # [B, M*(N + 1) +1, D]
        # 取出cls_mask_token输入self.head
        dec_out = self.proj(enc_out)# [B, M*(N + 1) +1, P]

        # 将mask_token部分去掉
        dec_out = dec_out[:, 1:, :] # [B, M*(N + 1), P]
        dec_out = dec_out.reshape(B * M, -1, self.patch_len)[:, :N, :] # [B * M, N, P]

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc)  # [B * M, N, D]
        _, N, D = x.shape
        
        # 前后添加self.cls_mask_token和self.variate_mask_token
        variate_mask_token = self.variate_mask_token.unsqueeze(0).unsqueeze(0).expand(B * M, 1, -1)
        x = torch.cat([x, variate_mask_token], dim=1)  # [B * M, N + 1, D]
        x = x.reshape(B, -1, D)  # [B, M*(N + 1), D]
        cls_mask_token = self.cls_mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls_mask_token, x], dim=1)  # [B, M*(N + 1) +1, D]
        _, N_, D = x.shape
        
        enc_out = x + self.pos_embed.to(x.device)[:, :N_, :].expand(B, N_, -1)
        enc_out, _ = self.encoder(enc_out)  # [B, N_, D]
        
        # clean cls mask token
        enc_out = enc_out[:, 1:, :] # [B, M(N+1), D]
        # clean variate mask token
        enc_out = enc_out.reshape(B, M, N+1, D)[:, :, :-1, :] # [B, M, N, D]
        # if self.configs.is_training:
        #     dec_out = self.head1(enc_out)
        dec_out = self.proj(enc_out).reshape(B, M, -1)[:, :, :L].transpose(1, 2)
        
        
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out
    
    def classification(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape
        # x_enc = x_enc[:, :-8, :]
        
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc)  # [B * M, N, D]
        _, N, D = x.shape
        # 前后添加self.cls_mask_token和self.variate_mask_token
        variate_mask_token = self.variate_mask_token.unsqueeze(0).unsqueeze(0).expand(B * M, 1, -1)
        x = torch.cat([x, variate_mask_token], dim=1)  # [B * M, N + 1, D]
        x = x.reshape(B, -1, D)  # [B, M*(N + 1), D]
        
        if self.use_variate_embedding:
            variate_id = torch.zeros(B, M, device=x.device).long()
            variate_id = variate_id.unsqueeze(-1).expand(-1, -1, N + 1).reshape(B, -1)
            x = x + self.variate_embedding(variate_id)
        
        cls_mask_token = self.cls_mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        if self.cls_mask_token_only:
            # return cls_mask_token
            x = cls_mask_token  # [B, 1, D]
        elif self.var_mask_token_only:
            x = variate_mask_token.reshape(B, -1, D)
        else:
            x = torch.cat([cls_mask_token, x], dim=1)  # [B, M*(N + 1) +1, D]

        _, N_, D = x.shape
        enc_out = x + self.pos_embed.to(x.device)[:, :N_, :].expand(B, N_, -1)
        enc_out, _ = self.encoder(enc_out)  # [B, N_, D]
        if self.task_name == 'classification_ablation':
            return enc_out
        # 取出cls_mask_token输入self.head
        dec_out = self.head(enc_out.reshape(B, -1))  # [B, 1, P]
        # dec_out = self.head(enc_out[:, 0, :]) # [B, 1, P]
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        # if self.task_name == 'anomaly_detection':
        if 'anomaly_detection' in self.task_name:
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification' or self.task_name == 'classification_ablation':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'mask_reconstruction':
            dec_out = self.mask_reconstruction(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        return None
