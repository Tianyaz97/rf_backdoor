import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable
import copy
import math
import numpy as np

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    
    # def __deepcopy__(self, memo):
    #     copied = self.__class__(self.pe.size(-1), self.dropout.p, self.pe.size(1) - 1)
    #     for k, v in self.__dict__.items():
    #         if k == 'pe':
    #             # 对于`pe`，直接引用相同的Tensor，避免深拷贝
    #             setattr(copied, k, v)
    #         else:
    #             setattr(copied, k, copy.deepcopy(v, memo))
    #     return copied


    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        self.attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', self.attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, n_channel=2, len_sw=256, n_classes=16, dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(n_channel, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position = PositionalEncoding(d_model=dim, max_len=len_sw)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        self.classifier = nn.Linear(dim, n_classes)


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)  # (256, 256, 2) -> (256, 256, 256)
        x = self.position(x)  # (256, 256, 256) -> (256, 256, 256)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)  # (256, 256, 256) -> (256, 257, 256)
        x = self.transformer(x)  # (256, 257, 256) -> (256, 257, 256)
        c_t = self.to_c_token(x[:, 0])  # (256, 257, 256) -> (256, 256)
        return c_t


class _Seq_Transformer(nn.Module): # the transformer used in TS-TCC: without positional encoding
    def __init__(self, patch_size, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(patch_size, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

class TC(nn.Module):
    def __init__(self, bb_dim, device, tc_hidden=100, temp_unit='tsfm'):
        super(TC, self).__init__()
        self.num_channels = bb_dim
        self.timestep = 6
        self.Wk = nn.ModuleList([nn.Linear(tc_hidden, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        self.temp_unit = temp_unit
        if self.temp_unit == 'tsfm':
            self.seq_transformer = _Seq_Transformer(patch_size=self.num_channels, dim=tc_hidden, depth=4, heads=4, mlp_dim=64)
        elif self.temp_unit == 'lstm':
            self.lstm = nn.LSTM(input_size=self.num_channels, hidden_size=tc_hidden, num_layers=1,
                                batch_first=True, bidirectional=False)
        elif self.temp_unit == 'blstm':
            self.blstm = nn.LSTM(input_size=self.num_channels, hidden_size=tc_hidden, num_layers=1,
                                batch_first=True, bidirectional=True)
        elif self.temp_unit == 'gru':
            self.gru = nn.GRU(input_size=self.num_channels, hidden_size=tc_hidden, num_layers=1,
                              batch_first=True, bidirectional=False)
        elif self.temp_unit == 'bgru':
            self.bgru = nn.GRU(input_size=self.num_channels, hidden_size=tc_hidden, num_layers=1,
                              batch_first=True, bidirectional=True)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # shape of features: (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(1, self.timestep + 1):
            idx = (t_samples + i).long()
            encode_samples[i - 1] = z_aug2[:, idx, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        if self.temp_unit == 'tsfm':
            c_t = self.seq_transformer(forward_seq)
        elif self.temp_unit == 'lstm':
            _, (c_t, _) = self.lstm(forward_seq)
            c_t = torch.squeeze(c_t)
        elif self.temp_unit == 'blstm':
            _, (c_t, _) = self.blstm(forward_seq)
            c_t = c_t[0, :, :]
            c_t = torch.squeeze(c_t)
        elif self.temp_unit == 'gru':
            _, c_t = self.gru(forward_seq)
            c_t = torch.squeeze(c_t)
        elif self.temp_unit == 'bgru':
            _, c_t = self.bgru(forward_seq)
            c_t = c_t[0, :, :]
            c_t = torch.squeeze(c_t)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, c_t