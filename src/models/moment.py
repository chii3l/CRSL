import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _valid_num_heads


class RevIN(nn.Module):
    """Reversible instance normalization used by the official MOMENT encoder."""

    def __init__(self, num_features, eps=1e-5, affine=False):
        super(RevIN, self).__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def forward(self, x, mode="norm", mask=None):
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unsupported RevIN mode: {mode}")

    def _get_statistics(self, x, mask=None):
        if mask is None:
            self.mean = x.mean(dim=-1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
            return

        mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.mean = ((x * mask).sum(dim=-1, keepdim=True) / denom).detach()
        centered = (x - self.mean) * mask
        self.stdev = torch.sqrt((centered.square().sum(dim=-1, keepdim=True) / denom) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return x * self.stdev + self.mean


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, :x.size(2), :].unsqueeze(1)


class Patching(nn.Module):
    def __init__(self, patch_len, stride, pad_end=True):
        super(Patching, self).__init__()
        self.patch_len = max(1, int(patch_len))
        self.stride = max(1, int(stride))
        self.pad_end = bool(pad_end)

    def forward(self, x):
        x = self._right_pad(x, value=0.0)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def seq_mask_to_patch_mask(self, mask):
        mask = self._right_pad(mask, value=0.0)
        return mask.unfold(dimension=-1, size=self.patch_len, step=self.stride).amax(dim=-1)

    def _right_pad(self, x, value=0.0):
        length = x.size(-1)
        pad = self._pad_amount(length)
        if pad <= 0:
            return x
        return F.pad(x, (0, pad), value=value)

    def _pad_amount(self, length):
        if length < self.patch_len:
            return self.patch_len - length
        if not self.pad_end:
            return 0
        remainder = (length - self.patch_len) % self.stride
        if remainder == 0:
            return 0
        return self.stride - remainder


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            d_model=768,
            seq_len=512,
            patch_len=8,
            stride=8,
            patch_dropout=0.1,
            add_positional_embedding=True,
            value_embedding_bias=False,
            orth_gain=1.41):
        super(PatchEmbedding, self).__init__()
        self.patch_len = int(patch_len)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.add_positional_embedding = bool(add_positional_embedding)
        self.value_embedding = nn.Linear(self.patch_len, self.d_model, bias=bool(value_embedding_bias))
        self.mask_embedding = nn.Parameter(torch.zeros(self.d_model))
        if orth_gain is not None:
            nn.init.orthogonal_(self.value_embedding.weight, gain=float(orth_gain))
        if value_embedding_bias:
            self.value_embedding.bias.data.zero_()
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(self.d_model)
        self.dropout = nn.Dropout(float(patch_dropout))

    def forward(self, x, patch_mask=None):
        tokens = self.value_embedding(x)
        if patch_mask is not None:
            patch_mask = patch_mask.to(device=x.device, dtype=tokens.dtype)
            if patch_mask.dim() == 2:
                patch_mask = patch_mask.unsqueeze(1).unsqueeze(-1)
            elif patch_mask.dim() == 3:
                patch_mask = patch_mask.unsqueeze(-1)
            else:
                raise ValueError("patch_mask must have shape (B, P) or (B, C, P).")
            tokens = patch_mask * tokens + (1.0 - patch_mask) * self.mask_embedding.view(1, 1, 1, -1)
        if self.add_positional_embedding:
            tokens = tokens + self.position_embedding(tokens)
        return self.dropout(tokens)


class PretrainHead(nn.Module):
    def __init__(self, d_model=768, patch_len=8, head_dropout=0.1, orth_gain=1.41):
        super(PretrainHead, self).__init__()
        self.dropout = nn.Dropout(float(head_dropout))
        self.linear = nn.Linear(int(d_model), int(patch_len))
        if orth_gain is not None:
            nn.init.orthogonal_(self.linear.weight, gain=float(orth_gain))
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(self.dropout(x))
        return x.flatten(start_dim=2, end_dim=3)


class MOMENTBackbone1D(nn.Module):
    """MOMENT-style patch encoder adapted for six-channel local PPG input."""

    def __init__(
            self,
            input_channels,
            output_channels,
            seq_len=300,
            hidden_channels=128,
            num_layers=2,
            num_heads=4,
            patch_len=16,
            patch_stride=16,
            d_ff=None,
            dropout_rate=0.10,
            patch_dropout=0.10,
            revin_affine=False,
            add_positional_embedding=True,
            value_embedding_bias=False,
            orth_gain=1.41,
            mask_ratio=0.0,
            channel_reduction="mean",
            pad_end=True):
        super(MOMENTBackbone1D, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.patch_len = int(patch_len)
        self.patch_stride = max(1, int(patch_stride))
        self.mask_ratio = float(mask_ratio)
        self.channel_reduction = str(channel_reduction).lower()

        num_heads = _valid_num_heads(self.hidden_channels, int(num_heads))
        d_ff = int(d_ff or self.hidden_channels * 4)

        self.normalizer = RevIN(self.input_channels, affine=revin_affine)
        self.tokenizer = Patching(self.patch_len, self.patch_stride, pad_end=pad_end)
        self.patch_embedding = PatchEmbedding(
            d_model=self.hidden_channels,
            seq_len=int(seq_len),
            patch_len=self.patch_len,
            stride=self.patch_stride,
            patch_dropout=patch_dropout,
            add_positional_embedding=add_positional_embedding,
            value_embedding_bias=value_embedding_bias,
            orth_gain=orth_gain)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_channels,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=float(dropout_rate),
            activation="gelu",
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(self.hidden_channels)

        if self.channel_reduction == "mean":
            self.channel_projection = None
        elif self.channel_reduction == "concat":
            self.channel_projection = nn.Linear(self.input_channels * self.hidden_channels, self.hidden_channels)
        else:
            raise ValueError("channel_reduction must be 'mean' or 'concat'.")

        self.projection = nn.Sequential(
            nn.Conv1d(self.hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.GELU())

    def forward(self, x, input_mask=None):
        batch_size, n_channels, seq_len = x.shape
        if input_mask is None:
            input_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)

        x = self.normalizer(x, mask=input_mask, mode="norm")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        patches = self.tokenizer(x)
        patch_valid_mask = self.tokenizer.seq_mask_to_patch_mask(input_mask)
        patch_embed_mask = self._make_patch_embedding_mask(patch_valid_mask)

        tokens = self.patch_embedding(patches, patch_mask=patch_embed_mask)
        n_patches = tokens.size(2)
        tokens = tokens.reshape(batch_size * n_channels, n_patches, self.hidden_channels)

        padding_mask = patch_valid_mask <= 0
        padding_mask = padding_mask.repeat_interleave(n_channels, dim=0)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        encoded = encoded.reshape(batch_size, n_channels, n_patches, self.hidden_channels)

        if self.channel_reduction == "mean":
            features = encoded.mean(dim=1)
        else:
            features = encoded.permute(0, 2, 1, 3).reshape(
                batch_size, n_patches, n_channels * self.hidden_channels)
            features = self.channel_projection(features)

        return self.projection(features.transpose(1, 2))

    def _make_patch_embedding_mask(self, patch_valid_mask):
        patch_embed_mask = patch_valid_mask
        if self.training and self.mask_ratio > 0.0:
            random_keep = torch.rand_like(patch_embed_mask) >= self.mask_ratio
            patch_embed_mask = patch_embed_mask * random_keep.to(dtype=patch_embed_mask.dtype)
        return patch_embed_mask
