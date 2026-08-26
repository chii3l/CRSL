import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _get_config_value, _interpolate_pos_embedding, _valid_num_heads


class PatchTSTBackbone1D(nn.Module):
    """PatchTST-style channel-independent Transformer backbone for short PPG windows."""
    def __init__(
            self,
            input_channels,
            output_channels,
            seq_len=300,
            dropout_rate=0.0,
            config=None):
        super(PatchTSTBackbone1D, self).__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.seq_len = int(seq_len)
        self.patch_len = int(_get_config_value(config, "ppg_patch_len", 16))
        self.patch_stride = int(_get_config_value(config, "ppg_patch_stride", max(1, self.patch_len // 2)))
        self.patch_len = max(2, self.patch_len)
        self.patch_stride = max(1, self.patch_stride)
        self.use_norm = bool(int(_get_config_value(config, "ppg_tsl_use_norm", 1)))

        d_model = int(_get_config_value(config, "ppg_tsl_d_model", min(self.output_channels, 64)))
        d_model = max(16, d_model)
        num_heads = _valid_num_heads(d_model, int(_get_config_value(config, "ppg_tsl_n_heads", 4)))
        num_layers = max(1, int(_get_config_value(config, "ppg_tsl_e_layers", 1)))
        d_ff = int(_get_config_value(config, "ppg_tsl_d_ff", max(d_model * 4, 128)))
        max_patches = max(1, 1 + max(0, self.seq_len - self.patch_len) // self.patch_stride + 1)

        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches + 4, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, self.output_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def _normalize(self, x):
        if not self.use_norm:
            return x
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        return (x - mean) / std

    def forward(self, x):
        # x: [B, C, T]. Each channel is patched independently, following PatchTST.
        if x.size(-1) < self.patch_len:
            x = F.pad(x, (0, self.patch_len - x.size(-1)), mode="replicate")
        x = self._normalize(x)
        if x.size(-1) < self.seq_len:
            x = F.pad(x, (0, self.seq_len - x.size(-1)), mode="replicate")

        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        batch_size, channels, num_patches, _ = patches.shape
        patches = patches.reshape(batch_size * channels, num_patches, self.patch_len)

        tokens = self.patch_embedding(patches)
        tokens = tokens + _interpolate_pos_embedding(self.pos_embedding, num_patches)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        tokens = tokens.reshape(batch_size, channels, num_patches, -1)
        tokens = tokens.mean(dim=1)
        out = self.projection(tokens)
        return out.transpose(1, 2).contiguous()
