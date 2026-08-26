# Adapted from:
# https://github.com/emadeldeen24/TSLANet
# TSLANet: Rethinking Transformers for Time Series Representation Learning, ICML 2024.

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ICB(nn.Module):
    """Interactive Convolution Block from the official TSLANet implementation."""
    def __init__(self, in_features, hidden_features, drop=0.0):
        super(ICB, self).__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_act = self.act(x1)
        x1_drop = self.drop(x1_act)

        x2 = self.conv2(x)
        x2_act = self.act(x2)
        x2_drop = self.drop(x2_act)

        out1 = x1 * x2_drop
        out2 = x2 * x1_drop
        x = self.conv3(out1 + out2)
        return x.transpose(1, 2)


class PatchEmbed(nn.Module):
    """Official TSLANet 1D patch embedding adapted for variable sequence length."""
    def __init__(self, seq_len, patch_size=8, patch_stride=None, in_chans=3, embed_dim=384):
        super(PatchEmbed, self).__init__()
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.stride = int(patch_stride) if patch_stride is not None else max(1, self.patch_size // 2)
        self.num_patches = max(1, int((self.seq_len - self.patch_size) / self.stride + 1))
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.stride)

    def forward(self, x):
        if x.size(-1) < self.patch_size:
            x = F.pad(x, (0, self.patch_size - x.size(-1)), "constant", 0)
        return self.proj(x).flatten(2).transpose(1, 2)


class AdaptiveSpectralBlock(nn.Module):
    """Adaptive Spectral Block from TSLANet, with explicit adaptive_filter flag."""
    def __init__(self, dim, adaptive_filter=True):
        super(AdaptiveSpectralBlock, self).__init__()
        self.adaptive_filter = bool(adaptive_filter)
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        nn.init.trunc_normal_(self.complex_weight_high, std=0.02)
        nn.init.trunc_normal_(self.complex_weight, std=0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        batch_size = x_fft.shape[0]
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(batch_size, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(batch_size, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach()
        adaptive_mask = adaptive_mask + self.threshold_param
        return adaptive_mask.unsqueeze(-1)

    def forward(self, x_in):
        batch_size, num_patches, channels = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted = x_weighted + x_masked * weight_high

        x = torch.fft.irfft(x_weighted, n=num_patches, dim=1, norm="ortho")
        return x.to(dtype).view(batch_size, num_patches, channels)


class TSLANetLayer(nn.Module):
    """Official TSLANet layer with configurable ASB and ICB components."""
    def __init__(
            self,
            dim,
            mlp_ratio=3.0,
            drop=0.0,
            drop_path_rate=0.0,
            use_icb=True,
            use_asb=True,
            adaptive_filter=True):
        super(TSLANetLayer, self).__init__()
        self.use_icb = bool(use_icb)
        self.use_asb = bool(use_asb)
        self.norm1 = nn.LayerNorm(dim)
        self.asb = AdaptiveSpectralBlock(dim, adaptive_filter=adaptive_filter)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * float(mlp_ratio))
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANetBackbone1D(nn.Module):
    """Official TSLANet ASB/ICB encoder adapted to the local PPG backbone API."""
    def __init__(
            self,
            input_channels,
            output_channels,
            seq_len=300,
            hidden_channels=128,
            num_layers=3,
            patch_len=16,
            patch_stride=None,
            mlp_ratio=3.0,
            use_icb=True,
            use_asb=True,
            adaptive_filter=True,
            dropout_rate=0.10):
        super(TSLANetBackbone1D, self).__init__()
        hidden_channels = int(hidden_channels)
        num_layers = max(1, int(num_layers))
        patch_stride = int(patch_stride) if patch_stride is not None else max(1, int(patch_len) // 2)

        self.patch_embed = PatchEmbed(
            seq_len=seq_len,
            patch_size=patch_len,
            patch_stride=patch_stride,
            in_chans=input_channels,
            embed_dim=hidden_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, hidden_channels))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, num_layers)]
        self.tsla_blocks = nn.ModuleList([
            TSLANetLayer(
                dim=hidden_channels,
                mlp_ratio=mlp_ratio,
                drop=dropout_rate,
                drop_path_rate=dpr[idx],
                use_icb=use_icb,
                use_asb=use_asb,
                adaptive_filter=adaptive_filter)
            for idx in range(num_layers)
        ])
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.GELU())

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _pos_embed_for(self, num_patches):
        if num_patches == self.pos_embed.size(1):
            return self.pos_embed
        return F.interpolate(
            self.pos_embed.transpose(1, 2),
            size=num_patches,
            mode="linear",
            align_corners=False).transpose(1, 2)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self._pos_embed_for(x.size(1))
        x = self.pos_drop(x)
        for block in self.tsla_blocks:
            x = block(x)
        return self.projection(x.transpose(1, 2))


# Compatibility aliases with the official repository naming.
Adaptive_Spectral_Block = AdaptiveSpectralBlock
TSLANet_layer = TSLANetLayer
