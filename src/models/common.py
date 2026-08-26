import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # 计算 query, key, value
        Q = self.query(x)  # (batch_size, seq_len, input_dim)
        K = self.key(x)    # (batch_size, seq_len, input_dim)
        V = self.value(x)  # (batch_size, seq_len, input_dim)
        
        # 计算注意力分数
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        
        # 应用 softmax 以获得注意力权重
        attn_weights = self.softmax(attn_scores)  # (batch_size, seq_len, seq_len)
        
        # 计算加权的 value
        attn_output = torch.bmm(attn_weights, V)  # (batch_size, seq_len, input_dim)
        
        return attn_output

class SquareLayer(nn.Module):
    def forward(self, x):
        return torch.square(x)


class SafeLogLayer(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ChannelLayerNorm1D(nn.Module):
    def __init__(self, channels):
        super(ChannelLayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SqueezeExcite1D, self).__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7, mlp_ratio=4, dropout_rate=0.0):
        super(ConvNeXtBlock1D, self).__init__()
        hidden = channels * int(mlp_ratio)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels)
        self.norm = ChannelLayerNorm1D(channels)
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Dropout(dropout_rate)
        )
        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 1e-4)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(x)
        return residual + self.gamma * x


class MultiScaleTemporalStem1D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernels=(7, 15, 31), stride=2, dropout_rate=0.0):
        super(MultiScaleTemporalStem1D, self).__init__()
        branch_channels = max(8, hidden_channels // len(kernels))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    branch_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                    bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.GELU())
            for kernel in kernels
        ])
        self.projection = nn.Sequential(
            nn.Conv1d(branch_channels * len(kernels), hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.projection(torch.cat([branch(x) for branch in self.branches], dim=1))


def _valid_num_heads(hidden_channels, requested_heads):
    requested_heads = max(1, int(requested_heads))
    if hidden_channels % requested_heads == 0:
        return requested_heads
    for heads in range(min(requested_heads, hidden_channels), 0, -1):
        if hidden_channels % heads == 0:
            return heads
    return 1


def _interpolate_pos_embedding(pos_embedding, seq_len):
    if seq_len <= pos_embedding.size(1):
        return pos_embedding[:, :seq_len, :]
    return F.interpolate(
        pos_embedding.transpose(1, 2),
        size=seq_len,
        mode='linear',
        align_corners=False).transpose(1, 2)


def _parse_int_list(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        items = [item.strip() for item in value.split(',') if item.strip()]
        return tuple(int(item) for item in items) if items else default
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return default


def _get_config_value(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


