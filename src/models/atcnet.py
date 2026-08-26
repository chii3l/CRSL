import torch.nn as nn


class TemporalResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, dropout_rate=0.25):
        super(TemporalResidualBlock1D, self).__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout_rate),
        )
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.net(x))


class ATCNetBackbone1D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_channels=96,
            num_layers=3,
            num_heads=4,
            temporal_kernel=15,
            dropout_rate=0.25):
        super(ATCNetBackbone1D, self).__init__()
        hidden_channels = int(hidden_channels)
        num_heads = int(num_heads)
        if hidden_channels % num_heads != 0:
            num_heads = 1
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=temporal_kernel, padding=temporal_kernel // 2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.tcn = nn.Sequential(*[
            TemporalResidualBlock1D(
                hidden_channels,
                kernel_size=7,
                dilation=2 ** i,
                dropout_rate=dropout_rate)
            for i in range(max(1, int(num_layers)))
        ])
        self.attn_norm = nn.LayerNorm(hidden_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True)
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        out = self.tcn(self.stem(x))
        seq = out.transpose(1, 2)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.attn_norm(seq + attn_out)
        return self.projection(seq.transpose(1, 2))


