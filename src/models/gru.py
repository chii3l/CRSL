import torch.nn as nn


class GRUBackbone1D(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=2, dropout_rate=0.0):
        super(GRUBackbone1D, self).__init__()
        hidden_size = max(1, output_channels // 2)
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0)
        gru_channels = hidden_size * 2
        self.output_proj = nn.Identity() if gru_channels == output_channels else nn.Linear(gru_channels, output_channels)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        out = self.output_proj(out)
        out = self.norm(out)
        return out.transpose(1, 2)


