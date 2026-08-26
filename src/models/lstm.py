import torch.nn as nn


class LSTMBackbone1D(nn.Module):
    """Bidirectional LSTM backbone for PPG sequence modeling."""

    def __init__(self, input_channels, output_channels, num_layers=2, dropout_rate=0.0):
        super(LSTMBackbone1D, self).__init__()
        hidden_size = max(1, int(output_channels) // 2)
        self.lstm = nn.LSTM(
            input_size=int(input_channels),
            hidden_size=hidden_size,
            num_layers=max(1, int(num_layers)),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout_rate) if int(num_layers) > 1 else 0.0)
        lstm_channels = hidden_size * 2
        self.output_proj = nn.Identity() if lstm_channels == int(output_channels) else nn.Linear(lstm_channels, int(output_channels))
        self.norm = nn.LayerNorm(int(output_channels))

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.output_proj(out)
        out = self.norm(out)
        return out.transpose(1, 2)
