import torch
import torch.nn as nn


class LSTMAttentionBackbone1D(nn.Module):
    """Bidirectional LSTM backbone with temporal attention context."""

    def __init__(self, input_channels, output_channels, num_layers=2, dropout_rate=0.0):
        super(LSTMAttentionBackbone1D, self).__init__()
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
        self.attn_score = nn.Linear(int(output_channels), 1)
        self.norm = nn.LayerNorm(int(output_channels))

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.output_proj(out)
        weights = torch.softmax(self.attn_score(out), dim=1)
        context = torch.sum(out * weights, dim=1, keepdim=True)
        out = self.norm(out + context)
        return out.transpose(1, 2)
