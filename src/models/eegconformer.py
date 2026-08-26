import torch.nn as nn


class EEGConformerBackbone1D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_channels=96,
            num_layers=2,
            num_heads=4,
            temporal_kernel=25,
            dropout_rate=0.25):
        super(EEGConformerBackbone1D, self).__init__()
        hidden_channels = int(hidden_channels)
        num_heads = int(num_heads)
        if hidden_channels % num_heads != 0:
            num_heads = 1
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(input_channels, 1), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout_rate),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(hidden_channels)
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.patch_embedding(x.unsqueeze(1)).squeeze(2)
        seq = self.encoder(out.transpose(1, 2))
        seq = self.norm(seq)
        return self.projection(seq.transpose(1, 2))


