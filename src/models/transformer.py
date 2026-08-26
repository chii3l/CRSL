import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBackbone1D(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=2, num_heads=4, dropout_rate=0.0, max_len=512):
        super(TransformerBackbone1D, self).__init__()
        if output_channels % num_heads != 0:
            num_heads = 1
        stem_channels = max(output_channels // 2, 32)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, output_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, output_channels))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_channels,
            nhead=num_heads,
            dim_feedforward=output_channels * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(output_channels)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        out = self.conv_stem(x).transpose(1, 2)
        seq_len = out.size(1)
        if seq_len <= self.pos_embedding.size(1):
            pos_embedding = self.pos_embedding[:, :seq_len, :]
        else:
            pos_embedding = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False).transpose(1, 2)
        out = out + pos_embedding
        out = self.encoder(out)
        out = self.norm(out)
        return out.transpose(1, 2)


