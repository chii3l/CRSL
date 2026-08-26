import torch
import torch.nn as nn

from .common import _interpolate_pos_embedding, _valid_num_heads


class MedformerBackbone1D(nn.Module):
    """Multi-granularity patching Transformer adapted from Medformer."""
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_channels=128,
            num_layers=2,
            num_heads=4,
            patch_sizes=(8, 16, 32),
            dropout_rate=0.10):
        super(MedformerBackbone1D, self).__init__()
        hidden_channels = int(hidden_channels)
        num_heads = _valid_num_heads(hidden_channels, int(num_heads))
        self.patch_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    hidden_channels,
                    kernel_size=int(patch),
                    stride=max(1, int(patch) // 2),
                    padding=int(patch) // 2,
                    bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU())
            for patch in patch_sizes
        ])
        self.scale_embeddings = nn.Parameter(torch.zeros(1, len(patch_sizes), hidden_channels))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 384, hidden_channels))
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
            nn.GELU()
        )
        nn.init.normal_(self.scale_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        token_list = []
        for idx, patch_embed in enumerate(self.patch_embeds):
            tokens = patch_embed(x).transpose(1, 2)
            tokens = tokens + self.scale_embeddings[:, idx:idx + 1, :]
            token_list.append(tokens)
        tokens = torch.cat(token_list, dim=1)
        tokens = tokens + _interpolate_pos_embedding(self.pos_embedding, tokens.size(1))
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return self.projection(tokens.transpose(1, 2))
