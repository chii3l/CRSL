import torch.nn as nn


class DeepConvNetBackbone1D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_channels=64,
            num_layers=4,
            temporal_kernel=7,
            dropout_rate=0.25):
        super(DeepConvNetBackbone1D, self).__init__()
        hidden_channels = int(hidden_channels)
        layers = [
            nn.Conv2d(1, hidden_channels, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(input_channels, 1), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate),
        ]
        channels = hidden_channels
        for layer_idx in range(max(1, int(num_layers) - 1)):
            next_channels = min(output_channels, hidden_channels * (2 ** (layer_idx + 1)))
            layers.extend([
                nn.Conv2d(channels, next_channels, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ELU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(dropout_rate),
            ])
            channels = next_channels
        self.features = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Conv1d(channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        out = self.features(x.unsqueeze(1)).squeeze(2)
        return self.projection(out)


