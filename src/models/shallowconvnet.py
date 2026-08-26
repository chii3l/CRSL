import torch.nn as nn

from .common import SafeLogLayer, SquareLayer


class ShallowConvNetBackbone1D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_channels=80,
            temporal_kernel=25,
            pool_size=20,
            pool_stride=5,
            dropout_rate=0.25):
        super(ShallowConvNetBackbone1D, self).__init__()
        hidden_channels = int(hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                hidden_channels,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(input_channels, 1), bias=False),
            nn.BatchNorm2d(hidden_channels),
            SquareLayer(),
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            SafeLogLayer(),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        out = self.features(x.unsqueeze(1)).squeeze(2)
        return self.projection(out)


