import torch.nn as nn

from .resnet import ResNet1DBlock


class ECGResNetBackbone1D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_channels=64,
            num_layers=5,
            dropout_rate=0.25):
        super(ECGResNetBackbone1D, self).__init__()
        hidden_channels = int(hidden_channels)
        layers = [
            nn.Conv1d(input_channels, hidden_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        ]
        channels = hidden_channels
        for layer_idx in range(max(1, int(num_layers))):
            next_channels = min(output_channels, hidden_channels * (2 ** (layer_idx // 2)))
            stride = 2 if layer_idx in [1, 3] else 1
            layers.append(ResNet1DBlock(channels, next_channels, kernel_size=7, stride=stride))
            channels = next_channels
        layers.extend([
            nn.Conv1d(channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


