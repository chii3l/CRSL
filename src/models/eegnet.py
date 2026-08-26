import torch.nn as nn


class EEGNetBackbone1D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            temporal_filters=16,
            depth_multiplier=2,
            separable_filters=64,
            temporal_kernel=64,
            separable_kernel=16,
            dropout_rate=0.25):
        super(EEGNetBackbone1D, self).__init__()
        temporal_filters = int(temporal_filters)
        depth_multiplier = int(depth_multiplier)
        depthwise_filters = temporal_filters * depth_multiplier
        separable_filters = int(separable_filters)
        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                depthwise_filters,
                kernel_size=(input_channels, 1),
                groups=temporal_filters,
                bias=False),
            nn.BatchNorm2d(depthwise_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                depthwise_filters,
                depthwise_filters,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=depthwise_filters,
                bias=False),
            nn.Conv2d(depthwise_filters, separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            nn.Conv1d(separable_filters, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        out = self.features(x.unsqueeze(1)).squeeze(2)
        return self.projection(out)


