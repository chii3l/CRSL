import torch.nn as nn
import torch.nn.functional as F


class TCN1DBlock(nn.Module):
    def __init__(self, input_channels, filters, kernel_size=3, dilation=1, dropout_rate=0.0):
        super(TCN1DBlock, self).__init__()
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(filters)
        self.dropout = nn.Dropout(dropout_rate)
        if input_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_channels, filters, kernel_size=1),
                nn.BatchNorm1d(filters)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

