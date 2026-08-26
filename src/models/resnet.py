import torch.nn as nn
import torch.nn.functional as F


class ResNet1DBlock(nn.Module):
    def __init__(self, input_channels, filters, kernel_size=3, stride=1):
        super(ResNet1DBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(filters)
        
        # 如果输入通道数与输出通道数不匹配，或者 stride != 1，则需要使用 shortcut 调整输入维度
        if stride != 1 or input_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_channels, filters, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(filters)
            )
        else:
            self.shortcut = nn.Identity()  # 如果不需要调整，直接通过 identity 层保持不变

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut路径
        shortcut = self.shortcut(x)
        
        out += shortcut
        out = self.relu(out)
        
        return out

class ResNet2DBlock(nn.Module):
    def __init__(self, input_channels, filters, kernel_size=3, stride=1):
        super(ResNet2DBlock, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(filters)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(filters)

        # 如果输入通道数与输出通道数不匹配，或者 stride != 1，则需要使用 shortcut 调整输入维度
        if stride != 1 or input_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, filters, kernel_size=1, stride=stride),
                nn.BatchNorm2d(filters)
            )
        else:
            self.shortcut = nn.Identity()  # 如果不需要调整，直接通过 identity 层保持不变

    def forward(self, x):
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut路径
        shortcut = self.shortcut(x)

        # 残差连接
        out += shortcut
        out = F.relu(out)
        
        return out
    
