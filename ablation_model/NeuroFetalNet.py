import torch.nn as nn
import torch
from torchsummary import summary
from torch.nn import init
from torch.nn import functional as F
import math


# input size (4800, 1)
# modify the CNN model


class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        padding=4,
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        out = self.relu(x + shortcut)
        return out


class AvgChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AvgChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(
                in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc(avg_out)
        out = self.sigmoid(avg_out)

        return out


class AvgChannelAttentionBlock1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AvgChannelAttentionBlock1D, self).__init__()
        self.ca = AvgChannelAttention1D(in_channels, reduction_ratio)

    def forward(self, x):
        # print("before ca: ", x.shape)
        # print("after ca: ", self.ca(x).shape)
        x = self.ca(x) * x
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels: int = 1, kernel_size: int = 3):
        super(ResNet, self).__init__()
        self.resnet = nn.Sequential(
            ResBlock(
                in_channels, 32, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.AvgPool1d(2, 2),
            ResBlock(32, 64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.AvgPool1d(2, 2),
            ResBlock(64, 128, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.AvgPool1d(4, 4),
            ResBlock(128, 128, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.AvgPool1d(4, 4),
            ResBlock(128, 128, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


class NeuroFetalNet(nn.Module):
    def __init__(
        self, num_classes: int = 2, in_channels: int = 1, dropout: float = 0.2
    ):
        super(NeuroFetalNet, self).__init__()

        # self.ca_block_0 = ChannelAttentionBlock1D(
        #     in_channels=in_channels, reduction_ratio=2
        # )

        self.resnet_k3 = ResNet(in_channels=in_channels, kernel_size=3)
        self.resnet_k5 = ResNet(in_channels=in_channels, kernel_size=5)
        self.resnet_k7 = ResNet(in_channels=in_channels, kernel_size=7)
        self.resnet_k9 = ResNet(in_channels=in_channels, kernel_size=9)

        self.ca = AvgChannelAttentionBlock1D(in_channels=128 * 4)
        self.fc = nn.Linear(128 * 4, num_classes)

    def forward(self, x):
        x_k3 = self.resnet_k3(x)
        x_k5 = self.resnet_k5(x)
        x_k7 = self.resnet_k7(x)
        x_k9 = self.resnet_k9(x)

        out = torch.cat((x_k3, x_k5, x_k7, x_k9), dim=1)
        # out = x_k9

        # out = self.ca(out)
        out = out.mean(dim=2)
        x = self.fc(out)

        return x


# main函数
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = NeuroFetalNet(num_classes=2, in_channels=1).to(device)

    print(model)
    input = torch.randn(32, 1, 4800).to(device)
    output = model(input)

    print(input.size())
    print(output.size())
