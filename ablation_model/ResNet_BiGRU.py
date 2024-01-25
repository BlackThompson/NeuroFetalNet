import torch.nn as nn
import torch
from torchsummary import summary
from torch.nn import init
from torch.nn import functional as F


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
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, padding=4):
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


class ResNet_BiGRU(nn.Module):
    def __init__(
        self, num_classes: int = 2, in_channels: int = 1, dropout: float = 0.5
    ):
        super(ResNet_BiGRU, self).__init__()
        self.resnet = nn.Sequential(
            ResBlock(in_channels, 32, kernel_size=3, padding=1),
            nn.AvgPool1d(2, 2),
            ResBlock(32, 64, kernel_size=3, padding=1),
            nn.AvgPool1d(2, 2),
            ResBlock(64, 128),
            nn.AvgPool1d(4, 4),
            ResBlock(128, 128),
            nn.AvgPool1d(4, 4),
            ResBlock(128, 128),
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        input_dim = 256

        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)

        return x


# main函数
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNet_BiGRU(2).to(device)

    print(model)
    input = torch.randn(32, 1, 4800).to(device)
    output = model(input)

    # print(input.size())
    # print(output.size())
