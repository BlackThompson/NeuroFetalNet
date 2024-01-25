import torch.nn as nn
import torch
from torchsummary import summary
from torch.nn import init
from torch.nn import functional as F
import math


# input size (4800, 1)
# modify the CNN model


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x -> (batch_size,d_model,seq_len)
        x = x.permute(0, 2, 1)  # x -> (batch_size,seq_len,d_model)
        pe = self.pe[:, : x.size(1)]
        pe = pe.expand(x.size(0), -1, -1)
        pe = pe.permute(0, 2, 1)  # pe -> (batch_size,d_model,seq_len)
        return pe


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
        attn_flag=False,
        cross_flag=False,
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

        self.attention = (
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=4)
            if attn_flag
            else None
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = x.permute(2, 0, 1)  # (seq_len, batch_size, channels)
        if self.attention is not None:
            x, _ = self.attention(x, x, x)  # 根据标志应用或不应用注意力机制
        x = x.permute(1, 2, 0)  # (batch_size, channels, seq_len)
        out = self.relu(x + shortcut)
        return out


class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 将输入的维度从 (batch_size, in_channels, spatial_dim) 转换为 (batch_size, in_channels, 1)
        # x = x.unsqueeze(2)
        # print("before avg_pool: ", x.shape)

        avg_out = self.avg_pool(x)

        # print("after avg_pool: ", avg_out.shape)

        max_out = self.max_pool(x)

        # 将输出的维度从 (batch_size, in_channels, 1) 转换为 (batch_size, in_channels)
        # avg_out = avg_out.squeeze(2)
        # max_out = max_out.squeeze(2)

        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)

        out = avg_out + max_out
        return out


class ChannelAttentionBlock1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionBlock1D, self).__init__()
        self.ca = ChannelAttention1D(in_channels, reduction_ratio)

    def forward(self, x):
        # print("before ca: ", x.shape)
        # print("after ca: ", self.ca(x).shape)
        x = self.ca(x) * x
        return x


class FusionNetwork(nn.Module):
    def __init__(self, fusion_num):
        super(FusionNetwork, self).__init__()
        # 2 input features, 2 output features (for weight1 and weight2)
        self.weights = nn.Parameter(torch.rand(fusion_num), requires_grad=True)
        self.activation = nn.Softmax(dim=0)  # Apply softmax to get normalized weights

    def forward(self, *inputs):
        # Linear layer to get unnormalized weights
        weights = self.weights
        # Apply softmax to get normalized weights
        weights = self.activation(weights)

        # 对每个输入张量按权重进行加权融合
        output = sum(
            input_tensor * weight for input_tensor, weight in zip(inputs, weights)
        )

        return output


class NeuroFetalNet_without_fusion_3(nn.Module):
    def __init__(
        self, num_classes: int = 2, in_channels: int = 1, dropout: float = 0.5
    ):
        super(NeuroFetalNet_without_fusion_3, self).__init__()

        # self.ca_block_0 = ChannelAttentionBlock1D(
        #     in_channels=in_channels, reduction_ratio=2
        # )

        self.resnet_k3 = nn.Sequential(
            ResBlock(in_channels, 64, kernel_size=3, padding=1),
            nn.AvgPool1d(2, 2),
            ResBlock(64, 64, kernel_size=3, padding=1),
            nn.AvgPool1d(2, 2),
            ResBlock(64, 128, kernel_size=3, padding=1),
            ChannelAttentionBlock1D(128),
            nn.AvgPool1d(2, 2),
            ResBlock(128, 128, kernel_size=3, padding=1),
            nn.AvgPool1d(2, 2),
            ResBlock(128, 256, kernel_size=3, padding=1),
            nn.AvgPool1d(2, 2),
            ResBlock(256, 256, kernel_size=3, padding=1),
            nn.AvgPool1d(2, 2),
            ResBlock(256, 256, kernel_size=3, padding=1),
        )

        self.resnet_k9 = nn.Sequential(
            ResBlock(in_channels, 64, kernel_size=9, padding=4),
            nn.AvgPool1d(4, 4),
            ResBlock(64, 128, kernel_size=9, padding=4),
            nn.AvgPool1d(4, 4),
            ResBlock(128, 256, kernel_size=9, padding=4),
            ChannelAttentionBlock1D(256),
            nn.AvgPool1d(4, 4),
            ResBlock(256, 256, kernel_size=9, padding=4),
        )

        # self.ca_block_1 = ChannelAttentionBlock1D(256 * 2)

        self.gru_k3 = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            # dropout=dropout,
        )

        self.gru_k9 = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            # dropout=dropout,
        )

        self.ln_k3 = nn.LayerNorm(256 * 2)

        self.ln_k9 = nn.LayerNorm(256 * 2)

        self.position_embedding_k3 = PositionalEmbedding(256)

        self.position_embedding_k9 = PositionalEmbedding(256)

        self.fc_k3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.fc_k9 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.fusion = FusionNetwork(fusion_num=2)

    def forward(self, x):
        # pos_embed = self.position_embedding(x)
        # x = x + pos_embed
        # x = self.ca_block_0(x)

        x_k3 = self.resnet_k3(x)  # x -> (batch_size, channels, seq_len)
        pos_embed_k3 = self.position_embedding_k3(x_k3)
        x_k3 = x_k3 + pos_embed_k3
        x_k3 = x_k3.permute(0, 2, 1)  # x_k3 -> (batch_size, seq_len, channels)
        x_k3, _ = self.gru_k3(x_k3)
        x_k3 = self.ln_k3(x_k3)
        x_k3 = x_k3[:, -1, :]
        x_k3 = self.fc_k3(x_k3)

        # x_k3 = x_k3.permute(2, 0, 1)  # x_k3 -> (seq_len, batch_size, channels)
        # x_k3, _ = self.attention(x_k3, x_k3, x_k3)

        x_k9 = self.resnet_k9(x)
        pos_embed_k9 = self.position_embedding_k9(x_k9)
        x_k9 = x_k9 + pos_embed_k9
        x_k9 = x_k9.permute(0, 2, 1)
        x_k9, _ = self.gru_k9(x_k9)
        x_k9 = self.ln_k9(x_k9)
        x_k9 = x_k9[:, -1, :]
        x_k9 = self.fc_k9(x_k9)

        # x_k9 = x_k9.permute(2, 0, 1)
        # x_k9, _ = self.attention(x_k9, x_k9, x_k9)

        # socre fusion
        # x = self.fusion(x_k3, x_k9)

        x = x_k3

        return x


# main函数
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = NeuroFetalNet_without_fusion_3(num_classes=2, in_channels=1).to(device)

    print(model)
    input = torch.randn(32, 1, 4800).to(device)
    output = model(input)

    print(input.size())
    print(output.size())
