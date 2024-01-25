import torch.nn as nn
import torch.nn.functional as F
import torch


class BiGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU Layer

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        # Forward propagate GRU
        out, _ = self.gru(x)

        # Layer normalization
        out = self.layer_norm(out)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


# main函数
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = BiGRU().to(device)

    print(model)
    input = torch.randn(32, 1, 4800).to(device)
    output = model(input)

    print(input.size())
    print(output.size())
