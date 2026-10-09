"""Small causal TCN with a persistence residual, suitable for CPU experiments."""
import torch
from torch import nn


class CausalConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation: int):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(input_channels, output_channels, 3,
                              dilation=dilation, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[..., :-self.padding]


class Block(nn.Module):
    def __init__(self, input_channels: int, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.sequence = nn.Sequential(CausalConv(input_channels, channels, dilation),
                                      nn.ReLU(), nn.Dropout(dropout),
                                      CausalConv(channels, channels, dilation),
                                      nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(input_channels, channels, 1) if input_channels != channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.sequence(x) + self.skip(x))


class TCN(nn.Module):
    def __init__(self, input_channels: int, horizon: int = 6, channels: int = 16,
                 dropout: float = .1):
        super().__init__()
        self.backbone = nn.Sequential(Block(input_channels, channels, 1, dropout),
                                      Block(channels, channels, 2, dropout),
                                      Block(channels, channels, 4, dropout),
                                      Block(channels, channels, 8, dropout))
        self.head = nn.Linear(channels, horizon)
        # Start from persistence. Training learns standardized future deltas.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x.transpose(1, 2))[..., -1]
        return self.head(hidden) + x[:, -1, 0:1]
