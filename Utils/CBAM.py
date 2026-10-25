import torch
import torch.nn as nn
from Parameters import CBAM_REDUCTION,CBAM_KERNEL

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = CBAM_REDUCTION):
        super().__init__()
        hidden = max(int(channels) // int(reduction_ratio), 8)
        self.channels = int(channels)
        self.hidden = hidden
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = CBAM_KERNEL):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.max(x, dim=1, keepdim=True).values
        return torch.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))

class StageCBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = CBAM_REDUCTION):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(CBAM_KERNEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)

