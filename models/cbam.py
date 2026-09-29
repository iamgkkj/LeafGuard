"""
CBAM: Convolutional Block Attention Module
Reference: Woo et al., ECCV 2018 - "CBAM: Convolutional Block Attention Module"
- Channel Attention: avg-pool + max-pool -> shared MLP (reduction r=16) -> sigmoid
- Spatial Attention: channel-pooled avg+max (2 channels) -> 7x7 conv -> sigmoid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)
        avg_out = self.mlp(avg)
        max_out = self.mlp(mx)
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> channel pooled
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        y = torch.cat([avg_out, max_out], dim=1)  # (B,2,H,W)
        y = self.conv(y)  # (B,1,H,W)
        return self.sigmoid(y)


class CBAM(nn.Module):
    """CBAM block: ChannelAttention -> SpatialAttention with residual multiply."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ca = x * self.ca(x)
        x_sa = x_ca * self.sa(x_ca)
        return x_sa

    def get_attention_maps(self, x: torch.Tensor):
        """Return (channel_attn, spatial_attn) for visualization."""
        ca = self.ca(x)
        x_ca = x * ca
        sa = self.sa(x_ca)
        return ca, sa


def get_cbam_params(model: nn.Module):
    """Count CBAM-specific params."""
    total = 0
    for m in model.modules():
        if isinstance(m, (ChannelAttention, SpatialAttention, CBAM)):
            total += sum(p.numel() for p in m.parameters())
    return total


if __name__ == "__main__":
    # quick shape test
    cbam = CBAM(64)
    x = torch.randn(2, 64, 56, 56)
    y = cbam(x)
    assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"
    ca, sa = cbam.get_attention_maps(x)
    assert ca.shape == (2, 64, 1, 1)
    assert sa.shape == (2, 1, 56, 56)
    assert float(ca.min()) >= 0 and float(ca.max()) <= 1
    assert float(sa.min()) >= 0 and float(sa.max()) <= 1
    print("CBAM test passed - shapes OK, attention in [0,1]")
    print(f" CBAM(64) params: {sum(p.numel() for p in cbam.parameters())}")
