from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_depthwise_conv1d(channels: int, kernel_size: int, bias: bool = False) -> nn.Conv1d:
    """Depthwise 1D conv with 'same' padding."""
    padding = kernel_size // 2
    return nn.Conv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        padding=padding,
        groups=channels,
        bias=bias,
    )


class Smoothing(nn.Module):
    """
    Per-channel depthwise smoothing of an attention map with a learnable blend factor α ∈ (0, 1):
        att_out = (1 - α) * σ(x) + α * σ(depthwise_conv(σ(x)))
    Notes:
      - Uses a logit-parameterized α to keep it in (0,1) without clamps.
      - Initializes depthwise kernel to moving-average (1 / kernel_size).
    """
    def __init__(self, channels: int, kernel_size: int = 3, alpha: float = 0.1) -> None:
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size must be odd and >= 1"
        self.channels = channels

        self.smoothing = _make_depthwise_conv1d(channels, kernel_size, bias=False)
        with torch.no_grad():
            # moving-average initialization
            w = torch.zeros_like(self.smoothing.weight)
            w.fill_(1.0 / kernel_size)
            self.smoothing.weight.copy_(w)

        self.raw_alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape: (B, C, T)
        att = torch.sigmoid(x)
        smooth = self.smoothing(att)
        out = (1.0 - self.raw_alpha) * att + self.raw_alpha * torch.sigmoid(smooth)
        return out


class ChannelAttention1D(nn.Module):
    """
    Channel Attention (CAM-1D):
      M_c(x) = MLP(AvgPool(x)) + MLP(MaxPool(x))
      y = σ( M_c(x) ) ⊗ x   (optionally smoothed)
    """
    def __init__(self, channels: int, reduction: int = 16, use_smoothing: bool = True,
                 smoothing_alpha: float = 0.1) -> None:
        super().__init__()
        assert channels > 0, "channels must be positive"
        red = max(1, channels // max(1, reduction))

        self.mlp = nn.Sequential(
            nn.Linear(channels, red, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(red, channels, bias=True),
        )

        self.use_smoothing = use_smoothing
        self.smoothing = Smoothing(channels, kernel_size=3, alpha=smoothing_alpha) if use_smoothing else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, _ = x.shape
        avg = F.adaptive_avg_pool1d(x, 1).view(b, c)   # (B, C)
        mx  = F.adaptive_max_pool1d(x, 1).view(b, c)   # (B, C)

        att = self.mlp(avg).view(b, c, 1) + self.mlp(mx).view(b, c, 1)  # (B, C, 1)

        if self.use_smoothing:
            att = self.smoothing(att)                   # (B, C, 1), smoothed + sigmoid inside
            out = att * x
        else:
            out = torch.sigmoid(att) * x
        return out


class SpatialAttention1D(nn.Module):
    """
    Spatial Attention (SAM-1D):
      M_s(x) = Conv([AvgPool_c(x); MaxPool_c(x)]), conv takes 2->1 channels
      y = σ( M_s(x) ) ⊗ x   (optionally smoothed per-time-step)
    """
    def __init__(self, kernel_size: int = 7, use_smoothing: bool = True,
                 smoothing_alpha: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size >= 1, "kernel_size must be odd and >= 1"
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

        self.use_smoothing = use_smoothing
        # For spatial attention, the attention map is 1-channel; smooth that map, then broadcast over C.
        self.smoothing = Smoothing(channels=1, kernel_size=3, alpha=smoothing_alpha) if use_smoothing else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Pool along channel dimension
        max_map, _ = torch.max(x, dim=1, keepdim=True)   # (B, 1, T)
        avg_map = torch.mean(x, dim=1, keepdim=True)     # (B, 1, T)
        att = self.conv(torch.cat([max_map, avg_map], dim=1))  # (B, 1, T)

        if self.use_smoothing:
            att = self.smoothing(att)                    # (B, 1, T)
            out = att * x                                # broadcast over C
        else:
            out = torch.sigmoid(att) * x
        return out


class CBAM1D(nn.Module):
    """
    Convolutional Block Attention Module (1D variant).
    Order: Channel Attention -> Spatial Attention -> (optional residual).
    """
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        cam_smoothing_alpha: float = 0.1,
        sam_kernel_size: int = 7,
        sam_smoothing_alpha: float = 0.1,
        use_smoothing: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual

        self.cam = ChannelAttention1D(
            channels=channels,
            reduction=reduction,
            use_smoothing=use_smoothing,
            smoothing_alpha=cam_smoothing_alpha,
        )
        self.sam = SpatialAttention1D(
            kernel_size=sam_kernel_size,
            use_smoothing=use_smoothing,
            smoothing_alpha=sam_smoothing_alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sam(self.cam(x))
        return x + y if self.residual else y
