"""Simple U-Net baseline model for segmentation comparison."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, num_channels: int, num_groups: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "group":
        groups = max(1, min(num_groups, num_channels))
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    """Two-layer convolutional block with configurable normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, norm_type: str, norm_groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm_type, out_channels, norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm_type, out_channels, norm_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample and fuse skip features in the decoder."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_type: str, norm_groups: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, norm_type, norm_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    """A compact U-Net for baseline comparison against DINSNet."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        depth: int = 4,
        norm_type: str = "group",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("SimpleUNet depth must be >= 2.")

        channels: List[int] = [base_channels * (2**idx) for idx in range(depth)]
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for out_channels in channels:
            self.encoders.append(ConvBlock(current_channels, out_channels, norm_type, norm_groups))
            current_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, norm_type, norm_groups)

        self.decoders = nn.ModuleList()
        decoder_in = channels[-1] * 2
        for skip_channels in reversed(channels[:-1]):
            self.decoders.append(UpBlock(decoder_in, skip_channels, skip_channels, norm_type, norm_groups))
            decoder_in = skip_channels

        self.head = nn.Conv2d(decoder_in, num_classes, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            if idx < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)

        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)
