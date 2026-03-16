"""Residual 3D U-Net backbone."""

from __future__ import annotations

import torch
from torch import nn


def _norm(num_channels: int) -> nn.Module:
    return nn.InstanceNorm3d(num_channels, affine=True)


def _act() -> nn.Module:
    return nn.LeakyReLU(0.01, inplace=True)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.act1 = _act()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        self.act2 = _act()
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                _norm(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + identity)
        return out


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        self.downsample = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                _norm(out_channels),
                _act(),
            )
            if downsample
            else None
        )
        block_in = out_channels if downsample else in_channels
        self.block1 = ResidualBlock3D(block_in, out_channels)
        self.block2 = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block1 = ResidualBlock3D(out_channels + skip_channels, out_channels)
        self.block2 = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff = [skip.shape[idx] - x.shape[idx] for idx in range(2, 5)]
        if any(diff):
            x = nn.functional.pad(
                x,
                [
                    max(diff[2] // 2, 0),
                    max(diff[2] - diff[2] // 2, 0),
                    max(diff[1] // 2, 0),
                    max(diff[1] - diff[1] // 2, 0),
                    max(diff[0] // 2, 0),
                    max(diff[0] - diff[0] // 2, 0),
                ],
            )
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


class ResidualUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        stage_channels: tuple[int, ...] = (32, 64, 96, 160, 256, 320),
        dropout_bottleneck: float = 0.1,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 6:
            raise ValueError("Expected 6 stage channels for the 6-stage encoder")
        self.stem = EncoderStage(in_channels, stage_channels[0], downsample=False)
        self.encoders = nn.ModuleList(
            [
                EncoderStage(stage_channels[0], stage_channels[1], downsample=True),
                EncoderStage(stage_channels[1], stage_channels[2], downsample=True),
                EncoderStage(stage_channels[2], stage_channels[3], downsample=True),
                EncoderStage(stage_channels[3], stage_channels[4], downsample=True),
                EncoderStage(stage_channels[4], stage_channels[5], downsample=True),
            ]
        )
        self.dropout = nn.Dropout3d(dropout_bottleneck) if dropout_bottleneck > 0 else nn.Identity()
        self.decoders = nn.ModuleList(
            [
                DecoderStage(stage_channels[5], stage_channels[4], stage_channels[4]),
                DecoderStage(stage_channels[4], stage_channels[3], stage_channels[3]),
                DecoderStage(stage_channels[3], stage_channels[2], stage_channels[2]),
                DecoderStage(stage_channels[2], stage_channels[1], stage_channels[1]),
                DecoderStage(stage_channels[1], stage_channels[0], stage_channels[0]),
            ]
        )
        self.head = nn.Conv3d(stage_channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1]))
        x = self.dropout(skips[-1])
        x = self.decoders[0](x, skips[-2])
        x = self.decoders[1](x, skips[-3])
        x = self.decoders[2](x, skips[-4])
        x = self.decoders[3](x, skips[-5])
        x = self.decoders[4](x, skips[-6])
        return self.head(x)
