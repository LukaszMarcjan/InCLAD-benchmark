from __future__ import annotations

import torch
from torch import nn

from pyclad.models.draem.standard.blocks import DoubleConvBlock, UpsampleBlock


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 128):
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels=in_channels, base_width=base_width)
        self.decoder = DecoderReconstructive(base_width=base_width, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encoder(x)
        return self.decoder(bottleneck)


class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        self.block1 = DoubleConvBlock(in_channels, base_width)
        self.block2 = DoubleConvBlock(base_width, base_width * 2)
        self.block3 = DoubleConvBlock(base_width * 2, base_width * 4)
        self.block4 = DoubleConvBlock(base_width * 4, base_width * 8)
        self.block5 = DoubleConvBlock(base_width * 8, base_width * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.block5(x)
        return x


class DecoderReconstructive(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 3):
        super().__init__()
        self.up1 = UpsampleBlock(base_width * 8, base_width * 8)
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.up2 = UpsampleBlock(base_width * 4, base_width * 4)
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.up3 = UpsampleBlock(base_width * 2, base_width * 2)
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.up4 = UpsampleBlock(base_width, base_width)
        self.db4 = DoubleConvBlock(base_width, base_width)
        self.output_head = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        x = self.up1(bottleneck)
        x = self.db1(x)
        x = self.up2(x)
        x = self.db2(x)
        x = self.up3(x)
        x = self.db3(x)
        x = self.up4(x)
        x = self.db4(x)
        return self.output_head(x)
