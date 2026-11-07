from __future__ import annotations

import torch
from torch import nn

from pyclad.models.draem.standard.blocks import DoubleConvBlock, UpsampleBlock


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 2, base_width: int = 64):
        super().__init__()
        self.encoder = EncoderDiscriminative(in_channels=in_channels, base_width=base_width)
        self.decoder = DecoderDiscriminative(base_width=base_width, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(*features)


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        self.block1 = DoubleConvBlock(in_channels, base_width)
        self.block2 = DoubleConvBlock(base_width, base_width * 2)
        self.block3 = DoubleConvBlock(base_width * 2, base_width * 4)
        self.block4 = DoubleConvBlock(base_width * 4, base_width * 8)
        self.block5 = DoubleConvBlock(base_width * 8, base_width * 8)
        self.block6 = DoubleConvBlock(base_width * 8, base_width * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b1 = self.block1(x)
        b2 = self.block2(self.pool(b1))
        b3 = self.block3(self.pool(b2))
        b4 = self.block4(self.pool(b3))
        b5 = self.block5(self.pool(b4))
        b6 = self.block6(self.pool(b5))
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 2):
        super().__init__()
        self.up_b = UpsampleBlock(base_width * 8, base_width * 8)
        self.db_b = DoubleConvBlock(base_width * 16, base_width * 8)

        self.up1 = UpsampleBlock(base_width * 8, base_width * 4)
        self.db1 = DoubleConvBlock(base_width * 12, base_width * 4)

        self.up2 = UpsampleBlock(base_width * 4, base_width * 2)
        self.db2 = DoubleConvBlock(base_width * 6, base_width * 2)

        self.up3 = UpsampleBlock(base_width * 2, base_width)
        self.db3 = DoubleConvBlock(base_width * 3, base_width)

        self.up4 = UpsampleBlock(base_width, base_width)
        self.db4 = DoubleConvBlock(base_width * 2, base_width)

        self.output_head = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        b1: torch.Tensor,
        b2: torch.Tensor,
        b3: torch.Tensor,
        b4: torch.Tensor,
        b5: torch.Tensor,
        b6: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up_b(b6)
        x = self.db_b(torch.cat((x, b5), dim=1))

        x = self.up1(x)
        x = self.db1(torch.cat((x, b4), dim=1))

        x = self.up2(x)
        x = self.db2(torch.cat((x, b3), dim=1))

        x = self.up3(x)
        x = self.db3(torch.cat((x, b2), dim=1))

        x = self.up4(x)
        x = self.db4(torch.cat((x, b1), dim=1))

        return self.output_head(x)
