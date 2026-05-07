from __future__ import annotations

import torch
from torch import nn


class AddCoords(nn.Module):
    def __init__(self, with_radius: bool = False):
        super().__init__()
        self.with_radius = with_radius

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = input_tensor.shape
        device = input_tensor.device
        dtype = input_tensor.dtype

        yy = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype).view(1, 1, 1, width)

        yy = yy.expand(batch_size, -1, -1, width)
        xx = xx.expand(batch_size, -1, height, -1)
        output = torch.cat((input_tensor, yy, xx), dim=1)

        if self.with_radius:
            rr = torch.sqrt(torch.clamp((xx - 0.5) ** 2 + (yy - 0.5) ** 2, min=0.0))
            output = torch.cat((output, rr), dim=1)
        return output


class CoordConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_radius: bool = False,
    ):
        super().__init__()
        extra_channels = 2 + int(with_radius)
        self.addcoords = AddCoords(with_radius=with_radius)
        self.conv = nn.Conv2d(
            in_channels + extra_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(self.addcoords(input_tensor))
