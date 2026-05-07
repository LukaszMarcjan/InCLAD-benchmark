from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def normalize_batch(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


class SmallPatchDescriptionNetwork(nn.Module):
    def __init__(
        self,
        out_channels: int,
        padding: bool = False,
        normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.normalize_mean = tuple(normalize_mean)
        self.normalize_std = tuple(normalize_std)

        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_batch(x, mean=self.normalize_mean, std=self.normalize_std)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class MediumPatchDescriptionNetwork(nn.Module):
    def __init__(
        self,
        out_channels: int,
        padding: bool = False,
        normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.normalize_mean = tuple(normalize_mean)
        self.normalize_std = tuple(normalize_std)

        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_batch(x, mean=self.normalize_mean, std=self.normalize_std)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)
