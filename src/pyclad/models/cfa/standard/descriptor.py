from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pyclad.models.cfa.standard.coordconv import CoordConv2d


class Descriptor(nn.Module):
    def __init__(self, gamma_d: int, feature_map_channels: int, backbone_name: str):
        super().__init__()
        if gamma_d <= 0:
            raise ValueError("gamma_d must be positive")

        self.backbone_name = backbone_name
        self.feature_map_channels = feature_map_channels
        self.output_channels = max(1, feature_map_channels // gamma_d)
        self._scale_efficientnet_features = backbone_name.startswith("efficientnet")
        self.layer = CoordConv2d(feature_map_channels, self.output_channels, kernel_size=1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) == 0:
            raise ValueError("Descriptor requires at least one feature map")

        target_size = features[0].shape[-2:]
        aligned_features: list[torch.Tensor] = []
        for feature in features:
            pooled = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
            if self._scale_efficientnet_features:
                pooled = pooled / max(1, pooled.shape[1])
            if pooled.shape[-2:] != target_size:
                pooled = F.interpolate(pooled, size=target_size, mode="bilinear", align_corners=False)
            aligned_features.append(pooled)

        descriptor_input = torch.cat(aligned_features, dim=1)
        return self.layer(descriptor_input)
