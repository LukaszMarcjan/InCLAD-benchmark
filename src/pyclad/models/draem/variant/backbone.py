from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pyclad.models.draem.standard.blocks import DoubleConvBlock, weights_init
from pyclad.models.vision.backbones import TorchvisionFeatureExtractor, default_backbone_return_nodes


class _BackboneEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        return_nodes: Optional[Sequence[str]],
        pretrained: bool,
        freeze: bool,
        input_size: tuple[int, int],
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError("The DRAEM backbone variant currently requires in_channels=3")

        used_nodes = tuple(return_nodes) if return_nodes is not None else tuple(default_backbone_return_nodes(backbone_name))
        self._freeze = freeze
        self._extractor = TorchvisionFeatureExtractor(
            backbone_name=backbone_name,
            return_nodes=used_nodes,
            pretrained=pretrained,
            freeze=freeze,
        )
        self._out_channels = self._extractor.infer_out_channels(input_size=input_size)

    @property
    def out_channels(self) -> tuple[int, ...]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self._freeze:
            was_training = self._extractor.training
            self._extractor.eval()
            with torch.no_grad():
                features = self._extractor(x)
            self._extractor.train(was_training)
            return features
        return self._extractor(x)


class _BackboneDecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = DoubleConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class BackboneReconstructiveSubNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: tuple[int, int],
        backbone_name: str,
        return_nodes: Optional[Sequence[str]] = None,
        pretrained: bool = False,
        freeze: bool = False,
        decoder_channels: Optional[tuple[int, ...]] = None,
        normalize_mean: Optional[Sequence[float]] = (0.485, 0.456, 0.406),
        normalize_std: Optional[Sequence[float]] = (0.229, 0.224, 0.225),
    ):
        super().__init__()

        if (normalize_mean is None) != (normalize_std is None):
            raise ValueError("normalize_mean and normalize_std must be both set or both None")
        if normalize_mean is not None and (len(tuple(normalize_mean)) != in_channels or len(tuple(normalize_std)) != in_channels):
            raise ValueError(
                f"normalize_mean/std length must match in_channels={in_channels}, got "
                f"{len(tuple(normalize_mean))} and {len(tuple(normalize_std))}"
            )

        self.encoder = _BackboneEncoder(
            backbone_name=backbone_name,
            in_channels=in_channels,
            return_nodes=return_nodes,
            pretrained=pretrained,
            freeze=freeze,
            input_size=input_size,
        )
        encoder_channels = self.encoder.out_channels
        if len(encoder_channels) < 2:
            raise ValueError("Backbone reconstructive variant requires at least 2 feature levels")

        used_decoder_channels = (
            tuple(decoder_channels) if decoder_channels is not None else tuple(reversed(encoder_channels[:-1]))
        )
        if len(used_decoder_channels) != len(encoder_channels) - 1:
            raise ValueError(
                "backbone_decoder_channels length must be exactly len(backbone_return_nodes) - 1 "
                f"({len(encoder_channels) - 1}), got {len(used_decoder_channels)}"
            )

        self.bottleneck = DoubleConvBlock(encoder_channels[-1], encoder_channels[-1])

        decoder_stages = []
        current_channels = encoder_channels[-1]
        for skip_channels, stage_out_channels in zip(reversed(encoder_channels[:-1]), used_decoder_channels):
            decoder_stages.append(
                _BackboneDecoderStage(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=stage_out_channels,
                )
            )
            current_channels = stage_out_channels

        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.output_head = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

        if normalize_mean is None:
            self.register_buffer("_normalize_mean", None, persistent=False)
            self.register_buffer("_normalize_std", None, persistent=False)
        else:
            self.register_buffer(
                "_normalize_mean",
                torch.tensor(tuple(normalize_mean), dtype=torch.float32).view(1, -1, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_normalize_std",
                torch.tensor(tuple(normalize_std), dtype=torch.float32).view(1, -1, 1, 1),
                persistent=False,
            )

    def initialize_decoder_weights(self) -> None:
        self.bottleneck.apply(weights_init)
        self.decoder_stages.apply(weights_init)
        self.output_head.apply(weights_init)

    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self._normalize_mean is None or self._normalize_std is None:
            return x
        mean = self._normalize_mean.to(device=x.device, dtype=x.dtype)
        std = self._normalize_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(self._normalize_inputs(x))

        reconstruction = self.bottleneck(features[-1])
        for stage, skip in zip(self.decoder_stages, reversed(features[:-1])):
            reconstruction = stage(reconstruction, skip)

        if reconstruction.shape[-2:] != input_size:
            reconstruction = F.interpolate(reconstruction, size=input_size, mode="bilinear", align_corners=False)

        return self.output_head(reconstruction)
