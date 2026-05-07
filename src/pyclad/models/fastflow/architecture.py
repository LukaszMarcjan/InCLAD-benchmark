from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from pyclad.models.fastflow.standard import create_fast_flow_block
from pyclad.models.vision.backbones import TorchvisionFeatureExtractor, default_backbone_return_nodes as vision_default_backbone_return_nodes

_SUPPORTED_BACKBONES: tuple[str, ...] = (
    "resnet18",
    "resnet34",
    "resnet50",
    "wide_resnet50_2",
    "mobilenet_v2",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
)


def supported_backbone_names() -> tuple[str, ...]:
    return _SUPPORTED_BACKBONES


def default_fastflow_return_nodes(backbone_name: str) -> tuple[str, ...]:
    try:
        default_nodes = tuple(vision_default_backbone_return_nodes(backbone_name))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported FastFlow backbone '{backbone_name}'. Supported default backbone presets: {', '.join(supported_backbone_names())}. "
            "You can still use a custom torchvision backbone by setting backbone_return_nodes explicitly."
        ) from exc

    # FastFlow is typically applied to a small set of intermediate 2D feature maps.
    if len(default_nodes) >= 4:
        return default_nodes[1:4]
    if len(default_nodes) >= 3:
        return default_nodes[:3]
    if len(default_nodes) >= 1:
        return default_nodes
    raise ValueError(f"Backbone '{backbone_name}' did not expose any default feature nodes")


class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size: tuple[int, int]):
        super().__init__()
        self.input_size = tuple(input_size)

    def forward(self, hidden_variables: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(hidden_variables) == 0:
            raise ValueError("FastFlow received no hidden variables to build the anomaly map")

        flow_maps: list[torch.Tensor] = []
        for hidden_variable in hidden_variables:
            log_prob = -0.5 * torch.mean(hidden_variable**2, dim=1, keepdim=True)
            probability = torch.exp(log_prob)
            flow_map = F.interpolate(
                -probability,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)

        return torch.mean(torch.stack(flow_maps, dim=-1), dim=-1)


class FastFlowArchitecture(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: tuple[int, int],
        backbone_name: str,
        backbone_return_nodes: tuple[str, ...] | None,
        pretrained_backbone: bool,
        freeze_backbone: bool,
        normalize_features: bool,
        flow_steps: int,
        conv3x3_only: bool,
        hidden_ratio: float,
        affine_clamping: float,
        score_mode: str,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError("FastFlow currently supports RGB inputs only and requires in_channels=3")

        return_nodes = backbone_return_nodes or default_fastflow_return_nodes(backbone_name)

        self.input_size = tuple(input_size)
        self.freeze_backbone = freeze_backbone
        self.score_mode = score_mode
        self.return_nodes = tuple(return_nodes)
        self.feature_extractor = TorchvisionFeatureExtractor(
            backbone_name=backbone_name,
            return_nodes=self.return_nodes,
            pretrained=pretrained_backbone,
            freeze=freeze_backbone,
        )

        feature_shapes = self._infer_feature_shapes(input_size=self.input_size, in_channels=in_channels)
        self.feature_shapes = tuple(feature_shapes)
        self.channels = tuple(shape[0] for shape in self.feature_shapes)

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(list(shape), elementwise_affine=True) if normalize_features else nn.Identity()
                for shape in self.feature_shapes
            ]
        )
        self.fast_flow_blocks = nn.ModuleList(
            [
                create_fast_flow_block(
                    input_dimensions=shape,
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                    clamp=affine_clamping,
                )
                for shape in self.feature_shapes
            ]
        )
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=self.input_size)

    def _infer_feature_shapes(self, input_size: tuple[int, int], in_channels: int) -> list[tuple[int, int, int]]:
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        with torch.no_grad():
            try:
                device = next(self.feature_extractor.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            dummy = torch.zeros((1, in_channels, input_size[0], input_size[1]), dtype=torch.float32, device=device)
            features = self.feature_extractor(dummy)
        self.feature_extractor.train(was_training)

        return [tuple(int(dimension) for dimension in feature.shape[1:]) for feature in features]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train(mode)
        self.norms.train(mode)
        self.fast_flow_blocks.train(mode)
        self.anomaly_map_generator.train(mode)
        return self

    def _extract_features(self, batch: torch.Tensor) -> list[torch.Tensor]:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.feature_extractor(batch)
        else:
            features = self.feature_extractor(batch)
        return [norm(feature) for norm, feature in zip(self.norms, features)]

    def _flow_outputs(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features = self._extract_features(batch)

        hidden_variables: list[torch.Tensor] = []
        log_jacobians: list[torch.Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)
        return hidden_variables, log_jacobians

    def forward_train(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not self.training:
            self.train()
        return self._flow_outputs(batch)

    def forward(self, batch: torch.Tensor):
        hidden_variables, log_jacobians = self._flow_outputs(batch)
        if self.training:
            return hidden_variables, log_jacobians

        anomaly_map = self.anomaly_map_generator(hidden_variables)[:, 0]
        if self.score_mode == "mean":
            anomaly_scores = anomaly_map.mean(dim=(1, 2))
        else:
            anomaly_scores = anomaly_map.view(anomaly_map.shape[0], -1).max(dim=1).values
        return anomaly_map, anomaly_scores

    @staticmethod
    def fastflow_loss(hidden_variables: Sequence[torch.Tensor], log_jacobians: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(hidden_variables) == 0 or len(log_jacobians) == 0:
            raise ValueError("FastFlow loss requires non-empty hidden variables and log Jacobians")

        loss = hidden_variables[0].new_tensor(0.0)
        for hidden_variable, log_jacobian in zip(hidden_variables, log_jacobians):
            loss = loss + torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - log_jacobian)
        return loss
