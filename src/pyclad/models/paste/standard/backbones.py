from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class PaSTeBackboneSpec:
    default_ad_layers: tuple[int, ...]
    stage_builder: Callable[[nn.Module], list[nn.Module]]


def _resnet_stages(model: nn.Module) -> list[nn.Module]:
    return [
        nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]


def _features_stages(model: nn.Module) -> list[nn.Module]:
    return list(model.features.children())


_BACKBONE_SPECS: dict[str, PaSTeBackboneSpec] = {
    "resnet18": PaSTeBackboneSpec(default_ad_layers=(1, 2, 3), stage_builder=_resnet_stages),
    "resnet34": PaSTeBackboneSpec(default_ad_layers=(1, 2, 3), stage_builder=_resnet_stages),
    "resnet50": PaSTeBackboneSpec(default_ad_layers=(1, 2, 3), stage_builder=_resnet_stages),
    "wide_resnet50_2": PaSTeBackboneSpec(default_ad_layers=(1, 2, 3), stage_builder=_resnet_stages),
    "mobilenet_v2": PaSTeBackboneSpec(default_ad_layers=(3, 6, 13), stage_builder=_features_stages),
    "efficientnet_b0": PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages),
    "efficientnet_b1": PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages),
    "efficientnet_b2": PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages),
    "efficientnet_b3": PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages),
    "efficientnet_b4": PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages),
    "efficientnet_v2_s": PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages),
    "efficientnet_v2_m": PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages),
    "efficientnet_v2_l": PaSTeBackboneSpec(default_ad_layers=(2, 3, 5), stage_builder=_features_stages),
}


def supported_backbone_names() -> tuple[str, ...]:
    return tuple(_BACKBONE_SPECS.keys())


def default_ad_layers(backbone_name: str) -> tuple[int, ...]:
    return resolve_backbone_spec(backbone_name).default_ad_layers


def resolve_backbone_spec(backbone_name: str) -> PaSTeBackboneSpec:
    try:
        return _BACKBONE_SPECS[backbone_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PaSTe backbone '{backbone_name}'. Supported backbones: {', '.join(supported_backbone_names())}"
        ) from exc


def _resolve_torchvision_weights(tv_models, model_fn, backbone_name: str):
    get_model_weights = getattr(tv_models, "get_model_weights", None)
    if get_model_weights is not None:
        return get_model_weights(model_fn).DEFAULT

    attr_name = f"{backbone_name}_weights".lower()
    for attr in dir(tv_models):
        if attr.lower() == attr_name:
            enum_cls = getattr(tv_models, attr)
            if hasattr(enum_cls, "DEFAULT"):
                return enum_cls.DEFAULT
            if hasattr(enum_cls, "IMAGENET1K_V1"):
                return enum_cls.IMAGENET1K_V1
    return None


def _create_torchvision_model(backbone_name: str, pretrained: bool) -> nn.Module:
    import torchvision.models as tv_models

    model_fn = getattr(tv_models, backbone_name, None)
    if model_fn is None:
        raise ValueError(f"Unsupported torchvision backbone '{backbone_name}'")

    params = inspect.signature(model_fn).parameters
    if "weights" in params:
        weights = _resolve_torchvision_weights(tv_models, model_fn, backbone_name) if pretrained else None
        return model_fn(weights=weights)
    return model_fn(pretrained=pretrained)


def supported_stage_indices(backbone_name: str, pretrained: bool = False) -> tuple[int, ...]:
    spec = resolve_backbone_spec(backbone_name)
    model = _create_torchvision_model(backbone_name, pretrained=pretrained)
    stages = spec.stage_builder(model)
    return tuple(range(len(stages)))


class PaSTeBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        ad_layers: tuple[int, ...],
        pretrained: bool,
        freeze: bool,
        bootstrap_layer: int | None,
        is_teacher: bool,
    ):
        super().__init__()

        spec = resolve_backbone_spec(backbone_name)
        stages = spec.stage_builder(_create_torchvision_model(backbone_name, pretrained=pretrained))
        max_layer = max(ad_layers)

        if max_layer >= len(stages):
            raise ValueError(
                f"Requested PaSTe ad layer {max_layer} for backbone '{backbone_name}', but it has only "
                f"{len(stages)} stages indexed from 0 to {len(stages) - 1}"
            )

        self.backbone_name = backbone_name
        self.ad_layers = tuple(sorted(ad_layers))
        self.bootstrap_layer = bootstrap_layer
        self.is_teacher = is_teacher

        if is_teacher:
            stage_slice = slice(0, max_layer + 1)
            self.layer_offset = 0
        else:
            start_layer = 0 if bootstrap_layer is None else bootstrap_layer + 1
            stage_slice = slice(start_layer, max_layer + 1)
            self.layer_offset = start_layer

        self.stages = nn.ModuleList(stages[stage_slice])

        if freeze:
            for parameter in self.stages.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        features: list[torch.Tensor] = []
        bootstrap_feature: torch.Tensor | None = None

        for layer_index, stage in enumerate(self.stages, start=self.layer_offset):
            x = stage(x)
            if layer_index in self.ad_layers:
                features.append(x)
            if self.bootstrap_layer is not None and layer_index == self.bootstrap_layer:
                bootstrap_feature = x.clone()

        return features, bootstrap_feature
