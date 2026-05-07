from pyclad.models.padim.config import PaDiMConfig
from pyclad.models.vision.backbones import TorchvisionFeatureExtractor


def default_backbone_return_nodes(backbone_name: str) -> list[str]:
    defaults = {
        "resnet18": ["layer1", "layer2", "layer3"],
        "resnet34": ["layer1", "layer2", "layer3"],
        "resnet50": ["layer1", "layer2", "layer3"],
        "wide_resnet50_2": ["layer1", "layer2", "layer3"],
        "mobilenet_v2": ["features.3", "features.6", "features.13"],
        "efficientnet_b0": ["features.2", "features.3", "features.5"],
        "efficientnet_b1": ["features.2", "features.3", "features.5"],
        "efficientnet_b2": ["features.2", "features.3", "features.5"],
        "efficientnet_b3": ["features.2", "features.3", "features.5"],
        "efficientnet_b4": ["features.2", "features.3", "features.5"],
        "efficientnet_v2_s": ["features.2", "features.3", "features.5"],
        "efficientnet_v2_m": ["features.2", "features.3", "features.5"],
        "efficientnet_v2_l": ["features.2", "features.3", "features.5"],
    }
    if backbone_name not in defaults:
        raise ValueError(f"No default return nodes for '{backbone_name}'. Set backbone_return_nodes explicitly.")
    return defaults[backbone_name]


def build(config: PaDiMConfig) -> TorchvisionFeatureExtractor:
    used_nodes = config.backbone_return_nodes or tuple(default_backbone_return_nodes(config.backbone_name))
    return TorchvisionFeatureExtractor(
        backbone_name=config.backbone_name,
        return_nodes=used_nodes,
        pretrained=config.pretrained_backbone,
        freeze=config.freeze_backbone,
    )
