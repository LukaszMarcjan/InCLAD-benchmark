from torch import nn

from pyclad.models.stfpm.config import STFPMConfig
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


class STFPMArchitecture(nn.Module):
    def __init__(
        self,
        teacher: TorchvisionFeatureExtractor,
        student: TorchvisionFeatureExtractor,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student

    @property
    def return_nodes(self) -> tuple[str, ...]:
        return self.teacher.return_nodes

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x):
        return self.teacher(x), self.student(x)


def build(config: STFPMConfig) -> nn.Module:
    used_nodes = config.backbone_return_nodes or tuple(default_backbone_return_nodes(config.backbone_name))
    teacher = TorchvisionFeatureExtractor(
        backbone_name=config.backbone_name,
        return_nodes=used_nodes,
        pretrained=config.pretrained_teacher,
        freeze=config.freeze_teacher,
    )
    student = TorchvisionFeatureExtractor(
        backbone_name=config.backbone_name,
        return_nodes=used_nodes,
        pretrained=config.pretrained_student,
        freeze=False,
    )
    return STFPMArchitecture(teacher=teacher, student=student)
