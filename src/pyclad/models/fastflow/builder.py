from pyclad.models.fastflow.architecture import FastFlowArchitecture
from pyclad.models.fastflow.config import FastFlowConfig


def build(config: FastFlowConfig) -> FastFlowArchitecture:
    return FastFlowArchitecture(
        in_channels=config.in_channels,
        input_size=config.input_size,
        backbone_name=config.backbone_name,
        backbone_return_nodes=config.backbone_return_nodes,
        pretrained_backbone=config.pretrained_backbone,
        freeze_backbone=config.freeze_backbone,
        normalize_features=config.normalize_features,
        flow_steps=config.flow_steps,
        conv3x3_only=config.conv3x3_only,
        hidden_ratio=config.hidden_ratio,
        affine_clamping=config.affine_clamping,
        score_mode=config.score_mode,
    )
