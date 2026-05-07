from pyclad.models.rd4ad.architecture import RD4ADArchitecture
from pyclad.models.rd4ad.config import RD4ADConfig


def build(config: RD4ADConfig) -> RD4ADArchitecture:
    return RD4ADArchitecture(
        backbone_name=config.backbone_name,
        input_size=config.input_size,
        pretrained_encoder=config.pretrained_encoder,
        freeze_encoder=config.freeze_encoder,
        score_smoothing_kernel=config.score_smoothing_kernel,
        score_smoothing_sigma=config.score_smoothing_sigma,
    )
