from pyclad.models.cfa.architecture import CFAArchitecture
from pyclad.models.cfa.config import CFAConfig


def build(config: CFAConfig) -> CFAArchitecture:
    return CFAArchitecture(
        in_channels=config.in_channels,
        input_size=config.input_size,
        backbone_name=config.backbone_name,
        backbone_return_nodes=config.backbone_return_nodes,
        pretrained_backbone=config.pretrained_backbone,
        freeze_backbone=config.freeze_backbone,
        gamma_c=config.gamma_c,
        gamma_d=config.gamma_d,
        k_neighbors=config.k_neighbors,
        repulsion_neighbors=config.repulsion_neighbors,
        nu=config.nu,
        alpha=config.alpha,
        radius_init=config.radius_init,
        score_smoothing_kernel=config.score_smoothing_kernel,
        score_smoothing_sigma=config.score_smoothing_sigma,
        score_mode=config.score_mode,
        random_seed=config.random_seed,
    )
