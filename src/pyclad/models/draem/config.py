from typing import Literal, Optional

from pydantic import BaseModel, Field


class DRAEMConfig(BaseModel):
    variant: Literal["standard", "backbone"] = "standard"
    in_channels: int = Field(default=3, gt=0)
    input_size: tuple[int, int] = (256, 256)

    reconstructive_base_width: int = Field(default=128, gt=0)
    discriminative_base_width: int = Field(default=64, gt=0)
    initialize_weights: bool = True
    backbone_name: str = "resnet18"
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_backbone: bool = False
    freeze_backbone: bool = False
    backbone_decoder_channels: Optional[tuple[int, ...]] = None
    backbone_normalize_mean: Optional[tuple[float, ...]] = (0.485, 0.456, 0.406)
    backbone_normalize_std: Optional[tuple[float, ...]] = (0.229, 0.224, 0.225)

    batch_size: int = Field(default=4, gt=0)
    epochs: int = Field(default=50, ge=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    use_lr_scheduler: bool = True
    lr_milestones: Optional[tuple[int, ...]] = None
    lr_gamma: float = Field(default=0.2, gt=0.0, lt=1.0)
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_restore_best: bool = True

    reconstruction_l2_weight: float = Field(default=1.0, ge=0.0)
    reconstruction_ssim_weight: float = Field(default=1.0, ge=0.0)
    reconstruction_ssim_window: int = Field(default=11, gt=0)
    segmentation_loss_weight: float = Field(default=1.0, ge=0.0)
    focal_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    focal_gamma: float = Field(default=2.0, ge=0.0)

    anomaly_source_root: Optional[str] = None
    anomaly_source_paths: Optional[tuple[str, ...]] = None
    allow_random_color_fallback: bool = True
    texture_augmentation: bool = True
    texture_augmentation_count: int = Field(default=3, ge=0)
    perlin_scale_min: int = Field(default=0, ge=0)
    perlin_scale_max: int = Field(default=6, ge=0)
    perlin_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    anomaly_blend_range: tuple[float, float] = (0.2, 1.0)
    anomaly_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    rotation_degrees: float = Field(default=90.0, ge=0.0)
    rotation_probability: float = Field(default=0.3, ge=0.0, le=1.0)

    normalize_mean: Optional[tuple[float, ...]] = None
    normalize_std: Optional[tuple[float, ...]] = None

    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)
    score_smoothing_kernel: int = Field(default=21, gt=0)

    random_seed: int = 0
    device: Optional[str] = None
