from typing import Literal, Optional

from pydantic import BaseModel, Field


class RD4ADConfig(BaseModel):
    in_channels: int = Field(default=3, gt=0)
    input_size: tuple[int, int] = (224, 224)

    backbone_name: Literal["resnet18", "resnet34", "resnet50", "wide_resnet50_2"] = "resnet18"
    pretrained_encoder: bool = True
    freeze_encoder: bool = True

    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=200, ge=0)
    learning_rate: float = Field(default=5e-3, gt=0.0)
    adam_beta1: float = Field(default=0.5, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_restore_best: bool = True

    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    score_smoothing_kernel: int = Field(default=1, gt=0)
    score_smoothing_sigma: float = Field(default=4.0, ge=0.0)
    score_mode: Literal["max", "mean"] = "max"
    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    device: Optional[str] = None
