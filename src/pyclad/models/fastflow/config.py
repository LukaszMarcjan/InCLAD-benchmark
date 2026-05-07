from typing import Literal, Optional

from pydantic import BaseModel, Field


class FastFlowConfig(BaseModel):
    in_channels: int = Field(default=3, gt=0)
    input_size: tuple[int, int] = (256, 256)

    backbone_name: str = "wide_resnet50_2"
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    normalize_features: bool = True

    batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=200, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_restore_best: bool = True

    flow_steps: int = Field(default=8, gt=0)
    conv3x3_only: bool = False
    hidden_ratio: float = Field(default=1.0, gt=0.0)
    affine_clamping: float = Field(default=2.0, gt=0.0)

    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    score_mode: Literal["max", "mean"] = "max"
    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    device: Optional[str] = None
