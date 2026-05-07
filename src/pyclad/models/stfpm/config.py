from typing import Literal, Optional

from pydantic import BaseModel, Field


class STFPMConfig(BaseModel):
    backbone_name: str = "resnet18"
    backbone_return_nodes: Optional[tuple[str, ...]] = None

    pretrained_teacher: bool = True
    pretrained_student: bool = False
    freeze_teacher: bool = True

    input_size: tuple[int, int] = (256, 256)
    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=100, ge=0)
    learning_rate: float = Field(default=0.4, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_restore_best: bool = True

    score_mode: Literal["max", "mean"] = "max"
    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    normalize_mean: Optional[tuple[float, ...]] = (0.485, 0.456, 0.406)
    normalize_std: Optional[tuple[float, ...]] = (0.229, 0.224, 0.225)

    device: Optional[str] = None
