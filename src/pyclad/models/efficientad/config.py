from typing import Literal, Optional

from pydantic import BaseModel, Field


class EfficientADConfig(BaseModel):
    in_channels: int = Field(default=3, gt=0)
    input_size: tuple[int, int] = (256, 256)

    model_size: Literal["small", "medium"] = "small"
    teacher_out_channels: int = Field(default=384, gt=0)
    padding: bool = False
    pad_maps: bool = True
    teacher_weights_path: Optional[str] = None
    teacher_weights_strict: bool = True

    batch_size: int = Field(default=1, gt=0)
    epochs: int = Field(default=50, ge=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    use_lr_scheduler: bool = True
    lr_decay_fraction: float = Field(default=0.95, gt=0.0, le=1.0)
    lr_decay_gamma: float = Field(default=0.1, gt=0.0, lt=1.0)
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_restore_best: bool = True

    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    penalty_source_root: Optional[str] = None
    penalty_source_paths: Optional[tuple[str, ...]] = None
    penalty_resize_scale: float = Field(default=2.0, gt=1.0)
    penalty_grayscale_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    ae_augmentation_min: float = Field(default=0.8, gt=0.0)
    ae_augmentation_max: float = Field(default=1.2, gt=0.0)

    student_hard_quantile: float = Field(default=0.999, gt=0.0, lt=1.0)
    map_quantile_low: float = Field(default=0.9, gt=0.0, lt=1.0)
    map_quantile_high: float = Field(default=0.995, gt=0.0, lt=1.0)
    map_padding: int = Field(default=4, ge=0)

    score_mode: Literal["max", "mean"] = "max"
    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    random_seed: int = 0
    device: Optional[str] = None
