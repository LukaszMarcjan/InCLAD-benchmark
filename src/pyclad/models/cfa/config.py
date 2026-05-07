from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class CFAConfig(BaseModel):
    in_channels: Literal[3] = 3
    input_size: tuple[int, int] = (224, 224)

    backbone_name: str = "wide_resnet50_2"
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_backbone: bool = True
    freeze_backbone: bool = True

    batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=10, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=5e-4, ge=0.0)
    use_amsgrad: bool = True
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_restore_best: bool = True

    gamma_c: int = Field(default=1, gt=0)
    gamma_d: int = Field(default=1, gt=0)
    k_neighbors: int = Field(default=3, gt=0)
    repulsion_neighbors: int = Field(default=3, gt=0)
    nu: float = Field(default=1e-3, gt=0.0)
    alpha: float = Field(default=1e-1, gt=0.0)
    radius_init: float = Field(default=1e-5, gt=0.0)

    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    score_smoothing_kernel: int = Field(default=3, gt=0)
    score_smoothing_sigma: float = Field(default=4.0, ge=0.0)
    score_mode: Literal["max", "mean"] = "max"
    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    random_seed: int = 0
    device: Optional[str] = None

    @field_validator("score_smoothing_kernel")
    @classmethod
    def _kernel_must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("score_smoothing_kernel must be odd")
        return v

    @field_validator("backbone_return_nodes")
    @classmethod
    def _return_nodes_not_empty(cls, v: Optional[tuple[str, ...]]) -> Optional[tuple[str, ...]]:
        if v is not None and len(v) == 0:
            raise ValueError("backbone_return_nodes must contain at least one feature node")
        return v
