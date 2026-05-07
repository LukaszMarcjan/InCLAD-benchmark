from typing import Literal, Optional

from pydantic import BaseModel, Field


class PaDiMConfig(BaseModel):
    backbone_name: str = "resnet18"
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_backbone: bool = False
    freeze_backbone: bool = True

    input_size: tuple[int, int] = (224, 224)
    batch_size: int = Field(default=32, gt=0)

    n_features: Optional[int] = Field(default=100, gt=0)
    covariance_regularization: float = Field(default=0.01, gt=0.0)

    score_mode: Literal["max", "mean"] = "max"
    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    normalize_mean: Optional[tuple[float, ...]] = None
    normalize_std: Optional[tuple[float, ...]] = None

    random_seed: int = 0
    device: Optional[str] = None
