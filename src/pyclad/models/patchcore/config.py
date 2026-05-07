from typing import Optional

from pydantic import BaseModel, Field


class PatchCoreConfig(BaseModel):
    backbone_name: str = "resnet18"
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_backbone: bool = False
    freeze_backbone: bool = True

    input_size: tuple[int, int] = (224, 224)
    batch_size: int = Field(default=32, gt=0)

    pretrain_embed_dimension: int = Field(default=1024, gt=0)
    target_embed_dimension: int = Field(default=1024, gt=0)
    patchsize: int = Field(default=3, gt=0)
    patchstride: int = Field(default=1, gt=0)
    coreset_sampling_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
    coreset_projection_dimension: int = Field(default=128, gt=0)
    coreset_starting_points: int = Field(default=10, gt=0)
    n_neighbors: int = Field(default=1, gt=0)

    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    normalize_mean: Optional[tuple[float, ...]] = None
    normalize_std: Optional[tuple[float, ...]] = None

    random_seed: int = 0
    smoothing_sigma: float = Field(default=4.0, ge=0.0)
    device: Optional[str] = None
