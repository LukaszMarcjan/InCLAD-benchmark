from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class UCADConfig(BaseModel):
    """Configuration for the clean-room UCAD implementation.

    The paper extracts features after zero-based ViT block index 5 and stores
    196 key and knowledge vectors per task.  Structure-based contrastive learning
    (SCL) is enabled by selecting either ``precomputed`` or ``sam`` structure
    masks.  ``structure_mode='none'`` is the CPM-only ablation and therefore
    performs no prompt optimization.
    """

    backbone_name: str = "vit_base_patch16_224.augreg_in21k"
    pretrained_backbone: bool = True
    freeze_backbone: bool = True

    input_size: tuple[int, int] = (224, 224)
    batch_size: int = Field(default=8, gt=0)
    feature_layer: int = Field(
        default=6,
        gt=0,
        description="One-based ViT block used for patch features (the paper's zero-based layer 5).",
    )
    prompt_depth: int = Field(default=6, gt=0)
    prompt_length: int = Field(default=1, gt=0)
    prompt_init: Literal["zero", "uniform"] = "uniform"
    warm_start_prompts: bool = False

    target_embed_dimension: int = Field(default=1024, gt=0)
    key_size: int = Field(default=196, gt=0)
    knowledge_size: int = Field(default=196, gt=0)
    coreset_projection_dimension: int = Field(default=128, gt=0)
    coreset_starting_points: int = Field(default=10, gt=0)
    n_neighbors: int = Field(default=1, gt=0)

    epochs: int = Field(default=25, ge=0)
    learning_rate: float = Field(default=5e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    contrastive_temperature: float = Field(default=0.5, gt=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)

    structure_mode: Literal["none", "precomputed", "sam"] = "none"
    structure_mask_root: Optional[str] = None
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_b"

    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)
    smoothing_sigma: float = Field(default=4.0, ge=0.0)
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    random_seed: int = 0
    device: Optional[str] = None

    @model_validator(mode="after")
    def validate_ucad_configuration(self):
        if not self.freeze_backbone:
            raise ValueError("UCAD requires a frozen ViT backbone; set freeze_backbone=True")
        if self.prompt_depth > self.feature_layer:
            raise ValueError(
                "prompt_depth cannot exceed feature_layer because later prompts cannot affect the features"
            )
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if self.structure_mode == "precomputed":
            if not self.structure_mask_root:
                raise ValueError("structure_mask_root is required when structure_mode='precomputed'")
            if not Path(self.structure_mask_root).expanduser().exists():
                raise ValueError(f"structure_mask_root does not exist: {self.structure_mask_root}")
        if self.structure_mode == "sam":
            if not self.sam_checkpoint:
                raise ValueError("sam_checkpoint is required when structure_mode='sam'")
            if not Path(self.sam_checkpoint).expanduser().is_file():
                raise ValueError(f"sam_checkpoint does not exist: {self.sam_checkpoint}")
        return self
