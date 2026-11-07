from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class OneForMoreConfig(BaseModel):
    """Configuration for the native One-for-More/CDAD implementation.

    The paper-scale defaults that materially affect the continual-learning
    protocol are retained (500 base epochs, 100 incremental epochs, gamma
    0.999 and ten DDIM steps).  ``base_channels`` is intentionally smaller
    than Stable Diffusion's 320-channel U-Net so the model remains usable as a
    pyCLAD benchmark component.  A diffusers VAE can be selected explicitly;
    otherwise a deterministic factor-eight image-space codec is used.
    """

    in_channels: int = Field(default=3, gt=0)
    input_size: tuple[int, int] = (256, 256)
    latent_downsample: int = Field(default=8, gt=0)

    # Optional frozen VAE. Keeping this opt-in avoids an implicit model
    # download during construction and makes offline benchmark runs explicit.
    vae_model_id: Optional[str] = None
    vae_subfolder: Optional[str] = None
    vae_local_files_only: bool = False
    vae_latent_channels: int = Field(default=4, gt=0)

    base_channels: int = Field(default=64, ge=8)
    conditioner_depth: int = Field(default=8, ge=0)
    conditioner_heads: int = Field(default=8, gt=0)
    conditioner_feedforward: int = Field(default=1024, gt=0)
    conditioner_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    neighbor_size: Optional[tuple[int, int]] = (7, 7)

    diffusion_steps: int = Field(default=1000, gt=1)
    inference_steps: int = Field(default=10, gt=0)
    inference_start_step: int = Field(default=999, ge=1)
    beta_start: float = Field(default=8.5e-4, gt=0.0, lt=1.0)
    beta_end: float = Field(default=1.2e-2, gt=0.0, lt=1.0)
    condition_loss_weight: float = Field(default=1.0, ge=0.0)

    epochs: int = Field(default=500, ge=0)
    incremental_epochs: int = Field(default=100, ge=0)
    batch_size: int = Field(default=12, gt=0)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    show_training_progress: bool = True

    patch_count_min: int = Field(default=1, gt=0)
    patch_count_max: int = Field(default=4, gt=0)
    patch_width_fraction: tuple[float, float] = (0.03, 0.40)
    patch_height_fraction: tuple[float, float] = (0.03, 0.40)

    projection_energy_threshold: float = Field(default=0.999, gt=0.0, le=1.0)
    projection_batches: int = Field(default=10, gt=0)
    projection_max_columns: Optional[int] = Field(default=50_000, gt=0)
    freeze_unprojected_parameters: bool = True

    backbone_name: Optional[str] = "resnet50"
    backbone_return_nodes: tuple[str, ...] = ("layer1", "layer2", "layer3")
    pretrained_backbone: bool = True
    score_smoothing_sigma: float = Field(default=5.0, ge=0.0)
    score_mode: str = "max"

    normalize_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    normalize_std: tuple[float, ...] = (0.5, 0.5, 0.5)
    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)
    threshold_calibration_samples: Optional[int] = Field(default=64, gt=0)

    random_seed: int = 0
    device: Optional[str] = None

    @model_validator(mode="after")
    def validate_shape_and_ranges(self):
        height, width = self.input_size
        if height <= 0 or width <= 0:
            raise ValueError("input_size dimensions must be positive")
        if height % self.latent_downsample != 0 or width % self.latent_downsample != 0:
            raise ValueError("input_size dimensions must be divisible by latent_downsample")
        if height // self.latent_downsample < 4 or width // self.latent_downsample < 4:
            raise ValueError("encoded spatial dimensions must be at least 4x4")
        if self.in_channels != 3:
            raise ValueError("One-for-More currently supports RGB input only")
        if self.base_channels % self.conditioner_heads != 0:
            raise ValueError("base_channels must be divisible by conditioner_heads")
        if self.inference_start_step >= self.diffusion_steps:
            raise ValueError("inference_start_step must be smaller than diffusion_steps")
        if self.inference_steps > self.inference_start_step + 1:
            raise ValueError("inference_steps cannot exceed inference_start_step + 1")
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be smaller than beta_end")
        if self.patch_count_min > self.patch_count_max:
            raise ValueError("patch_count_min cannot exceed patch_count_max")
        for field_name, bounds in (
            ("patch_width_fraction", self.patch_width_fraction),
            ("patch_height_fraction", self.patch_height_fraction),
        ):
            if not (0.0 < bounds[0] <= bounds[1] <= 1.0):
                raise ValueError(f"{field_name} must satisfy 0 < min <= max <= 1")
        if self.neighbor_size is not None and (
            self.neighbor_size[0] <= 0 or self.neighbor_size[1] <= 0
        ):
            raise ValueError("neighbor_size dimensions must be positive")
        if self.backbone_name is not None and len(self.backbone_return_nodes) == 0:
            raise ValueError("backbone_return_nodes cannot be empty when a backbone is enabled")
        if self.score_mode not in {"max", "mean"}:
            raise ValueError("score_mode must be 'max' or 'mean'")
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3:
            raise ValueError("normalize_mean and normalize_std must contain three values")
        return self
