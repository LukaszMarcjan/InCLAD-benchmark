from typing import Optional

from pydantic import BaseModel, Field


class GANomalyConfig(BaseModel):
    in_channels: int = Field(default=3, gt=0)
    input_size: tuple[int, int] = (256, 256)

    n_features: int = Field(default=64, gt=0)
    latent_vec_size: int = Field(default=100, gt=0)
    extra_layers: int = Field(default=0, ge=0)
    add_final_conv_layer: bool = True

    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=15, ge=0)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    adam_beta1: float = Field(default=0.5, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = Field(default=None, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_restore_best: bool = True

    adversarial_weight: float = Field(default=1.0, ge=0.0)
    contextual_weight: float = Field(default=50.0, ge=0.0)
    encoding_weight: float = Field(default=1.0, ge=0.0)

    threshold: Optional[float] = None
    threshold_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)

    # GANomaly reconstructs through a tanh decoder, so inputs are mapped to [-1, 1].
    normalize_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    normalize_std: tuple[float, ...] = (0.5, 0.5, 0.5)

    device: Optional[str] = None
