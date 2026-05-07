from __future__ import annotations

from pathlib import Path

import torch

from pyclad.models.efficientad.architecture import EfficientADArchitecture
from pyclad.models.efficientad.config import EfficientADConfig
from pyclad.models.efficientad.standard import AutoEncoder, MediumPatchDescriptionNetwork, SmallPatchDescriptionNetwork


def _load_teacher_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        stripped = {}
        for key, value in checkpoint.items():
            if key.startswith("teacher."):
                stripped[key.removeprefix("teacher.")] = value
            elif not key.startswith("student.") and not key.startswith("autoencoder.") and not key.startswith("ae."):
                stripped[key] = value
        checkpoint = stripped

    if not isinstance(checkpoint, dict):
        raise ValueError("teacher_weights_path must point to a PyTorch state_dict checkpoint")
    return checkpoint


def build(config: EfficientADConfig) -> EfficientADArchitecture:
    network_cls = SmallPatchDescriptionNetwork if config.model_size == "small" else MediumPatchDescriptionNetwork

    teacher = network_cls(
        out_channels=config.teacher_out_channels,
        padding=config.padding,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
    )
    student = network_cls(
        out_channels=config.teacher_out_channels * 2,
        padding=config.padding,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
    )
    autoencoder = AutoEncoder(
        out_channels=config.teacher_out_channels,
        padding=config.padding,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
    )

    if config.teacher_weights_path is not None:
        teacher_path = Path(config.teacher_weights_path)
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher weights file does not exist: {teacher_path}")
        teacher.load_state_dict(_load_teacher_state_dict(str(teacher_path)), strict=config.teacher_weights_strict)

    return EfficientADArchitecture(
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_out_channels=config.teacher_out_channels,
        padding=config.padding,
        pad_maps=config.pad_maps,
        map_padding=config.map_padding,
        student_hard_quantile=config.student_hard_quantile,
        ae_augmentation_min=config.ae_augmentation_min,
        ae_augmentation_max=config.ae_augmentation_max,
    )
