from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.models.one_for_more.architecture import OneForMoreArchitecture
from pyclad.models.one_for_more.config import OneForMoreConfig
from pyclad.models.one_for_more.projection import (
    IterativeSubspaceProjector,
    activation_matrix,
    protected_modules,
)
from pyclad.models.vision.preprocessing import ImagePreprocessor


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device}', but CUDA is not available")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(f"Requested device '{device}', but MPS is not available")
        return resolved
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class OneForMore(Model):
    """Continual Diffusion Anomaly Detector from One-for-More.

    Repeated :meth:`fit` calls are interpreted as consecutive concepts.  After
    every concept, input-feature subspaces are updated with iSVD; from the
    second concept onward gradients are projected onto their orthogonal
    complements.  No previous images are retained.
    """

    def __init__(self, config: Optional[OneForMoreConfig] = None):
        self.config = config or OneForMoreConfig()
        self._device = _resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=self.config.in_channels,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )
        self.module = OneForMoreArchitecture(self.config).to(self._device)
        self._projector = IterativeSubspaceProjector(
            energy_threshold=self.config.projection_energy_threshold,
            max_columns=self.config.projection_max_columns,
            random_seed=self.config.random_seed,
        )
        self._tasks_seen = 0
        self._threshold = self.config.threshold
        self._last_loss: Optional[float] = None
        self._last_diffusion_loss: Optional[float] = None
        self._last_condition_loss: Optional[float] = None
        self._last_prediction_data: Optional[np.ndarray] = None
        self._last_base_score_maps: Optional[np.ndarray] = None
        self._last_prediction_size: Optional[tuple[int, int]] = None

    def _invalidate_prediction_cache(self) -> None:
        self._last_prediction_data = None
        self._last_base_score_maps = None
        self._last_prediction_size = None

    def _prepare_batches(self, data: np.ndarray, shuffle: bool, seed_offset: int = 0) -> DataLoader:
        images = self._preprocessor.transform(data)
        generator = torch.Generator(device="cpu").manual_seed(self.config.random_seed + seed_offset)
        return DataLoader(
            TensorDataset(images),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator if shuffle else None,
        )

    def _corrupt_with_patches(self, images: torch.Tensor, seed: int) -> torch.Tensor:
        corrupted = images.clone()
        donor = torch.roll(images, shifts=1, dims=0)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        height, width = images.shape[-2:]

        if len(images) == 1:
            donor = torch.rand(images.shape, generator=generator, dtype=images.dtype) * 2.0 - 1.0

        for batch_index in range(len(images)):
            patch_count = int(
                torch.randint(
                    self.config.patch_count_min,
                    self.config.patch_count_max + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            for _ in range(patch_count):
                height_fraction = float(
                    torch.empty(1).uniform_(
                        self.config.patch_height_fraction[0],
                        self.config.patch_height_fraction[1],
                        generator=generator,
                    ).item()
                )
                width_fraction = float(
                    torch.empty(1).uniform_(
                        self.config.patch_width_fraction[0],
                        self.config.patch_width_fraction[1],
                        generator=generator,
                    ).item()
                )
                patch_height = min(height, max(1, round(height * height_fraction)))
                patch_width = min(width, max(1, round(width * width_fraction)))
                top = int(torch.randint(0, height - patch_height + 1, (1,), generator=generator).item())
                left = int(torch.randint(0, width - patch_width + 1, (1,), generator=generator).item())
                corrupted[
                    batch_index,
                    :,
                    top : top + patch_height,
                    left : left + patch_width,
                ] = donor[
                    batch_index,
                    :,
                    top : top + patch_height,
                    left : left + patch_width,
                ]
        return corrupted

    def _epochs_for_current_task(self) -> int:
        return self.config.epochs if self._tasks_seen == 0 else self.config.incremental_epochs

    def fit(self, data: np.ndarray):
        if len(data) == 0:
            return
        self._invalidate_prediction_cache()

        self.module = self.module.to(self._device)
        epochs = self._epochs_for_current_task()
        parameters = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        final_losses: dict[str, float] = {}
        epoch_iterator = range(epochs)
        if self.config.show_training_progress and epochs > 0:
            from tqdm.auto import tqdm

            epoch_iterator = tqdm(epoch_iterator, desc=f"One-for-More task {self._tasks_seen}", leave=False)

        self.module.train()
        global_batch = 0
        for epoch in epoch_iterator:
            loader = self._prepare_batches(data, shuffle=True, seed_offset=self._tasks_seen * 100_000 + epoch)
            for (clean_cpu,) in loader:
                corruption_seed = self.config.random_seed + self._tasks_seen * 1_000_000 + global_batch
                corrupted_cpu = self._corrupt_with_patches(clean_cpu, corruption_seed)
                clean = clean_cpu.to(self._device, dtype=torch.float32)
                corrupted = corrupted_cpu.to(self._device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                loss, components = self.module.training_loss(clean, corrupted)
                loss.backward()
                if self._tasks_seen > 0:
                    self._projector.apply_to_model(
                        self.module.continual_module,
                        freeze_unprojected=self.config.freeze_unprojected_parameters,
                    )
                optimizer.step()

                final_losses = {
                    "total": float(loss.detach().cpu()),
                    "diffusion": float(components["diffusion"].cpu()),
                    "condition": float(components["condition"].cpu()),
                }
                global_batch += 1

        if final_losses:
            self._last_loss = final_losses["total"]
            self._last_diffusion_loss = final_losses["diffusion"]
            self._last_condition_loss = final_losses["condition"]

        self._collect_projection_subspaces(data)
        self._tasks_seen += 1
        self._calibrate_threshold(data)

    @torch.no_grad()
    def _collect_projection_subspaces(self, data: np.ndarray) -> None:
        self.module.eval()
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name: str):
            def hook(layer: nn.Module, inputs, output):
                matrix = activation_matrix(layer, inputs)
                if matrix is not None:
                    captured[name] = matrix

            return hook

        for name, layer in protected_modules(self.module.continual_module):
            handles.append(layer.register_forward_hook(make_hook(name)))

        try:
            loader = self._prepare_batches(data, shuffle=False)
            for batch_index, (clean_cpu,) in enumerate(loader):
                if batch_index >= self.config.projection_batches:
                    break
                corrupted_cpu = self._corrupt_with_patches(
                    clean_cpu,
                    self.config.random_seed + self._tasks_seen * 1_000_000 + 500_000 + batch_index,
                )
                clean = clean_cpu.to(self._device, dtype=torch.float32)
                corrupted = corrupted_cpu.to(self._device, dtype=torch.float32)
                clean_latents = self.module.codec.encode(clean)
                timesteps = torch.full(
                    (len(clean),),
                    self.config.diffusion_steps // 2,
                    dtype=torch.long,
                    device=self._device,
                )
                noise = torch.zeros_like(clean_latents)
                noisy = self.module.q_sample(clean_latents, timesteps, noise)
                self.module.predict_noise(noisy, timesteps, corrupted)

                for name, matrix in captured.items():
                    self._projector.update(name, matrix)
                captured.clear()
        finally:
            for handle in handles:
                handle.remove()

    def _calibrate_threshold(self, data: np.ndarray) -> None:
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
            return
        calibration_data = data
        if self.config.threshold_calibration_samples is not None:
            calibration_data = data[: self.config.threshold_calibration_samples]
        scores = self.score_data(calibration_data)
        self._threshold = float(np.quantile(scores, self.config.threshold_quantile)) if len(scores) else 0.0

    @torch.no_grad()
    def _batch_reconstruct(self, images: torch.Tensor, batch_index: int) -> torch.Tensor:
        self.module = self.module.to(self._device).eval()
        return self.module.reconstruct(
            images.to(self._device, dtype=torch.float32),
            seed=self.config.random_seed + batch_index,
        )

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        reconstructions: list[np.ndarray] = []
        for batch_index, (images,) in enumerate(self._prepare_batches(data, shuffle=False)):
            reconstructed = self._batch_reconstruct(images, batch_index)
            reconstructions.append(reconstructed.cpu().numpy().astype(np.float32, copy=False))
        if not reconstructions:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(reconstructions, axis=0)

    @torch.no_grad()
    def _base_maps_for(self, data: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        if (
            data is self._last_prediction_data
            and self._last_base_score_maps is not None
            and self._last_prediction_size is not None
        ):
            return self._last_base_score_maps, self._last_prediction_size
        original_size = self._preprocessor.spatial_size(data)
        maps: list[np.ndarray] = []
        for batch_index, (images_cpu,) in enumerate(self._prepare_batches(data, shuffle=False)):
            images = images_cpu.to(self._device, dtype=torch.float32)
            reconstructed = self._batch_reconstruct(images_cpu, batch_index)
            batch_maps = self.module.anomaly_maps(images, reconstructed)
            maps.append(batch_maps.cpu().numpy().astype(np.float32, copy=False))

        base_maps = np.concatenate(maps, axis=0)
        self._last_prediction_data = data
        self._last_base_score_maps = base_maps
        self._last_prediction_size = original_size
        return base_maps, original_size

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        base_maps, original_size = self._base_maps_for(data)
        if resize_to_input and tuple(base_maps.shape[-2:]) != tuple(original_size):
            score_maps = F.interpolate(
                torch.from_numpy(base_maps[:, None]),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )[:, 0].numpy().astype(np.float32, copy=False)
        else:
            score_maps = base_maps

        if self.config.score_smoothing_sigma > 0.0:
            score_maps = np.asarray(
                [
                    ndimage.gaussian_filter(score_map, sigma=self.config.score_smoothing_sigma)
                    for score_map in score_maps
                ],
                dtype=np.float32,
            )
        return score_maps

    def score_data(self, data: np.ndarray) -> np.ndarray:
        score_maps = self.score_maps(data, resize_to_input=False)
        if len(score_maps) == 0:
            return np.asarray([], dtype=np.float32)
        if self.config.score_mode == "mean":
            return score_maps.mean(axis=(1, 2)).astype(np.float32, copy=False)
        return score_maps.max(axis=(1, 2)).astype(np.float32, copy=False)

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score_data(data)
        if self.config.threshold is not None:
            threshold = float(self.config.threshold)
        elif self._threshold is not None:
            threshold = float(self._threshold)
        else:
            threshold = float(np.quantile(scores, self.config.threshold_quantile)) if len(scores) else 0.0
        return (scores > threshold).astype(int), scores

    def name(self) -> str:
        return "One-for-More (CDAD)"

    def additional_info(self) -> Dict:
        summary = self._projector.summary()
        return {
            "implementation": "native_cdm_amn_isvd",
            "tasks_seen": self._tasks_seen,
            "input_size": self.config.input_size,
            "vae_model_id": self.config.vae_model_id,
            "base_channels": self.config.base_channels,
            "conditioner_depth": self.config.conditioner_depth,
            "diffusion_steps": self.config.diffusion_steps,
            "inference_steps": self.config.inference_steps,
            "epochs": self.config.epochs,
            "incremental_epochs": self.config.incremental_epochs,
            "learning_rate": self.config.learning_rate,
            "projection_energy_threshold": self.config.projection_energy_threshold,
            "projection_layers": summary.layers,
            "projection_total_rank": summary.total_rank,
            "projection_ranks": summary.ranks,
            "backbone": self.config.backbone_name,
            "pretrained_backbone": self.config.pretrained_backbone,
            "threshold": self._threshold,
            "threshold_quantile": self.config.threshold_quantile,
            "last_loss": self._last_loss,
            "last_diffusion_loss": self._last_diffusion_loss,
            "last_condition_loss": self._last_condition_loss,
        }
