from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.models.padim.builder import build
from pyclad.models.padim.config import PaDiMConfig
from pyclad.models.vision.features import align_feature_maps
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import resolve_device


class PaDiM(Model):
    def __init__(self, config: Optional[PaDiMConfig] = None):
        self.config = config or PaDiMConfig()

        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=3,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )
        self.module = build(self.config).to(self._device).eval()

        self._selected_feature_indices: Optional[np.ndarray] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_precision: Optional[np.ndarray] = None
        self._feature_map_size: Optional[tuple[int, int]] = None
        self._total_feature_channels: Optional[int] = None
        self._threshold = self.config.threshold

    def _prepare_batches(self, data: np.ndarray) -> DataLoader:
        x_t = self._preprocessor.transform(data)
        dataset = TensorDataset(x_t)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _extract_embeddings(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        self.module = self.module.to(self._device).eval()

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_x,) in self._prepare_batches(data):
                x = batch_x.to(self._device, dtype=torch.float32)
                batch_embeddings = align_feature_maps(self.module(x))
                embeddings.append(batch_embeddings.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(embeddings, axis=0) if embeddings else np.asarray([], dtype=np.float32)

    def _selected_channels(self, total_channels: int) -> np.ndarray:
        if self._selected_feature_indices is not None:
            if int(self._selected_feature_indices.max(initial=-1)) >= total_channels:
                raise ValueError("Stored feature selection is incompatible with the current backbone output size")
            return self._selected_feature_indices

        requested = self.config.n_features if self.config.n_features is not None else total_channels
        if requested > total_channels:
            raise ValueError(f"Requested n_features={requested} exceeds available backbone channels={total_channels}")

        rng = np.random.default_rng(self.config.random_seed)
        indices = np.sort(rng.choice(total_channels, size=requested, replace=False)).astype(np.int64, copy=False)
        self._selected_feature_indices = indices
        self._total_feature_channels = total_channels
        return indices

    def _select_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 4:
            raise ValueError(f"Expected embeddings with shape (N, C, H, W), got {embeddings.shape}")

        indices = self._selected_channels(total_channels=embeddings.shape[1])
        return embeddings[:, indices, :, :]

    def _estimate_gaussians(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_samples, n_features, height, width = embeddings.shape
        patches = (
            np.transpose(embeddings, (0, 2, 3, 1))
            .reshape(n_samples, height * width, n_features)
            .astype(np.float64, copy=False)
        )

        means = patches.mean(axis=0)
        precision = np.empty((height * width, n_features, n_features), dtype=np.float32)
        eye = np.eye(n_features, dtype=np.float64)

        for patch_index in range(height * width):
            patch_samples = patches[:, patch_index, :]
            if n_samples == 1:
                covariance = eye.copy()
            else:
                centered = patch_samples - means[patch_index]
                covariance = centered.T @ centered / max(n_samples - 1, 1)
            covariance = covariance + self.config.covariance_regularization * eye
            precision[patch_index] = np.linalg.pinv(covariance).astype(np.float32, copy=False)

        return means.astype(np.float32, copy=False), precision

    def _ensure_fitted(self) -> None:
        if self._feature_mean is None or self._feature_precision is None or self._feature_map_size is None:
            raise RuntimeError("PaDiM must be fitted before scoring or predicting")

    def _distance_maps_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        self._ensure_fitted()

        n_samples, n_features, height, width = embeddings.shape
        if self._feature_map_size != (height, width):
            raise ValueError(
                "Embedding spatial size differs from fitted distribution. "
                f"Expected {self._feature_map_size}, got {(height, width)}"
            )

        patches = np.transpose(embeddings, (0, 2, 3, 1)).reshape(n_samples, height * width, n_features)
        deltas = patches - self._feature_mean[None, :, :]
        distances_sq = np.einsum("npd,pde,npe->np", deltas, self._feature_precision, deltas, optimize=True)
        distances = np.sqrt(np.clip(distances_sq, a_min=0.0, a_max=None))
        return distances.reshape(n_samples, height, width).astype(np.float32, copy=False)

    @staticmethod
    def _resize_maps(distance_maps: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        if distance_maps.shape[-2:] == output_size:
            return distance_maps.astype(np.float32, copy=False)

        map_tensor = torch.from_numpy(distance_maps[:, None, :, :])
        resized = F.interpolate(map_tensor, size=output_size, mode="bilinear", align_corners=False)
        return resized[:, 0].numpy().astype(np.float32, copy=False)

    def _image_scores_from_maps(self, distance_maps: np.ndarray) -> np.ndarray:
        if self.config.score_mode == "max":
            return distance_maps.max(axis=(1, 2)).astype(np.float32, copy=False)
        return distance_maps.mean(axis=(1, 2)).astype(np.float32, copy=False)

    def fit(self, data: np.ndarray):
        if len(data) == 0:
            return

        embeddings = self._select_embeddings(self._extract_embeddings(data))
        self._feature_mean, self._feature_precision = self._estimate_gaussians(embeddings)
        self._feature_map_size = (int(embeddings.shape[-2]), int(embeddings.shape[-1]))

        image_scores = None
        if self.config.threshold is None:
            distance_maps = self._distance_maps_from_embeddings(embeddings)
            image_scores = self._image_scores_from_maps(distance_maps)
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
        elif image_scores is None or len(image_scores) == 0:
            self._threshold = 0.0
        else:
            self._threshold = float(np.quantile(image_scores, self.config.threshold_quantile))

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        embeddings = self._select_embeddings(self._extract_embeddings(data))
        distance_maps = self._distance_maps_from_embeddings(embeddings)

        if resize_to_input:
            return self._resize_maps(distance_maps, self._preprocessor.spatial_size(data))
        return distance_maps

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        embeddings = self._select_embeddings(self._extract_embeddings(data))
        distance_maps = self._distance_maps_from_embeddings(embeddings)
        return self._image_scores_from_maps(distance_maps)

    def _resolve_threshold(self, scores: np.ndarray) -> float:
        if self.config.threshold is not None:
            return float(self.config.threshold)
        if self._threshold is not None:
            return float(self._threshold)
        if len(scores) == 0:
            return 0.0
        return float(np.quantile(scores, self.config.threshold_quantile))

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        anomaly_scores = self.score_data(data)
        threshold = self._resolve_threshold(anomaly_scores)
        y_pred = (anomaly_scores > threshold).astype(int)
        return y_pred, anomaly_scores

    def name(self) -> str:
        return "PaDiM"

    def additional_info(self) -> Dict:
        return {
            "threshold": self._threshold,
            "device": str(self._device),
            "backbone": self.config.backbone_name,
            "feature_layers": getattr(self.module, "return_nodes", self.config.backbone_return_nodes),
            "input_size": self.config.input_size,
            "n_features": (
                len(self._selected_feature_indices)
                if self._selected_feature_indices is not None
                else self.config.n_features
            ),
            "pretrained_backbone": self.config.pretrained_backbone,
            "covariance_regularization": self.config.covariance_regularization,
            "score_mode": self.config.score_mode,
            "threshold_quantile": self.config.threshold_quantile,
        }
