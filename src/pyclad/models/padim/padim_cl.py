from __future__ import annotations

from typing import Dict

import numpy as np

from pyclad.models.padim.padim import PaDiM

class ContinualPaDiM(PaDiM):
    """PaDiM with continual-learning extensions.

    Design
    ------
    Each task gets a fresh per-task Gaussian (mean + precision matrix fitted
    only on that task's training data).  At inference time every image is
    assigned to the task whose Gaussian gives the smallest image-level
    Mahalanobis distance, and scored with that task's distribution.

    This keeps the task distributions completely separate so that learning a
    new concept cannot corrupt the existing ones (no catastrophic forgetting
    of the statistical model).
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._task_gaussians: list[tuple[np.ndarray, np.ndarray]] = []

    def begin_continual_run(self) -> None:
        """Reset internal CL state; call once before the first task."""
        self.reset_continual_state()

    def reset_continual_state(self) -> None:
        self._task_gaussians = []
        self._selected_feature_indices = None
        self._feature_mean = None
        self._feature_precision = None
        self._feature_map_size = None
        self._total_feature_channels = None
        self._threshold = self.config.threshold

    def fit_task_gaussian(self, data: np.ndarray) -> None:
        """Fit a fresh Gaussian for this task and append it to the task list.

        Also updates the base-class fields (``_feature_mean``,
        ``_feature_precision``, ``_feature_map_size``) to point at the new
        Gaussian so that :meth:`~pyclad.models.padim.padim.PaDiM._ensure_fitted`
        keeps passing and threshold calibration works correctly.
        """
        if len(data) == 0:
            return

        embeddings = self._select_embeddings(self._extract_embeddings(data))
        if embeddings.ndim != 4:
            raise ValueError(f"Expected embeddings with shape (N, C, H, W), got {embeddings.shape}")

        feature_map_size = (int(embeddings.shape[-2]), int(embeddings.shape[-1]))
        if self._feature_map_size is not None and self._feature_map_size != feature_map_size:
            raise ValueError(
                "All PaDiM CL tasks must share the same embedding map size. "
                f"Expected {self._feature_map_size}, got {feature_map_size}"
            )

        mean, precision = self._estimate_gaussians(embeddings)
        self._task_gaussians.append((mean, precision))

        # Keep base-class fields in sync so _ensure_fitted() and
        # update_threshold_from_reference_data() continue to work.
        self._feature_mean = mean
        self._feature_precision = precision
        self._feature_map_size = feature_map_size

    def _distance_maps_with_params(
        self,
        embeddings: np.ndarray,
        feature_mean: np.ndarray,
        feature_precision: np.ndarray,
        feature_map_size: tuple[int, int],
    ) -> np.ndarray:
        """Compute per-pixel Mahalanobis distance maps for given Gaussian params."""
        n_samples, n_features, height, width = embeddings.shape
        if feature_map_size != (height, width):
            raise ValueError(
                "Embedding spatial size differs from the fitted distribution. "
                f"Expected {feature_map_size}, got {(height, width)}"
            )
        patches = np.transpose(embeddings, (0, 2, 3, 1)).reshape(n_samples, height * width, n_features)
        deltas = patches - feature_mean[None, :, :]
        distances_sq = np.einsum("npd,pde,npe->np", deltas, feature_precision, deltas, optimize=True)
        distances = np.sqrt(np.clip(distances_sq, a_min=0.0, a_max=None))
        return distances.reshape(n_samples, height, width).astype(np.float32, copy=False)

    def _all_task_maps(self, embeddings: np.ndarray) -> np.ndarray:
        """Return distance maps for every task: shape ``(n_tasks, n_samples, H, W)``."""
        return np.stack(
            [
                self._distance_maps_with_params(embeddings, mean, precision, self._feature_map_size)
                for mean, precision in self._task_gaussians
            ],
            axis=0,
        )

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)
        if not self._task_gaussians:
            return super().score_data(data)

        embeddings = self._select_embeddings(self._extract_embeddings(data))
        n_tasks = len(self._task_gaussians)
        n_samples = len(data)

        all_maps = self._all_task_maps(embeddings)  # (n_tasks, n_samples, H, W)
        # Compute per-task image scores then pick the minimum per image.
        all_scores = self._image_scores_from_maps(
            all_maps.reshape(n_tasks * n_samples, *all_maps.shape[2:])
        ).reshape(n_tasks, n_samples)  # (n_tasks, n_samples)
        return all_scores.min(axis=0)  # (n_samples,)

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)
        if not self._task_gaussians:
            return super().score_maps(data, resize_to_input=resize_to_input)

        embeddings = self._select_embeddings(self._extract_embeddings(data))
        n_tasks = len(self._task_gaussians)
        n_samples = len(data)

        all_maps = self._all_task_maps(embeddings)  # (n_tasks, n_samples, H, W)
        all_scores = self._image_scores_from_maps(
            all_maps.reshape(n_tasks * n_samples, *all_maps.shape[2:])
        ).reshape(n_tasks, n_samples)
        selected_task_indices = np.argmin(all_scores, axis=0)  # (n_samples,)
        selected_maps = all_maps[selected_task_indices, np.arange(n_samples)]  # (n_samples, H, W)

        if resize_to_input:
            return self._resize_maps(selected_maps, self._preprocessor.spatial_size(data))
        return selected_maps

    def update_threshold_from_reference_data(self, data: np.ndarray) -> None:
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
            return
        if len(data) == 0:
            self._threshold = 0.0
            return
        scores = self.score_data(data)
        self._threshold = 0.0 if len(scores) == 0 else float(np.quantile(scores, self.config.threshold_quantile))

    def continual_state_info(self) -> Dict:
        return {
            "continual_state": "running" if self._task_gaussians else "empty",
            "continual_num_tasks": len(self._task_gaussians),
            "continual_feature_map_size": (
                tuple(int(v) for v in self._feature_map_size) if self._feature_map_size else None
            ),
        }

    def additional_info(self) -> Dict:
        return {**super().additional_info(), **self.continual_state_info()}
