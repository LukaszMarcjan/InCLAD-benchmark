from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pyclad.models.patchcore.patchcore import ApproximateGreedyCoresetSampler, PatchCore
from pyclad.strategies.vision.common import TaskMemoryBank

class ContinualPatchCore(PatchCore):
    """PatchCore with continual-learning extensions.

    The model stores a list of per-task patch banks and, during prediction,
    selects the task bank closest (on average) to each input image before
    computing the anomaly score.  This lets a single model cover multiple
    concepts without mixing their patch distributions.

    Bank management (building, shrinking, rebalancing) is delegated to the
    companion :class:`PatchCoreCLStrategy` so that the model itself stays
    stateless with respect to scheduling decisions.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._task_memory_state: TaskMemoryBank | None = None

    def begin_continual_run(self) -> None:
        """Reset internal CL state; call once before the first task."""
        self._task_memory_state = TaskMemoryBank(task_banks=[])
        self._memory_bank = None
        self._nn_index = None
        self._feature_map_size = None
        self._threshold = self.config.threshold

    def build_current_task_bank(self, data: np.ndarray) -> np.ndarray:
        """Build the full native coreset for the current task.

        This is intentionally identical to what :class:`PatchCore` does
        internally in :meth:`fit`, so that the first row of the evaluation
        matrix matches the naive baseline exactly.
        """
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        embeddings, patch_shapes = self._extract_embeddings(data, provide_patch_shapes=True)
        if len(embeddings) == 0:
            return np.asarray([], dtype=np.float32)

        feature_map_size = tuple(int(v) for v in patch_shapes[0])
        if self._feature_map_size is None:
            self._feature_map_size = feature_map_size
        elif self._feature_map_size != feature_map_size:
            raise ValueError(
                "All PatchCore CL tasks must share the same embedding map size. "
                f"Expected {self._feature_map_size}, got {feature_map_size}"
            )

        return self._coreset_sampler.run(embeddings)

    def compute_nn_weights(self, bank: np.ndarray, reference_bank: np.ndarray) -> np.ndarray:
        """Return per-patch NN distances from *bank* to *reference_bank*.

        These distances serve as importance weights when shrinking *bank*:
        patches far from the reference (current-task) bank are unique to the
        previous task and should be retained preferentially.
        """
        if len(reference_bank) == 0 or len(bank) == 0:
            return np.ones(len(bank), dtype=np.float32)

        nn = NearestNeighbors(n_neighbors=1, n_jobs=1)
        nn.fit(reference_bank)
        distances, _ = nn.kneighbors(bank)
        return distances[:, 0].astype(np.float32)

    def shrink_bank_weighted(
        self,
        bank: np.ndarray,
        target_size: int,
        reference_bank: np.ndarray,
    ) -> np.ndarray:
        """Reduce *bank* to *target_size* patches using distance-weighted greedy coreset.

        Selection criterion: ``w(p) * min_dist(p, S)`` where
        ``w(p) = nn_dist(p → reference_bank)``.  Patches both far from already-
        selected patches *and* far from the reference (current-task) bank are
        picked first.

        If *bank* is already ≤ *target_size*, it is returned unchanged.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        if len(bank) <= target_size:
            return bank.astype(np.float32, copy=False)

        weights = self.compute_nn_weights(bank, reference_bank)
        ratio = float(target_size) / float(len(bank))
        sampler = ApproximateGreedyCoresetSampler(
            percentage=ratio,
            device=self._device,
            number_of_starting_points=self.config.coreset_starting_points,
            dimension_to_project_features_to=self.config.coreset_projection_dimension,
            random_seed=self.config.random_seed,
        )
        return sampler.run_weighted(bank, weights)

    def set_task_banks(self, task_banks: list[np.ndarray]) -> None:
        """Replace all task banks and rebuild the unified NN index."""
        if self._task_memory_state is None:
            raise RuntimeError("Call begin_continual_run() before set_task_banks()")
        self._task_memory_state.task_banks = [b.astype(np.float32, copy=False) for b in task_banks]
        if self._task_memory_state.task_banks:
            self._memory_bank = np.concatenate(self._task_memory_state.task_banks, axis=0).astype(
                np.float32, copy=False
            )
        else:
            self._memory_bank = None
        self._nn_index = None

    def update_threshold_from_reference_data(self, data: np.ndarray) -> None:
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
            return
        if len(data) == 0:
            self._threshold = 0.0
            return
        scores = self.score_data(data)
        self._threshold = 0.0 if len(scores) == 0 else float(np.quantile(scores, self.config.threshold_quantile))

    def _score_embeddings_against_bank(self, embeddings: np.ndarray, bank: np.ndarray) -> np.ndarray:
        if len(bank) == 0:
            raise ValueError("Task bank must not be empty during scoring")
        index = NearestNeighbors(n_neighbors=min(self.config.n_neighbors, len(bank)), n_jobs=1)
        index.fit(bank)
        distances, _ = index.kneighbors(embeddings)
        return np.mean(distances, axis=-1).astype(np.float32, copy=False)

    def _predict_scores_and_masks(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Fall back to standard single-bank prediction when CL has not started
        # or only one task has been seen (semantically identical to naive).
        if self._task_memory_state is None or not self._task_memory_state.task_banks:
            return super()._predict_scores_and_masks(data)

        embeddings, patch_shapes = self._extract_embeddings(data, provide_patch_shapes=True)
        if len(embeddings) == 0:
            empty = np.asarray([], dtype=np.float32)
            return empty, empty

        batch_size = len(data)
        scales = patch_shapes[0]
        patch_count = int(scales[0] * scales[1])
        if embeddings.shape[0] != batch_size * patch_count:
            raise ValueError(
                "PatchCore CL expects embeddings to reshape cleanly to (batch, patches, dim). "
                f"Got {embeddings.shape[0]} embeddings for batch_size={batch_size}, patch_count={patch_count}"
            )

        # For each image, score against every task bank and select the bank
        # with the lowest mean NN distance (= "nearest task").
        bank_scores = np.stack(
            [
                self._score_embeddings_against_bank(embeddings, bank).reshape(batch_size, patch_count)
                for bank in self._task_memory_state.task_banks
            ],
            axis=0,
        )  # (n_tasks, batch, patches)

        selected_bank_indices = np.argmin(bank_scores.mean(axis=2), axis=0)  # (batch,)
        selected_patch_scores = np.stack(
            [bank_scores[selected_bank_indices[i], i] for i in range(batch_size)],
            axis=0,
        ).astype(np.float32, copy=False)  # (batch, patches)

        image_scores = selected_patch_scores.max(axis=1).astype(np.float32, copy=False)
        masks = self._segmentor.convert_to_segmentation(
            selected_patch_scores.reshape(batch_size, scales[0], scales[1]).astype(np.float32, copy=False)
        )
        return image_scores, masks

    def continual_state_info(self) -> Dict:
        if self._task_memory_state is None:
            return {
                "continual_state": "empty",
                "continual_num_task_banks": 0,
                "continual_task_bank_sizes": [],
            }
        return {
            "continual_state": "running",
            "continual_num_task_banks": len(self._task_memory_state.task_banks),
            "continual_task_bank_sizes": [int(len(b)) for b in self._task_memory_state.task_banks],
        }

    def additional_info(self) -> Dict:
        return {**super().additional_info(), **self.continual_state_info()}
