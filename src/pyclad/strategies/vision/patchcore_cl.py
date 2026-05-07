from __future__ import annotations

from typing import Dict

import numpy as np

from pyclad.models.patchcore.patchcore_cl import ContinualPatchCore
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class PatchCoreCLStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """Continual-learning strategy for PatchCore based on adaptive task memory banks.

    Design
    ------
    *Current task* — always stored at full native coreset resolution (same
    ``coreset_sampling_ratio`` as the naive baseline), so the first row of the
    evaluation matrix is identical to naive.

    *Previous tasks* — each shrunk to at most
    ``previous_ratio * len(current_bank) / n_previous`` patches using a
    distance-weighted greedy coreset.  The weight of each patch is its NN
    distance to the current-task bank: patches that are unique to the previous
    task (far from the current distribution) are retained preferentially.

    All previous banks are re-evaluated every time a new task arrives so that
    the total memory devoted to history stays bounded at
    ``previous_ratio * len(current_bank)`` regardless of the number of tasks.

    Parameters
    ----------
    model:
        A :class:`~pyclad.models.patchcore.patchcore_cl.ContinualPatchCore`
        instance.
    previous_ratio:
        Fraction of the current-task bank size to allocate across *all*
        previous tasks combined.  Default ``0.25`` means previous tasks share
        a budget of 25 % of the current coreset, equally divided among them.
    """

    def __init__(self, model: ContinualPatchCore, previous_ratio: float = 0.25):
        if not 0.0 < previous_ratio <= 1.0:
            raise ValueError(f"previous_ratio must be in (0, 1], got {previous_ratio}")
        self._model = model
        self._previous_ratio = previous_ratio
        self._tasks_seen = 0
        self._task_banks: list[np.ndarray] = []

    def learn(self, data: np.ndarray, **kwargs) -> None:
        if self._tasks_seen == 0:
            self._model.begin_continual_run()

        self._tasks_seen += 1

        # Build full coreset for the current task
        # This is intentionally identical to naive PatchCore.fit(), so the
        # diagonal entry of the first matrix row matches the naive baseline.
        current_bank = self._model.build_current_task_bank(data)

        # Shrink all previous banks to fit within the budget
        if self._task_banks:
            # Total budget for history = previous_ratio × |current bank|
            previous_budget = max(1, int(self._previous_ratio * len(current_bank)))
            per_task_budget = max(1, previous_budget // len(self._task_banks))

            self._task_banks = [
                self._model.shrink_bank_weighted(
                    bank=bank,
                    target_size=per_task_budget,
                    reference_bank=current_bank,
                )
                for bank in self._task_banks
            ]

        # Append the current (full) bank
        self._task_banks.append(current_bank)

        # Commit to the model and update threshold
        self._model.set_task_banks(self._task_banks)
        self._model.update_threshold_from_reference_data(data)

    def predict(self, data: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "PatchCoreCL"

    def additional_info(self) -> Dict:
        total_previous = sum(len(b) for b in self._task_banks[:-1]) if len(self._task_banks) > 1 else 0
        current_size = len(self._task_banks[-1]) if self._task_banks else 0
        return {
            "continual_strategy": "adaptive_task_memory_bank",
            "previous_ratio": self._previous_ratio,
            "tasks_seen": self._tasks_seen,
            "current_bank_size": current_size,
            "total_previous_bank_size": total_previous,
        }
