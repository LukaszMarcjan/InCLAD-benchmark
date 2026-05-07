from __future__ import annotations

from typing import Dict

import numpy as np

from pyclad.models.padim.padim_cl import ContinualPaDiM
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class PaDiMCLStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """Continual-learning strategy for PaDiM based on per-task Gaussian models.

    Design
    ------
    Each task receives a freshly estimated Gaussian distribution.  At
    inference time, the task whose Gaussian produces the lowest image-level
    score is selected, and that task's Mahalanobis distances are used for
    anomaly scoring.  This is the PaDiM analogue of the adaptive task-memory
    bank used by :class:`~pyclad.strategies.vision.patchcore_cl.PatchCoreCLStrategy`.

    Parameters
    ----------
    model:
        A :class:`~pyclad.models.padim.padim_cl.ContinualPaDiM` instance.
    """

    def __init__(self, model: ContinualPaDiM):
        self._model = model
        self._tasks_seen = 0

    def learn(self, data: np.ndarray, **kwargs) -> None:
        if self._tasks_seen == 0:
            self._model.begin_continual_run()
        self._tasks_seen += 1

        # Fit a fresh Gaussian for this task only (no cross-task contamination).
        self._model.fit_task_gaussian(data)
        # Calibrate threshold from this task's normal training data.
        # score_data() uses per-task argmin selection, so for training data of
        # the most recent task it correctly picks the just-fitted Gaussian.
        self._model.update_threshold_from_reference_data(data)

    def predict(self, data: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "PaDiMCL"

    def additional_info(self) -> Dict:
        return {
            "continual_strategy": "per_task_gaussian",
            "tasks_seen": self._tasks_seen,
        }
