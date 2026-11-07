from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pyclad.models.ucad.ucad import UCAD
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class UCADStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """pyCLAD strategy for UCAD's key-prompt-knowledge task memory."""

    def __init__(self, model: UCAD):
        self._model = model
        self._tasks_seen = 0

    def learn(self, data: np.ndarray, concept_id: Optional[str] = None, **kwargs) -> None:
        del kwargs
        if self._tasks_seen == 0:
            self._model.begin_continual_run()
        self._model.fit_task(data=data, concept_id=concept_id)
        self._tasks_seen += 1

    def predict(
        self,
        data: np.ndarray,
        concept_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        # concept_id is intentionally ignored: UCAD retrieves the task from
        # its learned keys and is task-agnostic at inference time.
        del concept_id, kwargs
        return self._model.predict(data)

    def name(self) -> str:
        return "UCAD"

    def additional_info(self) -> Dict:
        return {
            "continual_strategy": "key_prompt_knowledge",
            "task_agnostic_inference": True,
            "tasks_seen": self._tasks_seen,
        }
