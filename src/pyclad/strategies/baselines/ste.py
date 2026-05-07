from copy import deepcopy
from typing import Callable, Dict, Optional

import numpy as np

from pyclad.models.model import Model
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class SingleTaskExpertStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """Train a fresh expert on each concept without carrying state across concepts."""

    def __init__(self, model: Model, model_creation_fn: Optional[Callable[[], Model]] = None):
        self._model = model
        self._model_creation_fn = model_creation_fn or (lambda: deepcopy(model))
        self._training_runs = 0

    def learn(self, data: np.ndarray, **kwargs) -> None:
        # Reset the model for each concept so every row in the matrix comes
        # from a single-concept expert instead of sequential fine-tuning.
        self._model = self._model_creation_fn()
        self._model.fit(data)
        self._training_runs += 1

    def predict(self, data: np.ndarray, **kwargs) -> (np.ndarray, np.ndarray):
        return self._model.predict(data)

    def current_model(self) -> Model:
        return self._model

    def name(self) -> str:
        return "STE"

    def additional_info(self) -> Dict:
        return {
            "model": self._model.name(),
            "training_runs": self._training_runs,
            "reset_before_each_concept": True,
        }
