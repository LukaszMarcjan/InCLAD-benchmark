from __future__ import annotations

from typing import Dict, List

import numpy as np

from pyclad.models.cfa.cfa_cl import ContinualCFA
from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy


class CFACLStrategy(ConceptIncrementalStrategy, ConceptAwareStrategy):
    """Continual-learning strategy for CFA based on per-task inference banks.

    Design
    ------
    Training (per task k)
    ~~~~~~~~~~~~~~~~~~~~~
    1. Combine the current task's data with any replay samples.
    2. Initialise a *training bank* from the combined data using the current
       descriptor (this serves as the fixed hypersphere target for the CFA
       soft-boundary loss during step 3).
    3. Fine-tune the descriptor on the combined data.
    4. Rebuild ALL per-task inference banks using the now-updated descriptor
       so that every bank is consistent with the same embedding space.
    5. Calibrate the anomaly threshold on the current task's training data.
    6. Update the replay buffer.

    The key difference from the original approach is:
    * The training bank is built **before** descriptor training (no stale lag).
    * Per-task inference banks are rebuilt **after** descriptor training.
    * At inference time, each image is assigned to the task bank that gives
      the lowest anomaly score.

    Parameters
    ----------
    model:
        A :class:`~pyclad.models.cfa.cfa_cl.ContinualCFA` instance.
    buffer:
        Replay buffer used to mitigate forgetting in the descriptor.
    """

    def __init__(self, model: ContinualCFA, buffer: ReplayBuffer):
        self._model = model
        self._buffer = buffer
        self._initialized = False
        # Raw training samples kept per task; used to rebuild inference banks
        # after each descriptor update.
        self._task_samples: List[np.ndarray] = []

    def learn(self, data: np.ndarray, **kwargs) -> None:
        if not self._initialized:
            self._model.begin_continual_run()
            self._initialized = True

        # Record current task's training images
        self._task_samples.append(data)

        # Combine with replay data
        replay_data = self._buffer.data()
        train_data = np.concatenate([replay_data, data]) if len(replay_data) > 0 else data

        # Initialise training bank from all available training data
        # using the CURRENT descriptor (before training).  This bank is the
        # fixed target for the CFA soft-boundary loss
        self._model.initialize_training_bank(train_data)

        # Train the descriptor
        self._model.fit_descriptor_only(train_data)

        # Rebuild per-task inference banks
        # Now that the descriptor has been updated, recompute every task bank
        # so they all live in the same (current) embedding space.
        self._model.rebuild_task_inference_banks(self._task_samples)

        # Calibrate threshold
        # score_data() uses the per-task argmin selection, so for the current
        # task's normal training data it will correctly pick the just-rebuilt
        # bank for this task.
        self._model.update_threshold_from_reference_data(data)

        # Update replay buffer
        self._buffer.update(data)

    def predict(self, data: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    def name(self) -> str:
        return "CFACL"

    def additional_info(self) -> Dict:
        return {
            "continual_strategy": "per_task_inference_bank",
            "replay_buffer": self._buffer.info(),
            "tasks_seen": len(self._task_samples),
        }
