from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from pyclad.models.model import Model
from pyclad.strategies.replaycad.memory import ReplayCADMemory
from pyclad.strategies.strategy import ConceptAwareStrategy


class ReplayCADStrategy(ConceptAwareStrategy):
    """Replay-enhanced AD using diffusion-compressed historical concepts."""

    def __init__(self, model: Model, memory: ReplayCADMemory):
        self._model = model
        self._memory = memory

    def learn(self, data: np.ndarray, concept_id: str, **kwargs) -> None:
        del kwargs
        current = np.asarray(data)
        if len(current) == 0:
            raise ValueError(f"ReplayCAD cannot learn empty concept '{concept_id}'")

        replay = self._memory.generate_previous()
        if len(replay):
            replay = self._match_current_shape(replay, current)
            train_data = np.concatenate([replay, current], axis=0)
        else:
            train_data = current

        # The historical generator is used only for training. Compression of
        # the current class happens afterwards, ready for the next CL step.
        # Offload its frozen diffusion pipeline while PatchCore builds its
        # coreset; otherwise both large models compete for GPU memory.
        self._memory.release_device_memory()
        self._model.fit(train_data)
        self._memory.compress(concept_id=concept_id, images=current)
        # Evaluation that follows a CL step only needs the anomaly detector.
        self._memory.release_device_memory()

    def predict(self, data: np.ndarray, concept_id: str | None = None, **kwargs):
        del concept_id, kwargs
        return self._model.predict(data)

    def name(self) -> str:
        return "ReplayCAD"

    def additional_info(self) -> Dict[str, Any]:
        return {
            "model": self._model.name(),
            "replaycad_memory": self._memory.info(),
        }

    @staticmethod
    def _match_current_shape(replay: np.ndarray, current: np.ndarray) -> np.ndarray:
        if current.ndim != 4 or replay.ndim != 4:
            raise ValueError(
                "ReplayCAD visual data must have shape (N,H,W,C); "
                f"got replay={replay.shape}, current={current.shape}"
            )
        target_height, target_width, target_channels = current.shape[1:]
        if target_channels not in {1, 3}:
            raise ValueError(f"ReplayCAD supports one or three image channels, got {target_channels}")

        resized = []
        for image in replay:
            array = np.asarray(image)
            if target_channels == 1 and array.ndim == 3:
                pil = Image.fromarray(array[..., :3]).convert("L")
            else:
                if array.ndim == 2:
                    array = np.repeat(array[..., None], 3, axis=-1)
                pil = Image.fromarray(array[..., :3]).convert("RGB")
            pil = pil.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
            restored = np.asarray(pil)
            if target_channels == 1:
                restored = restored[..., None]
            resized.append(restored)
        return np.stack(resized).astype(current.dtype, copy=False)
