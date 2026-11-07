"""CFA CL with a bounded memory, and retained-memory accounting for CL baselines.

Additive module. :class:`~pyclad.strategies.vision.cfa_cl.CFACLStrategy` keeps
every training image of every past task (``_task_samples``) and rebuilds all
per-task inference banks from them after each descriptor update. Its replay
buffer is used only to fine-tune the descriptor. In a comparison where the
other methods keep a bounded memory, that gives CFA CL the full history of the
stream.

:class:`BoundedCFACLStrategy` keeps the same training procedure and changes one
thing: after each descriptor update, the bank of an earlier task is rebuilt
from that task's share of the replay buffer instead of from all of its
training images. The current task still uses all of its own images, as every
continual method does while it learns a task. The buffer is the one the
original strategy already holds, sized by the same ``--replay-buffer-fraction``
rule as the replay baselines, so no new budget parameter is introduced.

With ``bounded=False`` the class reproduces the original strategy exactly (it
calls the parent ``learn``) and only adds the memory accounting, so the
full-history row can be reported with the same measurement.

:func:`retained_memory` reports what a CL strategy keeps from past tasks, in
bytes, for CFA CL, PaDiM CL and PatchCore CL. It counts state that grows with
the stream (raw images, banks, Gaussians) and the learned parameters; the
frozen pretrained backbone is shared by every method and is not counted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.models.cfa.cfa_cl import ContinualCFA
from pyclad.output.output_writer import InfoProvider
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.vision.cfa_cl import CFACLStrategy

logger = logging.getLogger(__name__)


class PerTaskBalancedReplayBuffer(AdaptiveBalancedReplayBuffer):
    """The balanced buffer CFA CL already uses, with read access to each task's share."""

    def per_task(self) -> List[np.ndarray]:
        return list(self._buffers)

    def name(self) -> str:
        return "PerTaskBalancedReplayBuffer"


def _nbytes(obj: Any) -> int:
    """Bytes held by an array, a tensor, or a (nested) list/tuple of them."""
    if obj is None:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if hasattr(obj, "element_size") and hasattr(obj, "numel"):  # torch.Tensor
        return int(obj.element_size() * obj.numel())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(o) for o in obj)
    return 0


def _trainable_parameter_bytes(model: Any) -> int:
    module = getattr(model, "module", None)
    if module is None or not hasattr(module, "parameters"):
        return 0
    return sum(_nbytes(p) for p in module.parameters() if p.requires_grad)


def retained_memory(strategy: Any) -> Optional[Dict[str, int]]:
    """Bytes kept from past tasks by a CL strategy, by component; None if unknown."""
    model = getattr(strategy, "_model", None)
    name = type(strategy).__name__
    out: Dict[str, int] = {}

    if isinstance(strategy, CFACLStrategy):
        buffer = getattr(strategy, "_buffer", None)
        out["raw_images_full_history"] = _nbytes(getattr(strategy, "_task_samples", []))
        out["raw_images_replay_buffer"] = _nbytes(buffer.data()) if buffer is not None else 0
        out["inference_banks"] = _nbytes(getattr(model, "_task_inference_banks", []))
        out["learned_parameters"] = _trainable_parameter_bytes(model)
    elif name == "PaDiMCLStrategy":
        out["gaussians"] = _nbytes(getattr(model, "_task_gaussians", []))
    elif name == "PatchCoreCLStrategy":
        state = getattr(model, "_task_memory_state", None)
        out["patch_banks"] = _nbytes(getattr(state, "task_banks", []))
    else:
        return None

    out["total"] = sum(out.values())
    return out


class BoundedCFACLStrategy(CFACLStrategy):
    """CFA CL whose earlier-task banks are rebuilt from the replay buffer only."""

    def __init__(self, model: ContinualCFA, buffer: PerTaskBalancedReplayBuffer, bounded: bool = True):
        super().__init__(model=model, buffer=buffer)
        self._bounded = bool(bounded)
        self._bank_images: List[int] = []

    def learn(self, data: np.ndarray, **kwargs) -> None:
        if not self._bounded:
            super().learn(data, **kwargs)
            self._bank_images = [len(s) for s in self._task_samples]
            return

        if not self._initialized:
            self._model.begin_continual_run()
            self._initialized = True

        # Earlier tasks are represented only by what the buffer kept of them.
        earlier = self._buffer.per_task()
        empty = [i for i, s in enumerate(earlier) if len(s) == 0]
        if empty:
            raise ValueError(
                f"replay buffer holds no image for earlier task(s) {empty}; "
                "raise --replay-buffer-fraction so every task keeps at least one image"
            )

        replay_data = self._buffer.data()
        train_data = np.concatenate([replay_data, data]) if len(replay_data) > 0 else data

        # Same training steps as CFACLStrategy.learn.
        self._model.initialize_training_bank(train_data)
        self._model.fit_descriptor_only(train_data)

        bank_sources = earlier + [data]
        self._model.rebuild_task_inference_banks(bank_sources)
        self._bank_images = [len(s) for s in bank_sources]

        self._model.update_threshold_from_reference_data(data)
        self._buffer.update(data)

    def name(self) -> str:
        return "CFACLBounded" if self._bounded else "CFACLFullHistory"

    def additional_info(self) -> Dict:
        info = super().additional_info()
        info["bank_source"] = "replay_buffer" if self._bounded else "full_history"
        info["tasks_seen"] = len(self._bank_images)
        info["bank_images_per_task"] = list(self._bank_images)
        info["retained_memory_bytes"] = retained_memory(self)
        return info


class RetainedMemoryCallback(Callback, InfoProvider):
    """Records :func:`retained_memory` after every task, to show how it grows."""

    def __init__(self, strategy: Any):
        self._strategy = strategy
        self._by_concept: Dict[str, Dict[str, int]] = {}

    def after_training(self, *args, **kwargs):
        concept = kwargs.get("learned_concept", kwargs.get("concept"))
        if concept is None and args and isinstance(args[0], Concept):
            concept = args[0]
        key = concept.name if concept is not None else str(len(self._by_concept))
        mem = retained_memory(self._strategy)
        if mem is not None:
            self._by_concept[key] = mem

    def info(self) -> Dict[str, Any]:
        return {"retained_memory_callback": {"bytes_after_task": self._by_concept}}


__all__ = [
    "PerTaskBalancedReplayBuffer",
    "BoundedCFACLStrategy",
    "RetainedMemoryCallback",
    "retained_memory",
]
