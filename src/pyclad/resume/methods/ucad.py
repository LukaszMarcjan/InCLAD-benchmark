"""Checkpointable UCAD: the key-prompt-knowledge memory, and nothing else.

UCAD is the cheapest of the three methods to make resumable, because almost
everything it holds is either frozen, recomputed, or re-derivable from a
counter. What actually has to survive a restart is the ``(key, prompt,
knowledge)`` triple per learned task plus the two scalars that ``predict()``
depends on. Everything else is deliberately left out — see the *not persisted*
notes below, each of which is a decision rather than an omission.

**Why the strategy needs a hook of its own.** ``UCADStrategy.learn`` calls
``begin_continual_run()`` while ``_tasks_seen == 0``, and
``begin_continual_run`` clears ``_task_memories``, ``_feature_map_size`` and
``_threshold``. Restore the model perfectly, leave that counter at zero, and the
next concept erases the restored memory with no error and no log line — the
worst possible failure shape, because the run continues and produces numbers.
``ResumableUCADStrategy.resume_restore`` exists for that one assignment.

**Why no RNG state is captured.** Every random draw UCAD makes is seeded from
``config.random_seed + len(self._task_memories)`` and never touches the global
stream: prompt initialisation builds a local ``torch.Generator``; the SCL
DataLoader gets that same generator; and the two coreset samplings per concept
use ``np.random.default_rng(seed)`` with the projection built under
``torch.random.fork_rng()``. Restoring the memory list — hence its length —
therefore restores the RNG. The corollary is that the *order* of that list is
load-bearing twice over: it shifts every downstream seed offset, and it decides
the ``np.argmin`` tie-break in ``_task_selection_indices``. That is why the
payload stores tasks as a list and never as a dict.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, Mapping, Optional

import numpy as np
import torch

from pyclad.models.ucad.config import UCADConfig
from pyclad.models.ucad.ucad import UCAD, UCADTaskMemory
from pyclad.resume.fingerprint import fingerprint_from_config
from pyclad.resume.protocol import (
    DETERMINISM_BIT_IDENTICAL,
    DETERMINISM_STATISTICAL,
    CheckpointPayloadError,
    require_payload_keys,
)
from pyclad.strategies.vision.ucad import UCADStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "UCAD_FINGERPRINT_FIELDS",
    "RECOMMENDED_KEEP",
    "ResumableUCAD",
    "ResumableUCADStrategy",
    "ucad_determinism",
]

#: Config fields that silently change future numbers while leaving restored
#: state looking perfectly valid. A mismatch on any of them aborts the resume
#: with no override, because the failure it prevents is a plausible number in a
#: published table rather than a crash.
#:
#: Two notes on the contents:
#:
#: * the design named ``coreset_sampling_ratio``, which ``UCADConfig`` does not
#:   have — UCAD sizes its coresets by ``key_size``/``knowledge_size`` and
#:   parameterises the sampler with ``coreset_projection_dimension`` and
#:   ``coreset_starting_points``, so those two stand in for it;
#: * path-valued fields (``structure_mask_root``, ``sam_checkpoint``) are
#:   deliberately excluded. A chained scheduler job legitimately sees a
#:   different node-local path for the same masks, and a hard, non-overridable
#:   abort on that is a worse failure than the one it would prevent.
UCAD_FINGERPRINT_FIELDS: tuple[str, ...] = (
    # identity of the frozen feature extractor
    "backbone_name",
    "pretrained_backbone",
    "feature_layer",
    "input_size",
    "normalize_mean",
    "normalize_std",
    "target_embed_dimension",
    # prompt geometry and initialisation (the seed offset lives here too)
    "prompt_depth",
    "prompt_length",
    "prompt_init",
    "warm_start_prompts",
    # SCL optimisation: every one of these changes the learned prompt
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "contrastive_temperature",
    "gradient_clip_norm",
    "structure_mode",
    # memory sizing and the coreset sampler
    "key_size",
    "knowledge_size",
    "coreset_projection_dimension",
    "coreset_starting_points",
    # scoring
    "n_neighbors",
    "threshold",
    "threshold_quantile",
    "smoothing_sigma",
    "random_seed",
)

#: Recommended ``CheckpointStore(keep=...)`` for UCAD: unlimited. One concept
#: costs ~1.57 MiB (key 196x1024 + knowledge 196x1024 + prompt, all float32), so
#: retaining all fifteen MVTec concepts costs ~24 MiB. At that price there is no
#: reason to prune, and keeping every step makes a mid-run diagnosis possible.
RECOMMENDED_KEEP: int = 0


def ucad_determinism(config: UCADConfig) -> dict[str, Any]:
    """Describe honestly what a resumed UCAD run reproduces.

    Bit-identical in the cases the benchmarks actually run, because the seeds are
    re-derived rather than restored. The one config that breaks it is online SAM:
    ``SAMStructureMaskGenerator`` runs ``SamAutomaticMaskGenerator`` with no seed
    of its own, so the structure masks — and through them the prompt — are not
    reproducible across processes.
    """
    caveats = ["cudnn_nondeterminism"]
    guarantee = DETERMINISM_BIT_IDENTICAL
    if config.structure_mode == "sam":
        guarantee = DETERMINISM_STATISTICAL
        caveats.append("sam_masks_unseeded")
    return {"guarantee": guarantee, "caveats": caveats}


class ResumableUCAD(UCAD):
    """UCAD that can serialise its task memory and be rebuilt from it.

    Persisted: the ordered task memories, ``_feature_map_size`` and
    ``_threshold``.

    Not persisted, each for a reason:

    * **the ViT backbone.** Frozen from construction and never touched by
      ``fit_task``, so it is a pure function of ``backbone_name`` and
      ``pretrained_backbone`` — both fingerprinted. Storing ~330 MiB of weights
      per concept to reproduce a download would dwarf the 1.57 MiB that is
      genuinely new information.
    * **the optimizer.** Adam is a local inside ``_train_prompt`` and is rebuilt
      per concept *by design*, so an uninterrupted run also resets its moments at
      every boundary. Restoring them would make the resumed run diverge from the
      uninterrupted one — the opposite of the goal. The payload records
      ``optimizer_state: None`` explicitly so nobody later "fixes" this.
    * **RNG state.** Re-derived from the restored memory length; see the module
      docstring.
    * **the prediction cache.** It keys on object *identity*
      (``data is self._last_prediction_data``), so a restored cache could only
      ever alias a reused array and hand back stale scores. ``load_state``
      clears it.
    * **preprocessor, segmentor, structure-mask generator, coreset seed.** All
      constructed from the config, all stateless across concepts.
    """

    STATE_KIND: ClassVar[str] = "ucad"
    STATE_VERSION: ClassVar[int] = 1

    #: Keys ``load_state`` refuses to proceed without. ``threshold`` earns its
    #: place: losing it degrades *only* F1 (``predict`` falls back to 0.0, and
    #: every UCAD score is a positive nearest-neighbour distance, so everything
    #: reads as anomalous) while ROC-AUC and AP, computed from raw scores, still
    #: look perfect. That is a resume bug that reads as a modelling result.
    _REQUIRED_KEYS: ClassVar[tuple[str, ...]] = ("tasks", "tasks_seen", "feature_map_size", "threshold")
    _REQUIRED_TASK_KEYS: ClassVar[tuple[str, ...]] = ("key", "prompt", "knowledge")

    # ------------------------------------------------------------------ read-only

    @property
    def tasks_seen(self) -> int:
        """Number of learned tasks — the value the strategy counter must match."""
        return len(self._task_memories)

    @property
    def learned_concepts(self) -> tuple[Optional[str], ...]:
        return tuple(memory.concept_id for memory in self._task_memories)

    def determinism(self) -> dict[str, Any]:
        return ucad_determinism(self.config)

    # ------------------------------------------------------------------ protocol

    def save_state(self) -> dict[str, Any]:
        """Return tensors and JSON-native values only, tasks in learned order."""
        tasks = [
            {
                "key": _as_tensor(memory.key),
                "prompt": memory.prompt.detach().to(device="cpu", dtype=torch.float32).clone(),
                "knowledge": _as_tensor(memory.knowledge),
                "concept_id": None if memory.concept_id is None else str(memory.concept_id),
                "final_scl_loss": None if memory.final_scl_loss is None else float(memory.final_scl_loss),
            }
            for memory in self._task_memories
        ]
        return {
            "tasks": tasks,
            # Redundant with len(tasks) inside this class, but it is the key the
            # strategy hook reads, and a payload that carries it explicitly can
            # be validated against the model without reaching into either.
            "tasks_seen": len(self._task_memories),
            "feature_map_size": (
                None if self._feature_map_size is None else [int(size) for size in self._feature_map_size]
            ),
            "threshold": None if self._threshold is None else float(self._threshold),
            # Intentionally absent, not forgotten: see the class docstring.
            "optimizer_state": None,
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        """Rebuild the task memory, validating everything before mutating.

        Every check below runs against locals. Half-restored state is strictly
        worse than none: a model with restored memories and a stale
        ``_feature_map_size`` does not fail here, it fails at the next
        ``fit_task`` with a message about mismatched feature grids, tens of
        minutes into the resumed run.
        """
        require_payload_keys(payload, self._REQUIRED_KEYS, state_kind=self.STATE_KIND)

        raw_tasks = payload["tasks"]
        if not isinstance(raw_tasks, (list, tuple)):
            raise CheckpointPayloadError(
                f"{self.STATE_KIND} payload 'tasks' must be an ordered sequence (task order decides both the "
                f"per-task seed offset and the task-selection tie-break), got {type(raw_tasks).__name__}"
            )

        tasks_seen = _as_int(payload["tasks_seen"], path="tasks_seen", state_kind=self.STATE_KIND)
        if tasks_seen != len(raw_tasks):
            raise CheckpointPayloadError(
                f"{self.STATE_KIND} payload claims tasks_seen={tasks_seen} but carries {len(raw_tasks)} tasks"
            )

        expected_prompt_shape = self.expected_prompt_shape()
        memories: list[UCADTaskMemory] = []
        for index, entry in enumerate(raw_tasks):
            memories.append(self._task_memory_from(entry, index=index, expected_prompt_shape=expected_prompt_shape))

        feature_map_size = _as_feature_map_size(payload["feature_map_size"], state_kind=self.STATE_KIND)
        threshold = _as_optional_float(payload["threshold"], path="threshold", state_kind=self.STATE_KIND)

        if memories and feature_map_size is None:
            raise CheckpointPayloadError(
                f"{self.STATE_KIND} payload carries {len(memories)} tasks but no feature_map_size; "
                "prediction would fail its fitted-state check"
            )

        # Single commit point: nothing above this line touched the instance.
        self._task_memories = memories
        self._feature_map_size = feature_map_size
        self._threshold = threshold
        self._invalidate_prediction_cache()
        logger.info(
            "Restored %s with %d task memories (concepts=%s, feature_map_size=%s, threshold=%s)",
            self.STATE_KIND,
            len(memories),
            list(self.learned_concepts),
            feature_map_size,
            threshold,
        )

    def state_fingerprint(self) -> dict[str, Any]:
        return fingerprint_from_config(self.config, UCAD_FINGERPRINT_FIELDS)

    # ------------------------------------------------------------------ internals

    def expected_prompt_shape(self) -> tuple[int, int, int, int, int]:
        """The prompt shape this backbone + config combination produces."""
        return (
            int(self.config.prompt_depth),
            2,
            int(self.config.prompt_length),
            int(self._num_heads),
            int(self._head_dim),
        )

    def _task_memory_from(
        self,
        entry: Any,
        *,
        index: int,
        expected_prompt_shape: tuple[int, ...],
    ) -> UCADTaskMemory:
        where = f"{self.STATE_KIND} payload tasks[{index}]"
        if not isinstance(entry, Mapping):
            raise CheckpointPayloadError(f"{where} must be a mapping, got {type(entry).__name__}")
        require_payload_keys(entry, self._REQUIRED_TASK_KEYS, state_kind=f"{self.STATE_KIND}.tasks[{index}]")

        key = _as_feature_matrix(entry["key"], where=f"{where}['key']")
        knowledge = _as_feature_matrix(entry["knowledge"], where=f"{where}['knowledge']")
        target_dimension = int(self.config.target_embed_dimension)
        for name, matrix in (("key", key), ("knowledge", knowledge)):
            if matrix.shape[1] != target_dimension:
                raise CheckpointPayloadError(
                    f"{where}['{name}'] has feature dimension {matrix.shape[1]}, but this run maps features to "
                    f"target_embed_dimension={target_dimension}; the checkpoint belongs to a different configuration"
                )

        prompt = entry["prompt"]
        if not isinstance(prompt, torch.Tensor):
            raise CheckpointPayloadError(f"{where}['prompt'] must be a torch.Tensor, got {type(prompt).__name__}")
        # Prompts live on the CPU inside a task memory (``fit_task`` stores
        # ``prompt.cpu()``); the forward pass moves them to the device itself.
        prompt = prompt.detach().to(device="cpu", dtype=torch.float32).clone()
        if tuple(prompt.shape) != tuple(expected_prompt_shape):
            raise CheckpointPayloadError(
                f"{where}['prompt'] has shape {tuple(prompt.shape)} but this backbone and config expect "
                f"{tuple(expected_prompt_shape)}; the checkpoint was written against a different model"
            )

        return UCADTaskMemory(
            key=key,
            prompt=prompt,
            knowledge=knowledge,
            concept_id=_as_optional_str(entry.get("concept_id"), path=f"{where}['concept_id']"),
            final_scl_loss=_as_optional_float(
                entry.get("final_scl_loss"),
                path=f"{where}['final_scl_loss']",
                state_kind=self.STATE_KIND,
            ),
        )


class ResumableUCADStrategy(UCADStrategy):
    """``UCADStrategy`` plus the one assignment that makes a resume survive.

    ``name()`` is inherited and still returns ``"UCAD"``. That is deliberate: the
    name is the join key used by the results-aggregation scripts, and renaming it
    would quietly drop every resumed run out of the comparisons against the
    published benchmark. The checkpointable variant announces itself through
    ``additional_info()["resume"]`` instead, which lands in the result JSON
    without touching identity.
    """

    def __init__(self, model: ResumableUCAD, *, resume_info: Optional[Mapping[str, Any]] = None):
        if not isinstance(model, ResumableUCAD):
            raise TypeError(
                f"ResumableUCADStrategy needs a ResumableUCAD (the checkpointable model), got {type(model).__name__}. "
                "A plain UCAD has no save_state/load_state, so the run would checkpoint nothing."
            )
        super().__init__(model)
        self._resume_info: dict[str, Any] = {
            "enabled": False,
            "resumed": False,
            "state_kind": ResumableUCAD.STATE_KIND,
            "state_version": ResumableUCAD.STATE_VERSION,
        }
        if resume_info is not None:
            self.set_resume_info(resume_info)

    def set_resume_info(self, info: Mapping[str, Any]) -> None:
        """Record ``ResumeOutcome.info()`` for the result JSON."""
        self._resume_info = {
            "state_kind": ResumableUCAD.STATE_KIND,
            "state_version": ResumableUCAD.STATE_VERSION,
            **dict(info),
        }

    def resume_restore(self, payload: Mapping[str, Any], plan: Any) -> None:
        """Restore ``_tasks_seen`` — the single most valuable line in this file.

        Called by :func:`pyclad.resume.plan.apply_resume` *after*
        ``model.load_state``, so the counter can be cross-checked against the
        state that was actually restored rather than trusted blindly.
        """
        require_payload_keys(payload, ("tasks_seen",), state_kind=ResumableUCAD.STATE_KIND)
        tasks_seen = _as_int(payload["tasks_seen"], path="tasks_seen", state_kind=ResumableUCAD.STATE_KIND)

        restored = self._model.tasks_seen
        if tasks_seen != restored:
            raise CheckpointPayloadError(
                f"Checkpoint says tasks_seen={tasks_seen} but the restored model holds {restored} task memories; "
                "the strategy counter and the model state would disagree, and the next concept would either "
                "wipe the memory or shift every seed offset"
            )

        # A pending checkpoint is written *after* its concept was trained, so it
        # holds one more task than the plan's resume index; a committed one holds
        # exactly as many as the plan skips.
        expected = int(plan.resume_index) + (1 if bool(getattr(plan, "evaluate_pending_concept", False)) else 0)
        if tasks_seen != expected:
            raise CheckpointPayloadError(
                f"Checkpoint holds {tasks_seen} task memories but the resume plan expects {expected} "
                f"(resume_index={plan.resume_index}, evaluate_pending_concept="
                f"{getattr(plan, 'evaluate_pending_concept', False)}); refusing to start with a "
                "detector and a scenario that disagree about how far the run got"
            )

        self._tasks_seen = tasks_seen
        self._resume_info.update(
            {
                "enabled": True,
                "resumed": True,
                "restored_tasks": tasks_seen,
                "restored_concepts": [name for name in self._model.learned_concepts],
            }
        )
        logger.info("Resume: UCAD strategy continuing from %d learned tasks", tasks_seen)

    def additional_info(self) -> Dict:
        info = super().additional_info()
        info["resume"] = dict(self._resume_info)
        return info


# --------------------------------------------------------------------- coercion


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    """Copy a float32 feature matrix into a tensor the payload can hold.

    ``torch.load(weights_only=True)`` refuses numpy objects outright, so the
    numpy-native halves of a task memory have to cross the boundary as tensors.
    The clone is cheap here (~0.8 MiB per matrix) and keeps the payload from
    aliasing live model state.
    """
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).clone()


def _as_feature_matrix(value: Any, *, where: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise CheckpointPayloadError(f"{where} must be a torch.Tensor, got {type(value).__name__}")
    matrix = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    if matrix.ndim != 2:
        raise CheckpointPayloadError(f"{where} must be a 2-D (rows, features) matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise CheckpointPayloadError(f"{where} is empty; a task memory with no rows cannot answer a query")
    # np.ascontiguousarray on a fresh tensor view yields an owned array, so the
    # restored memory does not keep the payload's tensors alive.
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _as_int(value: Any, *, path: str, state_kind: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointPayloadError(f"{state_kind} payload '{path}' must be an integer, got {type(value).__name__}")
    integer = int(value)
    if integer != value or integer < 0:
        raise CheckpointPayloadError(f"{state_kind} payload '{path}' must be a non-negative integer, got {value!r}")
    return integer


def _as_optional_float(value: Any, *, path: str, state_kind: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.item())
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointPayloadError(
            f"{state_kind} payload '{path}' must be a float or None, got {type(value).__name__}"
        )
    return float(value)


def _as_optional_str(value: Any, *, path: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CheckpointPayloadError(f"{path} must be a string or None, got {type(value).__name__}")
    return value


def _as_feature_map_size(value: Any, *, state_kind: str) -> Optional[tuple[int, int]]:
    """Coerce back to a *tuple*, which is not a formality.

    ``fit_task`` compares ``self._feature_map_size != grid_size`` against a
    freshly inferred tuple. Restore a list and that comparison is always true,
    so the first concept after the resume dies with "All UCAD tasks must share
    the same feature grid" — a message that points nowhere near the checkpoint.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CheckpointPayloadError(
            f"{state_kind} payload 'feature_map_size' must be a two-element sequence or None, got {value!r}"
        )
    height = _as_int(value[0], path="feature_map_size[0]", state_kind=state_kind)
    width = _as_int(value[1], path="feature_map_size[1]", state_kind=state_kind)
    if height <= 0 or width <= 0:
        raise CheckpointPayloadError(f"{state_kind} payload 'feature_map_size' must be positive, got {(height, width)}")
    return (height, width)
