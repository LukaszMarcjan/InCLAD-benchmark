"""The callback that turns a running scenario into a resumable one.

It writes at ``after_training`` — before the evaluation sweep — because that is
where the expensive, unrepeatable work has just finished. The existing
``JsonCheckpointCallback`` writes the result snapshot at
``after_concept_processing``, after evaluation. The two together implement the
two-phase commit described in :mod:`pyclad.resume.store`: this callback writes
the state file (manifest ``pending``) and then commits it (manifest
``committed``) once the sweep is done and the result JSON has been refreshed.

One deliberate difference from ``JsonCheckpointCallback``: that one catches and
logs write failures, which is right for a result snapshot — losing one snapshot
costs one snapshot. This one lets them propagate. A run that cannot checkpoint
is a run that cannot be resumed, and every remaining hour of wall clock it burns
is wasted. Failing at concept 7 costs one concept; failing silently costs the
whole job.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.resume.fingerprint import current_provenance
from pyclad.resume.protocol import (
    DETERMINISM_BIT_IDENTICAL,
    PHASE_AFTER_TRAINING,
    ResumableState,
    build_envelope,
)
from pyclad.resume.scenario import ResumePlan
from pyclad.resume.store import CheckpointEntry, CheckpointStore

logger = logging.getLogger(__name__)

__all__ = ["StateCheckpointCallback"]


class StateCheckpointCallback(Callback):
    """Persist a :class:`ResumableState` at every concept boundary.

    The callback must be registered **last** in the callback list. It reads the
    training wall clock it measured itself back into the timing callback when a
    pending concept is replayed, and that only works if the timing callback has
    already run for the same event.
    """

    def __init__(
        self,
        store: CheckpointStore,
        state: ResumableState,
        *,
        concept_order: Sequence[str],
        seed: int | None = None,
        result_json_path: str | Path | None = None,
        determinism: Mapping[str, Any] | None = None,
        plan: ResumePlan | None = None,
        fingerprint: Mapping[str, Any] | None = None,
        extra_run_info: Mapping[str, Any] | None = None,
        time_callback: Any | None = None,
        pending_train_seconds: float | None = None,
        variants: Callable[[], Mapping[str, Mapping[str, Any]]] | None = None,
        enabled: bool = True,
    ):
        self._store = store
        self._state = state
        self._concept_order = [str(name) for name in concept_order]
        self._seed = seed
        self._result_json_path = Path(result_json_path).expanduser() if result_json_path is not None else None
        self._determinism = dict(determinism or {"guarantee": DETERMINISM_BIT_IDENTICAL, "caveats": []})
        self._plan = plan
        self._fingerprint_override = dict(fingerprint) if fingerprint is not None else None
        self._extra_run_info = dict(extra_run_info or {})
        self._time_callback = time_callback
        self._pending_train_seconds = pending_train_seconds
        self._variants = variants
        self._enabled = bool(enabled)

        start_index = plan.resume_index if plan is not None else 0
        self._concept_index = start_index - 1
        self._concept_name: str | None = None
        self._train_start: float | None = None
        self._last_entry: CheckpointEntry | None = None

    @property
    def last_entry(self) -> CheckpointEntry | None:
        return self._last_entry

    def before_concept_processing(self, concept: Concept, *args, **kwargs) -> None:
        self._concept_index += 1
        self._concept_name = concept.name
        expected = self._concept_order[self._concept_index] if self._concept_index < len(self._concept_order) else None
        if expected is not None and expected != concept.name:
            raise RuntimeError(
                f"Checkpoint bookkeeping desynchronised: concept index {self._concept_index} should be "
                f"'{expected}' but the scenario is processing '{concept.name}'"
            )

    def before_training(self, *args, **kwargs) -> None:
        self._train_start = time.perf_counter()

    def after_training(self, learned_concept: Concept, *args, **kwargs) -> None:
        if not self._enabled:
            return

        train_seconds = None if self._train_start is None else time.perf_counter() - self._train_start

        if self._is_pending_replay():
            # The state for this concept is already on disk, written before the
            # crash. Rewriting it would be pointless and would overwrite the
            # recorded training time with the ~0 s this replay took.
            self._restore_pending_train_time(learned_concept.name)
            logger.info(
                "Resume: reusing the pending checkpoint for concept %s instead of rewriting it",
                learned_concept.name,
            )
            return

        run_info = {
            "run_id": self._store.run_id,
            "seed": self._seed,
            "concept_index": self._concept_index,
            "concept_id": learned_concept.name,
            "concept_order": list(self._concept_order),
            "phase": PHASE_AFTER_TRAINING,
            "train_seconds": train_seconds,
            **self._extra_run_info,
        }
        envelope = build_envelope(
            state_kind=type(self._state).STATE_KIND,
            state_version=type(self._state).STATE_VERSION,
            provenance=current_provenance(),
            run=run_info,
            determinism=self._determinism,
            fingerprint=(
                self._fingerprint_override
                if self._fingerprint_override is not None
                else self._state.state_fingerprint()
            ),
            state=self._state.save_state(),
        )
        self._last_entry = self._store.write(envelope)
        self._write_variants(run_info)

    def after_concept_processing(self, concept: Concept, *args, **kwargs) -> None:
        if not self._enabled:
            return
        self._store.commit(
            index=self._concept_index,
            concept_id=concept.name,
            result_json=self._result_json_path,
        )

    # ------------------------------------------------------------------ internals

    def _is_pending_replay(self) -> bool:
        return (
            self._plan is not None
            and self._plan.evaluate_pending_concept
            and self._concept_index == self._plan.resume_index
        )

    def _restore_pending_train_time(self, concept_name: str) -> None:
        """Put the *original* training time back into the timing callback.

        Without this the resumed result JSON would report the few milliseconds
        this replay spent not training, and the published wall clock for that
        concept would be meaningless.
        """
        if self._pending_train_seconds is None or self._time_callback is None:
            return
        by_concept = getattr(self._time_callback, "_time_by_concept", None)
        if by_concept is None:
            return
        replayed = float(by_concept[concept_name].get("train_time", 0.0))
        by_concept[concept_name]["train_time"] = float(self._pending_train_seconds)
        total = getattr(self._time_callback, "_train_time_total", None)
        if total is not None:
            self._time_callback._train_time_total = float(total) - replayed + float(self._pending_train_seconds)

    def _write_variants(self, run_info: Mapping[str, Any]) -> None:
        """Write extra, never-pruned snapshots such as a basis-only file.

        A component opts in by exposing ``save_variants()`` (or by the caller
        passing ``variants=``) returning ``{variant_name: payload}``. This exists
        for state that cannot be recomputed after the fact — a GPM basis needs
        every previous concept's training data, which a continual benchmark has
        by definition thrown away, whereas weights can always be retrained.
        """
        provider = self._variants or getattr(self._state, "save_variants", None)
        if not callable(provider):
            return
        variants = provider() or {}
        for name, payload in variants.items():
            envelope = build_envelope(
                state_kind=type(self._state).STATE_KIND,
                state_version=type(self._state).STATE_VERSION,
                provenance=current_provenance(),
                run=dict(run_info),
                determinism=self._determinism,
                fingerprint=(
                    self._fingerprint_override
                    if self._fingerprint_override is not None
                    else self._state.state_fingerprint()
                ),
                state=payload,
            )
            self._store.write(envelope, variant=str(name))
