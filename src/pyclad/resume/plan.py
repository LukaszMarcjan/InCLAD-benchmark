"""Validate a checkpoint, then restore a run from it in a fixed order.

Two invariants govern this module:

* **no live object is mutated before validation completes.** Everything up to
  and including fingerprint, order and provenance checks is side-effect free, so
  an aborted resume leaves the process exactly as it found it.
* **no ``learn()`` runs before the restore finishes.** Ordering matters within
  the restore too — the pixel callbacks resolve their training concepts through
  their learned-concept list, so the lists have to come back first.

The order the two functions here implement, matching the design:

1. resolve the concept order (caller's job — it is the runner that owns seeding
   and shuffling, and every step below is keyed on the result);
2. read the manifest and compare the recorded order against it;
3. load and validate the envelope: kind, payload version, fingerprint,
   provenance, sidecar checksums — all side-effect free;
4. construct model and strategy fresh (caller's job; pretrained backbones are
   deliberately absent from the payload and rebuild here);
5. ``model.load_state(...)``;
6. strategy-level restore through the ``resume_restore`` hook;
7. invalidate the prediction cache;
8. rehydrate the metric callbacks from the result JSON;
9. rehydrate the pixel callbacks' training concepts;
10. prime the run-status provider and the JSON checkpoint callback.

Steps 5-10 live in :func:`apply_resume`. Any exception aborts the run; there is
no partial-restore path, because a partially restored run produces numbers
rather than errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pyclad.data.concept import Concept
from pyclad.resume.fingerprint import (
    assert_concept_order_matches,
    assert_fingerprint_matches,
    assert_provenance_matches,
)
from pyclad.resume.fingerprint import current_provenance as _current_provenance
from pyclad.resume.metrics_restore import (
    RestoreReport,
    load_result_payload,
    prime_json_checkpoint_callback,
    restore_callbacks,
    restore_run_status,
    restore_training_concepts,
)
from pyclad.resume.protocol import (
    CheckpointCorruptionError,
    CheckpointMissingError,
    CheckpointResultMismatch,
    merge_sidecars,
    validate_envelope,
)
from pyclad.resume.path_map import map_path, map_values
from pyclad.resume.scenario import ResumePlan
from pyclad.resume.store import CheckpointEntry, CheckpointStore

logger = logging.getLogger(__name__)

__all__ = [
    "RESUME_MODES",
    "RESUME_PHASES",
    "ResumeOutcome",
    "build_resume_plan",
    "apply_resume",
]

RESUME_MODES = ("off", "auto", "require")
RESUME_PHASES = ("committed", "pending")


@dataclass(frozen=True)
class ResumeOutcome:
    """Everything a runner needs to start, whether or not it resumed."""

    plan: ResumePlan
    mode: str
    entry: CheckpointEntry | None = None
    envelope: Mapping[str, Any] | None = None
    state_payload: Mapping[str, Any] | None = None
    result_payload: Mapping[str, Any] | None = None
    completed_concepts: tuple[str, ...] = ()
    determinism: Mapping[str, Any] | None = None
    version_drift: tuple[str, ...] = ()
    fallback: str | None = None
    pending_train_seconds: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def resumed(self) -> bool:
        return self.plan.resumed or self.plan.evaluate_pending_concept

    def info(self) -> dict[str, Any]:
        """Payload for ``strategy.additional_info()["resume"]``.

        A resumed run has to be identifiable from its published artifact alone —
        including the cases where it silently fell back to a fresh start or
        crossed a code version boundary.
        """
        payload: dict[str, Any] = {
            "enabled": self.mode != "off",
            "mode": self.mode,
            "resumed": self.resumed,
            "resumed_from_concept": self.entry.index if self.entry is not None else None,
            "resume_index": self.plan.resume_index,
            "skipped_concepts": list(self.plan.skip_concepts),
            "evaluate_pending_concept": self.plan.evaluate_pending_concept,
        }
        if self.determinism:
            payload["determinism"] = dict(self.determinism)
        if self.version_drift:
            payload["version_drift"] = list(self.version_drift)
        if self.fallback:
            payload["fallback"] = self.fallback
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def build_resume_plan(
    store: CheckpointStore,
    *,
    concept_order: Sequence[str],
    mode: str = "auto",
    phase: str = "committed",
    expected_kind: str | None = None,
    expected_state_version: int | None = None,
    expected_fingerprint: Mapping[str, Any] | None = None,
    result_json_path: str | Path | None = None,
    allow_version_drift: bool = False,
) -> ResumeOutcome:
    """Decide where this process starts, validating everything first."""
    if mode not in RESUME_MODES:
        raise ValueError(f"Unknown resume mode {mode!r}; expected one of {RESUME_MODES}")
    if phase not in RESUME_PHASES:
        raise ValueError(f"Unknown resume phase {phase!r}; expected one of {RESUME_PHASES}")

    order = [str(name) for name in concept_order]
    if mode == "off":
        return ResumeOutcome(plan=ResumePlan.fresh(order), mode=mode)

    manifest = store.read_manifest()
    if manifest is None:
        return _no_checkpoint(mode, order, reason=f"no manifest under {store.directory}")

    recorded_order = [str(name) for name in manifest.get("concept_order", [])]
    if recorded_order:
        assert_concept_order_matches(recorded_order, order)

    committed = store.committed()
    pending = store.pending()
    if committed is None and pending is None:
        return _no_checkpoint(mode, order, reason=f"manifest in {store.directory} has no committed checkpoint")

    warnings: list[str] = []
    state_entry, metrics_entry, evaluate_pending = _select_entries(
        committed=committed, pending=pending, phase=phase, warnings=warnings
    )
    if state_entry is None:
        return _no_checkpoint(mode, order, reason=f"manifest in {store.directory} has no committed checkpoint")

    fallback: str | None = None
    candidates = _candidates(store, state_entry)
    last_error: CheckpointCorruptionError | None = None
    envelope = None
    sidecars: Mapping[str, Any] = {}
    for candidate in candidates:
        try:
            envelope, sidecars = store.load(candidate)
        except CheckpointCorruptionError as error:
            # Corruption is the only failure that falls back. A fingerprint or
            # version mismatch is a different run, and silently reaching for an
            # older checkpoint of a different run is worse than stopping.
            last_error = error
            logger.error(
                "Checkpoint for concept %d (%s) is unusable (%s); trying the previous retained checkpoint",
                candidate.index,
                candidate.concept_id,
                error,
            )
            continue
        if candidate is not state_entry:
            fallback = f"previous_checkpoint:{candidate.index}"
            evaluate_pending = False
            metrics_entry = candidate
            warnings.append(
                f"Fell back to the checkpoint for concept {candidate.index} ({candidate.concept_id}); "
                f"concepts after it will be recomputed."
            )
        state_entry = candidate
        break
    else:
        if mode == "require":
            raise CheckpointMissingError(
                f"Every retained checkpoint under {store.directory} is unreadable "
                f"(last error: {last_error}). --resume require will not silently restart from concept 0."
            )
        logger.error("All retained checkpoints are unreadable; starting a fresh run and recording the fallback")
        return ResumeOutcome(
            plan=ResumePlan.fresh(order),
            mode=mode,
            fallback="fresh_start",
            warnings=tuple(warnings + [f"all retained checkpoints unreadable: {last_error}"]),
        )

    assert envelope is not None
    validate_envelope(envelope, expected_kind=expected_kind, expected_state_version=expected_state_version)

    if expected_fingerprint is not None:
        # assert_fingerprint_matches(envelope.get("fingerprint", {}), expected_fingerprint)
        # Recorded paths (e.g. replaycad.artifact_dir) are mapped through PYCLAD_RESUME_PATH_MAP;
        # without the variable map_values returns the fingerprint unchanged.
        assert_fingerprint_matches(map_values(envelope.get("fingerprint", {})), expected_fingerprint)

    drift = assert_provenance_matches(
        envelope.get("provenance", {}),
        _current_provenance(),
        allow_drift=allow_version_drift,
    )
    if drift:
        for diff in drift:
            logger.warning("Resuming across a code change: %s", diff)

    run_block = envelope.get("run", {})
    envelope_order = [str(name) for name in run_block.get("concept_order", [])]
    if envelope_order:
        assert_concept_order_matches(envelope_order, order)

    completed_count = state_entry.index if evaluate_pending else state_entry.index + 1
    completed = order[:completed_count]

    result_payload = _load_result_payload(
        entry=metrics_entry,
        committed=committed,
        explicit_path=result_json_path,
        warnings=warnings,
    )

    plan = ResumePlan.after_completed(order, completed_count, evaluate_pending_concept=evaluate_pending)
    state_payload = merge_sidecars(envelope.get("state", {}), sidecars)
    train_seconds = run_block.get("train_seconds") if evaluate_pending else None

    logger.info(
        "Resuming run %s from concept %d (%s); %d concepts already completed",
        store.run_id,
        plan.resume_index,
        order[plan.resume_index] if plan.resume_index < len(order) else "<none left>",
        len(completed),
    )

    return ResumeOutcome(
        plan=plan,
        mode=mode,
        entry=state_entry,
        envelope=envelope,
        state_payload=state_payload,
        result_payload=result_payload,
        completed_concepts=tuple(completed),
        determinism=envelope.get("determinism"),
        version_drift=tuple(str(diff) for diff in drift),
        fallback=fallback,
        pending_train_seconds=float(train_seconds) if train_seconds is not None else None,
        warnings=tuple(warnings),
    )


def apply_resume(
    outcome: ResumeOutcome,
    *,
    state: Any,
    strategy: Any = None,
    callbacks: Iterable[Any] = (),
    train_concepts: Sequence[Concept] = (),
    run_status_provider: Any = None,
    json_checkpoint_callback: Any = None,
) -> RestoreReport:
    """Steps 5-10: push the validated checkpoint into live objects."""
    report = RestoreReport()
    if not outcome.resumed:
        return report

    if outcome.state_payload is None:
        raise CheckpointCorruptionError("Resume outcome carries no state payload; nothing to restore")

    callbacks = list(callbacks)

    # 5 - model state.
    state.load_state(outcome.state_payload)
    report.restored.append(f"{type(state).__name__}.load_state")

    # 6 - strategy bookkeeping. Skipping this is how a perfectly restored model
    # gets erased at the next concept with no error and no log line.
    if strategy is not None:
        hook = getattr(strategy, "resume_restore", None)
        if callable(hook):
            hook(outcome.state_payload, outcome.plan)
            report.restored.append(f"{type(strategy).__name__}.resume_restore")
        else:
            logger.debug(
                "Strategy %s has no resume_restore hook; assuming it keeps no state of its own",
                type(strategy).__name__,
            )

    # 7 - prediction caches key on object identity, so a restored cache can
    # alias a reused array and hand back stale scores. The strategy usually
    # holds the very object we just restored, so de-duplicate by identity.
    seen_ids: set[int] = set()
    for target in (state, getattr(strategy, "_model", None)):
        if target is None or id(target) in seen_ids:
            continue
        seen_ids.add(id(target))
        invalidate = getattr(target, "_invalidate_prediction_cache", None)
        if callable(invalidate):
            invalidate()

    # 8 - metric matrices, thresholds and wall clock.
    if outcome.result_payload is not None:
        restore_callbacks(callbacks, outcome.result_payload, outcome.completed_concepts, report=report)
    elif outcome.completed_concepts:
        raise CheckpointResultMismatch(
            "No partial result JSON was available, so the metric matrices for "
            f"{len(outcome.completed_concepts)} completed concepts cannot be restored."
        )

    # 9 - pixel thresholds quantile over *all* seen training concepts.
    if train_concepts:
        restore_training_concepts(callbacks, train_concepts, outcome.completed_concepts, report=report)

    # 10 - run status continues rather than restarting.
    if run_status_provider is not None:
        restore_run_status(run_status_provider, outcome.completed_concepts)
    if json_checkpoint_callback is not None:
        prime_json_checkpoint_callback(json_checkpoint_callback, len(outcome.completed_concepts))

    for warning in report.warnings:
        logger.warning("Resume restore: %s", warning)
    return report


# ---------------------------------------------------------------------- helpers


def _no_checkpoint(mode: str, order: Sequence[str], *, reason: str) -> ResumeOutcome:
    if mode == "require":
        raise CheckpointMissingError(
            f"--resume require was given but no usable checkpoint was found ({reason}). "
            "Refusing to start from concept 0: a chained job that silently restarts burns its whole "
            "wall-clock allocation before anyone notices."
        )
    logger.info("No checkpoint to resume from (%s); starting a fresh run", reason)
    return ResumeOutcome(plan=ResumePlan.fresh(order), mode=mode, fallback="no_checkpoint")


def _select_entries(
    *,
    committed: CheckpointEntry | None,
    pending: CheckpointEntry | None,
    phase: str,
    warnings: list[str],
) -> tuple[CheckpointEntry | None, CheckpointEntry | None, bool]:
    """Pick the state entry, the metrics entry, and whether to replay pending."""
    if phase != "pending" or pending is None:
        return committed, committed, False

    expected_index = (committed.index + 1) if committed is not None else 0
    if pending.index != expected_index:
        warnings.append(
            f"Ignoring the pending checkpoint for concept {pending.index}: it does not directly follow the "
            f"committed concept (expected index {expected_index})."
        )
        logger.warning("%s", warnings[-1])
        return committed, committed, False

    return pending, committed, True


def _candidates(store: CheckpointStore, entry: CheckpointEntry) -> list[CheckpointEntry]:
    """The chosen entry, then retained predecessors, newest first."""
    history = [item for item in store.history() if item.index < entry.index and item.filename != entry.filename]
    history.sort(key=lambda item: item.index, reverse=True)
    return [entry] + history


def _load_result_payload(
    *,
    entry: CheckpointEntry | None,
    committed: CheckpointEntry | None,
    explicit_path: str | Path | None,
    warnings: list[str],
) -> Mapping[str, Any] | None:
    if entry is None:
        return None

    path = explicit_path if explicit_path is not None else map_path(entry.result_json)
    if path is None:
        warnings.append("The checkpoint manifest recorded no partial result JSON; metric matrices cannot be restored.")
        logger.warning("%s", warnings[-1])
        return None

    is_current_commit = committed is not None and entry.filename == committed.filename
    expected_sha = entry.result_json_sha256 if is_current_commit else None
    if not is_current_commit:
        warnings.append(
            "Restoring metrics from a result JSON that is newer than the checkpoint being resumed; "
            "rows after the restored concept are dropped and their wall clock is approximate."
        )
        logger.warning("%s", warnings[-1])
    elif not expected_sha:
        warnings.append("The manifest recorded no sha256 for the partial result JSON; pairing is unverified.")
        logger.warning("%s", warnings[-1])

    return load_result_payload(path, expected_sha256=expected_sha)
