"""Validation and the ordered restore driver, plus one end-to-end resume.

The end-to-end test is the one that matters: a run is killed at a concept
boundary, a second "process" resumes from the checkpoint, and the final metric
matrix must be identical to the matrix of a run that was never interrupted. If
that holds, the skip logic, the state protocol, the store and the metric
rehydration all agree with each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.resume.callback import StateCheckpointCallback
from pyclad.resume.plan import apply_resume, build_resume_plan
from pyclad.resume.protocol import (
    CheckpointFingerprintMismatch,
    CheckpointKindError,
    CheckpointMissingError,
    CheckpointOrderMismatch,
    CheckpointVersionDrift,
    CheckpointVersionError,
)
from pyclad.resume.scenario import ResumableConceptAwareScenario, ResumePlan
from pyclad.resume.store import CheckpointStore

from .conftest import (
    CONCEPTS,
    CountingMetric,
    FakeState,
    MatrixMean,
    RecordingStrategy,
    make_dataset,
    make_envelope,
)


class SnapshotCallback:
    """Stand-in for the runner's ``JsonCheckpointCallback``.

    Writes the same block shapes ``write_result_snapshot`` produces, trimmed to
    the concepts that have actually completed.
    """

    def __init__(self, path: Path, metric_callback: ConceptMetricCallback, time_callback: TimeEvaluationCallback):
        self._path = path
        self._metric_callback = metric_callback
        self._time_callback = time_callback
        self._completed: list[str] = []
        self._concept_index = 0

    def before_scenario(self, *args, **kwargs):
        self._write()

    def before_concept_processing(self, concept, *args, **kwargs):
        self._concept_index += 1

    def after_concept_processing(self, concept, *args, **kwargs):
        self._completed.append(concept.name)
        self._write()

    def after_training(self, *args, **kwargs):
        pass

    def before_training(self, *args, **kwargs):
        pass

    def before_evaluation(self, *args, **kwargs):
        pass

    def after_evaluation(self, *args, **kwargs):
        pass

    def after_scenario(self, *args, **kwargs):
        pass

    def seed_completed(self, completed: Sequence[str]) -> None:
        self._completed = [str(name) for name in completed]

    def _write(self):
        order = [name for name in self._metric_callback._learned_concepts if name in set(self._completed)]
        matrix = {learned: dict(self._metric_callback._metric_matrix.get(learned, {})) for learned in order}
        payload = {
            "concept_metric_callback_CountingMetric": {
                "base_metric_name": "CountingMetric",
                "metrics": {},
                "concepts_order": order,
                "metric_matrix": matrix,
            },
            "time_evaluation_callback": {
                "time_by_concept": {
                    name: dict(values) for name, values in self._time_callback._time_by_concept.items()
                },
                "train_time_total": self._time_callback._train_time_total,
                "eval_time_total": self._time_callback._eval_time_total,
            },
            "run_status": {"completed_concepts": list(self._completed)},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CrashAfter:
    """Kills the run after concept *index* has fully committed."""

    def __init__(self, index: int):
        self._index = index
        self._seen = -1

    def before_scenario(self, *args, **kwargs):
        pass

    def after_scenario(self, *args, **kwargs):
        pass

    def before_concept_processing(self, concept, *args, **kwargs):
        self._seen += 1

    def before_training(self, *args, **kwargs):
        pass

    def after_training(self, *args, **kwargs):
        pass

    def before_evaluation(self, *args, **kwargs):
        pass

    def after_evaluation(self, *args, **kwargs):
        pass

    def after_concept_processing(self, concept, *args, **kwargs):
        if self._seen == self._index:
            raise RuntimeError("simulated wall-clock limit")


def _build_run(tmp_path: Path, *, plan: ResumePlan | None = None, crash_after: int | None = None):
    dataset = make_dataset()
    state = FakeState()
    strategy = RecordingStrategy(state)
    metric_callback = ConceptMetricCallback(base_metric=CountingMetric(), metrics=[MatrixMean()])
    time_callback = TimeEvaluationCallback()
    snapshot = SnapshotCallback(tmp_path / "out" / "results.json", metric_callback, time_callback)
    store = CheckpointStore(tmp_path / "ckpt", "runid", keep=2)
    state_callback = StateCheckpointCallback(
        store,
        state,
        concept_order=CONCEPTS,
        seed=1,
        result_json_path=tmp_path / "out" / "results.json",
        plan=plan,
        time_callback=time_callback,
    )
    callbacks: list[Any] = [metric_callback, time_callback, snapshot, state_callback]
    if crash_after is not None:
        callbacks.append(CrashAfter(crash_after))
    scenario = ResumableConceptAwareScenario(dataset, strategy, callbacks, plan=plan)
    return {
        "dataset": dataset,
        "state": state,
        "strategy": strategy,
        "metric_callback": metric_callback,
        "time_callback": time_callback,
        "snapshot": snapshot,
        "store": store,
        "state_callback": state_callback,
        "callbacks": callbacks,
        "scenario": scenario,
    }


def test_resumed_run_reproduces_the_uninterrupted_matrix(tmp_path: Path):
    reference = _build_run(tmp_path / "reference")
    reference["scenario"].run()
    expected_matrix = {row: dict(values) for row, values in reference["metric_callback"]._metric_matrix.items()}

    # --- job 1: dies right after the third concept commits ------------------
    first = _build_run(tmp_path / "chain", crash_after=2)
    with pytest.raises(RuntimeError, match="simulated wall-clock limit"):
        first["scenario"].run()

    store = CheckpointStore(tmp_path / "chain" / "ckpt", "runid", keep=2)
    assert store.committed().index == 2
    assert store.committed().concept_id == "capsule"

    # --- job 2: fresh objects, exactly as a new process would build them -----
    second = _build_run(tmp_path / "chain")
    outcome = build_resume_plan(
        second["store"],
        concept_order=CONCEPTS,
        mode="require",
        expected_kind=FakeState.STATE_KIND,
        expected_state_version=FakeState.STATE_VERSION,
        expected_fingerprint=second["state"].state_fingerprint(),
        result_json_path=tmp_path / "chain" / "out" / "results.json",
    )
    assert outcome.plan.resume_index == 3
    assert outcome.plan.skip_concepts == tuple(CONCEPTS[:3])

    apply_resume(
        outcome,
        state=second["state"],
        strategy=second["strategy"],
        callbacks=second["callbacks"],
        train_concepts=second["dataset"].train_concepts(),
    )
    second["snapshot"].seed_completed(outcome.completed_concepts)
    second["state_callback"]._concept_index = outcome.plan.resume_index - 1
    second["scenario"]._plan = outcome.plan

    ResumableConceptAwareScenario(second["dataset"], second["strategy"], second["callbacks"], plan=outcome.plan).run()

    assert second["strategy"].learn_calls == ["carpet", "grid"]
    assert second["state"].learned == list(CONCEPTS)
    assert second["metric_callback"]._learned_concepts == list(CONCEPTS)
    assert second["metric_callback"]._metric_matrix == expected_matrix
    assert second["state"].cache_invalidations == 1
    assert second["strategy"].resume_restore_calls == [3]
    # The rehydrated wall clock covers the first job's concepts too.
    assert second["time_callback"]._train_time_total >= 0.0
    assert set(second["time_callback"]._time_by_concept) >= set(CONCEPTS)


def test_require_mode_refuses_to_silently_restart(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "runid")
    with pytest.raises(CheckpointMissingError, match="wall-clock allocation"):
        build_resume_plan(store, concept_order=CONCEPTS, mode="require")


def test_auto_mode_starts_fresh_and_says_so(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "runid")
    outcome = build_resume_plan(store, concept_order=CONCEPTS, mode="auto")
    assert outcome.plan.resume_index == 0
    assert outcome.fallback == "no_checkpoint"
    assert outcome.info()["resumed"] is False


def test_resume_off_never_touches_the_store(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "runid")
    store.write(make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS))
    store.commit(index=0, concept_id="bottle")

    outcome = build_resume_plan(store, concept_order=CONCEPTS, mode="off")
    assert outcome.plan.resume_index == 0
    assert outcome.entry is None


def _seeded_store(tmp_path: Path, **envelope_kwargs) -> CheckpointStore:
    store = CheckpointStore(tmp_path / "ckpt", "runid")
    store.write(make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS, **envelope_kwargs))
    store.commit(index=0, concept_id="bottle")
    return store


def test_fingerprint_mismatch_lists_every_offending_field(tmp_path: Path):
    store = _seeded_store(tmp_path, fingerprint={"key_size": 4, "seed": 11})
    with pytest.raises(CheckpointFingerprintMismatch) as error:
        build_resume_plan(
            store,
            concept_order=CONCEPTS,
            mode="auto",
            expected_fingerprint={"key_size": 8, "seed": 12},
        )
    message = str(error.value)
    assert "key_size: checkpoint=4 run=8" in message
    assert "seed: checkpoint=11 run=12" in message
    assert "no override" in message


def test_state_version_mismatch_has_no_override(tmp_path: Path):
    store = _seeded_store(tmp_path, state_version=1)
    with pytest.raises(CheckpointVersionError, match="payload schema changed"):
        build_resume_plan(store, concept_order=CONCEPTS, mode="auto", expected_state_version=2)

    import inspect

    signature = inspect.signature(build_resume_plan)
    assert "force" not in signature.parameters
    assert "allow_state_version_drift" not in signature.parameters


def test_state_kind_mismatch_aborts(tmp_path: Path):
    store = _seeded_store(tmp_path, state_kind="ucad")
    with pytest.raises(CheckpointKindError, match="resumes 'replaycad'"):
        build_resume_plan(store, concept_order=CONCEPTS, mode="auto", expected_kind="replaycad")


def test_concept_order_mismatch_shows_both_orders(tmp_path: Path):
    store = _seeded_store(tmp_path)
    reordered = ["cable", "bottle", "capsule", "carpet", "grid"]
    with pytest.raises(CheckpointOrderMismatch) as error:
        build_resume_plan(store, concept_order=reordered, mode="auto")
    message = str(error.value)
    assert "first difference at index 0" in message
    assert "checkpoint:" in message and "run:" in message


def test_version_drift_aborts_unless_allowed(tmp_path: Path, monkeypatch):
    store = _seeded_store(tmp_path, provenance={"pyclad_version": "0.3.0", "git_commit": "aaaaaaa"})
    monkeypatch.setattr(
        "pyclad.resume.plan._current_provenance",
        lambda: {"pyclad_version": "0.3.0", "git_commit": "bbbbbbb"},
    )

    with pytest.raises(CheckpointVersionDrift, match="--resume-allow-version-drift"):
        build_resume_plan(store, concept_order=CONCEPTS, mode="auto")

    outcome = build_resume_plan(store, concept_order=CONCEPTS, mode="auto", allow_version_drift=True)
    assert outcome.version_drift
    assert "git_commit" in outcome.info()["version_drift"][0]


def test_corrupted_latest_checkpoint_falls_back_to_the_predecessor(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "runid", keep=2)
    for index in range(3):
        store.write(make_envelope(index=index, concept_id=CONCEPTS[index], concept_order=CONCEPTS))
        store.commit(index=index, concept_id=CONCEPTS[index])

    latest = store.path_of(store.committed())
    latest.write_bytes(b"not a torch file")

    outcome = build_resume_plan(store, concept_order=CONCEPTS, mode="auto")
    assert outcome.entry.index == 1
    assert outcome.plan.resume_index == 2
    assert outcome.fallback == "previous_checkpoint:1"
    assert outcome.info()["fallback"] == "previous_checkpoint:1"


def test_all_checkpoints_corrupt_records_a_fresh_start(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "runid", keep=2)
    store.write(make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS))
    store.commit(index=0, concept_id="bottle")
    store.path_of(store.committed()).write_bytes(b"garbage")

    outcome = build_resume_plan(store, concept_order=CONCEPTS, mode="auto")
    assert outcome.plan.resume_index == 0
    assert outcome.fallback == "fresh_start", "the artifact must show this happened, not just the log"

    with pytest.raises(CheckpointMissingError, match="unreadable"):
        build_resume_plan(store, concept_order=CONCEPTS, mode="require")


def test_pending_phase_replays_only_the_evaluation_sweep(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "runid", keep=2)
    store.write(make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS))
    store.commit(index=0, concept_id="bottle")
    store.write(make_envelope(index=1, concept_id="cable", concept_order=CONCEPTS, train_seconds=1234.5))

    committed_only = build_resume_plan(store, concept_order=CONCEPTS, mode="auto", phase="committed")
    assert committed_only.plan.resume_index == 1
    assert committed_only.plan.evaluate_pending_concept is False

    pending = build_resume_plan(store, concept_order=CONCEPTS, mode="auto", phase="pending")
    assert pending.plan.resume_index == 1
    assert pending.plan.evaluate_pending_concept is True
    assert pending.entry.index == 1
    assert pending.pending_train_seconds == pytest.approx(1234.5)


def test_pending_that_does_not_follow_the_commit_is_ignored(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "runid", keep=0)
    store.write(make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS))
    store.commit(index=0, concept_id="bottle")
    store.write(make_envelope(index=3, concept_id="carpet", concept_order=CONCEPTS))

    outcome = build_resume_plan(store, concept_order=CONCEPTS, mode="auto", phase="pending")
    assert outcome.plan.evaluate_pending_concept is False
    assert outcome.plan.resume_index == 1
    assert any("does not directly follow" in warning for warning in outcome.warnings)


def test_pending_replay_restores_the_original_training_time(tmp_path: Path):
    run = _build_run(tmp_path, plan=ResumePlan.after_completed(CONCEPTS, 0, evaluate_pending_concept=True))
    # A pending replay always starts from a store that already holds the state
    # written before the crash; the callback must reuse it rather than rewrite.
    run["store"].write(make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS, train_seconds=999.0))
    run["state_callback"]._pending_train_seconds = 999.0
    run["scenario"].run()

    # Four writes, not five: the replayed concept reuses the checkpoint that the
    # crashed job had already written instead of serialising the model again.
    assert run["state"].save_calls == len(CONCEPTS) - 1

    assert run["strategy"].learn_calls == list(CONCEPTS[1:])
    assert run["time_callback"]._time_by_concept["bottle"]["train_time"] == pytest.approx(999.0)
    assert run["time_callback"]._train_time_total >= 999.0


def test_apply_resume_without_a_result_json_aborts(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "runid")
    store.write(make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS))
    store.commit(index=0, concept_id="bottle")  # no result_json recorded

    outcome = build_resume_plan(store, concept_order=CONCEPTS, mode="auto")
    assert outcome.result_payload is None
    from pyclad.resume.protocol import CheckpointResultMismatch

    with pytest.raises(CheckpointResultMismatch, match="metric matrices"):
        apply_resume(outcome, state=FakeState(), callbacks=[])
