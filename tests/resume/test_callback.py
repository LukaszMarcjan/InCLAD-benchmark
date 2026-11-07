"""Behaviour of :class:`StateCheckpointCallback` at the concept boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyclad.data.concept import Concept
from pyclad.resume.callback import StateCheckpointCallback
from pyclad.resume.protocol import CheckpointDiskSpaceError
from pyclad.resume.scenario import ResumePlan
from pyclad.resume.store import CheckpointStore

from .conftest import CONCEPTS, FakeState, FakeStateWithSidecar


def _concept(name: str) -> Concept:
    return Concept(name=name, data=np.zeros((2, 2)))


def _callback(store: CheckpointStore, state, **kwargs) -> StateCheckpointCallback:
    return StateCheckpointCallback(store, state, concept_order=CONCEPTS, seed=1, **kwargs)


def _process(callback: StateCheckpointCallback, name: str) -> None:
    callback.before_concept_processing(_concept(name))
    callback.before_training()
    callback.after_training(_concept(name))
    callback.after_concept_processing(_concept(name))


def test_two_phase_commit_moves_pending_to_committed(store: CheckpointStore):
    state = FakeState()
    callback = _callback(store, state)

    callback.before_concept_processing(_concept("bottle"))
    callback.before_training()
    state.fit("bottle")
    callback.after_training(_concept("bottle"))

    assert store.pending().index == 0
    assert store.committed() is None, "not committed until the evaluation sweep finishes"

    callback.after_concept_processing(_concept("bottle"))
    assert store.committed().index == 0
    assert store.pending() is None


def test_write_failures_are_not_swallowed(store: CheckpointStore, monkeypatch):
    """Unlike the result-snapshot callback, this one must fail loudly.

    Continuing means burning the rest of the wall clock on a job that can no
    longer be resumed.
    """
    callback = _callback(store, FakeState())

    class TinyUsage:
        free = 1

    monkeypatch.setattr("pyclad.resume.store.shutil.disk_usage", lambda path: TinyUsage())

    callback.before_concept_processing(_concept("bottle"))
    callback.before_training()
    with pytest.raises(CheckpointDiskSpaceError):
        callback.after_training(_concept("bottle"))


def test_records_the_training_wall_clock_and_the_run_block(store: CheckpointStore):
    callback = _callback(store, FakeState())
    _process(callback, "bottle")

    envelope, _ = store.load(store.committed())
    run = envelope["run"]
    assert run["concept_index"] == 0
    assert run["concept_id"] == "bottle"
    assert run["concept_order"] == list(CONCEPTS)
    assert run["seed"] == 1
    assert run["phase"] == "after_training"
    assert run["train_seconds"] >= 0.0
    assert envelope["fingerprint"] == {"key_size": 4, "seed": 11}


def test_resumed_callback_numbers_concepts_absolutely(store: CheckpointStore):
    plan = ResumePlan.after_completed(CONCEPTS, 3)
    callback = _callback(store, FakeState(), plan=plan)

    _process(callback, "carpet")
    assert store.committed().index == 3, "the resumed concept keeps its absolute index"

    _process(callback, "grid")
    assert store.committed().index == 4


def test_desynchronised_bookkeeping_is_caught_immediately(store: CheckpointStore):
    callback = _callback(store, FakeState())
    callback.before_concept_processing(_concept("bottle"))
    with pytest.raises(RuntimeError, match="desynchronised"):
        callback.before_concept_processing(_concept("capsule"))


def test_result_json_sha_is_recorded_at_commit(store: CheckpointStore, tmp_path: Path):
    result_json = tmp_path / "results.json"
    result_json.write_text(json.dumps({"run_status": {"completed_concepts": []}}), encoding="utf-8")

    callback = _callback(store, FakeState(), result_json_path=result_json)
    _process(callback, "bottle")

    entry = store.committed()
    assert entry.result_json == str(result_json)
    assert entry.result_json_sha256


def test_sidecar_state_lands_beside_the_checkpoint(store: CheckpointStore):
    state = FakeStateWithSidecar(bank_rows=4)
    callback = _callback(store, state)
    state.fit("bottle")
    _process(callback, "bottle")

    entry = store.committed()
    assert [sidecar.name for sidecar in entry.sidecars] == ["memory_bank"]
    assert (store.directory / entry.sidecars[0].filename).suffix == ".npy"

    restored = FakeStateWithSidecar(bank_rows=4)
    envelope, sidecars = store.load(entry)
    from pyclad.resume.protocol import merge_sidecars

    restored.load_state(merge_sidecars(envelope["state"], sidecars))
    np.testing.assert_array_equal(restored.bank, state.bank)
    assert restored.learned == ["bottle"]


def test_variants_are_written_alongside_the_main_checkpoint(store: CheckpointStore):
    state = FakeState()
    callback = _callback(store, state, variants=lambda: {"basis_only": {"basis": [1.0, 2.0]}})
    _process(callback, "bottle")

    variant_files = list(store.directory.glob("*.basis_only.ckpt"))
    assert len(variant_files) == 1
    assert store.committed().variant is None, "a variant must never become the committed entry"


def test_disabled_callback_writes_nothing(store: CheckpointStore):
    callback = _callback(store, FakeState(), enabled=False)
    _process(callback, "bottle")
    assert store.read_manifest() is None
