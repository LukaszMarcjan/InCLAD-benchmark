"""Durability tests for :class:`CheckpointStore`.

The important ones simulate a kill at each of the three points where a naive
implementation loses data: during the payload write, after the payload landed
but before the manifest commit, and during the manifest rewrite itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from pyclad.resume.protocol import (
    SIDECAR_KEY,
    CheckpointCorruptionError,
    CheckpointDiskSpaceError,
    CheckpointLockError,
    CheckpointPayloadError,
)
from pyclad.resume.store import CheckpointStore, ensure_resume_dir_outside_output

from .conftest import CONCEPTS, make_envelope


def _write_and_commit(store: CheckpointStore, index: int, *, result_json: Path | None = None):
    envelope = make_envelope(index=index, concept_id=CONCEPTS[index], concept_order=CONCEPTS)
    store.write(envelope)
    return store.commit(index=index, concept_id=CONCEPTS[index], result_json=result_json)


def test_write_commit_load_round_trip(store: CheckpointStore, tmp_path: Path):
    result_json = tmp_path / "results.json"
    result_json.write_text(json.dumps({"run_status": {"completed_concepts": ["bottle"]}}), encoding="utf-8")

    entry = _write_and_commit(store, 0, result_json=result_json)

    assert entry.index == 0
    assert entry.concept_id == "bottle"
    assert entry.result_json_sha256

    envelope, sidecars = store.load(store.committed())
    assert envelope["state_kind"] == "fake_state"
    assert envelope["run"]["concept_order"] == list(CONCEPTS)
    assert torch.equal(envelope["state"]["weights"], torch.ones(2))
    assert sidecars == {}
    assert store.pending() is None


def test_sidecars_round_trip_and_detect_tampering(store: CheckpointStore):
    bank = np.arange(12, dtype=np.float32).reshape(4, 3)
    envelope = make_envelope(
        index=0,
        concept_id="bottle",
        concept_order=CONCEPTS,
        state={"learned": ["bottle"], "weights": torch.ones(2), SIDECAR_KEY: {"memory_bank": bank}},
    )
    store.write(envelope)
    entry = store.commit(index=0, concept_id="bottle")

    loaded_envelope, sidecars = store.load(entry)
    assert SIDECAR_KEY not in loaded_envelope["state"], "sidecars must not be duplicated inside the payload"
    np.testing.assert_array_equal(sidecars["memory_bank"], bank)

    sidecar_path = store.directory / entry.sidecars[0].filename
    tampered = np.zeros_like(bank)
    np.save(sidecar_path, tampered)
    with pytest.raises(CheckpointCorruptionError, match="sha256"):
        store.load(entry)


def test_numpy_in_payload_is_rejected_at_write_time(store: CheckpointStore):
    """weights_only=True would refuse this at *load* time, hours later."""
    envelope = make_envelope(
        index=0,
        concept_id="bottle",
        concept_order=CONCEPTS,
        state={"learned": ["bottle"], "weights": np.ones(3)},
    )
    with pytest.raises(CheckpointPayloadError, match="numpy"):
        store.write(envelope)


def test_crash_during_payload_write_leaves_previous_checkpoint_valid(store: CheckpointStore, monkeypatch):
    _write_and_commit(store, 0)
    manifest_before = json.loads(store.manifest_path.read_text(encoding="utf-8"))

    class Boom(OSError):
        pass

    def exploding_save(payload, handle):
        handle.write(b"partial bytes, then the node dies")
        raise Boom(28, "No space left on device")

    monkeypatch.setattr(CheckpointStore, "_torch_save", staticmethod(exploding_save))

    with pytest.raises(CheckpointDiskSpaceError):
        store.write(make_envelope(index=1, concept_id="cable", concept_order=CONCEPTS))

    manifest_after = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest_after["committed"] == manifest_before["committed"]
    assert manifest_after["pending"] is None
    assert not list(store.directory.glob("*.tmp-*")), "temporary files must be unlinked on failure"
    assert not (store.directory / "concept_001_cable.ckpt").exists()
    assert not list(store.directory.glob("concept_001_*.npy")), "orphan sidecars must not survive a failed write"

    envelope, _ = store.load(store.committed())
    assert envelope["run"]["concept_id"] == "bottle"


def test_failed_write_removes_the_sidecar_it_had_already_written(store: CheckpointStore, monkeypatch):
    """A multi-GB orphan would make the next attempt fail sooner, not later."""
    envelope = make_envelope(
        index=0,
        concept_id="bottle",
        concept_order=CONCEPTS,
        state={
            "learned": ["bottle"],
            "weights": torch.ones(2),
            SIDECAR_KEY: {"memory_bank": np.zeros((4, 3), dtype=np.float32)},
        },
    )

    def boom(payload, handle):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(CheckpointStore, "_torch_save", staticmethod(boom))
    with pytest.raises(CheckpointDiskSpaceError):
        store.write(envelope)

    assert not list(store.directory.glob("*.npy"))
    assert not list(store.directory.glob("*.ckpt"))
    assert store.read_manifest() is None


def test_crash_after_payload_write_but_before_manifest_commit(store: CheckpointStore, monkeypatch):
    """A file the manifest does not name is garbage and is never read."""
    _write_and_commit(store, 0)

    original_write_manifest = CheckpointStore._write_manifest

    def exploding_manifest(self, manifest):
        raise OSError(5, "I/O error while committing the manifest")

    monkeypatch.setattr(CheckpointStore, "_write_manifest", exploding_manifest)
    with pytest.raises(OSError):
        store.write(make_envelope(index=1, concept_id="cable", concept_order=CONCEPTS))

    monkeypatch.setattr(CheckpointStore, "_write_manifest", original_write_manifest)

    orphan = store.directory / "concept_001_cable.ckpt"
    assert orphan.exists(), "the payload did land on disk"
    committed = store.committed()
    assert committed.index == 0, "but the manifest still points at the last committed concept"
    assert store.pending() is None
    envelope, _ = store.load(committed)
    assert envelope["run"]["concept_id"] == "bottle"


def test_crash_during_manifest_rewrite_leaves_the_old_manifest(store: CheckpointStore, monkeypatch):
    _write_and_commit(store, 0)
    before = store.manifest_path.read_bytes()

    real_replace = os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(dst).endswith("manifest.json"):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("pyclad.resume.store.os.replace", failing_replace)
    with pytest.raises(CheckpointDiskSpaceError):
        store.write(make_envelope(index=1, concept_id="cable", concept_order=CONCEPTS))
    monkeypatch.setattr("pyclad.resume.store.os.replace", real_replace)

    assert store.manifest_path.read_bytes() == before
    assert json.loads(before)["committed"]["index"] == 0
    assert not list(store.directory.glob("*.tmp-*"))


def test_truncated_checkpoint_is_detected_not_silently_loaded(store: CheckpointStore):
    _write_and_commit(store, 0)
    entry = store.committed()
    path = store.path_of(entry)
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])

    with pytest.raises(CheckpointCorruptionError):
        store.load(entry)


def test_retention_keeps_the_window_and_never_the_committed_entry(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "run", keep=2)
    for index in range(4):
        _write_and_commit(store, index)

    files = sorted(path.name for path in store.directory.glob("concept_*.ckpt"))
    assert files == ["concept_002_capsule.ckpt", "concept_003_carpet.ckpt"]
    assert store.committed().index == 3
    envelope, _ = store.load(store.committed())
    assert envelope["run"]["concept_id"] == "carpet"


def test_unlimited_retention_keeps_everything(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "run", keep=0)
    for index in range(4):
        _write_and_commit(store, index)
    assert len(list(store.directory.glob("concept_*.ckpt"))) == 4


def test_variants_are_never_pruned(tmp_path: Path):
    """A GPM-style basis cannot be recomputed once earlier data is gone."""
    store = CheckpointStore(tmp_path / "ckpt", "run", keep=1)
    for index in range(3):
        envelope = make_envelope(index=index, concept_id=CONCEPTS[index], concept_order=CONCEPTS)
        store.write(envelope)
        store.write(envelope, variant="basis_only")
        store.commit(index=index, concept_id=CONCEPTS[index])

    variants = sorted(path.name for path in store.directory.glob("*.basis_only.ckpt"))
    assert len(variants) == 3
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["variants"]) == 3
    assert manifest["pending"] is None


def test_disk_space_preflight_refuses_before_writing_anything(store: CheckpointStore, monkeypatch):
    _write_and_commit(store, 0)

    class TinyUsage:
        free = 1

    monkeypatch.setattr("pyclad.resume.store.shutil.disk_usage", lambda path: TinyUsage())
    with pytest.raises(CheckpointDiskSpaceError, match="Nothing was written"):
        store.write(make_envelope(index=1, concept_id="cable", concept_order=CONCEPTS))
    assert store.committed().index == 0


def test_commit_refuses_a_mismatched_index(store: CheckpointStore):
    store.write(make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS))
    with pytest.raises(CheckpointCorruptionError, match="commit was asked"):
        store.commit(index=1, concept_id="cable")


def test_lock_rejects_a_second_live_process(tmp_path: Path):
    first = CheckpointStore(tmp_path / "ckpt", "run")
    first.acquire_lock()
    second = CheckpointStore(tmp_path / "ckpt", "run")

    lock = json.loads(second.lock_path.read_text(encoding="utf-8"))
    lock["owner_pid"] = os.getpid() + 1
    lock["hostname"] = "some-other-node"
    second.lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(CheckpointLockError, match="already active"):
        second.acquire_lock()


def test_stale_lock_is_taken_over(tmp_path: Path):
    store = CheckpointStore(tmp_path / "ckpt", "run", lock_stale_after=1.0)
    store.acquire_lock()
    lock = json.loads(store.lock_path.read_text(encoding="utf-8"))
    lock["owner_pid"] = os.getpid() + 1
    lock["hostname"] = "dead-node"
    lock["heartbeat"] = 0.0
    store.lock_path.write_text(json.dumps(lock), encoding="utf-8")

    other = CheckpointStore(tmp_path / "ckpt", "run", lock_stale_after=1.0)
    other.acquire_lock()
    assert json.loads(other.lock_path.read_text(encoding="utf-8"))["owner_pid"] == os.getpid()


def test_resume_dir_inside_output_dir_is_refused(tmp_path: Path):
    output = tmp_path / "job-26414"
    ensure_resume_dir_outside_output(tmp_path / "scratch", output)
    with pytest.raises(ValueError, match="inside --output-dir"):
        ensure_resume_dir_outside_output(output / "checkpoints", output)
