"""The contracts a method class has to satisfy, and the guards behind them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyclad.resume.fingerprint import compute_run_id, fingerprint_from_config, json_safe
from pyclad.resume.protocol import (
    ENVELOPE_VERSION,
    SIDECAR_KEY,
    CheckpointEnvelopeError,
    CheckpointKindError,
    CheckpointPayloadError,
    CheckpointVersionError,
    ResumableState,
    build_envelope,
    merge_sidecars,
    normalize_metadata,
    require_payload_keys,
    split_sidecars,
    validate_envelope,
    validate_state_payload,
)

from .conftest import CONCEPTS, FakeState, FakeStateWithSidecar, make_envelope


def test_fakes_satisfy_the_runtime_protocol():
    assert isinstance(FakeState(), ResumableState)
    assert isinstance(FakeStateWithSidecar(), ResumableState)


def test_validate_state_payload_accepts_what_weights_only_can_load():
    validate_state_payload(
        {
            "tensor": torch.zeros(3),
            "counter": 7,
            "ratio": 0.1,
            "flag": True,
            "name": "bottle",
            "nothing": None,
            "nested": {"list": [1, 2.0, "three", torch.ones(1)]},
            "blob": b"\x00\x01",
        },
        state_kind="test",
    )


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"bank": np.zeros(3)}, "numpy"),
        ({"config": object()}, "unsupported value"),
        ({"nested": {"deep": np.float32(1.0)}}, "numpy"),
    ],
)
def test_validate_state_payload_rejects_unloadable_values(payload, expected):
    with pytest.raises(CheckpointPayloadError, match=expected):
        validate_state_payload(payload, state_kind="test")


def test_payload_error_points_at_the_offending_path():
    with pytest.raises(CheckpointPayloadError, match=r"state\.outer\.inner\[1\]"):
        validate_state_payload({"outer": {"inner": [1, np.zeros(2)]}}, state_kind="test")


def test_sidecar_block_is_allowed_but_only_for_arrays():
    validate_state_payload({SIDECAR_KEY: {"bank": np.zeros(3)}}, state_kind="test")
    with pytest.raises(CheckpointPayloadError, match="must be a numpy array"):
        validate_state_payload({SIDECAR_KEY: {"bank": torch.zeros(3)}}, state_kind="test")
    with pytest.raises(CheckpointPayloadError, match="path separators"):
        validate_state_payload({SIDECAR_KEY: {"../escape": np.zeros(3)}}, state_kind="test")


def test_split_and_merge_sidecars_round_trip():
    bank = np.arange(4, dtype=np.float32)
    inline, sidecars = split_sidecars({"weights": torch.zeros(2), SIDECAR_KEY: {"bank": bank}})
    assert SIDECAR_KEY not in inline
    np.testing.assert_array_equal(sidecars["bank"], bank)

    merged = merge_sidecars(inline, sidecars)
    assert set(merged) == {"weights", SIDECAR_KEY}
    np.testing.assert_array_equal(merged[SIDECAR_KEY]["bank"], bank)


def test_require_payload_keys_names_the_missing_key():
    """A lost threshold degrades only F1 — it must never load silently."""
    with pytest.raises(CheckpointPayloadError, match="threshold"):
        require_payload_keys({"tasks": []}, ("tasks", "threshold"), state_kind="ucad")


def test_build_envelope_rejects_an_unknown_phase():
    with pytest.raises(CheckpointEnvelopeError, match="Unknown checkpoint phase"):
        build_envelope(
            state_kind="k",
            state_version=1,
            provenance={},
            run={"concept_index": 0, "phase": "mid_epoch"},
            determinism={},
            fingerprint={},
            state={},
        )


def test_normalize_metadata_coerces_str_subclasses_and_numpy():
    class VersionLike(str):
        pass

    envelope = build_envelope(
        state_kind="k",
        state_version=1,
        provenance={"torch_version": VersionLike("2.11.0")},
        run={"concept_index": 0, "phase": "after_training", "seed": np.int64(7)},
        determinism={"caveats": ("cudnn",)},
        fingerprint={"ratio": np.float32(0.1)},
        state={},
    )
    assert type(envelope["provenance"]["torch_version"]) is str
    assert type(envelope["run"]["seed"]) is int
    assert envelope["determinism"]["caveats"] == ["cudnn"]
    assert type(envelope["fingerprint"]["ratio"]) is float
    assert normalize_metadata(Path("/tmp/x")) == "/tmp/x"


def test_validate_envelope_checks_layout_kind_and_version():
    envelope = make_envelope(index=0, concept_id="bottle", concept_order=CONCEPTS)
    assert validate_envelope(envelope, expected_kind="fake_state", expected_state_version=1) is envelope

    with pytest.raises(CheckpointKindError):
        validate_envelope(envelope, expected_kind="ucad")
    with pytest.raises(CheckpointVersionError):
        validate_envelope(envelope, expected_state_version=2)

    future = dict(envelope)
    future["envelope_version"] = ENVELOPE_VERSION + 1
    with pytest.raises(CheckpointEnvelopeError, match="cannot be read"):
        validate_envelope(future)

    incomplete = {key: value for key, value in envelope.items() if key != "state"}
    with pytest.raises(CheckpointEnvelopeError, match="missing required keys"):
        validate_envelope(incomplete)


def test_run_id_is_content_addressed_not_path_addressed():
    base = dict(model="patchcore", strategy="replaycad", benchmark="mvtec", seed=7, concept_order=CONCEPTS)
    first = compute_run_id(**base)
    assert first == compute_run_id(**base), "the same run must land on the same directory across jobs"
    assert len(first) == 12

    assert compute_run_id(**{**base, "seed": 8}) != first
    assert compute_run_id(**{**base, "concept_order": list(reversed(CONCEPTS))}) != first
    assert compute_run_id(**base, eval_level="pixel") != first


def test_fingerprint_from_config_reads_objects_and_mappings():
    class Config:
        key_size = 128
        coreset_sampling_ratio = 0.01
        artifact_dir = Path("/scratch/artifacts")

    fingerprint = fingerprint_from_config(Config(), ["key_size", "coreset_sampling_ratio", "artifact_dir"])
    assert fingerprint == {
        "key_size": 128,
        "coreset_sampling_ratio": 0.01,
        "artifact_dir": "/scratch/artifacts",
    }
    assert fingerprint_from_config({"seed": 1}, ["seed"]) == {"seed": 1}

    with pytest.raises(CheckpointPayloadError, match="drifted apart"):
        fingerprint_from_config(Config(), ["key_size", "prompt_depth"])


def test_json_safe_flattens_pydantic_and_numpy():
    class Model:
        def model_dump(self, mode="json"):
            return {"resolution": 512, "layers": ("layer2", "layer3")}

    assert json_safe(Model()) == {"layers": ["layer2", "layer3"], "resolution": 512}
    assert json_safe(np.float32(0.5)) == pytest.approx(0.5)
    assert json_safe(np.array([1, 2])) == [1, 2]
