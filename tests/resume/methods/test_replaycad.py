"""Tests for the checkpointable ReplayCAD variant.

The detector half is exercised end to end through the real
:class:`CheckpointStore`, because the property that matters is not "save_state
returns a dict" but "a second process scoring the same images gets the same
numbers". The generative half is exercised against injected fakes: the point of
``ReplayCADMemory``'s backend/mask-generator injection is that the activation
and tokenizer logic can be tested without diffusers, SAM or a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pytest
import torch

from pyclad.models.patchcore.config import PatchCoreConfig
from pyclad.resume.fingerprint import assert_fingerprint_matches, current_provenance
from pyclad.resume.methods.replaycad import (
    CheckpointedPatchCore,
    ResumableReplayCADStrategy,
    replaycad_determinism,
)
from pyclad.resume.plan import apply_resume, build_resume_plan
from pyclad.resume.protocol import (
    SIDECAR_KEY,
    CheckpointCorruptionError,
    CheckpointFingerprintMismatch,
    CheckpointPayloadError,
    ResumeError,
    build_envelope,
    merge_sidecars,
    validate_state_payload,
)
from pyclad.resume.scenario import ResumePlan
from pyclad.resume.store import CheckpointStore
from pyclad.strategies.replaycad.artifacts import artifact_for
from pyclad.strategies.replaycad.backend import ReplayCADBackend
from pyclad.strategies.replaycad.config import ReplayCADConfig
from pyclad.strategies.replaycad.masks import ReplayCADMaskGenerator
from pyclad.strategies.replaycad.memory import ReplayCADMemory

CONCEPTS = ("bottle", "cable", "capsule")


# --------------------------------------------------------------------- helpers


def patchcore_config(**overrides: Any) -> PatchCoreConfig:
    """A PatchCore small enough to fit and score in well under a second."""
    payload: Dict[str, Any] = {
        "backbone_name": "resnet18",
        "pretrained_backbone": False,
        "input_size": (32, 32),
        "batch_size": 4,
        "pretrain_embed_dimension": 64,
        "target_embed_dimension": 64,
        "coreset_sampling_ratio": 0.5,
        "coreset_projection_dimension": 16,
        "coreset_starting_points": 2,
        "random_seed": 7,
        "smoothing_sigma": 0.0,
        "device": "cpu",
    }
    payload.update(overrides)
    return PatchCoreConfig(**payload)


def make_images(count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(count, 32, 32, 3), dtype=np.uint8)


def fitted_model(config: PatchCoreConfig | None = None) -> tuple[CheckpointedPatchCore, np.ndarray]:
    model = CheckpointedPatchCore(config or patchcore_config())
    train = make_images(8, seed=0)
    model.fit(train)
    return model, train


def roundtrip(model: CheckpointedPatchCore, tmp_path: Path) -> Mapping[str, Any]:
    """Write the state through the real store and read it back."""
    store = CheckpointStore(tmp_path / "ckpt", "replaycadrun1", keep=2)
    envelope = build_envelope(
        state_kind=CheckpointedPatchCore.STATE_KIND,
        state_version=CheckpointedPatchCore.STATE_VERSION,
        provenance=current_provenance(),
        run={
            "run_id": store.run_id,
            "seed": 7,
            "concept_index": 0,
            "concept_id": CONCEPTS[0],
            "concept_order": list(CONCEPTS),
            "phase": "after_training",
        },
        determinism=model.determinism(),
        fingerprint=model.state_fingerprint(),
        state=model.save_state(),
    )
    entry = store.write(envelope)
    loaded, sidecars = store.load(entry)
    return merge_sidecars(loaded["state"], sidecars)


def replaycad_config(root: Path, *, cache_generated: bool = True, seed: int = 42) -> ReplayCADConfig:
    base = ReplayCADConfig.paper_preset(
        benchmark="mvtec",
        artifact_dir=root,
        mask_backend="full-frame",
        device="cpu",
        seed=seed,
        compression_steps=2,
        replay_samples_per_concept=4,
        masks_per_concept=1,
    )
    return base.model_copy(update={"cache_generated": cache_generated, "torch_dtype": "float32"})


def write_artifact(root: Path, concept_id: str, index: int) -> Any:
    """Create the on-disk condition an already-compressed concept leaves behind."""
    artifact = artifact_for(root, concept_id)
    artifact.directory.mkdir(parents=True, exist_ok=True)
    artifact.save_masks(np.zeros((1, 8, 8), dtype=np.uint8))
    token_names = [f"<rc-{concept_id}-{position:02d}>" for position in range(3)]
    torch.save(
        {
            "semantic_embeddings": torch.zeros(len(token_names), 4),
            "projection_state": {},
            "token_names": token_names,
            "condition_dim": 1280,
        },
        artifact.condition_path,
    )
    artifact.write_metadata({"schema_version": 1, "concept_id": concept_id, "concept_index": index})
    return artifact


class FakeBackend(ReplayCADBackend):
    """Records the tokenizer calls ``install_prior_tokens`` is supposed to make."""

    def __init__(self) -> None:
        self.loads = 0
        self.installed: list[tuple[tuple[str, ...], bool]] = []
        self.released = 0

    def compress(self, concept_id, images, masks, artifact):  # pragma: no cover - unused here
        raise AssertionError("compress must not run during a resume restore")

    def generate(self, artifact, count, seed):  # pragma: no cover - unused here
        raise AssertionError("generate must not run during a resume restore")

    def release_device_memory(self) -> None:
        self.released += 1

    def _load(self) -> None:
        self.loads += 1

    def _install_tokens(self, token_names: Sequence[str], initialize: bool) -> list[int]:
        assert self.loads > 0, "_install_tokens was called on a cold backend"
        self.installed.append((tuple(token_names), bool(initialize)))
        return list(range(len(token_names)))


class FakeMaskGenerator(ReplayCADMaskGenerator):
    def generate(self, images: np.ndarray, concept_id: str) -> np.ndarray:  # pragma: no cover - unused
        return np.zeros((len(images), 8, 8), dtype=np.uint8)


def build_strategy(
    tmp_path: Path,
    *,
    stored: Sequence[str] = CONCEPTS,
    cache_generated: bool = True,
    validate_caches: bool = True,
    write_caches: bool = True,
) -> tuple[ResumableReplayCADStrategy, ReplayCADMemory, FakeBackend]:
    root = tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    for index, concept_id in enumerate(stored):
        write_artifact(root, concept_id, index)

    config = replaycad_config(root, cache_generated=cache_generated)
    backend = FakeBackend()
    memory = ReplayCADMemory(config, backend=backend, mask_generator=FakeMaskGenerator())
    if cache_generated and write_caches:
        # A compressed concept also leaves its replay cache behind (the inclad-cr profile
        # writes it right after compress()); the adapter refuses to resume without one.
        for concept_id in stored:
            write_replay_cache(memory, concept_id)
    model = CheckpointedPatchCore(patchcore_config())
    strategy = ResumableReplayCADStrategy(model, memory, validate_caches=validate_caches)
    return strategy, memory, backend


def write_replay_cache(memory: ReplayCADMemory, concept_id: str) -> Path:
    """Write a non-constant replay cache where ``validate_generated_caches`` looks for it."""
    artifact = artifact_for(memory.root, concept_id)
    seed = ReplayCADMemory._concept_seed(memory.config.seed, concept_id)
    path = memory._cache_path(artifact, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.random.default_rng(11).integers(0, 255, size=(4, 8, 8, 3), dtype=np.uint8)
    np.savez_compressed(path, images=images)
    return path


def replaycad_block(memory: ReplayCADMemory, activated: Sequence[str], **overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "activated_concepts": list(activated),
        "generation_cache_signature": memory._generation_cache_signature(),
        "artifact_dir": str(memory.root),
        "counters": {"generated_samples": 12, "generation_seconds": 3.5, "compression_seconds": 7.25},
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------- the detector round trip


def test_restored_detector_predicts_identically(tmp_path: Path) -> None:
    """The property the whole layer exists for: same images, same numbers."""
    model, _ = fitted_model()
    test = make_images(6, seed=1)
    expected_labels, expected_scores = model.predict(test)
    expected_maps = model.score_maps(make_images(3, seed=2))

    payload = roundtrip(model, tmp_path)

    restored = CheckpointedPatchCore(patchcore_config())
    restored.load_state(payload)

    actual_labels, actual_scores = restored.predict(test)
    np.testing.assert_array_equal(expected_labels, actual_labels)
    np.testing.assert_array_equal(expected_scores, actual_scores)
    np.testing.assert_array_equal(expected_maps, restored.score_maps(make_images(3, seed=2)))
    assert restored._threshold == pytest.approx(model._threshold, rel=0, abs=0)
    np.testing.assert_array_equal(model._memory_bank, restored._memory_bank)
    assert restored._feature_map_size == model._feature_map_size


def test_state_payload_is_weights_only_safe(tmp_path: Path) -> None:
    model, _ = fitted_model()
    payload = model.save_state()
    validate_state_payload(payload, state_kind=CheckpointedPatchCore.STATE_KIND)
    assert isinstance(payload[SIDECAR_KEY]["memory_bank"], np.ndarray)
    # The bank is the one thing that must never be inlined into the torch file.
    inline = {key: value for key, value in payload.items() if key != SIDECAR_KEY}
    assert not any(isinstance(value, np.ndarray) for value in inline.values())


def test_sidecar_is_written_beside_the_checkpoint(tmp_path: Path) -> None:
    model, _ = fitted_model()
    store = CheckpointStore(tmp_path / "ckpt", "sidecarrun", keep=2)
    envelope = build_envelope(
        state_kind=CheckpointedPatchCore.STATE_KIND,
        state_version=CheckpointedPatchCore.STATE_VERSION,
        provenance=current_provenance(),
        run={
            "run_id": store.run_id,
            "seed": 7,
            "concept_index": 0,
            "concept_id": "bottle",
            "concept_order": list(CONCEPTS),
            "phase": "after_training",
        },
        determinism=model.determinism(),
        fingerprint=model.state_fingerprint(),
        state=model.save_state(),
    )
    entry = store.write(envelope)
    assert [sidecar.name for sidecar in entry.sidecars] == ["memory_bank"]
    assert (store.directory / entry.sidecars[0].filename).suffix == ".npy"


def test_save_state_before_fit_is_refused() -> None:
    model = CheckpointedPatchCore(patchcore_config())
    with pytest.raises(ResumeError, match="before fit"):
        model.save_state()


def test_missing_key_raises_before_mutating(tmp_path: Path) -> None:
    model, _ = fitted_model()
    payload = dict(roundtrip(model, tmp_path))
    payload.pop("threshold")

    restored = CheckpointedPatchCore(patchcore_config())
    with pytest.raises(CheckpointPayloadError, match="threshold"):
        restored.load_state(payload)
    # Validate-before-mutate: a rejected payload leaves an unfitted model
    # unfitted rather than half-restored.
    assert restored._memory_bank is None
    assert restored._feature_map_size is None


def test_bank_width_mismatch_is_refused(tmp_path: Path) -> None:
    model, _ = fitted_model()
    payload = dict(roundtrip(model, tmp_path))
    sidecars = dict(payload[SIDECAR_KEY])
    sidecars["memory_bank"] = np.zeros((4, 32), dtype=np.float32)
    payload[SIDECAR_KEY] = sidecars
    payload["memory_bank_shape"] = [4, 32]

    restored = CheckpointedPatchCore(patchcore_config())
    with pytest.raises(CheckpointPayloadError, match="different feature configuration"):
        restored.load_state(payload)


def test_backbone_signature_mismatch_is_refused(tmp_path: Path) -> None:
    """A silently different feature extractor is the worst failure available.

    The bank is a set of coordinates in the extractor's embedding space, so a
    swapped backbone produces a complete, plausible, meaningless metric matrix.
    """
    model, _ = fitted_model()
    payload = dict(roundtrip(model, tmp_path))
    # Simulate the pretrained path, where the weights are not carried and the
    # signature is the only line of defence.
    payload["backbone_state"] = None

    restored = CheckpointedPatchCore(patchcore_config())
    with torch.no_grad():
        for parameter in restored.module.parameters():
            parameter.add_(1.0)

    with pytest.raises(CheckpointPayloadError, match="different feature extractor"):
        restored.load_state(payload)


def test_backbone_weights_travel_when_not_pretrained(tmp_path: Path) -> None:
    """Without this a resumed run scores against a freshly randomised backbone."""
    model, _ = fitted_model()
    payload = roundtrip(model, tmp_path)
    assert payload["backbone_state"] is not None

    restored = CheckpointedPatchCore(patchcore_config())
    before = _first_weight(restored)
    assert not torch.equal(before, _first_weight(model))
    restored.load_state(payload)
    assert torch.equal(_first_weight(restored), _first_weight(model))


def _first_weight(model: CheckpointedPatchCore) -> torch.Tensor:
    return next(iter(model.module.state_dict().values())).detach().clone()


def test_prediction_cache_is_dropped_on_restore(tmp_path: Path) -> None:
    model, train = fitted_model()
    model.predict(train)
    assert model._last_prediction_maps is not None

    payload = roundtrip(model, tmp_path)
    model.load_state(payload)
    assert model._last_prediction_data is None
    assert model._last_prediction_maps is None


# ------------------------------------------------------------------ fingerprint


def test_fingerprint_covers_both_halves(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    model = CheckpointedPatchCore(patchcore_config(), replaycad_config=replaycad_config(root))
    fingerprint = model.state_fingerprint()
    assert fingerprint["patchcore.coreset_sampling_ratio"] == 0.5
    assert fingerprint["replaycad.model_id"] == "CompVis/ldm-text2im-large-256"
    assert fingerprint["replaycad.cache_generated"] is True


def test_changed_coreset_ratio_aborts_the_resume(tmp_path: Path) -> None:
    model, _ = fitted_model()
    other = CheckpointedPatchCore(patchcore_config(coreset_sampling_ratio=0.25))
    with pytest.raises(CheckpointFingerprintMismatch, match="coreset_sampling_ratio"):
        assert_fingerprint_matches(model.state_fingerprint(), other.state_fingerprint())


def test_determinism_is_reported_honestly(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    ldm = replaycad_determinism(replaycad_config(root))
    assert ldm["guarantee"] == "bit_identical"

    visa = replaycad_config(root).model_copy(update={"diffusion_family": "stable-diffusion"})
    sd = replaycad_determinism(visa)
    assert sd["guarantee"] == "statistical"
    assert "vae_sample_unseeded" in sd["caveats"]

    cold = replaycad_determinism(replaycad_config(root, cache_generated=False))
    assert cold["guarantee"] == "statistical"


# ------------------------------------------------------------- strategy restore


def test_resume_restore_reactivates_replay_in_order(tmp_path: Path) -> None:
    """Without this, ``generate_previous()`` returns nothing after a restart."""
    strategy, memory, backend = build_strategy(tmp_path)
    assert memory.artifacts == ()

    plan = ResumePlan.after_completed(CONCEPTS, 2)
    strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:2])}, plan)

    assert [artifact.concept_id for artifact in memory.artifacts] == ["bottle", "cable"]
    assert memory._generated_samples == 12
    assert memory._compression_seconds == pytest.approx(7.25)
    assert backend.released >= 1


def test_resume_restore_reinstalls_prior_tokens(tmp_path: Path) -> None:
    """The tokenizer size decides how many draws ``compress()`` consumes."""
    strategy, memory, backend = build_strategy(tmp_path)
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:2])}, plan)

    assert backend.loads >= 1
    assert [names for names, _ in backend.installed] == [
        tuple(f"<rc-bottle-{index:02d}>" for index in range(3)),
        tuple(f"<rc-cable-{index:02d}>" for index in range(3)),
    ]
    # initialize=False: the stored embeddings must not be overwritten with the
    # initializer word.
    assert all(initialize is False for _, initialize in backend.installed)


def test_pending_phase_activates_the_trained_concept(tmp_path: Path) -> None:
    strategy, memory, _ = build_strategy(tmp_path)
    plan = ResumePlan.after_completed(CONCEPTS, 1, evaluate_pending_concept=True)
    strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:2])}, plan)
    assert [artifact.concept_id for artifact in memory.artifacts] == ["bottle", "cable"]


def test_replay_set_disagreeing_with_the_plan_aborts(tmp_path: Path) -> None:
    strategy, memory, _ = build_strategy(tmp_path)
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    with pytest.raises(ResumeError, match="disagrees with the resume plan"):
        strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:1])}, plan)


def test_missing_artifact_aborts_rather_than_replaying_less(tmp_path: Path) -> None:
    strategy, memory, _ = build_strategy(tmp_path, stored=("bottle",))
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    with pytest.raises(CheckpointCorruptionError, match="cable"):
        strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:2])}, plan)


def test_missing_replaycad_block_aborts(tmp_path: Path) -> None:
    strategy, _, _ = build_strategy(tmp_path)
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    with pytest.raises(CheckpointPayloadError, match="no 'replaycad' block"):
        strategy.resume_restore({}, plan)


def test_resume_without_generation_cache_aborts(tmp_path: Path) -> None:
    strategy, memory, _ = build_strategy(tmp_path, cache_generated=False)
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    with pytest.raises(ResumeError, match="cache-generated"):
        strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:2])}, plan)


def test_degenerate_replay_cache_aborts(tmp_path: Path) -> None:
    """Verified real: every observed VisA cache is 800x512x512x3 of zeros."""
    strategy, memory, _ = build_strategy(tmp_path)
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    artifact = artifact_for(memory.root, "bottle")
    seed = ReplayCADMemory._concept_seed(memory.config.seed, "bottle")
    path = memory._cache_path(artifact, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=np.zeros((4, 8, 8, 3), dtype=np.uint8))

    with pytest.raises(CheckpointCorruptionError, match="constant image batch"):
        strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:2])}, plan)


def test_healthy_replay_cache_is_accepted(tmp_path: Path) -> None:
    strategy, memory, _ = build_strategy(tmp_path)
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    for concept_id in CONCEPTS[:2]:
        artifact = artifact_for(memory.root, concept_id)
        seed = ReplayCADMemory._concept_seed(memory.config.seed, concept_id)
        path = memory._cache_path(artifact, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        images = np.random.default_rng(3).integers(0, 255, size=(4, 8, 8, 3), dtype=np.uint8)
        np.savez_compressed(path, images=images)

    strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:2])}, plan)
    assert [artifact.concept_id for artifact in memory.artifacts] == ["bottle", "cable"]


def test_missing_replay_caches_abort(tmp_path: Path) -> None:
    strategy, memory, _ = build_strategy(tmp_path, write_caches=False)
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    with pytest.raises(ResumeError, match="replay caches could be located"):
        strategy.resume_restore({"replaycad": replaycad_block(memory, CONCEPTS[:2])}, plan)


def test_cache_signature_drift_is_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, memory, _ = build_strategy(tmp_path)
    plan = ResumePlan.after_completed(CONCEPTS, 2)
    block = replaycad_block(memory, CONCEPTS[:2], generation_cache_signature="deadbeefcafe")
    with caplog.at_level("WARNING"):
        strategy.resume_restore({"replaycad": block}, plan)
    assert "generation cache signature changed" in caplog.text


def test_strategy_name_is_unchanged(tmp_path: Path) -> None:
    """The name is the join key in the aggregation scripts, not a version tag."""
    strategy, _, _ = build_strategy(tmp_path)
    assert strategy.name() == "ReplayCAD"


def test_resume_info_surfaces_in_additional_info(tmp_path: Path) -> None:
    strategy, _, _ = build_strategy(tmp_path)
    assert "resume" not in strategy.additional_info()
    strategy.set_resume_info({"enabled": True, "resumed_from_concept": 6})
    info = strategy.additional_info()
    assert info["resume"]["resumed_from_concept"] == 6
    assert info["model"] == "PatchCore"


def test_apply_resume_drives_both_halves(tmp_path: Path) -> None:
    """The wiring the runner will use, exercised through the real plan driver.

    ``apply_resume`` finds ``resume_restore`` by duck typing, so a renamed hook
    would leave the memory un-activated with nothing raising anywhere.
    """
    strategy, _, _ = build_strategy(tmp_path)
    model = strategy._model
    strategy.activate_completed(CONCEPTS[:2])
    model.fit(make_images(8, seed=5))
    test = make_images(4, seed=6)
    expected_labels, expected_scores = model.predict(test)

    store = CheckpointStore(tmp_path / "ckpt", "endtoend01", keep=2)
    envelope = build_envelope(
        state_kind=CheckpointedPatchCore.STATE_KIND,
        state_version=CheckpointedPatchCore.STATE_VERSION,
        provenance=current_provenance(),
        run={
            "run_id": store.run_id,
            "seed": 7,
            "concept_index": 1,
            "concept_id": "cable",
            "concept_order": list(CONCEPTS),
            "phase": "after_training",
        },
        determinism=model.determinism(),
        fingerprint=model.state_fingerprint(),
        state=model.save_state(),
    )
    store.write(envelope)
    result_json = tmp_path / "result.json"
    result_json.write_text("{}", encoding="utf-8")
    store.commit(index=1, concept_id="cable", result_json=result_json)

    # Same artifact directory on purpose: a content-addressed checkpoint dir is
    # only useful if the restarted job points at the artifacts it already owns.
    fresh_strategy, fresh_memory, _ = build_strategy(tmp_path)
    outcome = build_resume_plan(
        store,
        concept_order=CONCEPTS,
        mode="require",
        expected_kind=CheckpointedPatchCore.STATE_KIND,
        expected_state_version=CheckpointedPatchCore.STATE_VERSION,
        expected_fingerprint=fresh_strategy._model.state_fingerprint(),
    )
    assert outcome.plan.resume_index == 2

    apply_resume(outcome, state=fresh_strategy._model, strategy=fresh_strategy)

    assert [artifact.concept_id for artifact in fresh_memory.artifacts] == ["bottle", "cable"]
    actual_labels, actual_scores = fresh_strategy.predict(test, concept_id="capsule")
    np.testing.assert_array_equal(expected_labels, actual_labels)
    np.testing.assert_array_equal(expected_scores, actual_scores)


def test_state_carries_the_memory_activation(tmp_path: Path) -> None:
    """The two halves of ReplayCAD's state have to live in one envelope."""
    strategy, memory, _ = build_strategy(tmp_path)
    strategy.activate_completed(CONCEPTS[:2])
    model = strategy._model
    model.fit(make_images(8, seed=4))

    payload = model.save_state()
    assert payload["replaycad"]["activated_concepts"] == ["bottle", "cable"]
    assert payload["replaycad"]["artifact_dir"] == str(memory.root)
