"""Tests for the checkpointable One-for-More.

Everything here runs on the CPU with a 32x32, eight-channel model, because the
property under test is the *state contract*, not the architecture: whether a
process that restores this payload continues the run the interrupted process was
having. Two assertions carry the file. One is that a restored model scores
identically, which pins the weights, the calibrated threshold and the bases. The
other is that training the *next* concept after a restore reproduces an
uninterrupted run bit for bit, which is the only check that can catch a lost
global RNG state, a reset ``_tasks_seen`` or a reset projector update counter -
all three of which corrupt numbers silently rather than raising.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Tuple

import numpy as np
import pytest
import torch

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.models.one_for_more.config import OneForMoreConfig
from pyclad.resume.callback import StateCheckpointCallback
from pyclad.resume.fingerprint import assert_fingerprint_matches, current_provenance
from pyclad.resume.methods.one_for_more import (
    ONE_FOR_MORE_FINGERPRINT_FIELDS,
    ResumableNaiveStrategy,
    ResumableOneForMore,
)
from pyclad.resume.protocol import (
    DETERMINISM_BIT_IDENTICAL,
    CheckpointPayloadError,
    ResumeError,
    build_envelope,
    merge_sidecars,
    validate_envelope,
    validate_state_payload,
)
from pyclad.resume.scenario import ResumableConceptIncrementalScenario, ResumePlan
from pyclad.resume.store import CheckpointStore

SEED = 20260731
CONCEPTS = ("bottle", "cable", "capsule")


def _tiny_config(**updates) -> OneForMoreConfig:
    """A model small enough to train in a test, large enough to have real state.

    ``conditioner_depth=1`` is deliberate: the transformer block brings dropout,
    which draws from the *global* RNG stream. Without it the continuation test
    would pass even if the RNG state were never checkpointed.
    """
    values = {
        "input_size": (32, 32),
        "latent_downsample": 8,
        "base_channels": 8,
        "conditioner_depth": 1,
        "conditioner_heads": 1,
        "conditioner_feedforward": 16,
        "conditioner_dropout": 0.1,
        "diffusion_steps": 10,
        "inference_start_step": 9,
        "inference_steps": 1,
        "epochs": 1,
        "incremental_epochs": 1,
        "batch_size": 2,
        "projection_batches": 1,
        "projection_max_columns": 32,
        "projection_energy_threshold": 0.9,
        "backbone_name": None,
        "score_smoothing_sigma": 0.0,
        "threshold": None,
        "show_training_progress": False,
        "device": "cpu",
    }
    values.update(updates)
    return OneForMoreConfig(**values)


def _images(seed: int, count: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).random((count, 32, 32, 3), dtype=np.float32)


def _fitted(config: OneForMoreConfig, data: np.ndarray, *, seed: int = SEED) -> ResumableOneForMore:
    torch.manual_seed(seed)
    model = ResumableOneForMore(config)
    model.fit(data)
    return model


def _envelope_for(model: ResumableOneForMore, *, index: int, concept_id: str) -> dict[str, Any]:
    return build_envelope(
        state_kind=ResumableOneForMore.STATE_KIND,
        state_version=ResumableOneForMore.STATE_VERSION,
        provenance=current_provenance(),
        run={
            "run_id": "onefor1234ab",
            "seed": SEED,
            "concept_index": index,
            "concept_id": concept_id,
            "concept_order": list(CONCEPTS),
            "phase": "after_training",
        },
        determinism=model.determinism_report(),
        fingerprint=model.state_fingerprint(),
        state=model.save_state(),
    )


def test_payload_is_weights_only_safe_and_round_trips_through_the_store(tmp_path):
    """The payload must survive the store, which always loads weights_only=True."""
    model = _fitted(_tiny_config(), _images(1))
    payload = model.save_state()
    validate_state_payload(payload, state_kind=ResumableOneForMore.STATE_KIND)

    store = CheckpointStore(tmp_path / "ckpt", "onefor1234ab", keep=2, verify_writes=True)
    entry = store.write(_envelope_for(model, index=0, concept_id=CONCEPTS[0]))
    envelope, sidecars = store.load(entry)
    restored_payload = merge_sidecars(envelope["state"], sidecars)

    assert envelope["state_kind"] == "one_for_more"
    assert set(restored_payload) == set(payload)
    # No sidecars: One-for-More's bulk is torch tensors the envelope can hold
    # directly, unlike ReplayCAD's multi-gigabyte numpy memory bank.
    assert sidecars == {}

    validate_envelope(
        envelope,
        expected_kind=ResumableOneForMore.STATE_KIND,
        expected_state_version=ResumableOneForMore.STATE_VERSION,
    )
    # A fingerprint that cannot match itself across a save/load round trip -
    # tuples coming back as lists, say - would abort every resume of a correctly
    # configured run.
    assert_fingerprint_matches(envelope["fingerprint"], ResumableOneForMore(_tiny_config()).state_fingerprint())

    fresh = ResumableOneForMore(_tiny_config())
    fresh.load_state(restored_payload)
    assert fresh._tasks_seen == model._tasks_seen
    assert fresh._threshold == pytest.approx(model._threshold)


def test_restored_model_reproduces_predictions_exactly():
    """Weights, threshold and bases all come back, and the fresh model differs."""
    config = _tiny_config()
    probe = _images(seed=99, count=5)
    model = _fitted(config, _images(1))
    labels, scores = model.predict(probe)
    maps = model.score_maps(probe)

    payload = model.save_state()

    torch.manual_seed(SEED + 1)
    restored = ResumableOneForMore(config)
    _, untrained_scores = restored.predict(probe)
    assert not np.array_equal(untrained_scores, scores), "the probe cannot distinguish two models"

    restored.load_state(payload)
    restored_labels, restored_scores = restored.predict(probe)

    assert np.isfinite(scores).all()
    assert np.array_equal(restored_scores, scores)
    assert np.array_equal(restored_labels, labels)
    assert np.array_equal(restored.score_maps(probe), maps)

    assert restored._tasks_seen == model._tasks_seen == 1
    assert restored._threshold == model._threshold
    assert restored._projector._updates == model._projector._updates
    original_bases = model._projector.bases
    restored_bases = restored._projector.bases
    assert set(restored_bases) == set(original_bases) and original_bases
    for name, basis in original_bases.items():
        assert torch.equal(restored_bases[name], basis)


def test_resumed_training_matches_an_uninterrupted_run():
    """The real prize: concept k+1 after a restart equals concept k+1 without one.

    This is what pins the global RNG state, ``_tasks_seen`` (epoch budget,
    corruption seeds, and whether gradient projection runs at all) and the
    projector's update counter. Every one of those failures leaves a model that
    trains happily and reports plausible numbers.
    """
    config = _tiny_config()
    first, second = _images(1), _images(2)
    probe = _images(seed=99, count=5)

    reference = _fitted(config, first)
    reference.fit(second)
    _, expected_scores = reference.predict(probe)

    interrupted = _fitted(config, first)
    payload = interrupted.save_state()

    # A restarted process rebuilds the model - consuming the global stream on
    # the way - and only then restores the checkpoint.
    resumed = ResumableOneForMore(config)
    resumed.load_state(payload)
    resumed.fit(second)
    _, resumed_scores = resumed.predict(probe)

    assert np.array_equal(resumed_scores, expected_scores)
    assert resumed._tasks_seen == 2

    # Teeth: the same restore with the global stream repositioned elsewhere
    # diverges, so the equality above is evidence about the RNG state and not an
    # artifact of a model too small to care.
    diverged = ResumableOneForMore(config)
    diverged.load_state(payload)
    torch.manual_seed(SEED + 7)
    diverged.fit(second)
    _, diverged_scores = diverged.predict(probe)
    assert not np.array_equal(diverged_scores, expected_scores)


def test_frozen_and_derived_tensors_stay_out_of_the_payload():
    """A checkpoint carries the trainable module and nothing else.

    ``pretrained_backbone=False`` keeps the test offline; the exclusion rule is
    the same one that keeps a 34 MB pretrained resnet out of every concept's
    checkpoint in a real run.
    """
    config = _tiny_config(backbone_name="resnet18", backbone_return_nodes=("layer1",), pretrained_backbone=False)
    model = _fitted(config, _images(1))

    module_state = model.save_state()["module"]

    assert module_state, "the trainable module must be persisted"
    assert all(key.startswith(("conditioner.", "denoiser.")) for key in module_state)
    assert not any(key.startswith("feature_extractor.") for key in module_state)
    assert "alphas_cumulative" not in module_state
    # The derived buffer is recomputed identically by __init__, so leaving it
    # out costs nothing.
    fresh = ResumableOneForMore(config)
    assert torch.equal(fresh.module.alphas_cumulative, model.module.alphas_cumulative)


def test_optimizer_state_is_omitted_on_purpose():
    """Restoring AdamW moments would make the resume diverge, not converge.

    ``fit`` builds a fresh optimizer per concept, so an uninterrupted run resets
    the moments at every boundary too. The null key documents the decision in
    the artifact itself.
    """
    model = _fitted(_tiny_config(), _images(1))
    payload = model.save_state()

    assert "optimizer_state" in payload
    assert payload["optimizer_state"] is None


def test_load_state_validates_before_it_mutates():
    config = _tiny_config()
    model = _fitted(config, _images(1))
    payload = model.save_state()

    target = ResumableOneForMore(config)
    before = {key: tensor.clone() for key, tensor in target.module.state_dict().items()}

    def assert_untouched() -> None:
        assert target._tasks_seen == 0
        assert target._projector.summary().layers == 0
        for key, tensor in target.module.state_dict().items():
            assert torch.equal(tensor, before[key]), f"{key} was mutated by a rejected payload"

    incomplete = {key: value for key, value in payload.items() if key != "projector_updates"}
    with pytest.raises(CheckpointPayloadError, match="projector_updates"):
        target.load_state(incomplete)
    assert_untouched()

    reshaped = dict(payload)
    reshaped["module"] = dict(payload["module"])
    key = next(iter(reshaped["module"]))
    reshaped["module"][key] = torch.zeros(3, 5)
    with pytest.raises(CheckpointPayloadError, match="shape"):
        target.load_state(reshaped)
    assert_untouched()

    renamed = dict(payload)
    renamed["module"] = {f"{name}_v3": tensor for name, tensor in payload["module"].items()}
    with pytest.raises(CheckpointPayloadError, match="architecture"):
        target.load_state(renamed)
    assert_untouched()

    with pytest.raises(CheckpointPayloadError, match="load_basis_only"):
        target.load_state(model.save_variants()["basis_only"])
    assert_untouched()


def test_basis_only_variant_rescues_the_one_unrecoverable_piece_of_state():
    """The bases need every previous concept's data, so they get their own file."""
    config = _tiny_config()
    model = _fitted(config, _images(1))
    variants = model.save_variants()

    assert set(variants) == {"basis_only"}
    basis_payload = variants["basis_only"]
    validate_state_payload(basis_payload, state_kind=ResumableOneForMore.STATE_KIND)
    assert "module" not in basis_payload

    rescued = ResumableOneForMore(config)
    rescued._tasks_seen = model._tasks_seen  # the weights come from elsewhere
    rescued.load_basis_only(basis_payload)

    for name, basis in model._projector.bases.items():
        assert torch.equal(rescued._projector.bases[name], basis)
    assert rescued._projector._updates == model._projector._updates

    # Grafting a basis onto a model at a different point in the concept sequence
    # projects gradients onto the wrong subspace and would report nothing.
    mismatched = ResumableOneForMore(config)
    with pytest.raises(CheckpointPayloadError, match="concepts"):
        mismatched.load_basis_only(basis_payload)


def test_variant_can_be_disabled():
    model = ResumableOneForMore(_tiny_config(), keep_basis_variant=False)
    model.fit(_images(1))
    assert model.save_variants() == {}


def test_determinism_report_is_bit_identical_on_cpu():
    model = ResumableOneForMore(_tiny_config())
    report = model.determinism_report()

    assert report["guarantee"] == DETERMINISM_BIT_IDENTICAL
    assert report["caveats"] == []


def test_determinism_downgrades_when_a_cuda_rng_state_cannot_be_restored():
    """A three-GPU state on a two-GPU host is a different run, not a near miss."""
    config = _tiny_config()
    model = _fitted(config, _images(1))
    payload = model.save_state()
    payload["rng"] = dict(payload["rng"])
    payload["rng"]["device_type"] = "cuda"
    payload["rng"]["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)]

    restored = ResumableOneForMore(config)
    restored.load_state(payload)
    report = restored.determinism_report()

    assert report["guarantee"] == "statistical"
    assert "cuda_rng_state_not_restored" in report["caveats"]
    assert "device_changed_across_resume" in report["caveats"]


def test_fingerprint_covers_the_fields_that_move_numbers():
    model = ResumableOneForMore(_tiny_config())
    fingerprint = model.state_fingerprint()

    assert set(fingerprint) == set(ONE_FOR_MORE_FINGERPRINT_FIELDS)
    for field in ("random_seed", "base_channels", "conditioner_depth", "projection_max_columns", "epochs"):
        assert field in fingerprint
    # Fields that cannot change a number are deliberately absent, so a chained
    # job may move to another node or turn off a progress bar.
    for field in ("device", "show_training_progress", "vae_local_files_only"):
        assert field not in fingerprint

    other = ResumableOneForMore(_tiny_config(base_channels=16, conditioner_heads=2))
    assert other.state_fingerprint() != fingerprint


class _CountingModel:
    """Stand-in for a model the strategy might be holding by mistake."""

    def __init__(self, tasks_seen: int):
        self._tasks_seen = tasks_seen

    def fit(self, data):  # pragma: no cover - never called
        raise AssertionError("the wiring guard should have fired first")

    def predict(self, data):  # pragma: no cover - never called
        raise AssertionError("the wiring guard should have fired first")


def _plan(resume_index: int, *, pending: bool = False) -> ResumePlan:
    return ResumePlan.after_completed(CONCEPTS, resume_index, evaluate_pending_concept=pending)


def test_resumable_naive_strategy_keeps_its_identity_and_reports_the_resume():
    strategy = ResumableNaiveStrategy(_CountingModel(2), resume_info={"enabled": True, "resumed_from_concept": 1})

    # The join key for results aggregation: renaming it would drop resumed runs
    # out of every comparison against the published benchmark.
    assert strategy.name() == "Naive"
    assert strategy.info()["strategy"]["name"] == "Naive"
    assert strategy.additional_info()["resume"]["resumed_from_concept"] == 1

    strategy.set_resume_info({"enabled": True, "resumed_from_concept": 4})
    assert strategy.additional_info()["resume"]["resumed_from_concept"] == 4


def test_resumable_naive_strategy_catches_a_model_that_was_never_restored():
    payload: Mapping[str, Any] = {"tasks_seen": 2}

    ResumableNaiveStrategy(_CountingModel(2)).resume_restore(payload, _plan(2))
    # Pending replay: the concept was trained before the crash and the scenario
    # skips learn(), so the model is one concept ahead of the resume index.
    ResumableNaiveStrategy(_CountingModel(2)).resume_restore(payload, _plan(1, pending=True))

    with pytest.raises(ResumeError, match="different instance"):
        ResumableNaiveStrategy(_CountingModel(0)).resume_restore(payload, _plan(2))

    with pytest.raises(CheckpointPayloadError, match="completed concepts"):
        ResumableNaiveStrategy(_CountingModel(2)).resume_restore(payload, _plan(1))


def test_resumable_naive_strategy_accepts_a_restored_one_for_more():
    config = _tiny_config()
    model = _fitted(config, _images(1))
    payload = model.save_state()

    resumed = ResumableOneForMore(config)
    resumed.load_state(payload)
    strategy = ResumableNaiveStrategy(resumed)

    strategy.resume_restore(payload, _plan(1))
    assert strategy.predict(_images(seed=99, count=5))[1].shape == (5,)


# --------------------------------------------------------------- end-to-end


class _ScoreRecorder(Callback):
    """Records every evaluation sweep so two runs can be compared cell by cell."""

    def __init__(self):
        self.scores: List[Tuple[str, np.ndarray]] = []

    def after_evaluation(self, evaluated_concept, y_true, y_pred, anomaly_scores, *args, **kwargs):
        self.scores.append((evaluated_concept.name, np.asarray(anomaly_scores)))


class _SimulatedCrash(RuntimeError):
    """A wall-clock limit, a preempted node, a full filesystem - pick one."""


class _CrashBeforeConcept(Callback):
    def __init__(self, concept_name: str):
        self._concept_name = concept_name

    def before_concept_processing(self, concept, *args, **kwargs):
        if concept.name == self._concept_name:
            raise _SimulatedCrash(concept.name)


class _CrashAfterEvaluating(Callback):
    """Dies inside an evaluation sweep, which is what leaves a pending checkpoint."""

    def __init__(self, concept_name: str):
        self._concept_name = concept_name

    def after_evaluation(self, evaluated_concept, *args, **kwargs):
        if evaluated_concept.name == self._concept_name:
            raise _SimulatedCrash(evaluated_concept.name)


def _dataset(concepts: Tuple[str, ...]) -> ConceptsDataset:
    train = [Concept(name=name, data=_images(index)) for index, name in enumerate(concepts)]
    test = [
        Concept(name=name, data=_images(100 + index, count=3), labels=np.array([0, 0, 1]))
        for index, name in enumerate(concepts)
    ]
    return ConceptsDataset(name="tiny", train_concepts=train, test_concepts=test)


def _reference_sweeps(config: OneForMoreConfig, concepts: Tuple[str, ...]) -> List[Tuple[str, np.ndarray]]:
    """Every evaluation row of a run that was never interrupted."""
    recorder = _ScoreRecorder()
    torch.manual_seed(SEED)
    strategy = ResumableNaiveStrategy(ResumableOneForMore(config))
    ResumableConceptIncrementalScenario(_dataset(concepts), strategy, [recorder]).run()
    return recorder.scores


def _crashed_run(
    config: OneForMoreConfig,
    concepts: Tuple[str, ...],
    store: CheckpointStore,
    result_json,
    crash: Callback,
) -> None:
    """Run until *crash* fires, leaving whatever the checkpointer had written."""
    result_json.write_text("{}", encoding="utf-8")
    torch.manual_seed(SEED)
    model = ResumableOneForMore(config)
    strategy = ResumableNaiveStrategy(model)
    checkpointer = StateCheckpointCallback(
        store,
        model,
        concept_order=list(concepts),
        seed=SEED,
        result_json_path=result_json,
        determinism=model.determinism_report(),
    )
    with pytest.raises(_SimulatedCrash):
        ResumableConceptIncrementalScenario(_dataset(concepts), strategy, [crash, checkpointer]).run()


def _restart(config: OneForMoreConfig, store: CheckpointStore, entry, plan: ResumePlan):
    """What a new process does: rebuild, validate, restore, hand to the scenario."""
    envelope, sidecars = store.load(entry)
    validate_envelope(
        envelope,
        expected_kind=ResumableOneForMore.STATE_KIND,
        expected_state_version=ResumableOneForMore.STATE_VERSION,
    )
    torch.manual_seed(SEED + 3)  # a fresh process is not sitting where the old one stopped
    model = ResumableOneForMore(config)
    strategy = ResumableNaiveStrategy(model, resume_info={"enabled": True, "resumed_from_concept": entry.index})
    payload = merge_sidecars(envelope["state"], sidecars)
    model.load_state(payload)
    strategy.resume_restore(payload, plan)
    return model, strategy


def test_committed_resume_is_statistically_equivalent_and_says_so(tmp_path):
    """The whole seam at once: scenario, checkpoint callback, store, restore.

    It also settles the assumption the design rests on for this method - that
    nothing between ``after_training`` (where the state is written) and the next
    ``fit()`` consumes the global RNG stream. It does: ``_prepare_batches``
    builds inference loaders with ``generator=None``, and ``DataLoader.__iter__``
    draws a base seed from the global generator for each one. Skipping the
    committed concept's sweep therefore starts the next concept a few draws
    early. The rows stay statistically equivalent, and the run reports that it
    is no longer bit-identical instead of claiming otherwise.
    """
    config = _tiny_config()
    concepts = CONCEPTS[:2]
    reference = _reference_sweeps(config, concepts)

    store = CheckpointStore(tmp_path / "ckpt", "onefor1234ab", keep=2)
    _crashed_run(config, concepts, store, tmp_path / "result.json", _CrashBeforeConcept(concepts[1]))

    committed = store.committed()
    assert committed is not None and committed.index == 0
    assert store.pending() is None
    # The basis-only snapshot is written next to the full checkpoint and is
    # never pruned, because a lost basis cannot be recomputed.
    assert any(entry.get("variant") == "basis_only" for entry in store.read_manifest()["variants"])

    plan = ResumePlan.after_completed(concepts, committed.index + 1)
    model, strategy = _restart(config, store, committed, plan)

    recorder = _ScoreRecorder()
    ResumableConceptIncrementalScenario(_dataset(concepts), strategy, [recorder], plan=plan).run()

    # A skipped concept fires no callbacks at all, so only the second row runs.
    assert [name for name, _ in recorder.scores] == list(concepts)
    expected_rows = reference[len(concepts) :]
    for (name, actual), (expected_name, expected) in zip(recorder.scores, expected_rows):
        assert name == expected_name
        assert np.allclose(actual, expected, rtol=1e-3, atol=1e-3), f"row for {name} is not even close"

    report = model.determinism_report()
    assert report["guarantee"] == "statistical"
    assert "committed_resume_skips_an_evaluation_sweep_that_advances_the_global_rng" in report["caveats"]
    assert strategy.additional_info()["resume"]["resumed_from_concept"] == 0


def test_pending_resume_reproduces_the_scenario_bit_for_bit(tmp_path):
    """Replaying the interrupted sweep also replays its draws, so identity holds.

    This is the same run as the test above, interrupted one step earlier - inside
    concept 0's evaluation sweep rather than after it. The scenario re-runs that
    sweep, which advances the global stream exactly as the dead process did, so
    concept 1 trains from the position an uninterrupted run would have had and
    every recorded row matches exactly. It is the counterexample that proves the
    downgrade above is about the missing sweep and nothing else.
    """
    config = _tiny_config()
    concepts = CONCEPTS[:2]
    reference = _reference_sweeps(config, concepts)

    store = CheckpointStore(tmp_path / "ckpt", "onefor1234ab", keep=2)
    _crashed_run(config, concepts, store, tmp_path / "result.json", _CrashAfterEvaluating(concepts[1]))

    pending = store.pending()
    assert pending is not None and pending.index == 0
    assert store.committed() is None

    plan = ResumePlan.after_completed(concepts, 0, evaluate_pending_concept=True)
    model, strategy = _restart(config, store, pending, plan)

    recorder = _ScoreRecorder()
    ResumableConceptIncrementalScenario(_dataset(concepts), strategy, [recorder], plan=plan).run()

    assert [name for name, _ in recorder.scores] == [name for name, _ in reference]
    for (name, actual), (expected_name, expected) in zip(recorder.scores, reference):
        assert name == expected_name
        assert np.array_equal(actual, expected), f"row for {name} diverged after the pending resume"

    assert model.determinism_report()["guarantee"] == DETERMINISM_BIT_IDENTICAL
