"""Round-trip tests for the checkpointable UCAD.

The property that matters is not "the payload has the right keys" but "a run
that was interrupted after concept k produces the same numbers as one that was
never interrupted". UCAD can be held to exactly that, bit for bit, because every
seed it uses is re-derived from the length of the restored task memory rather
than carried across the restart — so these tests assert equality, not closeness.

Everything runs on a tiny stand-in ViT (the same one
``tests/models/test_ucad.py`` uses) so the whole file is a few seconds of CPU.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest
import torch
from torch import nn

from pyclad.data.concept import Concept
from pyclad.models.ucad.config import UCADConfig
from pyclad.models.ucad.ucad import UCAD
from pyclad.resume.callback import StateCheckpointCallback
from pyclad.resume.fingerprint import current_provenance
from pyclad.resume.methods.ucad import (
    UCAD_FINGERPRINT_FIELDS,
    ResumableUCADStrategy,
    ResumableUCAD,
    ucad_determinism,
)
from pyclad.resume.plan import apply_resume, build_resume_plan
from pyclad.resume.protocol import (
    DETERMINISM_BIT_IDENTICAL,
    DETERMINISM_STATISTICAL,
    CheckpointPayloadError,
    build_envelope,
    validate_state_payload,
)
from pyclad.resume.scenario import ResumePlan
from pyclad.resume.store import CheckpointStore
from pyclad.strategies.vision.ucad import UCADStrategy

CONCEPTS = ("dark", "bright", "mixed")


# ---------------------------------------------------------------- tiny backbone


class _TinyPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.proj(self.norm(tokens))


class _TinyViT(nn.Module):
    """Deterministic across instances: a fresh one has identical weights.

    That is what makes "construct a fresh model and load the checkpoint into it"
    a fair comparison here — in a real run the same guarantee comes from the
    frozen pretrained backbone, which is why no backbone weights are persisted.
    """

    embed_dim = 8
    num_prefix_tokens = 1

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.patch_embed = _TinyPatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])

    def _pos_embed(self, patches: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.expand(len(patches), -1, -1)
        return torch.cat([cls, patches], dim=1)


@pytest.fixture
def tiny_backbone(monkeypatch):
    monkeypatch.setattr("pyclad.models.ucad.ucad._build_vit_backbone", lambda config: _TinyViT())


def _config(**updates) -> UCADConfig:
    values: dict[str, Any] = {
        "backbone_name": "tiny-test-vit",
        "pretrained_backbone": False,
        "input_size": (8, 8),
        "batch_size": 2,
        "feature_layer": 2,
        "prompt_depth": 2,
        "prompt_length": 1,
        # 'uniform' rather than 'zero' on purpose: the initial prompt is drawn
        # from a generator seeded with random_seed + len(task_memories), so a
        # resume that lost the memory length would produce a different prompt
        # for the next concept and the continuation test below would catch it.
        "prompt_init": "uniform",
        "target_embed_dimension": 8,
        "key_size": 4,
        "knowledge_size": 4,
        "coreset_projection_dimension": 4,
        "coreset_starting_points": 2,
        "epochs": 0,
        "structure_mode": "none",
        "smoothing_sigma": 0.0,
        "device": "cpu",
        "random_seed": 5,
    }
    values.update(updates)
    return UCADConfig(**values)


def _concept_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    return {
        "dark": rng.integers(0, 80, size=(4, 8, 8, 3), dtype=np.uint8),
        "bright": rng.integers(175, 255, size=(4, 8, 8, 3), dtype=np.uint8),
        "mixed": rng.integers(60, 200, size=(4, 8, 8, 3), dtype=np.uint8),
    }


def _probe(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([data["dark"][:2], data["bright"][:2], data["mixed"][:2]])


def _train(strategy: UCADStrategy, data: Mapping[str, np.ndarray], concepts) -> None:
    for concept in concepts:
        strategy.learn(data[concept], concept_id=concept)


# --------------------------------------------------------------------- payload


def test_save_state_payload_is_loadable_by_weights_only(tiny_backbone):
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:2])

    payload = model.save_state()

    # The write-time guard that keeps a checkpoint from becoming unreadable by
    # the very layer that wrote it.
    validate_state_payload(payload, state_kind=ResumableUCAD.STATE_KIND)
    assert payload["tasks_seen"] == 2
    assert [task["concept_id"] for task in payload["tasks"]] == ["dark", "bright"]
    assert payload["feature_map_size"] == [2, 2]
    assert payload["optimizer_state"] is None
    assert all(
        isinstance(task[name], torch.Tensor) for task in payload["tasks"] for name in ("key", "prompt", "knowledge")
    )


def test_state_is_not_aliased_by_the_payload(tiny_backbone):
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:1])

    payload = model.save_state()
    model._task_memories[0].key[:] = 999.0
    model._task_memories[0].prompt.fill_(999.0)

    assert not torch.any(payload["tasks"][0]["key"] == 999.0)
    assert not torch.any(payload["tasks"][0]["prompt"] == 999.0)


def test_fingerprint_covers_every_field_and_reacts_to_a_change(tiny_backbone):
    baseline = ResumableUCAD(_config()).state_fingerprint()
    changed = ResumableUCAD(_config(key_size=6)).state_fingerprint()

    assert set(baseline) == set(UCAD_FINGERPRINT_FIELDS)
    assert baseline["key_size"] != changed["key_size"]
    # json_safe flattens tuples, so a fingerprint compares equal across the
    # JSON round trip a manifest puts it through.
    assert baseline["input_size"] == [8, 8]


# ------------------------------------------------------------------- round trip


def test_restored_model_predicts_identically(tiny_backbone, tmp_path):
    """Save after one concept, rebuild from disk, assert bit-identical output.

    The checkpoint goes through the real :class:`CheckpointStore` rather than a
    plain dict, so this also covers ``torch.load(weights_only=True)`` accepting
    the payload — the failure mode that is otherwise only discovered by the job
    that tries to resume.
    """
    data = _concept_data()
    probe = _probe(data)

    original = ResumableUCAD(_config())
    original_strategy = ResumableUCADStrategy(original)
    _train(original_strategy, data, CONCEPTS[:2])
    expected_labels, expected_scores = original_strategy.predict(probe.copy())
    expected_maps = original.score_maps(probe.copy())
    expected_tasks = original.selected_task_indices(probe.copy())

    store = CheckpointStore(tmp_path / "ckpt", "ucad-run", keep=0)
    envelope = build_envelope(
        state_kind=ResumableUCAD.STATE_KIND,
        state_version=ResumableUCAD.STATE_VERSION,
        provenance=current_provenance(),
        run={
            "concept_index": 1,
            "concept_id": "bright",
            "concept_order": list(CONCEPTS),
            "phase": "after_training",
        },
        determinism=original.determinism(),
        fingerprint=original.state_fingerprint(),
        state=original.save_state(),
    )
    entry = store.write(envelope)
    loaded_envelope, sidecars = store.load(entry)
    assert sidecars == {}

    restored = ResumableUCAD(_config())
    restored_strategy = ResumableUCADStrategy(restored)
    restored.load_state(loaded_envelope["state"])
    restored_strategy.resume_restore(
        loaded_envelope["state"],
        ResumePlan.after_completed(CONCEPTS, 2),
    )

    labels, scores = restored_strategy.predict(probe.copy())
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(scores, expected_scores)
    np.testing.assert_array_equal(restored.score_maps(probe.copy()), expected_maps)
    np.testing.assert_array_equal(restored.selected_task_indices(probe.copy()), expected_tasks)
    assert restored._threshold == original._threshold
    assert restored._feature_map_size == original._feature_map_size
    assert restored_strategy._tasks_seen == 2


def test_resumed_run_matches_an_uninterrupted_one(tiny_backbone):
    """The property the whole layer exists for, on the concept *after* the resume.

    Training a third concept from restored state has to reproduce the
    uninterrupted run exactly. It only can because the prompt generator and both
    coreset samplers are seeded from ``random_seed + len(task_memories)``, so
    restoring the memory list restores the RNG.
    """
    data = _concept_data()
    probe = _probe(data)

    uninterrupted = ResumableUCAD(_config())
    uninterrupted_strategy = ResumableUCADStrategy(uninterrupted)
    _train(uninterrupted_strategy, data, CONCEPTS)
    expected_labels, expected_scores = uninterrupted_strategy.predict(probe.copy())

    interrupted = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(interrupted), data, CONCEPTS[:2])
    payload = interrupted.save_state()

    resumed = ResumableUCAD(_config())
    resumed_strategy = ResumableUCADStrategy(resumed)
    resumed.load_state(payload)
    resumed_strategy.resume_restore(payload, ResumePlan.after_completed(CONCEPTS, 2))
    resumed_strategy.learn(data["mixed"], concept_id="mixed")

    labels, scores = resumed_strategy.predict(probe.copy())
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(scores, expected_scores)
    assert [memory.concept_id for memory in resumed._task_memories] == list(CONCEPTS)
    assert torch.equal(resumed._task_memories[2].prompt, uninterrupted._task_memories[2].prompt)


def test_feature_map_size_round_trips_as_a_tuple(tiny_backbone):
    """A list here would survive load and then break the *next* fit_task.

    ``fit_task`` compares ``self._feature_map_size != grid_size`` against a
    freshly inferred tuple, so a restored list makes every subsequent concept
    raise "All UCAD tasks must share the same feature grid".
    """
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:1])

    restored = ResumableUCAD(_config())
    restored.load_state(model.save_state())

    assert isinstance(restored._feature_map_size, tuple)
    restored.fit_task(data["bright"], concept_id="bright")  # would raise if it were a list


# ------------------------------------------------------------ the strategy hook


def test_without_the_strategy_hook_the_next_concept_wipes_the_memory(tiny_backbone):
    """Why ``resume_restore`` exists, demonstrated rather than asserted in prose.

    A perfectly restored model plus a strategy counter left at zero silently
    calls ``begin_continual_run()``, which clears the task memories, the feature
    map size and the threshold — no error, no log line, just different numbers.
    """
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:2])
    payload = model.save_state()

    forgetful = ResumableUCAD(_config())
    forgetful.load_state(payload)
    ResumableUCADStrategy(forgetful).learn(data["mixed"], concept_id="mixed")  # _tasks_seen still 0

    careful = ResumableUCAD(_config())
    careful_strategy = ResumableUCADStrategy(careful)
    careful.load_state(payload)
    careful_strategy.resume_restore(payload, ResumePlan.after_completed(CONCEPTS, 2))
    careful_strategy.learn(data["mixed"], concept_id="mixed")

    assert [memory.concept_id for memory in forgetful._task_memories] == ["mixed"]
    assert [memory.concept_id for memory in careful._task_memories] == list(CONCEPTS)


def test_resume_restore_rejects_a_plan_that_disagrees_with_the_state(tiny_backbone):
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:2])
    payload = model.save_state()

    restored = ResumableUCAD(_config())
    strategy = ResumableUCADStrategy(restored)
    restored.load_state(payload)

    with pytest.raises(CheckpointPayloadError, match="resume plan expects 1"):
        strategy.resume_restore(payload, ResumePlan.after_completed(CONCEPTS, 1))
    assert strategy._tasks_seen == 0


def test_resume_restore_accepts_a_pending_plan_one_ahead(tiny_backbone):
    """A pending checkpoint is written after training, so it is one task ahead."""
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:2])
    payload = model.save_state()

    restored = ResumableUCAD(_config())
    strategy = ResumableUCADStrategy(restored)
    restored.load_state(payload)
    strategy.resume_restore(
        payload,
        ResumePlan.after_completed(CONCEPTS, 1, evaluate_pending_concept=True),
    )

    assert strategy._tasks_seen == 2


def test_strategy_keeps_its_published_name_and_records_the_resume(tiny_backbone):
    model = ResumableUCAD(_config())
    strategy = ResumableUCADStrategy(model)

    assert strategy.name() == "UCAD"
    assert model.name() == "UCAD"
    info = strategy.additional_info()
    assert info["continual_strategy"] == "key_prompt_knowledge"
    assert info["resume"]["state_kind"] == "ucad"
    assert info["resume"]["resumed"] is False

    strategy.set_resume_info({"enabled": True, "resumed": True, "resumed_from_concept": 6})
    assert strategy.additional_info()["resume"]["resumed_from_concept"] == 6


def test_strategy_refuses_a_non_checkpointable_model(tiny_backbone):
    with pytest.raises(TypeError, match="ResumableUCAD"):
        ResumableUCADStrategy(UCAD(_config()))


# ----------------------------------------------------------------- bad payloads


@pytest.mark.parametrize("missing", ["tasks", "tasks_seen", "feature_map_size", "threshold"])
def test_load_state_names_the_missing_key_and_changes_nothing(tiny_backbone, missing):
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:1])
    payload = model.save_state()
    payload.pop(missing)

    fresh = ResumableUCAD(_config())
    with pytest.raises(CheckpointPayloadError, match=missing):
        fresh.load_state(payload)

    assert fresh._task_memories == []
    assert fresh._feature_map_size is None


def test_load_state_rejects_a_prompt_from_a_different_backbone(tiny_backbone):
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:1])
    payload = model.save_state()
    payload["tasks"][0]["prompt"] = torch.zeros(4, 2, 1, 1, 8)

    fresh = ResumableUCAD(_config())
    with pytest.raises(CheckpointPayloadError, match="prompt"):
        fresh.load_state(payload)
    assert fresh._task_memories == []


def test_load_state_rejects_features_of_the_wrong_width(tiny_backbone):
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:1])
    payload = model.save_state()
    payload["tasks"][0]["knowledge"] = torch.zeros(4, 16)

    fresh = ResumableUCAD(_config())
    with pytest.raises(CheckpointPayloadError, match="target_embed_dimension"):
        fresh.load_state(payload)


def test_load_state_rejects_tasks_given_as_a_mapping(tiny_backbone):
    """Order is load-bearing twice: seed offsets and the argmin tie-break."""
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:2])
    payload = model.save_state()
    payload["tasks"] = {task["concept_id"]: task for task in payload["tasks"]}

    with pytest.raises(CheckpointPayloadError, match="ordered sequence"):
        ResumableUCAD(_config()).load_state(payload)


def test_load_state_rejects_a_task_count_that_disagrees_with_the_list(tiny_backbone):
    data = _concept_data()
    model = ResumableUCAD(_config())
    _train(ResumableUCADStrategy(model), data, CONCEPTS[:2])
    payload = model.save_state()
    payload["tasks_seen"] = 3

    with pytest.raises(CheckpointPayloadError, match="tasks_seen=3"):
        ResumableUCAD(_config()).load_state(payload)


# ------------------------------------------------------------------ determinism


def test_determinism_is_reported_honestly():
    block = ucad_determinism(_config())

    assert block["guarantee"] == DETERMINISM_BIT_IDENTICAL
    # Present even in the bit-identical case: an uninterrupted run on a GPU has
    # the same caveat, so recording it keeps the claim truthful rather than
    # implying the resume introduced it.
    assert "cudnn_nondeterminism" in block["caveats"]


def test_sam_structure_mode_downgrades_the_guarantee(tiny_backbone, tmp_path):
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    config = _config(structure_mode="sam", sam_checkpoint=str(checkpoint))

    block = ucad_determinism(config)

    assert block["guarantee"] == DETERMINISM_STATISTICAL
    assert "sam_masks_unseeded" in block["caveats"]


# ------------------------------------------------------------ callback wiring


def test_checkpoint_callback_writes_and_commits_a_concept(tiny_backbone, tmp_path):
    """The callback path the runner will actually use, end to end."""
    data = _concept_data()
    model = ResumableUCAD(_config())
    strategy = ResumableUCADStrategy(model)
    store = CheckpointStore(tmp_path / "ckpt", "ucad-callback", keep=0)
    callback = StateCheckpointCallback(
        store,
        model,
        concept_order=CONCEPTS,
        seed=5,
        determinism=model.determinism(),
    )

    for index, name in enumerate(CONCEPTS[:2]):
        concept = Concept(name=name, data=data[name])
        callback.before_concept_processing(concept=concept)
        callback.before_training()
        strategy.learn(data[name], concept_id=name)
        callback.after_training(learned_concept=concept)
        assert store.pending() is not None and store.pending().index == index
        callback.after_concept_processing(concept=concept)

    committed = store.committed()
    assert committed is not None
    assert (committed.index, committed.concept_id) == (1, "bright")
    assert store.pending() is None

    envelope, _ = store.load(committed)
    assert envelope["state_kind"] == "ucad"
    assert envelope["determinism"]["guarantee"] == DETERMINISM_BIT_IDENTICAL
    assert envelope["run"]["concept_order"] == list(CONCEPTS)
    assert envelope["state"]["tasks_seen"] == 2


def test_plan_driver_restores_ucad_and_the_run_continues_identically(tiny_backbone, tmp_path):
    """The seam between these classes and ``pyclad.resume.plan``.

    ``apply_resume`` discovers ``resume_restore`` by duck typing, so a renamed or
    mis-signed hook would fail silently — the model would come back perfectly and
    the counter would stay at zero. This drives the real driver to prove it fires.
    """
    data = _concept_data()
    probe = _probe(data)

    uninterrupted = ResumableUCAD(_config())
    uninterrupted_strategy = ResumableUCADStrategy(uninterrupted)
    _train(uninterrupted_strategy, data, CONCEPTS)
    expected_labels, expected_scores = uninterrupted_strategy.predict(probe.copy())

    result_json = tmp_path / "result.json"
    result_json.write_text("{}", encoding="utf-8")

    interrupted = ResumableUCAD(_config())
    interrupted_strategy = ResumableUCADStrategy(interrupted)
    store = CheckpointStore(tmp_path / "ckpt", "ucad-plan", keep=0)
    callback = StateCheckpointCallback(
        store,
        interrupted,
        concept_order=CONCEPTS,
        seed=5,
        result_json_path=result_json,
        determinism=interrupted.determinism(),
    )
    for name in CONCEPTS[:2]:
        concept = Concept(name=name, data=data[name])
        callback.before_concept_processing(concept=concept)
        callback.before_training()
        interrupted_strategy.learn(data[name], concept_id=name)
        callback.after_training(learned_concept=concept)
        callback.after_concept_processing(concept=concept)

    resumed = ResumableUCAD(_config())
    resumed_strategy = ResumableUCADStrategy(resumed)
    outcome = build_resume_plan(
        CheckpointStore(tmp_path / "ckpt", "ucad-plan", keep=0),
        concept_order=CONCEPTS,
        mode="require",
        expected_kind=ResumableUCAD.STATE_KIND,
        expected_state_version=ResumableUCAD.STATE_VERSION,
        expected_fingerprint=resumed.state_fingerprint(),
    )
    report = apply_resume(outcome, state=resumed, strategy=resumed_strategy)

    assert outcome.plan.resume_index == 2
    assert outcome.plan.skip_concepts == CONCEPTS[:2]
    assert any("resume_restore" in item for item in report.restored)
    assert resumed_strategy._tasks_seen == 2
    assert resumed_strategy.additional_info()["resume"]["restored_tasks"] == 2

    resumed_strategy.learn(data["mixed"], concept_id="mixed")
    labels, scores = resumed_strategy.predict(probe.copy())
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(scores, expected_scores)
