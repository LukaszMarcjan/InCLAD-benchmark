from pathlib import Path

import numpy as np
import pytest
import torch

from pyclad.models.model import Model
from pyclad.strategies.replaycad.artifacts import ReplayCADArtifact
from pyclad.strategies.replaycad.backend import DiffusersReplayCADBackend, ReplayCADBackend
from pyclad.strategies.replaycad.config import ReplayCADConfig
from pyclad.strategies.replaycad.masks import FullFrameMaskGenerator
from pyclad.strategies.replaycad.memory import ReplayCADMemory
from pyclad.strategies.replaycad.strategy import ReplayCADStrategy


class FakeReplayCADBackend(ReplayCADBackend):
    def __init__(self):
        self.compressed = []
        self.generated = []
        self.release_calls = 0

    def compress(self, concept_id, images, masks, artifact: ReplayCADArtifact):
        self.compressed.append((concept_id, len(images), len(masks)))
        artifact.condition_path.write_bytes(f"condition:{concept_id}".encode())
        return {"compression_seconds": 0.01, "fake": True}

    def generate(self, artifact: ReplayCADArtifact, count: int, seed: int):
        self.generated.append((artifact.concept_id, count, seed))
        value = 32 + len(self.generated)
        return np.full((count, 8, 8, 3), value, dtype=np.uint8)

    def release_device_memory(self):
        self.release_calls += 1


class RecordingModel(Model):
    def __init__(self):
        self.fit_batches = []

    def fit(self, data: np.ndarray):
        self.fit_batches.append(np.asarray(data).copy())

    def predict(self, data: np.ndarray):
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=np.float32)

    def name(self) -> str:
        return "RecordingModel"


def replaycad_config(tmp_path: Path, benchmark: str = "mvtec") -> ReplayCADConfig:
    base = ReplayCADConfig.paper_preset(
        benchmark=benchmark,
        artifact_dir=tmp_path / "artifacts",
        mask_backend="full-frame",
        compression_steps=1,
        replay_samples_per_concept=2,
        masks_per_concept=1,
        device="cpu",
    )
    payload = base.model_dump(mode="python")
    payload.update({"generation_batch_size": 1, "compression_batch_size": 1})
    return ReplayCADConfig.model_validate(payload)


def test_paper_presets_have_valid_spatial_token_shapes(tmp_path):
    mvtec = ReplayCADConfig.paper_preset(
        benchmark="mvtec",
        artifact_dir=tmp_path / "mvtec",
        mask_backend="full-frame",
    )
    visa = ReplayCADConfig.paper_preset(
        benchmark="visa",
        artifact_dir=tmp_path / "visa",
        mask_backend="full-frame",
    )

    assert (mvtec.resolution, mvtec.condition_dim, mvtec.spatial_tokens) == (256, 1280, 5)
    assert (visa.resolution, visa.condition_dim, visa.spatial_tokens) == (512, 768, 32)
    assert (mvtec.compression_batch_size, mvtec.generation_batch_size) == (16, 8)
    assert (visa.compression_batch_size, visa.generation_batch_size) == (2, 2)
    assert mvtec.latent_scaling_factor == visa.latent_scaling_factor == 0.18215
    assert mvtec.timestep_sampling == "uniform"
    assert visa.timestep_sampling == "cubic"


@pytest.mark.parametrize(
    ("benchmark", "expected_mask_modes"),
    [("btech", {}), ("dagm", {"Class8": "full-frame"}), ("mpdd", {})],
)
def test_remaining_inclad_benchmarks_use_explicit_ldm_256_adaptation(
    tmp_path, benchmark, expected_mask_modes
):
    config = ReplayCADConfig.paper_preset(
        benchmark=benchmark,
        artifact_dir=tmp_path / benchmark,
        mask_backend="full-frame",
    )

    assert config.benchmark == benchmark
    assert config.preset == "inclad-ldm-256"
    assert config.diffusion_family == "ldm"
    assert config.resolution == 256
    assert config.initializer_word == "object"
    assert config.mask_modes == expected_mask_modes


def test_paper_spatial_projections_produce_expected_condition_shapes(tmp_path):
    for benchmark, latent_shape, expected_tokens in (
        ("mvtec", (2, 4, 32, 32), 5),
        ("visa", (2, 4, 64, 64), 32),
    ):
        config = replaycad_config(tmp_path / benchmark, benchmark)
        backend = DiffusersReplayCADBackend(config)
        backend._torch = torch
        projection = torch.nn.Sequential(
            torch.nn.Linear(config.mask_group_width, config.mask_projection_width),
            torch.nn.ReLU(),
        )

        spatial = backend._spatial_tokens(torch.zeros(latent_shape), projection)

        assert spatial.shape == (2, expected_tokens, config.condition_dim)


def test_strategy_replays_previous_concepts_and_preserves_current_shape(tmp_path):
    backend = FakeReplayCADBackend()
    memory = ReplayCADMemory(
        replaycad_config(tmp_path),
        backend=backend,
        mask_generator=FullFrameMaskGenerator(),
    )
    model = RecordingModel()
    strategy = ReplayCADStrategy(model=model, memory=memory)

    first = np.full((3, 4, 5, 3), 10, dtype=np.uint8)
    second = np.full((4, 4, 5, 3), 20, dtype=np.uint8)
    strategy.learn(first, concept_id="bottle")
    strategy.learn(second, concept_id="cable")

    assert model.fit_batches[0].shape == (3, 4, 5, 3)
    assert model.fit_batches[1].shape == (6, 4, 5, 3)
    assert backend.compressed == [("bottle", 3, 3), ("cable", 4, 4)]
    assert backend.release_calls == 4
    assert [artifact.concept_id for artifact in memory.artifacts] == ["bottle", "cable"]
    assert memory.storage_size_bytes() > 0


def test_existing_complete_artifacts_are_reused_on_resume(tmp_path):
    config = replaycad_config(tmp_path)
    backend = FakeReplayCADBackend()
    memory = ReplayCADMemory(config, backend=backend, mask_generator=FullFrameMaskGenerator())
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    memory.compress("bottle", images)

    resumed_backend = FakeReplayCADBackend()
    resumed = ReplayCADMemory(config, backend=resumed_backend, mask_generator=FullFrameMaskGenerator())
    assert resumed.artifacts == ()
    resumed.compress("bottle", images)

    assert [artifact.concept_id for artifact in resumed.artifacts] == ["bottle"]
    assert resumed_backend.compressed == []


def test_resume_activates_cached_artifacts_in_scenario_order(tmp_path):
    config = replaycad_config(tmp_path)
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    initial = ReplayCADMemory(
        config,
        backend=FakeReplayCADBackend(),
        mask_generator=FullFrameMaskGenerator(),
    )
    initial.compress("bottle", images)
    initial.compress("cable", images)

    resumed_backend = FakeReplayCADBackend()
    resumed = ReplayCADMemory(
        config,
        backend=resumed_backend,
        mask_generator=FullFrameMaskGenerator(),
    )
    model = RecordingModel()
    strategy = ReplayCADStrategy(model=model, memory=resumed)

    strategy.learn(images, concept_id="bottle")
    assert resumed_backend.generated == []
    assert model.fit_batches[0].shape == images.shape

    strategy.learn(images, concept_id="cable")
    assert [item[0] for item in resumed_backend.generated] == ["bottle"]
    assert model.fit_batches[1].shape[0] == len(images) + config.replay_samples_per_concept
    assert resumed_backend.compressed == []


def test_incomplete_artifact_is_completed_and_registered(tmp_path):
    config = replaycad_config(tmp_path)
    incomplete = config.artifact_dir / "concepts" / "bottle"
    incomplete.mkdir(parents=True)
    (incomplete / "masks.npz").write_bytes(b"interrupted")

    backend = FakeReplayCADBackend()
    memory = ReplayCADMemory(config, backend=backend, mask_generator=FullFrameMaskGenerator())
    memory.compress("bottle", np.zeros((2, 4, 4, 3), dtype=np.uint8))

    assert [artifact.concept_id for artifact in memory.artifacts] == ["bottle"]
    assert backend.compressed == [("bottle", 2, 2)]


def test_conditional_storage_excludes_generated_cache(tmp_path):
    base = replaycad_config(tmp_path)
    payload = base.model_dump(mode="python")
    payload["cache_generated"] = True
    config = ReplayCADConfig.model_validate(payload)
    memory = ReplayCADMemory(config, backend=FakeReplayCADBackend(), mask_generator=FullFrameMaskGenerator())
    memory.compress("bottle", np.zeros((2, 4, 4, 3), dtype=np.uint8))

    before_generation = memory.storage_size_bytes()
    memory.generate_previous()

    assert memory.storage_size_bytes() == before_generation
    assert list((config.artifact_dir / "concepts" / "bottle" / "generated").glob("*.npz"))


def test_generated_cache_is_invalidated_when_inference_settings_change(tmp_path):
    base = replaycad_config(tmp_path)
    payload = base.model_dump(mode="python")
    payload["cache_generated"] = True
    config = ReplayCADConfig.model_validate(payload)
    backend = FakeReplayCADBackend()
    memory = ReplayCADMemory(config, backend=backend, mask_generator=FullFrameMaskGenerator())
    memory.compress("bottle", np.zeros((2, 4, 4, 3), dtype=np.uint8))
    memory.generate_previous()

    changed_payload = config.model_dump(mode="python")
    changed_payload["inference_steps"] += 1
    changed_backend = FakeReplayCADBackend()
    changed = ReplayCADMemory(
        ReplayCADConfig.model_validate(changed_payload),
        backend=changed_backend,
        mask_generator=FullFrameMaskGenerator(),
    )
    changed.compress("bottle", np.zeros((2, 4, 4, 3), dtype=np.uint8))
    changed.generate_previous()

    cache_files = list((config.artifact_dir / "concepts" / "bottle" / "generated").glob("*.npz"))
    assert len(backend.generated) == 1
    assert len(changed_backend.generated) == 1
    assert len(cache_files) == 2


def test_artifact_directory_rejects_condition_affecting_config_change(tmp_path):
    config = replaycad_config(tmp_path)
    ReplayCADMemory(config, backend=FakeReplayCADBackend(), mask_generator=FullFrameMaskGenerator())
    payload = config.model_dump(mode="python")
    payload["compression_steps"] = 2

    with pytest.raises(ValueError, match="incompatible configuration"):
        ReplayCADMemory(
            ReplayCADConfig.model_validate(payload),
            backend=FakeReplayCADBackend(),
            mask_generator=FullFrameMaskGenerator(),
        )


def test_concept_slug_collision_is_rejected(tmp_path):
    config = replaycad_config(tmp_path)
    memory = ReplayCADMemory(config, backend=FakeReplayCADBackend(), mask_generator=FullFrameMaskGenerator())
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    memory.compress("metal nut", images)

    with pytest.raises(ValueError, match="same artifact directory"):
        memory.compress("metal-nut", images)


def test_predict_is_concept_agnostic(tmp_path):
    memory = ReplayCADMemory(
        replaycad_config(tmp_path),
        backend=FakeReplayCADBackend(),
        mask_generator=FullFrameMaskGenerator(),
    )
    strategy = ReplayCADStrategy(model=RecordingModel(), memory=memory)
    labels, scores = strategy.predict(np.zeros((2, 4, 4, 3), dtype=np.uint8), concept_id="future")

    assert labels.tolist() == [0, 0]
    assert scores.tolist() == [0.0, 0.0]
