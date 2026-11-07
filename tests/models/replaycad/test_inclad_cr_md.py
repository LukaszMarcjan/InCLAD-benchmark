import argparse
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.neighbors import NearestNeighbors

from pyclad.models.patchcore import gpu_knn
from pyclad.models.patchcore.gpu_knn import TorchExactNearestNeighbors
from pyclad.models.patchcore.patchcore import PatchCore
from pyclad.strategies.replaycad import inclad_cr
from pyclad.strategies.replaycad.artifacts import ReplayCADArtifact
from pyclad.strategies.replaycad.backend import ReplayCADBackend
from pyclad.strategies.replaycad.inclad_cr import (
    GREY_METADATA_KEY,
    PROFILE_PRESET,
    InCLADCRReplayCADConfig,
    InCLADCRReplayCADMemory,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESUMABLE_WRAPPER_PATH = REPO_ROOT / "examples" / "clvad" / "run_replaycad_cr_resumable.py"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("run_replaycad_cr_resumable_under_test", RESUMABLE_WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ColourBackend(ReplayCADBackend):
    def compress(self, concept_id, images, masks, artifact: ReplayCADArtifact):
        artifact.condition_path.write_bytes(f"condition:{concept_id}".encode())
        return {"compression_seconds": 0.0}

    def generate(self, artifact: ReplayCADArtifact, count: int, seed: int):
        replay = np.zeros((count, 4, 4, 3), dtype=np.uint8)
        replay[..., 0] = 200
        return replay


def md_config(tmp_path: pathlib.Path, **updates) -> InCLADCRReplayCADConfig:
    config = InCLADCRReplayCADConfig.inclad_cr_preset("inclad-md", tmp_path / "bank", device="cpu", mask_backend="full-frame")
    payload = config.model_dump(mode="python")
    payload.update({"mask_backend": "full-frame", "replay_samples_per_concept": 2, **updates})
    return InCLADCRReplayCADConfig.model_validate(payload)


# ------------------------------------------------------------------ GPU kNN


@pytest.mark.parametrize("n_neighbors", [1, 3])
def test_torch_knn_matches_sklearn_brute_force(n_neighbors):
    rng = np.random.default_rng(0)
    bank = (rng.normal(size=(5000, 64)) * 3 + 10).astype(np.float32)
    # Near-duplicates of bank rows are where float32 cancellation could reorder neighbours.
    queries = np.concatenate([(rng.normal(size=(700, 64)) * 3 + 10).astype(np.float32), bank[:40] + 1e-4])
    expected_distance, expected_index = NearestNeighbors(n_neighbors=n_neighbors).fit(bank).kneighbors(queries)

    index = TorchExactNearestNeighbors(n_neighbors, torch.device("cpu"), query_chunk=256, bank_chunk=1700)
    distance, found = index.fit(bank).kneighbors(queries)

    assert np.array_equal(found, expected_index)
    np.testing.assert_allclose(distance, expected_distance, rtol=1e-5, atol=1e-5)
    assert distance.dtype == np.float32


def test_torch_knn_with_fewer_bank_rows_than_candidates():
    bank = np.arange(12, dtype=np.float32).reshape(3, 4)
    distance, index = TorchExactNearestNeighbors(1, torch.device("cpu")).fit(bank).kneighbors(bank + 0.5)
    assert index[:, 0].tolist() == [0, 1, 2]
    np.testing.assert_allclose(distance[:, 0], np.full(3, 1.0))


def test_gpu_knn_install_uses_torch_only_on_cuda_and_uninstalls(monkeypatch):
    class RecordingIndex:
        def __init__(self, n_neighbors, device):
            self.n_neighbors, self.device = n_neighbors, device

        def fit(self, bank):
            self.bank = bank
            return self

    monkeypatch.setattr(gpu_knn, "TorchExactNearestNeighbors", RecordingIndex)
    bank = np.zeros((4, 2), dtype=np.float32)

    def fake_model(device):
        return SimpleNamespace(_device=torch.device(device), _memory_bank=bank,
                               config=SimpleNamespace(n_neighbors=9), _ensure_fitted=lambda: None)

    original = PatchCore.__dict__["_fit_nn_index"]
    uninstall = gpu_knn.install()
    try:
        cpu_model, cuda_model = fake_model("cpu"), fake_model("cuda")
        PatchCore._fit_nn_index(cpu_model)
        PatchCore._fit_nn_index(cuda_model)
        assert isinstance(cpu_model._nn_index, NearestNeighbors)
        assert isinstance(cuda_model._nn_index, RecordingIndex)
        assert cuda_model._nn_index.n_neighbors == 4 and cuda_model._nn_index.bank is bank
    finally:
        uninstall()
    assert PatchCore.__dict__["_fit_nn_index"] is original


def test_install_switches_gpu_knn_on_only_when_asked():
    original = PatchCore.__dict__["_fit_nn_index"]
    uninstall = inclad_cr.install()
    assert PatchCore.__dict__["_fit_nn_index"] is original
    uninstall()
    uninstall = inclad_cr.install(gpu_knn=True)
    assert PatchCore.__dict__["_fit_nn_index"] is not original
    uninstall()
    assert PatchCore.__dict__["_fit_nn_index"] is original


# ------------------------------------------------------------------ MD profile


@pytest.mark.parametrize("name", ["inclad-md", "InCLAD_MD", "multidataset", "md"])
def test_md_preset_accepts_the_runner_spellings(tmp_path, name):
    config = InCLADCRReplayCADConfig.inclad_cr_preset(name, tmp_path, device="cpu", mask_backend="full-frame")
    assert config.benchmark == "inclad-md"
    assert config.preset == PROFILE_PRESET
    assert config.model_id == "CompVis/ldm-text2im-large-256"


def test_md_concepts_keep_the_mask_rule_of_their_source_dataset(tmp_path):
    config = InCLADCRReplayCADConfig.inclad_cr_preset("inclad-md", tmp_path, device="cpu", mask_backend="full-frame")
    mode = lambda concept: config.mask_modes.get(concept, config.default_mask_mode)  # noqa: E731
    for concept in ("mvtec__carpet", "mvtec__leather", "btech__02", "dagm__Class4", "dagm__Class8"):
        assert mode(concept) == "full-frame", concept
    for concept in ("mvtec__bottle", "mvtec__screw", "btech__01", "btech__03", "visa__pcb4", "mpdd__tubes"):
        assert mode(concept) == "corner-background", concept


def test_md_artifact_bank_records_grey_flag_and_resume_reads_it(tmp_path):
    config = md_config(tmp_path)
    writer = InCLADCRReplayCADMemory(config, backend=ColourBackend())
    writer.compress("mvtec__bottle", np.full((2, 4, 4, 3), [10, 50, 90], dtype=np.uint8))
    writer.compress("dagm__Class4", np.full((2, 4, 4, 3), 90, dtype=np.uint8))
    flags = {artifact.concept_id: artifact.read_metadata()[GREY_METADATA_KEY] for artifact in writer.artifacts}
    assert flags == {"mvtec__bottle": False, "dagm__Class4": True}

    # A resumed process activates the artifacts without seeing the training images.
    resumed = InCLADCRReplayCADMemory(config, backend=ColourBackend())
    resumed._artifacts = list(writer.artifacts)
    assert resumed._grey_concepts == {}
    replay = resumed.generate_previous()
    bottle, dagm = replay[:2], replay[2:]
    assert bottle[..., 1].max() == 0
    assert np.array_equal(dagm[..., 0], dagm[..., 1])


def test_activated_artifact_without_grey_flag_is_refused(tmp_path):
    config = md_config(tmp_path)
    writer = InCLADCRReplayCADMemory(config, backend=ColourBackend())
    artifact = writer.compress("mvtec__bottle", np.full((2, 4, 4, 3), [10, 50, 90], dtype=np.uint8))
    metadata = artifact.read_metadata()
    del metadata[GREY_METADATA_KEY]
    artifact.write_metadata(metadata)

    resumed = InCLADCRReplayCADMemory(config, backend=ColourBackend())
    resumed._artifacts = [artifact]
    with pytest.raises(ValueError, match=GREY_METADATA_KEY):
        resumed.generate_previous()


# ------------------------------------------------------------------ resumable wrapper


def test_wrapper_requires_a_per_run_artifact_dir_and_prepends_profile_defaults():
    wrapper = load_wrapper()
    base = ["--strategy", "replaycad", "--benchmark", "inclad-md"]
    with pytest.raises(SystemExit, match="replaycad-artifact-dir"):
        wrapper.build_runner_argv(base)
    with pytest.raises(SystemExit, match="one bank per run"):
        wrapper.build_runner_argv([*base, "--replaycad-bank-dir", "x", "--replaycad-artifact-dir", "y"])

    argv, use_gpu_knn = wrapper.build_runner_argv([*base, "--replaycad-artifact-dir", "y"])
    assert use_gpu_knn is True
    assert argv[:1] == ["--replaycad-cache-generated"] and '{"threshold": 0.0}' in argv
    _, use_gpu_knn = wrapper.build_runner_argv([*base, "--replaycad-artifact-dir", "y", "--no-patchcore-gpu-knn"])
    assert use_gpu_knn is False


def test_wrapper_lets_md_through_the_base_check_and_keeps_its_other_rules():
    wrapper = load_wrapper()
    seen = []

    def base_check(args):
        seen.append(args.benchmark)
        if args.benchmark not in {"btech", "dagm", "mpdd", "mvtec", "visa"}:
            raise ValueError("ReplayCAD supports InCLAD benchmarks")
        if args.model != "patchcore":
            raise ValueError("only patchcore")
        args.backbone = args.backbone or "wide_resnet50_2"
        return args

    check = wrapper.accept_inclad_md(base_check)
    md = argparse.Namespace(strategy="replaycad", benchmark="multidataset", ordering_mode="hard_to_easy",
                            model="patchcore", backbone=None)
    assert check(md).benchmark == "inclad-md" and md.backbone == "wide_resnet50_2"
    assert seen == ["mvtec"]

    with pytest.raises(ValueError, match="only patchcore"):
        check(argparse.Namespace(strategy="replaycad", benchmark="inclad-md", ordering_mode=None,
                                 model="ucad", backbone=None))
    with pytest.raises(ValueError, match="easy_to_hard and hard_to_easy"):
        check(argparse.Namespace(strategy="replaycad", benchmark="inclad-md", ordering_mode="random",
                                 model="patchcore", backbone=None))
    single = argparse.Namespace(strategy="replaycad", benchmark="visa", ordering_mode="random",
                                model="patchcore", backbone="resnet18")
    assert check(single).benchmark == "visa"
