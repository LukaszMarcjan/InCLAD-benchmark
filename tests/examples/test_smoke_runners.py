"""
Smoke tests for visual experiment runner scripts and utilities.

These tests verify that:
- CLI flags (--inclad-bench, --ordering-mode random) are accepted by all runners
- build_command() propagates --inclad-bench to subprocesses
- KNOWN_BENCHMARKS / KNOWN_DATASETS include inclad-md and inclad-bench
- Path-based benchmark inference handles inclad-md

No actual experiments are run; no datasets or models are loaded.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples"
CLVAD_DIR = EXAMPLES_DIR / "clvad"
VISUAL_DIR = EXAMPLES_DIR / "visual_models"
NEURIPS_DIR = VISUAL_DIR / "neurips_utilities"


def _load_module(path: pathlib.Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base_runner():
    return _load_module(CLVAD_DIR / "run_continual_visual_ad.py", "_smoke_base_runner")


@pytest.fixture(scope="module")
def generate_heatmaps():
    return _load_module(VISUAL_DIR / "generate_heatmaps.py", "_smoke_generate_heatmaps")


@pytest.fixture(scope="module")
def group_results():
    return _load_module(NEURIPS_DIR / "group_results_by_dataset.py", "_smoke_group_results")


@pytest.fixture(scope="module")
def batch_runner(generate_heatmaps):
    # run_all_experiments does `from generate_heatmaps import ...` — needs the
    # visual_models dir on sys.path so the relative import resolves.
    sys.path.insert(0, str(VISUAL_DIR))
    try:
        return _load_module(VISUAL_DIR / "run_all_experiments.py", "_smoke_batch_runner")
    finally:
        sys.path.remove(str(VISUAL_DIR))


# ---------------------------------------------------------------------------
# generate_heatmaps: KNOWN_BENCHMARKS
# ---------------------------------------------------------------------------

class TestKnownBenchmarks:
    def test_inclad_md_present(self, generate_heatmaps):
        assert "inclad-md" in generate_heatmaps.KNOWN_BENCHMARKS

    def test_inclad_bench_present(self, generate_heatmaps):
        assert "inclad-bench" in generate_heatmaps.KNOWN_BENCHMARKS

    def test_originals_still_present(self, generate_heatmaps):
        for name in ("mvtec", "btech", "visa", "mpdd", "dagm"):
            assert name in generate_heatmaps.KNOWN_BENCHMARKS

    def test_infer_inclad_md_from_path(self, generate_heatmaps, tmp_path):
        json_path = tmp_path / "inclad-md" / "padim_naive_seed_42.json"
        json_path.parent.mkdir(parents=True)
        json_path.touch()
        result = generate_heatmaps.infer_benchmark_name(json_path, tmp_path)
        assert result == "inclad-md"

    def test_infer_inclad_md_underscore_from_path(self, generate_heatmaps, tmp_path):
        json_path = tmp_path / "inclad_md" / "padim_naive_seed_42.json"
        json_path.parent.mkdir(parents=True)
        json_path.touch()
        result = generate_heatmaps.infer_benchmark_name(json_path, tmp_path)
        assert result == "inclad_md"


# ---------------------------------------------------------------------------
# group_results_by_dataset: KNOWN_DATASETS
# ---------------------------------------------------------------------------

class TestKnownDatasets:
    def test_inclad_md_present(self, group_results):
        assert "inclad-md" in group_results.KNOWN_DATASETS

    def test_inclad_bench_present(self, group_results):
        assert "inclad-bench" in group_results.KNOWN_DATASETS

    def test_originals_still_present(self, group_results):
        for name in ("mvtec", "btech", "visa", "mpdd", "dagm"):
            assert name in group_results.KNOWN_DATASETS

    def test_infer_inclad_md_from_json_name(self, group_results, tmp_path):
        json_path = tmp_path / "padim_naive_seed_42.json"
        data = {"dataset": {"name": "InCLAD-MD-easy_to_hard"}}
        result = group_results.infer_dataset(data, json_path)
        assert result == "inclad-md"

    def test_infer_mvtec_still_works(self, group_results, tmp_path):
        json_path = tmp_path / "padim_naive_seed_42.json"
        data = {"dataset": {"name": "mvtec-easy_to_hard"}}
        result = group_results.infer_dataset(data, json_path)
        assert result == "mvtec"


# ---------------------------------------------------------------------------
# run_continual_visual_ad: --inclad-bench flag
# ---------------------------------------------------------------------------

class TestBaseRunnerArgs:
    def test_inclad_bench_flag_accepted(self, base_runner):
        with patch("sys.argv", [
            "run_continual_visual_ad.py",
            "--model", "patchcore",
            "--benchmark", "mvtec",
            "--inclad-bench",
        ]):
            args = base_runner.parse_args()
        assert args.inclad_bench is True

    def test_inclad_bench_default_false(self, base_runner):
        with patch("sys.argv", [
            "run_continual_visual_ad.py",
            "--model", "patchcore",
            "--benchmark", "mvtec",
        ]):
            args = base_runner.parse_args()
        assert args.inclad_bench is False

    def test_ordering_mode_random_accepted(self, base_runner):
        with patch("sys.argv", [
            "run_continual_visual_ad.py",
            "--model", "patchcore",
            "--benchmark", "mvtec",
            "--ordering-mode", "random",
        ]):
            args = base_runner.parse_args()
        assert args.ordering_mode == "random"


# ---------------------------------------------------------------------------
# run_all_experiments: --inclad-bench and --ordering-mode random
# ---------------------------------------------------------------------------

class TestBatchRunnerArgs:
    def _parse(self, batch_runner, extra_argv: list[str]):
        with patch("sys.argv", ["run_all_experiments.py"] + extra_argv):
            return batch_runner.parse_args()

    def test_inclad_bench_flag_accepted(self, batch_runner):
        args = self._parse(batch_runner, ["--inclad-bench", "--benchmarks", "mvtec"])
        assert args.inclad_bench is True

    def test_inclad_bench_default_false(self, batch_runner):
        args = self._parse(batch_runner, ["--benchmarks", "mvtec"])
        assert args.inclad_bench is False

    def test_ordering_mode_random_accepted(self, batch_runner):
        args = self._parse(batch_runner, ["--benchmarks", "mvtec", "--ordering-mode", "random"])
        assert args.ordering_mode == "random"

    def test_ordering_mode_easy_to_hard_accepted(self, batch_runner):
        args = self._parse(batch_runner, ["--benchmarks", "mvtec", "--ordering-mode", "easy_to_hard"])
        assert args.ordering_mode == "easy_to_hard"


# ---------------------------------------------------------------------------
# run_all_experiments: build_command propagates --inclad-bench
# ---------------------------------------------------------------------------

class TestBuildCommand:
    def _make_args(self, batch_runner, inclad_bench: bool = False, ordering_mode=None):
        argv = ["run_all_experiments.py", "--benchmarks", "mvtec"]
        if inclad_bench:
            argv.append("--inclad-bench")
        if ordering_mode:
            argv.extend(["--ordering-mode", ordering_mode])
        with patch("sys.argv", argv):
            return batch_runner.parse_args()

    def test_inclad_bench_propagated(self, batch_runner, tmp_path):
        args = self._make_args(batch_runner, inclad_bench=True)
        args.output_root = str(tmp_path)
        command = batch_runner.build_command(
            "patchcore", "naive", "mvtec", tmp_path / "mvtec", args
        )
        assert "--inclad-bench" in command

    def test_inclad_bench_not_included_by_default(self, batch_runner, tmp_path):
        args = self._make_args(batch_runner, inclad_bench=False)
        args.output_root = str(tmp_path)
        command = batch_runner.build_command(
            "patchcore", "naive", "mvtec", tmp_path / "mvtec", args
        )
        assert "--inclad-bench" not in command

    def test_ordering_mode_random_propagated(self, batch_runner, tmp_path):
        args = self._make_args(batch_runner, ordering_mode="random")
        args.output_root = str(tmp_path)
        command = batch_runner.build_command(
            "patchcore", "naive", "mvtec", tmp_path / "mvtec", args
        )
        assert "--ordering-mode" in command
        idx = command.index("--ordering-mode")
        assert command[idx + 1] == "random"
