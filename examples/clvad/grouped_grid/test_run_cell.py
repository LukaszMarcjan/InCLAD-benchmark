"""Checks for run_cell.py.

Run from the repository root:

    PYTHONPATH=src:. python -m pytest -q examples/clvad/grouped_grid/test_run_cell.py

The expected values are the ``run_config.cli_args`` recorded in the result
JSONs of the grouped-grid runs (paths and the runtime-resolved PatchCore
memory-bank size excluded).
"""

import importlib.util
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_cell = _load("grouped_grid_run_cell", HERE / "run_cell.py")

# Recorded cli_args shared by every grouped run.
COMMON = {
    "pretrained": True, "batch_size": 32, "epochs": 50, "n_runs": 1, "replay_buffer_fraction": 0.2,
    "device": "cuda:0", "eval_level": "both", "pixel_threshold_mode": "train-quantile",
    "pixel_threshold_quantile": 0.95, "pixel_threshold": 0.5, "pixel_skip_missing_masks": True,
    "shuffle_concepts": False, "group_test": False, "ordering_metric": "roc-auc", "ordering_column": "concept",
    "root": None, "resize": None, "max_train": None, "max_test": None, "freeze_backbone": None,
    "model_size": None, "teacher_weights_path": None, "penalty_source_root": None,
}
RECORDED = [
    (("mvtec", "rn18", "padim", "naive", "14-1", "eth"),
     {"model": "padim", "strategy": "naive", "benchmark": "mvtec", "backbone": "resnet18",
      "step_schedule": "14-1", "ordering_mode": "easy_to_hard", "master_seed": 42}),
    (("mvtec", "rn18", "padim", "cl", "14-1", "rand"),
     {"model": "padim", "strategy": "cl", "benchmark": "mvtec", "backbone": "resnet18",
      "step_schedule": "14-1", "ordering_mode": None, "master_seed": 123}),
    (("mpdd", "rn18", "patchcore", "cl", "4-1x2", "hte"),
     {"model": "patchcore", "strategy": "cl", "benchmark": "mpdd", "backbone": "resnet18",
      "step_schedule": "4-1x2", "ordering_mode": "hard_to_easy", "master_seed": 42}),
    (("dagm", "wr50", "rd4ad", "replay", "2x5", "rand"),
     {"model": "rd4ad", "strategy": "replay", "benchmark": "dagm", "backbone": "wide_resnet50_2",
      "step_schedule": "2x5", "ordering_mode": None, "master_seed": 123}),
]


def test_manifest_rows_all_map():
    rows = run_cell.read_manifest(run_cell.DEFAULT_MANIFEST)
    assert len(rows) == 1728
    commands = [run_cell.build_command(*row) for row in rows]
    assert len({tuple(c) for c in commands}) == 1728


def test_rand_has_no_ordering_flags():
    command = run_cell.build_command("visa", "wr50", "stfpm", "replay", "8-4", "rand")
    assert "--ordering-mode" not in command and "--ordering-dir" not in command
    assert "--shuffle-concepts" not in command
    assert command[command.index("--master-seed") + 1] == "123"


def test_output_dir_layout():
    command = run_cell.build_command("btech", "wr50", "cfa", "cl", "1x3", "hte", results_root="out")
    assert command[command.index("--output-dir") + 1] == "out/btech/wr50/hte/cfa/cl/1x3"


@pytest.mark.parametrize("row", [
    ("mvtec", "rn18", "stfpm", "cl", "14-1", "eth"),
    ("mvtec", "rn18", "padim", "replay", "14-1", "eth"),
    ("mvtec", "rn18", "padim", "naive", "3x2", "eth"),
    ("mvtec", "rn50", "padim", "naive", "14-1", "eth"),
    ("mvtec", "rn18", "padim", "naive", "14-1", "random"),
])
def test_rejects_cells_outside_the_grid(row):
    with pytest.raises(ValueError):
        run_cell.build_command(*row)


@pytest.mark.parametrize("row,recorded", RECORDED)
def test_runner_parses_command_to_recorded_cli_args(row, recorded, monkeypatch):
    runner = _load("grouped_grid_runner", REPO_ROOT / "examples/clvad/run_continual_visual_ad_step_schedule.py")
    command = run_cell.build_command(*row)
    monkeypatch.setattr(sys, "argv", command[1:])
    parsed = vars(runner.parse_args())
    for key, value in {**COMMON, **recorded}.items():
        assert parsed[key] == value, key
    expected_dir = "examples/visual_models/ordering"
    if recorded["ordering_mode"] is not None:
        assert parsed["ordering_dir"] == expected_dir
