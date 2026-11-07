"""Checks for the result-path handling of the camera-ready table builders.

Run from the repository root:

    PYTHONPATH=src:. python -m pytest -q examples/clvad/camera_ready/test_results_root.py

No result files are needed: small trees are created in a temporary directory.
"""

import importlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

results_root = importlib.import_module("results_root")


def _touch(path: pathlib.Path, text: str = "{}") -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_locate_maps_foreign_prefix_onto_results_root(tmp_path):
    target = _touch(tmp_path / "trees" / "run_tree" / "results_x" / "run_seed_1.json")
    found = results_root.locate("/some/other/machine/run_tree/results_x/run_seed_1.json", tmp_path / "trees")
    assert found == target


def test_locate_uses_run_tree_subdir_and_search_dirs(tmp_path):
    root = tmp_path / "root"
    in_subdir = _touch(root / results_root.RUN_TREES_SUBDIR / "tree_a" / "r" / "f.json")
    assert results_root.locate("/old/prefix/tree_a/r/f.json", root) == in_subdir
    extra = tmp_path / "elsewhere"
    in_extra = _touch(extra / "tree_b" / "r" / "g.json")
    assert results_root.locate("/old/prefix/tree_b/r/g.json", root) is None
    assert results_root.locate("/old/prefix/tree_b/r/g.json", root, [extra]) == in_extra


def test_locate_relative_source_and_no_basename_only_match(tmp_path):
    rel = _touch(tmp_path / "exports" / "m" / "run.json")
    assert results_root.locate("exports/m/run.json", tmp_path) == rel
    _touch(tmp_path / "lonely.json")
    # A bare file name below the root must not satisfy a longer recorded path.
    assert results_root.locate("/old/prefix/some_tree/lonely.json", tmp_path) is None


def test_require_reports_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        results_root.require("/old/prefix/tree/missing.json", tmp_path)


def test_source_string_relative_or_absolute(tmp_path):
    inside = _touch(tmp_path / "root" / "a" / "b.json")
    outside = _touch(tmp_path / "other" / "c.json")
    assert results_root.source_string(inside, [tmp_path / "root"]) == "a/b.json"
    assert results_root.source_string(outside, [tmp_path / "root"]) == outside.as_posix()


@pytest.mark.parametrize("module_name", [
    "rebuild_main_table", "build_baseline_table", "build_grouped_tables", "build_resource_table",
    "heldout_ordering_check", "multidataset_forgetting", "build_multidataset_selection",
])
def test_defaults_derive_from_results_root(module_name, tmp_path):
    module = importlib.import_module(module_name)
    args = module.parse_args(["--results-root", str(tmp_path)])
    assert args.results_root == tmp_path
    for name, value in vars(args).items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, pathlib.Path) and name not in ("results_root", "manifest"):
                assert item.is_relative_to(tmp_path), (name, item)


def test_grouped_main_matches_paper_layout():
    grouped = importlib.import_module("build_grouped_tables")
    row = {"dataset": "mvtec", "backbone": "rn18", "ordering": "eth", "model": "cfa", "variant": "naive",
           "schedule": "14-1", "steps": 2}
    rows = []
    for variant in ("naive", "continual"):
        rows.append({**row, "variant": variant,
                     **{f"{level}_{m}": 0.5 for level in ("image", "pixel") for m in ("ca", "bwt", "rfm", "rnta")}})
    body = [line for line in grouped.render_main(rows).splitlines() if "&" in line]
    assert body[0] == "MVTec & 50.0 & 50.0 & 50.0 & 50.0 & 50.0 & 50.0 & 50.0 & 50.0 \\\\"
    assert all(line.count("&") == 8 for line in body)
    wide = [line for line in grouped.render_main(rows, with_schedules=True).splitlines() if "&" in line]
    assert wide[0].startswith("MVTec & 14+1 & ")


def test_grouped_builder_fails_clearly_without_runs(tmp_path, monkeypatch, capsys):
    grouped = importlib.import_module("build_grouped_tables")
    manifest = _touch(tmp_path / "manifest.tsv", "mvtec\trn18\tcfa\tnaive\t14-1\teth\n")
    monkeypatch.setattr(sys, "argv", ["build_grouped_tables.py", "--results-root", str(tmp_path),
                                      "--manifest", str(manifest), "--exports", "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="no completed grouped runs found"):
        grouped.main()


def test_baseline_source_outside_results_root(tmp_path):
    baseline = importlib.import_module("build_baseline_table")
    bundle = tmp_path / "data" / "bundle"
    matrix = {"a": {"a": 0.9, "b": 0.5}, "b": {"a": 0.8, "b": 0.7}}
    run = {
        "run_config": {"cli_args": {"backbone": "resnet50", "benchmark": "btech", "ordering_mode": "easy_to_hard"}},
        "run_status": {"state": "completed"},
        "concept_metric_callback_ROC-AUC": {"concepts_order": ["a", "b"], "metric_matrix": matrix},
        "pixel_concept_metric_callback_Pixel-ROC-AUC": {"concepts_order": ["a", "b"], "metric_matrix": matrix},
    }
    path = _touch(bundle / "one_for_more" / "btech__easy_to_hard" / "run.json", json.dumps(run))
    rows = baseline.select_runs(bundle, [], tmp_path / "unrelated_root")
    chosen = [r for r in rows if r["source"]]
    assert len(chosen) == 1 and chosen[0]["source"] == path.as_posix()
    rows = baseline.select_runs(bundle, [], tmp_path / "data")
    assert [r["source"] for r in rows if r["source"]] == ["bundle/one_for_more/btech__easy_to_hard/run.json"]
