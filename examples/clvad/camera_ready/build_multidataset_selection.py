#!/usr/bin/env python3
"""Select the MultiDataset runs behind Table 2 and write the selection file.

rebuild_main_table.py and multidataset_forgetting.py read the MultiDataset
rows of Table 2 from a selection file (--multidataset-json). This script
rebuilds that file from the MultiDataset result roots with the rule used for
the paper:

  * every *_seed_*.json below the roots, except FastFlow runs;
  * dataset.name == MULTIDATASET_SEQ_V1-VisualBenchmark;
  * run_status.state == "completed" and
    run_status.completed_concepts_count == run_status.total_concepts;
  * methods CFA, PaDiM and PaSTe (model.name), both backbones;
  * ordering from the file path (easy-to-hard / hard-to-easy), variant from
    strategy.name (naive -> NCL, anything else -> CL).

Within a method the selected files are averaged; each cell then reports the
mean and sample standard deviation across the methods. As in the submitted
table, the BWT values in this file are pyclad's BackwardTransfer;
rebuild_main_table.py recomputes BWT with the paper's Eq. (bwt).

The paper's selection (23 runs) comes from the default --md-results-roots
(shown by --help). Run paths are written relative to
--results-root when the files lie below it, otherwise as absolute paths.

Example:
  PYTHONPATH=src:. python examples/clvad/camera_ready/build_multidataset_selection.py \\
      --results-root /path/to/results --output out/multidataset_selection.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import defaultdict

from rebuild_main_table import MD_METHODS, MD_ORDERINGS, read_file_metrics
from results_root import (
    MULTIDATASET_ROOTS,
    OUTPUT_NAME,
    absolute,
    add_results_root_argument,
    require_input,
    results_root_from_argv,
    source_string,
)

DATASET_NAME = "MULTIDATASET_SEQ_V1-VisualBenchmark"
VARIANTS = ["naive", "continual"]
METRICS = ("image_roc", "image_bwt", "pixel_roc", "pixel_bwt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_results_root_argument(parser)
    root = results_root_from_argv(argv)
    parser.add_argument("--md-results-roots", type=pathlib.Path, nargs="+",
                        default=[root / sub for sub in MULTIDATASET_ROOTS],
                        help="Roots scanned recursively for MultiDataset result files.")
    parser.add_argument("--output", type=pathlib.Path,
                        default=root / OUTPUT_NAME / "table2" / "multidataset_selection.json")
    parser.add_argument("--compare-with", type=pathlib.Path, default=None,
                        help="An existing selection file; report whether the run lists and values agree.")
    return parser.parse_args(argv)


def ordering_of(path: pathlib.Path) -> str | None:
    name = path.as_posix()
    for ordering in MD_ORDERINGS:
        if ordering in name:
            return ordering
    return None


def select(roots: list[pathlib.Path]) -> list[dict]:
    runs = []
    for root in roots:
        for path in sorted(root.rglob("*_seed_*.json")):
            if "/fastflow/" in path.as_posix():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict) or (data.get("dataset") or {}).get("name") != DATASET_NAME:
                continue
            status = data.get("run_status") or {}
            if status.get("state") != "completed" or status.get("completed_concepts_count") != status.get("total_concepts"):
                continue
            method = str((data.get("model") or {}).get("name", "")).strip().lower()
            ordering = ordering_of(path)
            if method not in MD_METHODS or ordering is None:
                continue
            variant = "naive" if str((data.get("strategy") or {}).get("name", "")).strip().lower() == "naive" else "continual"
            values = {}
            for modality in ("image", "pixel"):
                metrics, reason = read_file_metrics(path, modality, None)
                if metrics is None:
                    raise SystemExit(f"{path}: {reason}")
                values[f"{modality}_roc"] = metrics["continual_average"]
                values[f"{modality}_bwt"] = metrics["bwt_pyclad"]
            runs.append({"path": path, "method": method, "variant": variant, "order": ordering,
                         "backbone": str((data.get("model") or {}).get("backbone", "unknown")), **values})
    return runs


def build(runs: list[dict], results_root: pathlib.Path) -> dict:
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["order"], run["variant"], run["method"])].append(run)
    cells = []
    for ordering in MD_ORDERINGS:
        for variant in VARIANTS:
            methods = [m for m in MD_METHODS if grouped.get((ordering, variant, m))]
            details = {}
            for method in methods:
                files = grouped[(ordering, variant, method)]
                details[method] = {
                    **{metric: statistics.mean(f[metric] for f in files) for metric in METRICS},
                    "backbones": sorted({f["backbone"] for f in files}),
                    "paths": [source_string(f["path"], [results_root]) for f in files],
                }
            metrics = {}
            for metric in METRICS:
                values = [details[m][metric] for m in methods]
                metrics[metric] = {
                    "mean": statistics.mean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "values_by_method": {m: details[m][metric] for m in methods},
                }
            cells.append({"dataset": "multidataset", "dataset_label": "MultiDataset", "order": ordering,
                          "variant": variant, "methods": methods, "metrics": metrics, "method_details": details})
    return {
        "metadata": {
            "description": "MultiDataset treated as one whole dataset. Completed runs only. "
                           "Model-first aggregation across cfa, padim, paste. BWT: pyclad BackwardTransfer.",
            "runs_included": len(runs),
        },
        "runs": [{**{k: v for k, v in run.items() if k != "path"}, "path": source_string(run["path"], [results_root])}
                 for run in runs],
        "cells": cells,
    }


def compare(new: dict, old_file: pathlib.Path) -> bool:
    old = json.loads(old_file.read_text(encoding="utf-8"))

    def files_of(payload: dict) -> dict:
        return {(cell["order"], cell["variant"], method): details["paths"]
                for cell in payload["cells"] for method, details in cell["method_details"].items()}

    def same_paths(a: list[str], b: list[str]) -> bool:
        # Paths may be written relative to different bases: match on the trailing path.
        def match(x: str, y: str) -> bool:
            return x == y or x.endswith("/" + y) or y.endswith("/" + x)
        return len(a) == len(b) and all(sum(match(x, y) for y in b) == 1 for x in a)

    new_files, old_files = files_of(new), files_of(old)
    same_files = new_files.keys() == old_files.keys() and all(same_paths(new_files[k], old_files[k]) for k in new_files)
    worst = 0.0
    for new_cell, old_cell in zip(new["cells"], old["cells"]):
        for metric in METRICS:
            worst = max(worst, abs(new_cell["metrics"][metric]["mean"] - old_cell["metrics"][metric]["mean"]),
                        abs(new_cell["metrics"][metric]["std"] - old_cell["metrics"][metric]["std"]))
    print(f"run files per cell identical: {same_files}; max |difference| of cell mean/std: {worst:.3g}")
    return same_files and worst < 1e-12


def main() -> int:
    args = parse_args()
    roots = [absolute(require_input(r, "--md-results-roots")) for r in args.md_results_roots]
    runs = select(roots)
    if not runs:
        raise SystemExit("no completed MultiDataset runs found below --md-results-roots")
    payload = build(runs, args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{len(runs)} MultiDataset runs selected; wrote {args.output}")
    if args.compare_with is not None and not compare(payload, require_input(args.compare_with, "--compare-with")):
        raise SystemExit("selection differs from --compare-with")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
