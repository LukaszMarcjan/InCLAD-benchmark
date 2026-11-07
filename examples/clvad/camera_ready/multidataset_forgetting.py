#!/usr/bin/env python3
"""Where forgetting happens in the 27-task MultiDataset stream.

Uses the run files behind the MultiDataset rows of Table 2 (CFA, PaDiM, PaSTe;
both backbones; Easy-to-Hard and Hard-to-Easy). For each run and level:

  * per-source BWT: Eq. (bwt) restricted to the tasks j of one source dataset
    (mean over i > j of R[i][j] - R[j][j]);
  * step-to-step change R[i][j] - R[i-1][j] of every earlier task j, split by
    whether step i starts a new source dataset (boundary step) or continues the
    current one (within-dataset step).

Values are averaged within a method (backbones, orderings), then across methods.

Input: --multidataset-json is the MultiDataset run selection read by
rebuild_main_table.py (default derived from --results-root; see
build_multidataset_selection.py). Its run paths are taken relative to the
parent of the file's directory, then looked up below --results-root.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import defaultdict

from rebuild_main_table import MULTIDATASET_JSON, REPO_ROOT, multidataset_run_path, normalize_matrix
from results_root import OUTPUT_NAME, add_results_root_argument, require_input, results_root_from_argv

KEYS = {"image": "concept_metric_callback_ROC-AUC", "pixel": "pixel_concept_metric_callback_Pixel-ROC-AUC"}
SOURCES = ["btech", "mpdd", "dagm", "visa", "mvtec"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_results_root_argument(parser)
    root = results_root_from_argv(argv)
    parser.add_argument("--multidataset-json", type=pathlib.Path, default=root / MULTIDATASET_JSON)
    parser.add_argument("--output", type=pathlib.Path, default=root / OUTPUT_NAME / "multidataset_forgetting.json")
    return parser.parse_args(argv)


def run_stats(path: pathlib.Path, level: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data[KEYS[level]]
    order = [str(c) for c in payload["concepts_order"]]
    R = normalize_matrix(payload["metric_matrix"], order)
    source = [c.split("__", 1)[0] for c in order]
    n = len(order)
    out = {}
    for s in SOURCES:
        values = [R[i][j] - R[j][j] for j in range(n) if source[j] == s for i in range(j + 1, n)]
        out[f"bwt_{s}"] = statistics.mean(values) if values else None
    boundary, within = [], []
    for i in range(1, n):
        deltas = [R[i][j] - R[i - 1][j] for j in range(i)]
        (boundary if source[i] != source[i - 1] else within).extend(deltas)
    out["delta_boundary"] = statistics.mean(boundary)
    out["delta_within"] = statistics.mean(within)
    return out


def main() -> int:
    args = parse_args()
    require_input(args.multidataset_json, "--multidataset-json")
    payload = json.loads(args.multidataset_json.read_text(encoding="utf-8"))
    by = defaultdict(lambda: defaultdict(list))  # (variant, level) -> method -> list of run stats
    for cell in payload["cells"]:
        for method, details in cell["method_details"].items():
            for rel in details["paths"]:
                for level in KEYS:
                    path = multidataset_run_path(args.multidataset_json, rel, args.results_root, args.search_dir)
                    by[(cell["variant"], level)][method].append(run_stats(path, level))
    result = {}
    for (variant, level), methods in by.items():
        keys = list(next(iter(methods.values()))[0])
        per_method = {m: {k: statistics.mean(r[k] for r in runs if r[k] is not None) for k in keys} for m, runs in methods.items()}
        result[f"{variant}_{level}"] = {k: statistics.mean(v[k] for v in per_method.values()) for k in keys}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for key, values in sorted(result.items()):
        print(key, {k: round(v, 3) for k, v in values.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
