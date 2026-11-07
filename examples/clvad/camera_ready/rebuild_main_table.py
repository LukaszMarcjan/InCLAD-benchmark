#!/usr/bin/env python3
"""Rebuild the InCLAD main-paper table (Table 2) for the camera-ready version.

Two corrections are applied to the per-slot aggregates that fed the submitted
table, and everything else (slot selection, model-first aggregation, sample
standard deviation, MultiDataset method set) is kept as before:

1. Result files whose ROC-AUC matrix does not cover every task of the stream,
   or whose run_status is not "completed", are dropped. In the submitted table
   such files (failed or interrupted runs with an empty matrix) entered the
   averages with ContinualAverage = BackwardTransfer = 0.0.
2. BWT follows the definition printed in the paper (Eq. bwt, after
   Diaz-Rodriguez et al. 2018): the mean over all pairs i > j of
   R[i][j] - R[j][j]. pyclad.metrics.continual.BackwardTransfer averages the
   step-to-step differences R[j][i] - R[j-1][i] over the same number of pairs,
   which telescopes to the GEM numerator divided by N(N-1)/2.

The submitted table is first reproduced from the original inputs, and the
script stops if that reproduction does not match the bundled table body.

Outputs (in --output-dir):
  legacy/            the submitted table rebuilt from the original aggregates
  corrected/         Table 2 after both corrections
  article_model_aggregates_corrected.json
  multidataset_rows.json
  table2_body.tex    single-dataset body + MultiDataset rows, same layout as the paper
  table2_text_numbers.json   averages quoted in the Results Discussion
  excluded_files.csv

Inputs: the aggregates and the submitted table body of the 2026-10-07 bundle
(01_main_table_480/) and the MultiDataset run selection (the bundled file, or
the one written by build_multidataset_selection.py); the defaults below
--results-root are shown by --help. The per-run files listed in the aggregates
are absolute paths of the machine the runs were collected on; they are mapped
onto --results-root by their trailing tree path (see results_root.py).

Example:
  PYTHONPATH=src:. python examples/clvad/camera_ready/rebuild_main_table.py \
      --results-root /path/to/results --output-dir out/table2
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pathlib
import statistics
import subprocess
import sys

from results_root import (
    BUNDLE_NAME,
    MULTIDATASET_JSON,
    OUTPUT_NAME,
    add_results_root_argument,
    locate,
    require,
    require_input,
    results_root_from_argv,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BUNDLE = REPO_ROOT / "InCLAD_paper_results_20261007"
SUMMARY_BUILDER = REPO_ROOT / "examples" / "visual_models" / "neurips_utilities" / "build_article_main_paper_summary.py"

N_TASKS = {"mvtec": 15, "visa": 12, "btech": 3, "mpdd": 6, "dagm": 10}
PAYLOAD_KEYS = {
    "image": "concept_metric_callback_ROC-AUC",
    "pixel": "pixel_concept_metric_callback_Pixel-ROC-AUC",
}
DATASETS = ["mvtec", "visa", "btech", "mpdd", "dagm"]
DATASET_LABELS = {"mvtec": "MVTec", "visa": "VisA", "btech": "BTAD", "mpdd": "MPDD", "dagm": "DAGM"}
ORDERINGS = ["easy_to_hard", "hard_to_easy", "random"]
ORDERING_LABELS = {"easy_to_hard": "Easy-to-Hard", "hard_to_easy": "Hard-to-Easy", "random": "Random"}
MD_METHODS = ["cfa", "padim", "paste"]
MD_ORDERINGS = ["easy-to-hard", "hard-to-easy"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = results_root_from_argv(argv)
    bundle = root / BUNDLE_NAME
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_results_root_argument(parser)
    parser.add_argument(
        "--aggregates-json",
        type=pathlib.Path,
        default=bundle / "01_main_table_480" / "article_model_aggregates.json",
    )
    parser.add_argument(
        "--submitted-tex",
        type=pathlib.Path,
        default=bundle / "01_main_table_480" / "article_main_paper_summary_model_first.tex",
        help="Table body of the submitted paper, used to check the legacy reproduction.",
    )
    parser.add_argument(
        "--multidataset-json",
        type=pathlib.Path,
        default=root / MULTIDATASET_JSON,
        help="MultiDataset run selection (see build_multidataset_selection.py).",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=root / OUTPUT_NAME / "table2")
    return parser.parse_args(argv)


# ---------------------------------------------------------------- metrics

def is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def nanmean(values: list[float]) -> float | None:
    kept = [value for value in values if value is not None and not is_nan(value)]
    return sum(kept) / len(kept) if kept else None


def continual_average(matrix: list[list[float]]) -> float | None:
    n = len(matrix)
    return nanmean([matrix[i][j] for i in range(n) for j in range(i + 1)])


def bwt_paper(matrix: list[list[float]]) -> float:
    """Eq. (bwt) of the paper: mean over i > j of R[i][j] - R[j][j]."""
    n = len(matrix)
    if n < 2:
        return 0.0
    value = nanmean([matrix[i][j] - matrix[j][j] for i in range(1, n) for j in range(i)])
    return 0.0 if value is None else value


def bwt_pyclad(matrix: list[list[float]]) -> float:
    """pyclad BackwardTransfer: mean over i < j of R[j][i] - R[j-1][i]."""
    n = len(matrix)
    if n < 2:
        return 0.0
    value = nanmean([matrix[j][i] - matrix[j - 1][i] for i in range(n) for j in range(i + 1, n)])
    return 0.0 if value is None else value


def normalize_matrix(matrix, concepts_order: list[str]) -> list[list[float]] | None:
    """Same row/column convention as build_article_aggregate_json.normalize_matrix."""
    if isinstance(matrix, dict):
        order = concepts_order or [str(key) for key in matrix]
        rows = []
        for row_key in order:
            row = matrix.get(row_key)
            if not isinstance(row, dict):
                return None
            values = [row.get(col_key) for col_key in order]
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
                return None
            rows.append([float(value) for value in values])
        return rows
    if isinstance(matrix, list):
        rows = []
        for row in matrix:
            if not isinstance(row, list):
                return None
            rows.append([float(value) for value in row])
        return rows
    return None


def resolve_source(path_str: str, results_root: pathlib.Path | None = None,
                   search_dirs: list[pathlib.Path] | None = None) -> pathlib.Path | None:
    path = pathlib.Path(path_str)
    if path.exists():
        return path
    # The aggregates were built when the result trees sat in the repository root;
    # they now live under results_all_/neurips/.
    relocated = pathlib.Path(path_str.replace(str(REPO_ROOT) + "/", str(REPO_ROOT / "results_all_" / "neurips") + "/"))
    if relocated.exists():
        return relocated
    # Recorded paths from another machine: match their trailing tree path below the results root.
    return locate(path_str, REPO_ROOT if results_root is None else results_root, search_dirs or [])


def read_file_metrics(path: pathlib.Path, modality: str, expected_tasks: int | None):
    """Return (metrics, reason). metrics is None when the file must be excluded."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    payload = data.get(PAYLOAD_KEYS[modality])
    if not isinstance(payload, dict):
        return None, "no payload"
    concepts_order = [str(item) for item in payload.get("concepts_order", [])]
    matrix = normalize_matrix(payload.get("metric_matrix"), concepts_order)
    if not matrix:
        return None, "empty matrix"
    if expected_tasks is not None and len(matrix) != expected_tasks:
        return None, f"matrix covers {len(matrix)} of {expected_tasks} tasks"
    state = (data.get("run_status") or {}).get("state")
    if state not in (None, "completed"):
        return None, f"run_status={state}"
    return {
        "matrix": matrix,
        "continual_average": continual_average(matrix),
        "bwt_paper": bwt_paper(matrix),
        "bwt_pyclad": bwt_pyclad(matrix),
    }, "ok"


def scalar(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
        "values": values,
    }


def mean_matrix(matrices: list[list[list[float]]]) -> dict | None:
    if not matrices or any(len(m) != len(matrices[0]) for m in matrices):
        return None
    n = len(matrices[0])
    return {
        "count": len(matrices),
        "shape": [n, len(matrices[0][0])],
        "mean": [[sum(m[i][j] for m in matrices) / len(matrices) for j in range(n)] for i in range(n)],
    }


# ---------------------------------------------------------------- Table 2, single-dataset rows

def correct_aggregates(aggregates: dict, results_root: pathlib.Path | None = None,
                       search_dirs: list[pathlib.Path] | None = None) -> tuple[dict, list[dict]]:
    corrected = copy.deepcopy(aggregates)
    excluded: list[dict] = []
    for record in corrected["records"]:
        if not record.get("article_usable"):
            continue
        for modality in ("image", "pixel"):
            payload = record.get(modality)
            if not isinstance(payload, dict) or not payload.get("available"):
                continue
            kept_files, rows = [], []
            for source in payload["source_files"]:
                resolved = resolve_source(source, results_root, search_dirs)
                if resolved is None:
                    require(source, REPO_ROOT if results_root is None else results_root, search_dirs or [],
                            what="result file listed in --aggregates-json")
                metrics, reason = read_file_metrics(resolved, modality, N_TASKS[record["dataset"]])
                if metrics is None:
                    excluded.append({
                        "dataset": record["dataset"], "model": record["model"], "setting": record["setting"],
                        "strategy": record["strategy"], "ordering": record["ordering"], "modality": modality,
                        "reason": reason, "source": source,
                    })
                    continue
                kept_files.append(source)
                rows.append(metrics)
            payload["source_files"] = kept_files
            payload["count"] = len(kept_files)
            payload["available"] = bool(kept_files)
            payload["scalar_metrics"] = {
                "continual_average": scalar([row["continual_average"] for row in rows]),
                "backward_transfer": scalar([row["bwt_paper"] for row in rows]),
                "backward_transfer_pyclad": scalar([row["bwt_pyclad"] for row in rows]),
            }
            payload["metric_matrix"] = mean_matrix([row["matrix"] for row in rows])
    corrected.setdefault("metadata", {})["camera_ready_correction"] = (
        "Files with an incomplete ROC-AUC matrix or run_status != completed removed; "
        "backward_transfer recomputed with the paper's Eq. (bwt); "
        "backward_transfer_pyclad keeps pyclad's BackwardTransfer for comparison."
    )
    return corrected, excluded


def run_summary_builder(input_json: pathlib.Path, output_dir: pathlib.Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(SUMMARY_BUILDER),
            "--input-json", str(input_json),
            "--output-json", str(output_dir / "summary.json"),
            "--output-csv", str(output_dir / "summary.csv"),
            "--output-tex", str(output_dir / "summary.tex"),
            "--log-level", "WARNING",
        ],
        check=True,
    )
    with (output_dir / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def table_body(tex_path: pathlib.Path) -> list[str]:
    return [line.rstrip() for line in tex_path.read_text(encoding="utf-8").splitlines() if "&" in line]


# ---------------------------------------------------------------- MultiDataset rows

def multidataset_run_path(md_json: pathlib.Path, rel: str, results_root: pathlib.Path | None = None,
                          search_dirs: list[pathlib.Path] | None = None) -> pathlib.Path:
    """Paths in the selection are relative to the parent of its directory, or to --results-root."""
    path = md_json.parent.parent / rel
    if path.exists():
        return path
    return require(rel, REPO_ROOT if results_root is None else results_root, search_dirs or [],
                   what=f"MultiDataset run listed in {md_json}")


def multidataset_rows(md_json: pathlib.Path, results_root: pathlib.Path | None = None,
                      search_dirs: list[pathlib.Path] | None = None) -> dict:
    """Recompute the MultiDataset rows from the run files listed in the submitted aggregate.

    Within a method the run files (both backbones) are averaged, then the table
    reports mean and sample standard deviation across CFA, PaDiM and PaSTe.
    """
    payload = json.loads(md_json.read_text(encoding="utf-8"))
    out = {"cells": [], "legacy_check": []}
    for cell in payload["cells"]:
        by_method = {}
        for method in cell["methods"]:
            details = cell["method_details"][method]
            files = []
            for rel in details["paths"]:
                path = multidataset_run_path(md_json, rel, results_root, search_dirs)
                file_metrics = {}
                for modality in ("image", "pixel"):
                    metrics, reason = read_file_metrics(path, modality, None)
                    if metrics is None:
                        raise ValueError(f"{path}: {reason}")
                    file_metrics[modality] = metrics
                files.append(file_metrics)
            by_method[method] = {
                f"{modality}_{name}": statistics.mean(f[modality][key] for f in files)
                for modality in ("image", "pixel")
                for name, key in (("roc", "continual_average"), ("bwt", "bwt_paper"), ("bwt_pyclad", "bwt_pyclad"))
            }
            for metric in ("image_roc", "image_bwt", "pixel_roc", "pixel_bwt"):
                legacy_key = metric if metric.endswith("roc") else metric + "_pyclad"
                out["legacy_check"].append(abs(by_method[method][legacy_key] - details[metric]))
        metrics = {}
        for metric in ("image_roc", "image_bwt", "pixel_roc", "pixel_bwt"):
            values = [by_method[m][metric] for m in MD_METHODS if m in by_method]
            metrics[metric] = {"mean": statistics.mean(values), "std": statistics.stdev(values), "values_by_method": dict(zip(MD_METHODS, values))}
        out["cells"].append({"ordering": cell["order"], "variant": cell["variant"], "metrics": metrics})
    return out


# ---------------------------------------------------------------- LaTeX

def fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    return f"\\({mean:.3f}\\,{{\\scriptstyle\\pm\\,{std:.3f}}}\\)"


def cell_lookup(summary: dict) -> dict:
    lookup = {}
    for cell in summary["cells"]:
        key = (cell["dataset"], cell["ordering"], cell["modality"], cell["strategy_variant"])
        lookup[key] = cell["metrics"]
    return lookup


def render_table2(summary: dict, md: dict) -> str:
    lookup = cell_lookup(summary)
    lines = ["% Auto-generated by examples/clvad/camera_ready/rebuild_main_table.py", ""]
    for d_index, dataset in enumerate(DATASETS):
        for o_index, ordering in enumerate(ORDERINGS):
            for metric, label in (("roc", "ROC$\\uparrow$"), ("bwt", "BWT$\\uparrow$")):
                prefix_dataset = f"\\multirow{{6}}{{*}}{{{DATASET_LABELS[dataset]}}}" if (o_index == 0 and metric == "roc") else ""
                prefix_order = f"\\multirow{{2}}{{*}}{{{ORDERING_LABELS[ordering]}}}" if metric == "roc" else ""
                values = []
                for modality in ("image", "pixel"):
                    for variant in ("naive", "continual"):
                        m = lookup[(dataset, ordering, modality, variant)][metric]
                        values.append(fmt(m["mean"], m["std"]))
                lead = prefix_dataset if prefix_dataset else ""
                lines.append(f"{lead} & {prefix_order} & {label} & " + " & ".join(values) + " \\\\")
            if o_index < len(ORDERINGS) - 1:
                lines.append(" \\cmidrule(lr){4-7}")
        lines.append("\\cmidrule(lr){1-7}" if d_index < len(DATASETS) - 1 else "\\midrule")
    md_lookup = {(c["ordering"], c["variant"]): c["metrics"] for c in md["cells"]}
    for o_index, ordering in enumerate(MD_ORDERINGS):
        for metric, label in (("roc", "ROC$\\uparrow$"), ("bwt", "BWT$\\uparrow$")):
            prefix_dataset = "\\multirow{4}{*}{MultiDataset}" if (o_index == 0 and metric == "roc") else ""
            prefix_order = f"\\multirow{{2}}{{*}}{{{ORDERING_LABELS[ordering.replace('-', '_')]}}}" if metric == "roc" else ""
            values = []
            for modality in ("image", "pixel"):
                for variant in ("naive", "continual"):
                    m = md_lookup[(ordering, variant)][f"{modality}_{metric}"]
                    values.append(fmt(m["mean"], m["std"]))
            lines.append(f"{prefix_dataset} & {prefix_order} & {label} & " + " & ".join(values) + " \\\\")
        if o_index == 0:
            lines.append("\\cmidrule(lr){4-7}")
    return "\n".join(lines) + "\n"


def mark_changes(new_body: str, old_body: str, macro: str = "\\crnew") -> str:
    """Wrap every value cell that differs from the submitted table in `macro`."""
    out = []
    for new_line, old_line in zip(new_body.splitlines(), old_body.splitlines()):
        if "&" not in new_line:
            out.append(new_line)
            continue
        new_cells, old_cells = new_line.split(" & "), old_line.split(" & ")
        marked = []
        for index, (new_cell, old_cell) in enumerate(zip(new_cells, old_cells)):
            ending = " \\\\" if new_cell.endswith(" \\\\") else ""
            value = new_cell[: -len(ending)] if ending else new_cell
            old_value = old_cell[: -len(" \\\\")] if old_cell.endswith(" \\\\") else old_cell
            if index >= 3 and value != old_value:
                value = f"{macro}{{{value}}}"
            marked.append(value + ending)
        out.append(" & ".join(marked))
    return "\n".join(out) + "\n"


def text_numbers(summary: dict) -> dict:
    """Averages quoted in the Results Discussion: means of the single-dataset table entries."""
    lookup = cell_lookup(summary)
    out = {}
    for modality in ("image", "pixel"):
        for variant in ("naive", "continual"):
            for metric in ("roc", "bwt"):
                values = [lookup[(d, o, modality, variant)][metric]["mean"] for d in DATASETS for o in ORDERINGS]
                out[f"{modality}_{variant}_{metric}_all"] = statistics.mean(values)
                for o in ORDERINGS:
                    out[f"{modality}_{variant}_{metric}_{o}"] = statistics.mean(
                        lookup[(d, o, modality, variant)][metric]["mean"] for d in DATASETS
                    )
    return out


def main() -> int:
    args = parse_args()
    require_input(args.aggregates_json, "--aggregates-json")
    require_input(args.submitted_tex, "--submitted-tex")
    require_input(args.multidataset_json, "--multidataset-json")
    require_input(SUMMARY_BUILDER, "summary builder")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    legacy_summary = run_summary_builder(args.aggregates_json, out / "legacy")
    if table_body(out / "legacy" / "summary.tex") != table_body(args.submitted_tex):
        sys.exit("legacy reproduction does not match the submitted table body; refusing to continue")
    print("legacy single-dataset table reproduced exactly")

    aggregates = json.loads(args.aggregates_json.read_text(encoding="utf-8"))
    corrected, excluded = correct_aggregates(aggregates, args.results_root, args.search_dir)
    corrected_path = out / "article_model_aggregates_corrected.json"
    corrected_path.write_text(json.dumps(corrected, indent=1), encoding="utf-8")
    with (out / "excluded_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "model", "setting", "strategy", "ordering", "modality", "reason", "source"])
        writer.writeheader()
        writer.writerows(excluded)
    print(f"excluded {len(excluded)} result files")

    corrected_summary = run_summary_builder(corrected_path, out / "corrected")

    md = multidataset_rows(args.multidataset_json, args.results_root, args.search_dir)
    worst = max(md["legacy_check"])
    if worst > 1e-9:
        sys.exit(f"MultiDataset legacy values not reproduced (max diff {worst})")
    print("legacy MultiDataset method values reproduced exactly")
    (out / "multidataset_rows.json").write_text(json.dumps(md, indent=1), encoding="utf-8")

    new_body = render_table2(corrected_summary, md)
    (out / "table2_body.tex").write_text(new_body, encoding="utf-8")
    legacy_md = json.loads(args.multidataset_json.read_text(encoding="utf-8"))
    legacy_md = {"cells": [{"ordering": c["order"], "variant": c["variant"], "metrics": c["metrics"]} for c in legacy_md["cells"]]}
    (out / "table2_body_marked.tex").write_text(mark_changes(new_body, render_table2(legacy_summary, legacy_md)), encoding="utf-8")
    numbers = {"corrected": text_numbers(corrected_summary), "submitted": text_numbers(legacy_summary)}
    (out / "table2_text_numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
