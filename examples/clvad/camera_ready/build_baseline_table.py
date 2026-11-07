#!/usr/bin/env python3
"""Continual industrial AD baselines (UCAD, One-for-More, ReplayCAD) on the InCLAD scenarios.

One run per method, dataset and ordering is selected from the 2026-10-07 bundle
(04_baselines/) and from the later cluster exports, and the metrics are computed
the same way as Table 2: ContinualAverage of the ROC-AUC matrix, and BWT by the
paper's Eq. (bwt) (mean over i > j of R[i][j] - R[j][j]).

Selection, per (method, dataset, ordering):
  * only runs with the protocol of the bundled runs are eligible, decided from
    the recorded run_config.cli_args / model_config alone (directory names play
    no role): UCAD backbone vit_base_patch16_224*, One-for-More backbone
    resnet50, ReplayCAD backbone resnet18 with replaycad_compression_steps 18000
    (runs with 20000, 27000 or 30000 steps are excluded);
  * a completed run is preferred; otherwise the run with the most completed
    tasks; on a tie, a run of the bundle wins over an export run, and among
    the rest the first one found wins (inputs in the order given, files sorted
    by path within each input).
Runs that stopped early keep the tasks they finished; their entries are marked
as partial in the JSON output and in a LaTeX comment.

Inputs. --bundle-baselines and every --exports directory must directly contain
one sub-directory per method (ucad/, one_for_more/, replaycad/) with the run
JSON files below it. Table 3 and the per-ordering appendix table of the paper
were built from these inputs, with <exports> the export directory
below --results-root that holds the default --exports:
  --bundle-baselines <results-root>/InCLAD_paper_results_20261007/04_baselines
  --exports <exports>/inclad_export_*_20261022_* (the two exports of 2026-10-22)
            <exports>/replaycad_all_20261106/resumable-results
--table2-summary and --multidataset-rows are outputs of rebuild_main_table.py
(corrected/summary.json and multidataset_rows.json). ReplayCAD VisA
hard-to-easy is a partial run (7/12 tasks) in that build. Without --exports the default is every <exports>/inclad_export_*_2026*
directory, which does not include the ReplayCAD resumable-results export.

Outputs (in --output-dir): baselines.json, baselines.csv,
baseline_table_main.tex (averaged over orderings, with the CL mean of Table 2
for reference) and baseline_table_appendix.tex (per ordering, Table 2 layout).
The "source" field of each row is relative to --results-root when the file
lies below it, and absolute otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics

from rebuild_main_table import (
    BUNDLE,
    DATASET_LABELS,
    ORDERING_LABELS,
    REPO_ROOT,
    bwt_paper,
    continual_average,
    normalize_matrix,
)
from results_root import (
    BUNDLE_NAME,
    EXPORTS_NAME,
    OUTPUT_NAME,
    absolute,
    add_results_root_argument,
    require_input,
    results_root_from_argv,
    source_string,
)

METHODS = ["ucad", "one_for_more", "replaycad"]
# METHOD_LABELS = {"ucad": "UCAD", "one_for_more": "One-for-More", "replaycad": "ReplayCAD"}
# The paper names One-for-More by its method name, CDAD (Li et al., CVPR 2025).
METHOD_LABELS = {"ucad": "UCAD", "one_for_more": "CDAD", "replaycad": "ReplayCAD"}
SCENARIOS = ["mvtec", "visa", "btech", "mpdd", "dagm", "inclad-md"]
SCENARIO_LABELS = {**DATASET_LABELS, "inclad-md": "MultiDataset"}
ORDERINGS = {"inclad-md": ["easy_to_hard", "hard_to_easy"]}
DEFAULT_ORDERINGS = ["easy_to_hard", "hard_to_easy", "random"]
N_TASKS = {"mvtec": 15, "visa": 12, "btech": 3, "mpdd": 6, "dagm": 10, "inclad-md": 27}
PAYLOAD_KEYS = {
    "image": "concept_metric_callback_ROC-AUC",
    "pixel": "pixel_concept_metric_callback_Pixel-ROC-AUC",
}
SKIP_NAMES = {"metadata.json", "manifest.json", "replaycad_config.json"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = results_root_from_argv(argv)
    base = root / OUTPUT_NAME
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_results_root_argument(parser)
    parser.add_argument("--bundle-baselines", type=pathlib.Path, default=root / BUNDLE_NAME / "04_baselines",
                        help="Directory with ucad/, one_for_more/ and replaycad/ run sub-directories.")
    parser.add_argument(
        "--exports",
        type=pathlib.Path,
        nargs="*",
        default=sorted((root / EXPORTS_NAME).glob("inclad_export_*_2026*")),
        help="Further result directories, each directly containing <method>/ sub-directories "
             "(ucad/, one_for_more/, replaycad/) of run JSON files, e.g. an extracted cluster "
             "export or a ReplayCAD resumable-results directory.",
    )
    parser.add_argument(
        "--table2-summary",
        type=pathlib.Path,
        default=base / "table2" / "corrected" / "summary.json",
        help="corrected/summary.json written by rebuild_main_table.py.",
    )
    parser.add_argument(
        "--multidataset-rows",
        type=pathlib.Path,
        default=base / "table2" / "multidataset_rows.json",
        help="multidataset_rows.json written by rebuild_main_table.py.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=base / "baselines")
    return parser.parse_args(argv)


def eligible(method: str, data: dict) -> bool:
    run_config = data.get("run_config") or {}
    cli = run_config.get("cli_args") or {}
    model_config = run_config.get("model_config") or {}
    backbone = cli.get("backbone") or model_config.get("backbone_name") or model_config.get("backbone")
    if method == "ucad":
        return str(backbone).startswith("vit_base_patch16_224")
    if method == "one_for_more":
        return backbone == "resnet50"
    if method == "replaycad":
        return backbone == "resnet18" and cli.get("replaycad_compression_steps") == 18000
    return False


def cell_of(data: dict) -> tuple[str, str] | None:
    run_config = data.get("run_config") or {}
    cli = run_config.get("cli_args") or {}
    benchmark = str(run_config.get("benchmark") or cli.get("benchmark") or "").lower()
    if benchmark in ("inclad-md", "inclad_md") or "multi" in benchmark:
        benchmark = "inclad-md"
    ordering = cli.get("ordering_mode") or (run_config.get("ordering_metadata") or {}).get("ordering_mode")
    if not ordering and (run_config.get("shuffle_concepts") or cli.get("shuffle_concepts")):
        ordering = "random"
    if benchmark not in N_TASKS or not ordering:
        return None
    return benchmark, ordering


def run_metrics(data: dict) -> dict | None:
    out = {}
    for modality, key in PAYLOAD_KEYS.items():
        payload = data.get(key)
        if not isinstance(payload, dict):
            return None
        order = [str(item) for item in payload.get("concepts_order", [])]
        matrix = normalize_matrix(payload.get("metric_matrix"), order)
        if not matrix:
            return None
        out[f"{modality}_roc"] = continual_average(matrix)
        out[f"{modality}_bwt"] = bwt_paper(matrix)
        out["tasks_done"] = len(matrix)
    return out


def candidates(method: str, bundle: pathlib.Path, exports: list[pathlib.Path]):
    roots = [bundle / method] + [export / method for export in exports]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            if path.name in SKIP_NAMES:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict) or not eligible(method, data):
                continue
            cell = cell_of(data)
            metrics = run_metrics(data)
            if cell is None or metrics is None:
                continue
            state = (data.get("run_status") or {}).get("state")
            yield cell, path, state, metrics


def select_runs(bundle: pathlib.Path, exports: list[pathlib.Path],
                results_root: pathlib.Path | None = None) -> list[dict]:
    root = REPO_ROOT if results_root is None else results_root
    chosen: dict[tuple[str, str, str], dict] = {}
    for method in METHODS:
        for (scenario, ordering), path, state, metrics in candidates(method, bundle, exports):
            complete = state == "completed" and metrics["tasks_done"] == N_TASKS[scenario]
            rank = (complete, metrics["tasks_done"], path.is_relative_to(bundle))
            key = (method, scenario, ordering)
            if key not in chosen or rank > chosen[key]["_rank"]:
                chosen[key] = {
                    "method": method, "scenario": scenario, "ordering": ordering,
                    "complete": complete, "tasks_done": metrics["tasks_done"], "tasks_total": N_TASKS[scenario],
                    # "state": state, "source": str(path.relative_to(REPO_ROOT)), "_rank": rank,
                    "state": state, "source": source_string(path, [root]), "_rank": rank,
                    **{k: v for k, v in metrics.items() if k != "tasks_done"},
                }
    rows = []
    for method in METHODS:
        for scenario in SCENARIOS:
            for ordering in ORDERINGS.get(scenario, DEFAULT_ORDERINGS):
                row = chosen.get((method, scenario, ordering))
                if row is None:
                    row = {"method": method, "scenario": scenario, "ordering": ordering, "complete": False,
                           "tasks_done": 0, "tasks_total": N_TASKS[scenario], "state": None, "source": None,
                           "image_roc": None, "image_bwt": None, "pixel_roc": None, "pixel_bwt": None}
                row.pop("_rank", None)
                rows.append(row)
    return rows


def cl_reference(table2_summary: pathlib.Path, md_rows: pathlib.Path) -> dict:
    """CL mean of Table 2 per scenario and ordering (image/pixel ROC and BWT)."""
    summary = json.loads(table2_summary.read_text(encoding="utf-8"))
    ref = {}
    for cell in summary["cells"]:
        if cell["strategy_variant"] != "continual":
            continue
        for metric in ("roc", "bwt"):
            ref[(cell["dataset"], cell["ordering"], f"{cell['modality']}_{metric}")] = cell["metrics"][metric]["mean"]
    md = json.loads(md_rows.read_text(encoding="utf-8"))
    for cell in md["cells"]:
        if cell["variant"] != "continual":
            continue
        for name, payload in cell["metrics"].items():
            ref[("inclad-md", cell["ordering"].replace("-", "_"), name)] = payload["mean"]
    return ref


def fmt(value: float | None) -> str:
    if value is None:
        return "--"
    text = f"{value:.3f}"
    return "0.000" if text == "-0.000" else text


def mean_over_orderings(rows: list[dict], method: str, scenario: str, metric: str) -> float | None:
    values = [r[metric] for r in rows if r["method"] == method and r["scenario"] == scenario and r[metric] is not None]
    return statistics.mean(values) if values else None


def render_main(rows: list[dict], ref: dict) -> str:
    lines = ["% Auto-generated by examples/clvad/camera_ready/build_baseline_table.py",
             "% Values: mean over the orderings of each scenario. CL avg.: mean of the CL entries of Table 2.", ""]
    partial = [r for r in rows if r["source"] and not r["complete"]]
    for r in partial:
        lines.append(f"% partial: {r['method']} {r['scenario']} {r['ordering']} ({r['tasks_done']}/{r['tasks_total']} tasks)")
    lines.append("")
    for s_index, scenario in enumerate(SCENARIOS):
        for metric, label in (("roc", "ROC$\\uparrow$"), ("bwt", "BWT$\\uparrow$")):
            cells = []
            for modality in ("image", "pixel"):
                key = f"{modality}_{metric}"
                for method in METHODS:
                    cells.append(fmt(mean_over_orderings(rows, method, scenario, key)))
                ref_values = [ref[(scenario, o, key)] for o in ORDERINGS.get(scenario, DEFAULT_ORDERINGS) if (scenario, o, key) in ref]
                cells.append(fmt(statistics.mean(ref_values) if ref_values else None))
            lead = f"\\multirow{{2}}{{*}}{{{SCENARIO_LABELS[scenario]}}}" if metric == "roc" else ""
            lines.append(f"{lead} & {label} & " + " & ".join(cells) + " \\\\")
        if s_index < len(SCENARIOS) - 1:
            lines.append("\\cmidrule(lr){1-10}")
    return "\n".join(lines) + "\n"


def render_appendix(rows: list[dict]) -> str:
    lines = ["% Auto-generated by examples/clvad/camera_ready/build_baseline_table.py",
             "% One run per entry. Columns: UCAD, CDAD (One-for-More), ReplayCAD (image), then the same three (pixel).", ""]
    lookup = {(r["method"], r["scenario"], r["ordering"]): r for r in rows}
    for s_index, scenario in enumerate(SCENARIOS):
        orderings = ORDERINGS.get(scenario, DEFAULT_ORDERINGS)
        for o_index, ordering in enumerate(orderings):
            for metric, label in (("roc", "ROC$\\uparrow$"), ("bwt", "BWT$\\uparrow$")):
                cells = []
                for modality in ("image", "pixel"):
                    for method in METHODS:
                        cells.append(fmt(lookup[(method, scenario, ordering)][f"{modality}_{metric}"]))
                lead_s = f"\\multirow{{{2 * len(orderings)}}}{{*}}{{{SCENARIO_LABELS[scenario]}}}" if (o_index == 0 and metric == "roc") else ""
                lead_o = f"\\multirow{{2}}{{*}}{{{ORDERING_LABELS[ordering]}}}" if metric == "roc" else ""
                partial_note = ""
                if metric == "roc":
                    notes = [f"{METHOD_LABELS[m]} {lookup[(m, scenario, ordering)]['tasks_done']}/{lookup[(m, scenario, ordering)]['tasks_total']}"
                             for m in METHODS if lookup[(m, scenario, ordering)]["source"] and not lookup[(m, scenario, ordering)]["complete"]]
                    partial_note = f" % partial: {', '.join(notes)}" if notes else ""
                lines.append(f"{lead_s} & {lead_o} & {label} & " + " & ".join(cells) + " \\\\" + partial_note)
            if o_index < len(orderings) - 1:
                lines.append(" \\cmidrule(lr){4-9}")
        if s_index < len(SCENARIOS) - 1:
            lines.append("\\cmidrule(lr){1-9}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    require_input(args.bundle_baselines, "--bundle-baselines")
    for export in args.exports:
        require_input(export, "--exports")
    require_input(args.table2_summary, "--table2-summary")
    require_input(args.multidataset_rows, "--multidataset-rows")
    if not any((export / "replaycad").is_dir() for export in args.exports):
        print("note: no --exports directory has a replaycad/ sub-directory; ReplayCAD runs come from the bundle only")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = select_runs(absolute(args.bundle_baselines), [absolute(e) for e in args.exports], args.results_root)
    if not any(row["source"] for row in rows):
        raise SystemExit("no eligible baseline runs found in --bundle-baselines / --exports")
    ref = cl_reference(args.table2_summary, args.multidataset_rows)
    (out / "baselines.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    with (out / "baselines.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "baseline_table_main.tex").write_text(render_main(rows, ref), encoding="utf-8")
    (out / "baseline_table_appendix.tex").write_text(render_appendix(rows), encoding="utf-8")
    for row in rows:
        status = "complete" if row["complete"] else f"{row['tasks_done']}/{row['tasks_total']}"
        print(f"{row['method']:12s} {row['scenario']:9s} {row['ordering']:12s} {status:8s} "
              f"img {fmt(row['image_roc'])} {fmt(row['image_bwt'])} pix {fmt(row['pixel_roc'])} {fmt(row['pixel_bwt'])}  {row['source'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
