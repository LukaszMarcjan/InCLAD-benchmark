#!/usr/bin/env python3
"""Grouped-component (step-schedule) scenarios: appendix tables for the camera-ready paper.

Each training step may introduce several components; the metric matrix is
rectangular (steps x components) and first_seen_step maps a component to the
step that introduced it. From that matrix, per run and level (image, pixel):

  CA    continual ROC-AUC: mean of R[s][c] over steps s >= first_seen(c)
  BWT   mean of R[s][c] - R[first_seen(c)][c] over s > first_seen(c)
        (the rectangular counterpart of Eq. (bwt))
  RFM   rectangular forgetting: mean over components introduced before the last
        step of max_{first_seen <= s < T-1} R[s][c] - R[T-1][c]
  RNTA  new-task acquisition: mean of R[first_seen(c)][c]

One run per cell of examples/clvad/grouped_grid/manifest.tsv is taken from the
2026-10-07 bundle (--bundle-step-grid) and the cluster exports (--exports);
only completed runs whose image and pixel rectangular matrices are present
count. Seeds: E2H and H2E cells use seed 191664963 (master seed 42); every
Alph. cell uses seed 33158374 (master seed 123), because no "rand" run exists
with seed 191664963. A run with seed 191664963 is preferred over one with
another seed, which only decides E2H/H2E cells that also have a 33158374 run.

Duplicate completed runs with the same seed rank equal; the first one found is
used. The search order is: the bundle directory, then each --exports directory
in the order given (with --sort-exports: sorted by directory name), and within
an export results_step_v2_grid before results_step_v2_grid_repair; files are
sorted by name within a directory. The camera-ready tables were built with the
exports in name order, which is also the order of the default glob:
  --bundle-step-grid <results-root>/InCLAD_paper_results_20261007/05_step_schedule_grid
  --exports the two cluster exports of 2026-10-22 (inclad_export_*_20261022_*,
            the default --exports), in name order;
this selects 1692 runs. The order
matters for cells with several completed same-seed runs whose metrics differ,
e.g. MVTec WR50 E2H STFPM replay 14+1, where it picks the run from the first
export's results_step_v2_grid_repair. Cells with such ties are printed.

Aggregation follows Table 2: within a method, configurations are averaged first,
then the tables report mean and sample standard deviation across methods.
The ordering directory "rand" holds runs in the dataset's own (alphabetical)
category order, so it is labelled Alphabetical.

Outputs in --output-dir: grouped_runs.csv and five LaTeX bodies
(grouped_main.tex, grouped_summary.tex, grouped_by_ordering.tex, grouped_methods.tex,
grouped_depth.tex). grouped_main.tex has the 9 columns of the paper's table
(dataset, then image and pixel ROC/BWT for NCL and CL); --main-with-schedules
adds the list of schedules as a second column. The "source" column of
grouped_runs.csv is relative to --results-root when the file lies below it,
and absolute otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

from rebuild_main_table import BUNDLE, REPO_ROOT
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

PREFERRED_SEED = 191664963

KEYS = {"image": "rectangular_concept_metric_callback_ROC-AUC",
        "pixel": "rectangular_pixel_concept_metric_callback_Pixel-ROC-AUC"}
DATASETS = ["mvtec", "visa", "btech", "mpdd", "dagm"]
LABEL = {"mvtec": "MVTec", "visa": "VisA", "btech": "BTAD", "mpdd": "MPDD", "dagm": "DAGM"}
ORDERINGS = ["eth", "hte", "rand"]
ORDER_LABEL = {"eth": "E2H", "hte": "H2E", "rand": "Alph."}
BACKBONES = ["rn18", "wr50"]
BB_LABEL = {"rn18": "RN18", "wr50": "WR50"}
METHODS = ["cfa", "efficientad", "fastflow", "ganomaly", "padim", "paste", "patchcore", "rd4ad", "stfpm"]
METHOD_LABEL = {"cfa": "CFA", "efficientad": "EfficientAD", "fastflow": "FastFlow", "ganomaly": "GANomaly",
                "padim": "PaDiM", "paste": "PaSTe", "patchcore": "PatchCore", "rd4ad": "RD4AD", "stfpm": "STFPM"}


def schedule_label(schedule: str) -> str:
    """'10-1x5' -> '10+1x5', '3x5' stays."""
    return schedule.replace("-", "+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = results_root_from_argv(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_results_root_argument(parser)
    parser.add_argument("--manifest", type=pathlib.Path, default=REPO_ROOT / "examples/clvad/grouped_grid/manifest.tsv")
    parser.add_argument("--bundle-step-grid", type=pathlib.Path, default=root / BUNDLE_NAME / "05_step_schedule_grid",
                        help="Bundle directory with <dataset>/<backbone>/<ordering>/<model>/<strategy>/<schedule>/ runs; "
                             "searched before the exports.")
    parser.add_argument("--exports", type=pathlib.Path, nargs="*",
                        default=sorted((root / EXPORTS_NAME).glob("inclad_export_*_2026*")),
                        help="Cluster exports, each containing step_schedule_grid/pyCLAD/results_step_v2_grid[_repair]/; "
                             "searched in the order given.")
    parser.add_argument("--sort-exports", action="store_true",
                        help="Search the exports in directory-name order instead of the order given.")
    parser.add_argument("--main-with-schedules", action="store_true",
                        help="Add the schedule list as a second column of grouped_main.tex (not in the paper's table).")
    parser.add_argument("--output-dir", type=pathlib.Path, default=root / OUTPUT_NAME / "grouped")
    return parser.parse_args(argv)


def nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def mean(values):
    values = [v for v in values if not nan(v)]
    return sum(values) / len(values) if values else None


def metrics(block: dict) -> dict | None:
    rows = block.get("train_order") or list(block["metric_matrix"])
    cols = block.get("test_order")
    first = block.get("first_seen_step") or {}
    matrix = block["metric_matrix"]
    try:
        R = [[matrix[r][c] for c in cols] for r in rows]
        f = [int(first[c]) for c in cols]
    except (KeyError, TypeError):
        return None
    T = len(R)
    ca = mean([R[s][j] for j in range(len(cols)) for s in range(f[j], T)])
    bwt = mean([R[s][j] - R[f[j]][j] for j in range(len(cols)) for s in range(f[j] + 1, T)])
    rfm_values = []
    for j in range(len(cols)):
        if f[j] < T - 1:
            history = [R[s][j] for s in range(f[j], T - 1) if not nan(R[s][j])]
            if history and not nan(R[T - 1][j]):
                rfm_values.append(max(history) - R[T - 1][j])
    rnta = mean([R[f[j]][j] for j in range(len(cols))])
    return {"ca": ca, "bwt": 0.0 if bwt is None else bwt, "rfm": mean(rfm_values) or 0.0, "rnta": rnta, "steps": T}


def candidate_dirs(rel: str, exports: list[pathlib.Path], bundle_step_grid: pathlib.Path | None = None) -> list[pathlib.Path]:
    # dirs = [BUNDLE / "05_step_schedule_grid" / rel]
    dirs = [(BUNDLE / "05_step_schedule_grid" if bundle_step_grid is None else bundle_step_grid) / rel]
    for export in exports:
        for sub in ("results_step_v2_grid", "results_step_v2_grid_repair"):
            dirs.append(export / "step_schedule_grid" / "pyCLAD" / sub / rel)
    return dirs


def select(rel: str, exports: list[pathlib.Path], bundle_step_grid: pathlib.Path | None = None,
           ties: list | None = None) -> tuple[pathlib.Path, dict] | None:
    best = None
    equal = []
    for directory in candidate_dirs(rel, exports, bundle_step_grid):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".progress.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if (data.get("run_status") or {}).get("state") != "completed":
                continue
            result = {}
            for level, key in KEYS.items():
                block = data.get(key)
                m = metrics(block) if isinstance(block, dict) else None
                if m is None:
                    break
                result[level] = m
            else:
                rank = ((data.get("run_config") or {}).get("seed") == PREFERRED_SEED,)
                if best is None or rank > best[0]:
                    best = (rank, path, result)
                    equal = [(path, result)]
                elif rank == best[0]:
                    equal.append((path, result))
    if ties is not None and len(equal) > 1:
        ties.append((rel, equal))
    return (best[1], best[2]) if best else None


def collect(args, ties: list | None = None) -> list[dict]:
    rows = []
    exports = list(getattr(args, "exports", []))
    if getattr(args, "sort_exports", False):
        exports = sorted(exports, key=lambda p: p.name)
    bundle_step_grid = getattr(args, "bundle_step_grid", None)
    results_root = getattr(args, "results_root", None) or REPO_ROOT
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        dataset, backbone, model, strategy, schedule, ordering = line.split("\t")
        rel = f"{dataset}/{backbone}/{ordering}/{model}/{strategy}/{schedule}"
        chosen = select(rel, exports, bundle_step_grid, ties)
        if chosen is None:
            continue
        path, result = chosen
        rows.append({
            "dataset": dataset, "backbone": backbone, "ordering": ordering, "model": model,
            "variant": "naive" if strategy == "naive" else "continual", "schedule": schedule,
            "steps": result["image"]["steps"],
            **{f"{level}_{name}": result[level][name] for level in KEYS for name in ("ca", "bwt", "rfm", "rnta")},
            # "source": str(path.relative_to(REPO_ROOT)),
            "source": source_string(path, [results_root]),
        })
    return rows


def model_first(rows, keys, value):
    """Average within a method over the given rows, then mean/sample-std across methods."""
    by_method = defaultdict(list)
    for r in rows:
        if all(r[k] == v for k, v in keys.items()) and not nan(r[value]):
            by_method[r["model"]].append(r[value])
    per_method = [statistics.mean(v) for v in by_method.values()]
    if not per_method:
        return None, None
    return statistics.mean(per_method), (statistics.stdev(per_method) if len(per_method) > 1 else 0.0)


def pm(m, s, scale=100):
    if m is None:
        return "--"
    return f"\\({m * scale:.1f}\\,{{\\scriptstyle\\pm\\,{s * scale:.1f}}}\\)"


def num(m, scale=100):
    return "--" if m is None else f"{m * scale:.1f}"


def schedules_of(rows, dataset):
    seen = []
    for r in rows:
        if r["dataset"] == dataset and r["schedule"] not in seen:
            seen.append(r["schedule"])
    return seen


def render_summary(rows) -> str:
    lines = ["% Auto-generated by examples/clvad/camera_ready/build_grouped_tables.py",
             "% CA and BWT in percent; mean +- sample std across methods (model-first over backbones and orderings).", ""]
    for d_index, dataset in enumerate(DATASETS):
        schedules = schedules_of(rows, dataset)
        for s_index, schedule in enumerate(schedules):
            # for metric, label in (("ca", "CROC$\\uparrow$"), ("bwt", "BWT$\\uparrow$")):
            # The paper labels continual ROC-AUC as "ROC" in every table.
            for metric, label in (("ca", "ROC$\\uparrow$"), ("bwt", "BWT$\\uparrow$")):
                cells = []
                for level in ("image", "pixel"):
                    for variant in ("naive", "continual"):
                        cells.append(pm(*model_first(rows, {"dataset": dataset, "schedule": schedule, "variant": variant}, f"{level}_{metric}")))
                lead_d = f"\\multirow{{{2 * len(schedules)}}}{{*}}{{{LABEL[dataset]}}}" if (s_index == 0 and metric == "ca") else ""
                lead_s = f"\\multirow{{2}}{{*}}{{{schedule_label(schedule)}}}" if metric == "ca" else ""
                lines.append(f"{lead_d} & {lead_s} & {label} & " + " & ".join(cells) + " \\\\")
            if s_index < len(schedules) - 1:
                lines.append(" \\cmidrule(lr){4-7}")
        lines.append("\\cmidrule(lr){1-7}" if d_index < len(DATASETS) - 1 else "")
    return "\n".join(lines) + "\n"


def render_main(rows, with_schedules: bool = False) -> str:
    """Main-text summary: one row per dataset, averaged over schedules, backbones and orderings.

    The default layout is the paper's 9-column table; with_schedules adds the
    schedule list as a second column.
    """
    lines = ["% Auto-generated by examples/clvad/camera_ready/build_grouped_tables.py",
             "% Per dataset: ROC (continual ROC-AUC) and BWT (percent), NCL and CL, mean +- sample std across methods.", ""]
    for dataset in DATASETS:
        schedules = ", ".join(schedule_label(s) for s in schedules_of(rows, dataset))
        cells = []
        for level in ("image", "pixel"):
            for metric in ("ca", "bwt"):
                for variant in ("naive", "continual"):
                    m, _ = model_first(rows, {"dataset": dataset, "variant": variant}, f"{level}_{metric}")
                    cells.append(num(m))
        if with_schedules:
            lines.append(f"{LABEL[dataset]} & {schedules} & " + " & ".join(cells) + " \\\\")
        else:
            lines.append(f"{LABEL[dataset]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def render_by_ordering(rows) -> str:
    lines = ["% Auto-generated by examples/clvad/camera_ready/build_grouped_tables.py",
             "% CL continual ROC-AUC (percent), mean across methods; columns E2H, H2E, Alph. for image then pixel.", ""]
    for d_index, dataset in enumerate(DATASETS):
        schedules = schedules_of(rows, dataset)
        backbones = [b for b in BACKBONES if any(r["dataset"] == dataset and r["backbone"] == b for r in rows)]
        first = True
        for backbone in backbones:
            for schedule in schedules:
                cells = []
                for level in ("image", "pixel"):
                    for ordering in ORDERINGS:
                        m, _ = model_first(rows, {"dataset": dataset, "backbone": backbone, "schedule": schedule,
                                                  "ordering": ordering, "variant": "continual"}, f"{level}_ca")
                        cells.append(num(m))
                lead = f"\\multirow{{{len(backbones) * len(schedules)}}}{{*}}{{{LABEL[dataset]}}}" if first else ""
                first = False
                lines.append(f"{lead} & {BB_LABEL[backbone]} & {schedule_label(schedule)} & " + " & ".join(cells) + " \\\\")
        if d_index < len(DATASETS) - 1:
            lines.append("\\cmidrule(lr){1-9}")
    return "\n".join(lines) + "\n"


def render_methods(rows) -> str:
    lines = ["% Auto-generated by examples/clvad/camera_ready/build_grouped_tables.py",
             "% Paired NCL/CL configurations (same dataset, backbone, ordering, schedule); percent.", ""]
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["dataset"], r["backbone"], r["ordering"], r["schedule"], r["model"])][r["variant"]] = r
    for method in METHODS:
        matched = [p for k, p in pairs.items() if k[4] == method and "naive" in p and "continual" in p]
        if not matched:
            continue
        cells = [str(len(matched))]
        for key in ("image_ca", "pixel_ca", "image_bwt", "pixel_bwt"):
            n = statistics.mean(p["naive"][key] for p in matched)
            c = statistics.mean(p["continual"][key] for p in matched)
            cells.append(f"{n * 100:.1f} $\\rightarrow$ {c * 100:.1f}")
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def render_depth(rows) -> str:
    lines = ["% Auto-generated by examples/clvad/camera_ready/build_grouped_tables.py",
             "% Image-level RFM and RNTA (percent), mean across methods (model-first).", ""]
    for d_index, dataset in enumerate(DATASETS):
        schedules = schedules_of(rows, dataset)
        for s_index, schedule in enumerate(schedules):
            steps = next(r["steps"] for r in rows if r["dataset"] == dataset and r["schedule"] == schedule)
            cells = [str(steps)]
            for metric in ("rfm", "rnta"):
                for variant in ("naive", "continual"):
                    m, _ = model_first(rows, {"dataset": dataset, "schedule": schedule, "variant": variant}, f"image_{metric}")
                    cells.append(num(m))
            lead = f"\\multirow{{{len(schedules)}}}{{*}}{{{LABEL[dataset]}}}" if s_index == 0 else ""
            lines.append(f"{lead} & {schedule_label(schedule)} & " + " & ".join(cells) + " \\\\")
        if d_index < len(DATASETS) - 1:
            lines.append("\\cmidrule(lr){1-7}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    require_input(args.manifest, "--manifest")
    if not args.bundle_step_grid.is_dir():
        print(f"warning: --bundle-step-grid {args.bundle_step_grid} does not exist; using the exports only", file=sys.stderr)
    for export in args.exports:
        require_input(export, "--exports")
    args.bundle_step_grid = absolute(args.bundle_step_grid)
    args.exports = [absolute(e) for e in args.exports]
    if not args.sort_exports and args.exports != sorted(args.exports, key=lambda p: p.name):
        print("note: --exports are not in name order; the camera-ready tables used name order "
              "(pass --sort-exports to reproduce them)", file=sys.stderr)
    ties: list = []
    rows = collect(args, ties)
    if not rows:
        searched = [str(args.bundle_step_grid)] + [str(e) for e in args.exports]
        raise SystemExit("no completed grouped runs found for the manifest cells; searched: " + ", ".join(searched)
                         + " (pass --results-root, --bundle-step-grid and --exports)")
    differing = [(rel, equal) for rel, equal in ties
                 if len({json.dumps(result, sort_keys=True) for _, result in equal}) > 1]
    print(f"{len(ties)} cells have more than one completed run with the preferred seed rank; "
          f"in {len(differing)} of them the metrics differ (the first run in search order is used):")
    for rel, equal in differing:
        print(f"  {rel}: " + ", ".join(f"{source_string(path, [args.results_root])} "
                                         f"(image CA {result['image']['ca']:.5f})" for path, result in equal))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "grouped_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "grouped_main.tex").write_text(render_main(rows, args.main_with_schedules), encoding="utf-8")
    (out / "grouped_summary.tex").write_text(render_summary(rows), encoding="utf-8")
    (out / "grouped_by_ordering.tex").write_text(render_by_ordering(rows), encoding="utf-8")
    (out / "grouped_methods.tex").write_text(render_methods(rows), encoding="utf-8")
    (out / "grouped_depth.tex").write_text(render_depth(rows), encoding="utf-8")
    print(f"{len(rows)} grouped runs used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
