"""
Build appendix-style empirical time-complexity tables from NeurIPS result JSONs.

The script reads canonical experiment slots from `article_model_aggregates.json`,
matches them to the per-run result JSONs of the main grid (the
`01_main_table_480/` folder of the results bundle), extracts train/eval
wall-clock totals, and aggregates them across orderings.

Reproduce the paper's execution-time tables (App. "Empirical Time-Complexity";
only the wide_resnet50_2 and special tables appear in the paper):

    python examples/visual_models/neurips_utilities/build_time_complexity_tables.py \\
        --bundle-dir <results_root>/InCLAD_paper_results_20261007/01_main_table_480 \\
        --output-dir <out_dir> \\
        --tables wide_resnet50_2 special

--bundle-dir sets the defaults of --aggregate-json and --results-root; the
default bundle location is <repo>/InCLAD_paper_results_20261007/01_main_table_480
and the default output directory <repo>/InCLAD_camera_ready_20261024/time_complexity,
the same layout the camera-ready table builders in examples/clvad/camera_ready use.

Aggregation rule per slot:
- if a combined `eval-both` JSON exists, use its `train_time_total` and `eval_time_total`
- otherwise, if both `eval-image` and `eval-pixel` JSONs exist, sum their train/eval totals
- incomplete slots are skipped

Outputs:
1. slot-level CSV
2. grouped summary CSV
3. LaTeX tables grouped by setting/backbone family
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = REPO_ROOT / "InCLAD_paper_results_20261007" / "01_main_table_480"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "InCLAD_camera_ready_20261024" / "time_complexity"

COMPLETE_STATUSES = {"complete_both_json", "complete_split_jsons"}
BACKBONE_TABLE_ORDER = ["resnet18", "wide_resnet50_2", "special"]
DATASET_ORDER = ["mvtec", "visa", "btech", "mpdd", "dagm"]
DATASET_LABELS = {
    "mvtec": "MVTec",
    "visa": "VisA",
    "btech": "BTAD",
    "mpdd": "MPDD",
    "dagm": "DAGM",
}
MODEL_ORDER = [
    "cfa",
    "padim",
    "patchcore",
    "fastflow",
    "paste",
    "rd4ad",
    "stfpm",
    "ganomaly",
    "efficientad",
]
MODEL_LABELS = {
    "cfa": "CFA",
    "padim": "PaDiM",
    "patchcore": "PatchCore",
    "fastflow": "FastFlow",
    "paste": "PaSTe",
    "rd4ad": "RD4AD",
    "stfpm": "STFPM",
    "ganomaly": "GANomaly",
    "efficientad": "EfficientAD",
}
SETTING_GROUPS = {
    "resnet18": {"resnet18"},
    "wide_resnet50_2": {"wide_resnet50_2"},
    "special": {"default", "small"},
}
SETTING_GROUP_CAPTIONS = {
    "resnet18": "Execution times (train and eval separately, mean $\\pm$ population std across available orderings) for resnet18-based models. Continual denotes CL for CFA, PaDiM, and PatchCore, and replay for the remaining models. When image and pixel results were produced as separate runs, their times are summed to reflect the total wall-clock cost of obtaining both modalities.",
    "wide_resnet50_2": "Execution times (train and eval separately, mean $\\pm$ population std across available orderings) for wide\\_resnet50\\_2-based models. Continual denotes CL for CFA, PaDiM, and PatchCore, and replay for the remaining models. When image and pixel results were produced as separate runs, their times are summed to reflect the total wall-clock cost of obtaining both modalities.",
    "special": "Execution times (train and eval separately, mean $\\pm$ population std across available orderings) for models without the two-backbone setup: GANomaly (default) and EfficientAD (small). When image and pixel results were produced as separate runs, their times are summed to reflect the total wall-clock cost of obtaining both modalities.",
}
SETTING_GROUP_LABELS = {
    "resnet18": "resnet18",
    "wide_resnet50_2": "wide\\_resnet50\\_2",
    "special": "default/small",
}

LOCAL_RESULT_PATTERN = re.compile(
    r"^model-(?P<model>.+?)__strategy-(?P<strategy>.+?)__setting-(?P<setting>.+?)__ordering-(?P<ordering>.+?)__eval-(?P<eval_level>both|image|pixel)__src-.+\.json$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build empirical time-complexity CSV/LaTeX tables from canonical NeurIPS results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Original defaults (paths relative to the working directory that are not part
    # of the repository):
    #   --aggregate-json     ordering_files/neurips_results/article_model_aggregates.json
    #   --results-root       ordering_files/neurips_results
    #   --output-slot-csv    ordering_files/neurips_results/time_complexity_slots.csv
    #   --output-summary-csv ordering_files/neurips_results/time_complexity_summary.csv
    #   --output-tex         ordering_files/neurips_results/time_complexity_tables.tex
    # They can still be passed explicitly.
    parser.add_argument(
        "--bundle-dir",
        type=pathlib.Path,
        default=DEFAULT_BUNDLE_DIR,
        help="Main-grid results folder (01_main_table_480). Default for --aggregate-json "
        "(<bundle-dir>/article_model_aggregates.json) and --results-root (<bundle-dir>).",
    )
    parser.add_argument(
        "--aggregate-json",
        type=pathlib.Path,
        default=None,
        help="Canonical aggregate JSON with the 480 experiment slots (default: <bundle-dir>/article_model_aggregates.json).",
    )
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=None,
        help="Directory containing the raw per-run result JSONs (default: <bundle-dir>).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Default directory for the three outputs below.",
    )
    parser.add_argument(
        "--output-slot-csv",
        type=pathlib.Path,
        default=None,
        help="Where the slot-level timing CSV should be written (default: <output-dir>/time_complexity_slots.csv).",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=pathlib.Path,
        default=None,
        help="Where the grouped timing summary CSV should be written (default: <output-dir>/time_complexity_summary.csv).",
    )
    parser.add_argument(
        "--output-tex",
        type=pathlib.Path,
        default=None,
        help="Where the LaTeX tables should be written (default: <output-dir>/time_complexity_tables.tex).",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=BACKBONE_TABLE_ORDER,
        default=list(BACKBONE_TABLE_ORDER),
        help="LaTeX tables to write. The paper uses wide_resnet50_2 and special; resnet18 is extra.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    if args.aggregate_json is None:
        args.aggregate_json = args.bundle_dir / "article_model_aggregates.json"
    if args.results_root is None:
        args.results_root = args.bundle_dir
    if args.output_slot_csv is None:
        args.output_slot_csv = args.output_dir / "time_complexity_slots.csv"
    if args.output_summary_csv is None:
        args.output_summary_csv = args.output_dir / "time_complexity_summary.csv"
    if args.output_tex is None:
        args.output_tex = args.output_dir / "time_complexity_tables.tex"
    return args


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_strategy_variant(record: dict[str, Any]) -> str:
    variant = str(record.get("strategy_variant", "")).strip().lower()
    if variant in {"naive", "continual"}:
        return variant
    strategy = str(record.get("strategy", "")).strip().lower()
    if strategy == "naive":
        return "naive"
    return "continual"


def discover_local_results(results_root: pathlib.Path) -> dict[tuple[str, str, str, str, str], dict[str, pathlib.Path]]:
    mapping: dict[tuple[str, str, str, str, str], dict[str, pathlib.Path]] = defaultdict(dict)
    for json_path in sorted(results_root.rglob("*.json")):
        if json_path.name.startswith("article_") or json_path.name.endswith("_plot_heatmap_proxy.json"):
            continue
        if json_path.parent == results_root:
            continue
        match = LOCAL_RESULT_PATTERN.match(json_path.name)
        if match is None:
            continue
        parts = match.groupdict()
        dataset = json_path.parent.name
        key = (
            dataset,
            parts["model"],
            parts["setting"],
            parts["strategy"],
            parts["ordering"],
        )
        mapping[key][parts["eval_level"]] = json_path
    return mapping


def extract_time_totals(json_path: pathlib.Path) -> tuple[float | None, float | None]:
    data = load_json(json_path)
    payload = data.get("time_evaluation_callback")
    if not isinstance(payload, dict):
        return None, None
    train_total = safe_float(payload.get("train_time_total"))
    eval_total = safe_float(payload.get("eval_time_total"))
    time_by_concept = payload.get("time_by_concept")

    # Some runs serialize an empty timing callback as zeros, which should be
    # treated as missing rather than literal 0-second execution.
    if train_total == 0.0 and eval_total == 0.0 and not time_by_concept:
        return None, None

    return train_total, eval_total


def slot_times_from_paths(paths: dict[str, pathlib.Path]) -> tuple[str, float | None, float | None]:
    if "both" in paths:
        train, eval_ = extract_time_totals(paths["both"])
        return "both", train, eval_

    if "image" in paths and "pixel" in paths:
        image_train, image_eval = extract_time_totals(paths["image"])
        pixel_train, pixel_eval = extract_time_totals(paths["pixel"])
        train_values = [value for value in [image_train, pixel_train] if value is not None]
        eval_values = [value for value in [image_eval, pixel_eval] if value is not None]
        return (
            "image+pixel",
            sum(train_values) if len(train_values) == 2 else None,
            sum(eval_values) if len(eval_values) == 2 else None,
        )

    return "incomplete", None, None


def summarize(values: list[float]) -> tuple[int, float, float]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    if len(values) == 1:
        return 1, values[0], 0.0
    return len(values), mean(values), pstdev(values)


def format_float(value: float | None, decimals: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    if seconds < 60:
        return f"{seconds:.1f}s"
    rounded = int(round(seconds))
    total_minutes, remaining_seconds = divmod(rounded, 60)
    return f"{total_minutes}m\\,{remaining_seconds}s"


def format_mean_std(mean_seconds: float | None, std_seconds: float | None) -> str:
    if mean_seconds is None or std_seconds is None:
        return "--"
    return f"{format_seconds(mean_seconds)} ($\\pm$ {format_seconds(std_seconds)})"


def write_csv(path: pathlib.Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    try:
        return (DATASET_ORDER.index(dataset), dataset)
    except ValueError:
        return (len(DATASET_ORDER), dataset)


def model_sort_key(model: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def setting_group_for(setting: str) -> str:
    for group_name, settings in SETTING_GROUPS.items():
        if setting in settings:
            return group_name
    return "special"


def setting_label(setting: str) -> str:
    if setting == "wide_resnet50_2":
        return "wide\\_resnet50\\_2"
    return setting


def build_latex_table(
    group_name: str,
    rows: list[dict[str, Any]],
) -> str:
    datasets = sorted({str(row["dataset"]) for row in rows}, key=dataset_sort_key)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)

    lines: list[str] = []
    label_suffix = "special" if group_name == "special" else group_name.replace("_", "-")
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{SETTING_GROUP_CAPTIONS[group_name]}}}")
    lines.append(f"\\label{{tab:execution_times_{label_suffix}}}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{lllrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Time} & \\textbf{Naive} & \\textbf{Continual} \\\\")
    lines.append("\\midrule")

    for dataset_index, dataset in enumerate(datasets):
        dataset_rows = sorted(
            grouped[dataset],
            key=lambda row: (model_sort_key(str(row["model"])), str(row["setting"])),
        )
        dataset_display = DATASET_LABELS.get(dataset, dataset)
        dataset_line_count = 2 * len(dataset_rows)

        for model_index, row in enumerate(dataset_rows):
            model_display = MODEL_LABELS.get(str(row["model"]), str(row["model"]))
            if group_name == "special":
                model_display = f"{model_display} ({setting_label(str(row['setting']))})"
            if model_index == 0:
                lines.append(
                    f"\\multirow{{{dataset_line_count}}}{{*}}{{{dataset_display}}} & \\multirow{{2}}{{*}}{{{model_display}}} & Train & {row['naive_train_cell']} & {row['continual_train_cell']} \\\\"
                )
            else:
                lines.append(
                    f" & \\multirow{{2}}{{*}}{{{model_display}}} & Train & {row['naive_train_cell']} & {row['continual_train_cell']} \\\\"
                )
            lines.append(
                f" &  & Eval & {row['naive_eval_cell']} & {row['continual_eval_cell']} \\\\"
            )

        if dataset_index != len(datasets) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    aggregate_json = args.aggregate_json.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output_slot_csv = args.output_slot_csv.expanduser().resolve()
    output_summary_csv = args.output_summary_csv.expanduser().resolve()
    output_tex = args.output_tex.expanduser().resolve()

    for flag, path in (("--aggregate-json", aggregate_json), ("--results-root", results_root)):
        if not path.exists():
            raise SystemExit(
                f"{flag}: {path} does not exist. Pass --bundle-dir <results_root>/InCLAD_paper_results_20261007/"
                f"01_main_table_480 or set {flag} explicitly."
            )

    aggregate = load_json(aggregate_json)
    local_results = discover_local_results(results_root)

    slot_rows: list[dict[str, str]] = []
    grouped_values: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {
            "train": [],
            "eval": [],
            "orderings": [],
        }
    )

    for record in aggregate.get("records", []):
        status = str(record.get("status", "")).strip().lower()
        if status not in COMPLETE_STATUSES:
            continue

        dataset = str(record["dataset"])
        model = str(record["model"])
        setting = str(record["setting"])
        strategy = str(record["strategy"])
        strategy_variant = normalize_strategy_variant(record)
        ordering = str(record["ordering"])

        local_key = (dataset, model, setting, strategy, ordering)
        matched_paths = local_results.get(local_key, {})
        slot_source, train_total, eval_total = slot_times_from_paths(matched_paths)

        slot_rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "setting": setting,
                "setting_group": setting_group_for(setting),
                "strategy": strategy,
                "strategy_variant": strategy_variant,
                "ordering": ordering,
                "status": status,
                "slot_source": slot_source,
                "train_time_total_seconds": format_float(train_total),
                "eval_time_total_seconds": format_float(eval_total),
                "matched_both_file": str(matched_paths.get("both", "")),
                "matched_image_file": str(matched_paths.get("image", "")),
                "matched_pixel_file": str(matched_paths.get("pixel", "")),
            }
        )

        if train_total is None or eval_total is None:
            continue

        group_key = (dataset, model, setting, strategy_variant)
        grouped_values[group_key]["train"].append(train_total)
        grouped_values[group_key]["eval"].append(eval_total)
        grouped_values[group_key]["orderings"].append(ordering)

    summary_rows: list[dict[str, Any]] = []
    for dataset, model, setting, strategy_variant in sorted(
        grouped_values,
        key=lambda key: (dataset_sort_key(key[0]), model_sort_key(key[1]), key[2], key[3]),
    ):
        values = grouped_values[(dataset, model, setting, strategy_variant)]
        train_n, train_mean, train_std = summarize(values["train"])
        eval_n, eval_mean, eval_std = summarize(values["eval"])
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "setting": setting,
                "setting_group": setting_group_for(setting),
                "strategy_variant": strategy_variant,
                "ordering_count": str(len(values["orderings"])),
                "orderings": json.dumps(sorted(values["orderings"])),
                "train_count": str(train_n),
                "train_mean_seconds": format_float(train_mean),
                "train_std_seconds": format_float(train_std),
                "eval_count": str(eval_n),
                "eval_mean_seconds": format_float(eval_mean),
                "eval_std_seconds": format_float(eval_std),
                "train_cell": format_mean_std(train_mean, train_std),
                "eval_cell": format_mean_std(eval_mean, eval_std),
            }
        )

    slot_fieldnames = [
        "dataset",
        "dataset_label",
        "model",
        "model_label",
        "setting",
        "setting_group",
        "strategy",
        "strategy_variant",
        "ordering",
        "status",
        "slot_source",
        "train_time_total_seconds",
        "eval_time_total_seconds",
        "matched_both_file",
        "matched_image_file",
        "matched_pixel_file",
    ]
    write_csv(output_slot_csv, slot_rows, slot_fieldnames)

    summary_fieldnames = [
        "dataset",
        "dataset_label",
        "model",
        "model_label",
        "setting",
        "setting_group",
        "strategy_variant",
        "ordering_count",
        "orderings",
        "train_count",
        "train_mean_seconds",
        "train_std_seconds",
        "eval_count",
        "eval_mean_seconds",
        "eval_std_seconds",
        "train_cell",
        "eval_cell",
    ]
    write_csv(output_summary_csv, summary_rows, summary_fieldnames)

    by_group: dict[str, dict[tuple[str, str, str], dict[str, dict[str, str]]]] = defaultdict(dict)
    for row in summary_rows:
        group_name = str(row["setting_group"])
        logical_key = (str(row["dataset"]), str(row["model"]), str(row["setting"]))
        logical_row = by_group[group_name].setdefault(
            logical_key,
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "setting": row["setting"],
                "naive_train_cell": "--",
                "naive_eval_cell": "--",
                "continual_train_cell": "--",
                "continual_eval_cell": "--",
            },
        )
        if row["strategy_variant"] == "naive":
            logical_row["naive_train_cell"] = row["train_cell"]
            logical_row["naive_eval_cell"] = row["eval_cell"]
        else:
            logical_row["continual_train_cell"] = row["train_cell"]
            logical_row["continual_eval_cell"] = row["eval_cell"]

    latex_tables: list[str] = []
    for group_name in BACKBONE_TABLE_ORDER:
        if group_name not in args.tables:
            continue
        if group_name not in by_group:
            continue
        rows = list(by_group[group_name].values())
        latex_tables.append(build_latex_table(group_name, rows))

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n\n".join(latex_tables) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d table(s))", output_tex, len(latex_tables))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
