"""
Build a main-paper summary that pools across models and backbone/settings.

Two aggregation modes are supported:

1. model-first
   First average available settings/backbones within each model, then average
   across models. This avoids giving double weight to models that have two
   backbones while models like ganomaly/efficientad only expose one setting.

2. pooled-settings
   Treat each model-setting variant as an independent unit. With the current
   experiment grid this gives up to 16 units per table cell rather than 18,
   because ganomaly only has `default` and efficientad only has `small`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["cfa", "fastflow", "padim", "patchcore", "ganomaly", "paste", "rd4ad", "stfpm", "efficientad"]
DATASETS = ["mvtec", "visa", "btech", "mpdd", "dagm"]
DATASET_LABELS = {
    "mvtec": "MVTec",
    "visa": "VisA",
    "btech": "BTAD",
    "mpdd": "MPDD",
    "dagm": "DAGM",
}
ORDERINGS = ["easy_to_hard", "hard_to_easy", "random"]
ORDERING_LABELS = {
    "easy_to_hard": "Curriculum UP",
    "hard_to_easy": "Curriculum Down",
    "random": "Random",
}
MODALITIES = ["image", "pixel"]
STRATEGY_VARIANTS = ["naive", "continual"]
STRATEGY_LABELS = {
    "naive": "NCL",
    "continual": "CL",
}
TABLE_METRICS = {
    "roc": "ROC$\\uparrow$",
    "bwt": "BWT$\\uparrow$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a main-paper summary pooled across models and settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-json",
        type=pathlib.Path,
        default=pathlib.Path("../../../ordering_files/neurips_results/article_model_aggregates.json"),
        help="Per-slot aggregate JSON produced by build_article_aggregate_json.py.",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=pathlib.Path("../../ordering_files/neurips_results/article_main_paper_summary_model_first.json"),
        help="Where the pooled summary JSON should be written.",
    )
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=pathlib.Path("../../ordering_files/neurips_results/article_main_paper_summary_model_first.csv"),
        help="Flat CSV export of the pooled summary.",
    )
    parser.add_argument(
        "--output-tex",
        type=pathlib.Path,
        default=pathlib.Path("../../ordering_files/neurips_results/article_main_paper_summary_model_first.tex"),
        help="LaTeX table body for the pooled summary.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to include in the pooled aggregation.",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=["model-first", "pooled-settings"],
        default="model-first",
        help="Whether to average within model first or treat model-setting variants independently.",
    )
    parser.add_argument(
        "--roc-source",
        choices=["continual_average", "diagonal_average"],
        default="continual_average",
        help="Which ROC summary should populate the table cells.",
    )
    parser.add_argument(
        "--std-mode",
        choices=["sample", "population"],
        default="sample",
        help="How to compute standard deviation across aggregated units.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal precision for generated LaTeX values.",
    )
    parser.add_argument(
        "--latex-show-count",
        action="store_true",
        help="Append the coverage count to each LaTeX cell.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def load_payload(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def std(values: list[float], mode: str) -> float | None:
    if not values or len(values) < 2:
        return None
    center = mean(values)
    assert center is not None
    squared = sum((value - center) ** 2 for value in values)
    if mode == "sample":
        return math.sqrt(squared / (len(values) - 1))
    return math.sqrt(squared / len(values))


def normalize_matrix(matrix: Any) -> list[list[float]] | None:
    if not isinstance(matrix, list):
        return None
    normalized: list[list[float]] = []
    width: int | None = None
    for row in matrix:
        if not isinstance(row, list):
            return None
        normalized_row: list[float] = []
        for value in row:
            numeric = numeric_or_none(value)
            if numeric is None:
                return None
            normalized_row.append(numeric)
        if width is None:
            width = len(normalized_row)
        elif width != len(normalized_row):
            return None
        normalized.append(normalized_row)
    return normalized


def lower_triangle_only(matrix: list[list[Any]], *, strict: bool) -> list[list[Any]]:
    lowered: list[list[Any]] = []
    for row_index, row in enumerate(matrix):
        lowered_row: list[Any] = []
        for col_index, value in enumerate(row):
            keep = row_index > col_index if strict else row_index >= col_index
            lowered_row.append(value if keep else None)
        lowered.append(lowered_row)
    return lowered


def extract_scalar_metric(record: dict[str, Any], modality: str, scalar_key: str) -> float | None:
    modality_payload = record.get(modality)
    if not isinstance(modality_payload, dict):
        return None
    scalar_metrics = modality_payload.get("scalar_metrics")
    if not isinstance(scalar_metrics, dict):
        return None
    metric_payload = scalar_metrics.get(scalar_key)
    if not isinstance(metric_payload, dict):
        return None
    return numeric_or_none(metric_payload.get("mean"))


def extract_matrix(record: dict[str, Any], modality: str) -> list[list[float]] | None:
    modality_payload = record.get(modality)
    if not isinstance(modality_payload, dict):
        return None
    matrix_payload = modality_payload.get("metric_matrix")
    if not isinstance(matrix_payload, dict):
        return None
    return normalize_matrix(matrix_payload.get("mean"))


def extract_concepts_order(record: dict[str, Any], modality: str) -> list[str]:
    modality_payload = record.get(modality)
    if not isinstance(modality_payload, dict):
        return []
    concepts_order = modality_payload.get("concepts_order", [])
    if not isinstance(concepts_order, list):
        return []
    return [str(item) for item in concepts_order]


def aggregate_scalar(values_by_unit: dict[str, float], expected_units: list[str], std_mode: str) -> dict[str, Any]:
    available_units = [unit for unit in expected_units if unit in values_by_unit]
    values = [values_by_unit[unit] for unit in available_units]
    return {
        "count": len(values),
        "expected_count": len(expected_units),
        "mean": mean(values),
        "std": std(values, std_mode),
        "values_by_unit": {unit: values_by_unit[unit] for unit in available_units},
        "available_units": available_units,
        "missing_units": [unit for unit in expected_units if unit not in values_by_unit],
    }


def aggregate_matrix(
    matrices_by_unit: dict[str, list[list[float]]],
    expected_units: list[str],
    concepts_by_unit: dict[str, list[str]],
    std_mode: str,
) -> dict[str, Any] | None:
    available_units = [unit for unit in expected_units if unit in matrices_by_unit]
    if not available_units:
        return None

    matrices = [matrices_by_unit[unit] for unit in available_units]
    first_shape = (len(matrices[0]), len(matrices[0][0]) if matrices[0] else 0)
    if any((len(matrix), len(matrix[0]) if matrix else 0) != first_shape for matrix in matrices):
        return None

    n_rows, n_cols = first_shape
    mean_matrix: list[list[float]] = []
    std_matrix: list[list[float | None]] = []
    for row_index in range(n_rows):
        mean_row: list[float] = []
        std_row: list[float | None] = []
        for col_index in range(n_cols):
            values = [matrix[row_index][col_index] for matrix in matrices]
            cell_mean = mean(values)
            assert cell_mean is not None
            mean_row.append(cell_mean)
            std_row.append(std(values, std_mode))
        mean_matrix.append(mean_row)
        std_matrix.append(std_row)

    reference_order = concepts_by_unit.get(available_units[0], [])
    concepts_consistent = all(concepts_by_unit.get(unit, []) == reference_order for unit in available_units)
    return {
        "count": len(available_units),
        "expected_count": len(expected_units),
        "available_units": available_units,
        "missing_units": [unit for unit in expected_units if unit not in matrices_by_unit],
        "shape": [n_rows, n_cols],
        "mean": mean_matrix,
        "std": std_matrix,
        "lower_triangle_mean": lower_triangle_only(mean_matrix, strict=True),
        "lower_triangle_std": lower_triangle_only(std_matrix, strict=True),
        "lower_or_equal_triangle_mean": lower_triangle_only(mean_matrix, strict=False),
        "lower_or_equal_triangle_std": lower_triangle_only(std_matrix, strict=False),
        "concepts_order_reference": reference_order,
        "concepts_order_consistent_across_units": concepts_consistent,
    }


def build_record_groups(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["dataset"]),
            str(record["ordering"]),
            str(record["strategy_variant"]),
            str(record["model"]),
        )
        groups[key].append(record)
    for group_records in groups.values():
        group_records.sort(key=lambda record: str(record["setting"]))
    return groups


def expected_units_for_mode(records: list[dict[str, Any]], models: list[str], aggregation_mode: str) -> list[str]:
    if aggregation_mode == "model-first":
        return list(models)

    variants = {
        f"{record['model']}::{record['setting']}"
        for record in records
        if str(record.get("model")) in models
    }
    return sorted(variants)


def summarize_group_as_unit(
    group_records: list[dict[str, Any]],
    modality: str,
    roc_source: str,
) -> tuple[float | None, float | None, list[list[float]] | None, list[str], list[str]]:
    roc_values: list[float] = []
    bwt_values: list[float] = []
    matrices: list[list[list[float]]] = []
    concept_orders: list[list[str]] = []
    available_settings: list[str] = []

    for record in group_records:
        modality_payload = record.get(modality)
        if not isinstance(modality_payload, dict):
            continue
        if not record.get("article_usable") or not modality_payload.get("available"):
            continue

        roc_value = extract_scalar_metric(record, modality, roc_source)
        bwt_value = extract_scalar_metric(record, modality, "backward_transfer")
        matrix = extract_matrix(record, modality)

        if roc_value is not None:
            roc_values.append(roc_value)
        if bwt_value is not None:
            bwt_values.append(bwt_value)
        if matrix is not None:
            matrices.append(matrix)
            concept_orders.append(extract_concepts_order(record, modality))
        available_settings.append(str(record["setting"]))

    roc_mean = mean(roc_values)
    bwt_mean = mean(bwt_values)

    matrix_mean: list[list[float]] | None = None
    concepts_order: list[str] = []
    if matrices:
        first_shape = (len(matrices[0]), len(matrices[0][0]) if matrices[0] else 0)
        if all((len(matrix), len(matrix[0]) if matrix else 0) == first_shape for matrix in matrices):
            n_rows, n_cols = first_shape
            matrix_mean = []
            for row_index in range(n_rows):
                row: list[float] = []
                for col_index in range(n_cols):
                    values = [matrix[row_index][col_index] for matrix in matrices]
                    cell_mean = mean(values)
                    assert cell_mean is not None
                    row.append(cell_mean)
                matrix_mean.append(row)
            concepts_order = concept_orders[0] if concept_orders else []

    return roc_mean, bwt_mean, matrix_mean, concepts_order, available_settings


def build_cell(
    group_index: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    dataset: str,
    ordering: str,
    modality: str,
    strategy_variant: str,
    expected_units: list[str],
    models: list[str],
    aggregation_mode: str,
    roc_source: str,
    std_mode: str,
) -> dict[str, Any]:
    roc_values_by_unit: dict[str, float] = {}
    bwt_values_by_unit: dict[str, float] = {}
    matrices_by_unit: dict[str, list[list[float]]] = {}
    concepts_by_unit: dict[str, list[str]] = {}
    coverage_by_unit: dict[str, Any] = {}

    if aggregation_mode == "model-first":
        for model in models:
            key = (dataset, ordering, strategy_variant, model)
            group_records = group_index.get(key, [])
            roc_mean, bwt_mean, matrix_mean, concepts_order, available_settings = summarize_group_as_unit(
                group_records=group_records,
                modality=modality,
                roc_source=roc_source,
            )
            all_settings = [str(record["setting"]) for record in group_records]
            coverage_by_unit[model] = {
                "unit_type": "model",
                "all_settings": all_settings,
                "available_settings": available_settings,
                "missing_settings": [setting for setting in all_settings if setting not in available_settings],
            }
            if roc_mean is not None:
                roc_values_by_unit[model] = roc_mean
            if bwt_mean is not None:
                bwt_values_by_unit[model] = bwt_mean
            if matrix_mean is not None:
                matrices_by_unit[model] = matrix_mean
                concepts_by_unit[model] = concepts_order
    else:
        for unit in expected_units:
            model, setting = unit.split("::", 1)
            key = (dataset, ordering, strategy_variant, model)
            group_records = [record for record in group_index.get(key, []) if str(record["setting"]) == setting]
            coverage_by_unit[unit] = {
                "unit_type": "model_setting",
                "model": model,
                "setting": setting,
                "present_records": len(group_records),
            }
            if not group_records:
                continue
            record = group_records[0]
            modality_payload = record.get(modality)
            if not isinstance(modality_payload, dict):
                continue
            if not record.get("article_usable") or not modality_payload.get("available"):
                continue
            roc_value = extract_scalar_metric(record, modality, roc_source)
            bwt_value = extract_scalar_metric(record, modality, "backward_transfer")
            matrix = extract_matrix(record, modality)
            if roc_value is not None:
                roc_values_by_unit[unit] = roc_value
            if bwt_value is not None:
                bwt_values_by_unit[unit] = bwt_value
            if matrix is not None:
                matrices_by_unit[unit] = matrix
                concepts_by_unit[unit] = extract_concepts_order(record, modality)

    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "ordering": ordering,
        "ordering_label": ORDERING_LABELS.get(ordering, ordering),
        "modality": modality,
        "strategy_variant": strategy_variant,
        "strategy_label": STRATEGY_LABELS.get(strategy_variant, strategy_variant),
        "aggregation_mode": aggregation_mode,
        "expected_units": expected_units,
        "coverage_by_unit": coverage_by_unit,
        "metrics": {
            "roc": {
                "source_scalar_metric": roc_source,
                **aggregate_scalar(roc_values_by_unit, expected_units, std_mode),
            },
            "bwt": {
                "source_scalar_metric": "backward_transfer",
                **aggregate_scalar(bwt_values_by_unit, expected_units, std_mode),
            },
        },
        "metric_matrix": aggregate_matrix(matrices_by_unit, expected_units, concepts_by_unit, std_mode),
    }


def render_metric_value(metric_payload: dict[str, Any], decimals: int, show_count: bool) -> str:
    value_mean = metric_payload.get("mean")
    value_std = metric_payload.get("std")
    count = metric_payload.get("count")
    expected_count = metric_payload.get("expected_count")

    if value_mean is None or count in (None, 0):
        return "--"

    if value_std is None:
        rendered = f"\\({value_mean:.{decimals}f}\\)"
    else:
        rendered = (
            f"\\({value_mean:.{decimals}f}"
            f"\\,{{\\scriptstyle\\pm\\,{value_std:.{decimals}f}}}\\)"
        )

    if show_count:
        rendered += f" {{\\scriptsize({count}/{expected_count})}}"
    return rendered


def build_vertical_latex_body(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    decimals: int,
    show_count: bool,
) -> str:
    lines: list[str] = []
    for dataset_index, dataset in enumerate(DATASETS):
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        dataset_prefix = f"\\multirow{{6}}{{*}}{{{dataset_label}}}"
        first_row_for_dataset = True

        for ordering in ORDERINGS:
            ordering_label = ORDERING_LABELS.get(ordering, ordering)
            ordering_prefix = f"\\multirow{{2}}{{*}}{{{ordering_label}}}"

            for metric_key, metric_label in TABLE_METRICS.items():
                image_naive = lookup[(dataset, ordering, "image", "naive")]
                image_continual = lookup[(dataset, ordering, "image", "continual")]
                pixel_naive = lookup[(dataset, ordering, "pixel", "naive")]
                pixel_continual = lookup[(dataset, ordering, "pixel", "continual")]

                row_prefix: list[str] = []
                if first_row_for_dataset:
                    row_prefix.append(dataset_prefix)
                    first_row_for_dataset = False
                else:
                    row_prefix.append("")

                if metric_key == "roc":
                    row_prefix.append(ordering_prefix)
                else:
                    row_prefix.append("")

                row_prefix.append(metric_label)
                row_cells = [
                    render_metric_value(image_naive["metrics"][metric_key], decimals, show_count),
                    render_metric_value(image_continual["metrics"][metric_key], decimals, show_count),
                    render_metric_value(pixel_naive["metrics"][metric_key], decimals, show_count),
                    render_metric_value(pixel_continual["metrics"][metric_key], decimals, show_count),
                ]
                lines.append(" & ".join(row_prefix + row_cells) + " \\\\")

        if dataset_index < len(DATASETS) - 1:
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_csv(path: pathlib.Path, cells: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "dataset_label",
        "ordering",
        "ordering_label",
        "modality",
        "strategy_variant",
        "strategy_label",
        "aggregation_mode",
        "expected_count",
        "roc_count",
        "roc_mean",
        "roc_std",
        "roc_available_units",
        "roc_missing_units",
        "bwt_count",
        "bwt_mean",
        "bwt_std",
        "bwt_available_units",
        "bwt_missing_units",
        "matrix_count",
        "matrix_shape",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            matrix_payload = cell.get("metric_matrix") or {}
            writer.writerow(
                {
                    "dataset": cell["dataset"],
                    "dataset_label": cell["dataset_label"],
                    "ordering": cell["ordering"],
                    "ordering_label": cell["ordering_label"],
                    "modality": cell["modality"],
                    "strategy_variant": cell["strategy_variant"],
                    "strategy_label": cell["strategy_label"],
                    "aggregation_mode": cell["aggregation_mode"],
                    "expected_count": len(cell["expected_units"]),
                    "roc_count": cell["metrics"]["roc"]["count"],
                    "roc_mean": cell["metrics"]["roc"]["mean"],
                    "roc_std": cell["metrics"]["roc"]["std"],
                    "roc_available_units": "|".join(cell["metrics"]["roc"]["available_units"]),
                    "roc_missing_units": "|".join(cell["metrics"]["roc"]["missing_units"]),
                    "bwt_count": cell["metrics"]["bwt"]["count"],
                    "bwt_mean": cell["metrics"]["bwt"]["mean"],
                    "bwt_std": cell["metrics"]["bwt"]["std"],
                    "bwt_available_units": "|".join(cell["metrics"]["bwt"]["available_units"]),
                    "bwt_missing_units": "|".join(cell["metrics"]["bwt"]["missing_units"]),
                    "matrix_count": matrix_payload.get("count"),
                    "matrix_shape": json.dumps(matrix_payload.get("shape")),
                }
            )


def write_tex(path: pathlib.Path, latex_body: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated main-paper table body pooled across models/settings.\n\n")
        handle.write(latex_body)


def build_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    roc_count_distribution = Counter(cell["metrics"]["roc"]["count"] for cell in cells)
    bwt_count_distribution = Counter(cell["metrics"]["bwt"]["count"] for cell in cells)
    full_coverage_cells = sum(
        1
        for cell in cells
        if cell["metrics"]["roc"]["count"] == len(cell["expected_units"])
        and cell["metrics"]["bwt"]["count"] == len(cell["expected_units"])
    )
    return {
        "cells_total": len(cells),
        "cells_full_coverage": full_coverage_cells,
        "cells_partial_coverage": len(cells) - full_coverage_cells,
        "roc_count_distribution": dict(sorted(roc_count_distribution.items())),
        "bwt_count_distribution": dict(sorted(bwt_count_distribution.items())),
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    input_json = args.input_json.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_tex = args.output_tex.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    payload = load_payload(input_json)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Expected 'records' list in {input_json}")

    models = [str(model) for model in args.models]
    filtered_records = [record for record in records if str(record.get("model")) in models]
    group_index = build_record_groups(filtered_records)
    expected_units = expected_units_for_mode(filtered_records, models, args.aggregation_mode)

    cells: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for ordering in ORDERINGS:
            for modality in MODALITIES:
                for strategy_variant in STRATEGY_VARIANTS:
                    cells.append(
                        build_cell(
                            group_index=group_index,
                            dataset=dataset,
                            ordering=ordering,
                            modality=modality,
                            strategy_variant=strategy_variant,
                            expected_units=expected_units,
                            models=models,
                            aggregation_mode=args.aggregation_mode,
                            roc_source=args.roc_source,
                            std_mode=args.std_mode,
                        )
                    )

    lookup = {
        (
            cell["dataset"],
            cell["ordering"],
            cell["modality"],
            cell["strategy_variant"],
        ): cell
        for cell in cells
    }
    latex_body = build_vertical_latex_body(
        lookup=lookup,
        decimals=args.decimals,
        show_count=args.latex_show_count,
    )

    output_payload = {
        "metadata": {
            "input_json": str(input_json),
            "output_json": str(output_json),
            "output_csv": str(output_csv),
            "output_tex": str(output_tex),
            "models": models,
            "aggregation_mode": args.aggregation_mode,
            "expected_units": expected_units,
            "roc_source": args.roc_source,
            "bwt_source": "backward_transfer",
            "std_mode": args.std_mode,
            "note": (
                "model-first first averages settings/backbones within each model and then "
                "computes the table statistics across models. pooled-settings treats each "
                "model-setting variant as an independent unit."
            ),
        },
        "summary": build_summary(cells),
        "cells": cells,
        "latex_body": latex_body,
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    write_csv(output_csv, cells)
    write_tex(output_tex, latex_body)

    logger.info("Wrote JSON summary to %s", output_json)
    logger.info("Wrote CSV summary to %s", output_csv)
    logger.info("Wrote LaTeX body to %s", output_tex)
    logger.info("Summary: %s", output_payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
