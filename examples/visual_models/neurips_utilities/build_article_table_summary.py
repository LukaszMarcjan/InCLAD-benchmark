"""
Build article-table summaries across the 7 backbone-based models.

This script consumes the per-slot aggregate JSON produced by
`build_article_aggregate_json.py` and builds table-ready aggregates for:

    dataset x backbone x ordering x modality x strategy_variant

Each table cell is aggregated across the 7 selected models, so the reported
standard deviation reflects variability across models rather than duplicated
JSON files for a single slot.

Besides scalar table metrics (ROC and BWT), the output also aggregates the
full metric matrices across models so the lower triangle can be inspected for
forgetting analyses.

For downstream compatibility, the JSON exposes the aggregated list twice:
under `cells` (the historical name used by the table tooling) and under
`records` (matching the per-model aggregate JSON layout).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["cfa", "fastflow", "padim", "patchcore", "paste", "rd4ad", "stfpm"]
DATASETS = ["mvtec", "visa", "btech", "mpdd", "dagm"]
DATASET_LABELS = {
    "mvtec": "MVTec",
    "visa": "VisA",
    "btech": "BTAD",
    "mpdd": "MPDD",
    "dagm": "DAGM",
}
BACKBONES = ["resnet18", "wide_resnet50_2"]
ORDERINGS = ["easy_to_hard", "hard_to_easy", "random"]
ORDERING_LABELS = {
    "easy_to_hard": "Curriculum UP",
    "hard_to_easy": "Curriculum Down",
    "random": "Random",
}
MODALITIES = ["image", "pixel"]
MODALITY_LABELS = {
    "image": "Image-level",
    "pixel": "Pixel-level",
}
STRATEGY_VARIANTS = ["naive", "continual"]
STRATEGY_LABELS = {
    "naive": "NCL",
    "continual": "CL",
}
TABLE_METRICS = {
    "roc": "ROC$\\uparrow$",
    "bwt": "BWT$\\uparrow$",
}
SCALAR_METRIC_KEYS = {
    "roc": {
        "continual_average": "continual_average",
        "diagonal_average": "diagonal_average",
    },
    "bwt": {
        "backward_transfer": "backward_transfer",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build article-table aggregates across the 7 selected models.",
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
        default=pathlib.Path("../../../ordering_files/neurips_results/article_table_summary_across_models.json"),
        help="Where the table-ready aggregate JSON should be written.",
    )
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=pathlib.Path("../../../ordering_files/neurips_results/article_table_summary_across_models.csv"),
        help="Flat CSV export of the table-ready aggregates.",
    )
    parser.add_argument(
        "--output-tex",
        type=pathlib.Path,
        default=pathlib.Path("../../../ordering_files/neurips_results/article_table_summary_across_models.tex"),
        help="LaTeX table-body fragments for both backbones.",
    )
    parser.add_argument(
        "--latex-layout",
        choices=["vertical-orderings", "horizontal-orderings"],
        default="vertical-orderings",
        help="How to arrange the LaTeX table body.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to include in the across-model aggregation.",
    )
    parser.add_argument(
        "--roc-source",
        choices=["continual_average", "diagonal_average"],
        default="continual_average",
        help="Which scalar ROC summary from the per-model JSON should feed the table cells.",
    )
    parser.add_argument(
        "--std-mode",
        choices=["sample", "population"],
        default="sample",
        help="How to compute standard deviation across models.",
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
        help="Append the model count to LaTeX cells, useful while coverage is incomplete.",
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


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def std(values: list[float], mode: str) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return None
    center = mean(values)
    assert center is not None
    squared = sum((value - center) ** 2 for value in values)
    if mode == "sample":
        return math.sqrt(squared / (len(values) - 1))
    return math.sqrt(squared / len(values))


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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


def lower_triangle_only(matrix: list[list[float]], *, strict: bool) -> list[list[float | None]]:
    lowered: list[list[float | None]] = []
    for row_index, row in enumerate(matrix):
        lowered_row: list[float | None] = []
        for col_index, value in enumerate(row):
            keep = row_index > col_index if strict else row_index >= col_index
            lowered_row.append(value if keep else None)
        lowered.append(lowered_row)
    return lowered


def aggregate_scalar(values_by_model: dict[str, float], expected_models: list[str], std_mode: str) -> dict[str, Any]:
    available_models = [model for model in expected_models if model in values_by_model]
    values = [values_by_model[model] for model in available_models]
    return {
        "count": len(values),
        "expected_count": len(expected_models),
        "mean": mean(values),
        "std": std(values, std_mode),
        "values_by_model": {model: values_by_model[model] for model in available_models},
        "available_models": available_models,
        "missing_models": [model for model in expected_models if model not in values_by_model],
    }


def aggregate_matrix(
    matrices_by_model: dict[str, list[list[float]]],
    expected_models: list[str],
    concepts_by_model: dict[str, list[str]],
    std_mode: str,
) -> dict[str, Any] | None:
    available_models = [model for model in expected_models if model in matrices_by_model]
    if not available_models:
        return None

    matrices = [matrices_by_model[model] for model in available_models]
    first_shape = (len(matrices[0]), len(matrices[0][0]) if matrices[0] else 0)
    for matrix in matrices:
        shape = (len(matrix), len(matrix[0]) if matrix else 0)
        if shape != first_shape:
            return {
                "count": len(available_models),
                "expected_count": len(expected_models),
                "available_models": available_models,
                "missing_models": [model for model in expected_models if model not in matrices_by_model],
                "shape": None,
                "mean": None,
                "std": None,
                "lower_triangle_mean": None,
                "lower_triangle_std": None,
                "lower_or_equal_triangle_mean": None,
                "lower_or_equal_triangle_std": None,
                "concepts_order_reference": None,
                "concepts_order_consistent_across_models": False,
                "shape_consistent_across_models": False,
            }

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

    reference_order = concepts_by_model.get(available_models[0], [])
    concepts_consistent = all(
        concepts_by_model.get(model, []) == reference_order for model in available_models
    )
    return {
        "count": len(available_models),
        "expected_count": len(expected_models),
        "available_models": available_models,
        "missing_models": [model for model in expected_models if model not in matrices_by_model],
        "shape": [n_rows, n_cols],
        "mean": mean_matrix,
        "std": std_matrix,
        "lower_triangle_mean": lower_triangle_only(mean_matrix, strict=True),
        "lower_triangle_std": lower_triangle_only(std_matrix, strict=True),  # type: ignore[arg-type]
        "lower_or_equal_triangle_mean": lower_triangle_only(mean_matrix, strict=False),
        "lower_or_equal_triangle_std": lower_triangle_only(std_matrix, strict=False),  # type: ignore[arg-type]
        "concepts_order_reference": reference_order,
        "concepts_order_consistent_across_models": concepts_consistent,
        "shape_consistent_across_models": True,
    }


def build_record_index(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record["dataset"]),
            str(record["setting"]),
            str(record["ordering"]),
            str(record["strategy_variant"]),
            str(record["model"]),
        )
        index[key] = record
    return index


def extract_scalar_metric(
    record: dict[str, Any],
    modality: str,
    scalar_key: str,
) -> float | None:
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


def extract_mean_matrix(record: dict[str, Any], modality: str) -> list[list[float]] | None:
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


def build_cell(
    index: dict[tuple[str, str, str, str, str], dict[str, Any]],
    dataset: str,
    backbone: str,
    ordering: str,
    modality: str,
    strategy_variant: str,
    expected_models: list[str],
    roc_source: str,
    std_mode: str,
) -> dict[str, Any]:
    roc_values_by_model: dict[str, float] = {}
    bwt_values_by_model: dict[str, float] = {}
    matrices_by_model: dict[str, list[list[float]]] = {}
    concepts_by_model: dict[str, list[str]] = {}
    coverage_by_model: dict[str, dict[str, Any]] = {}

    for model in expected_models:
        key = (dataset, backbone, ordering, strategy_variant, model)
        record = index.get(key)
        if record is None:
            coverage_by_model[model] = {
                "present": False,
                "article_usable": False,
                "coverage_status": "MISSING_RECORD",
                "modality_available": False,
            }
            continue

        modality_payload = record.get(modality)
        modality_available = isinstance(modality_payload, dict) and bool(modality_payload.get("available"))
        coverage_by_model[model] = {
            "present": True,
            "article_usable": bool(record.get("article_usable")),
            "coverage_status": record.get("coverage_status"),
            "modality_available": modality_available,
            "source_files": modality_payload.get("source_files", []) if isinstance(modality_payload, dict) else [],
        }
        if not record.get("article_usable") or not modality_available:
            continue

        roc_value = extract_scalar_metric(record, modality, roc_source)
        if roc_value is not None:
            roc_values_by_model[model] = roc_value

        bwt_value = extract_scalar_metric(record, modality, "backward_transfer")
        if bwt_value is not None:
            bwt_values_by_model[model] = bwt_value

        matrix = extract_mean_matrix(record, modality)
        if matrix is not None:
            matrices_by_model[model] = matrix
            concepts_by_model[model] = extract_concepts_order(record, modality)

    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "backbone": backbone,
        "ordering": ordering,
        "ordering_label": ORDERING_LABELS.get(ordering, ordering),
        "modality": modality,
        "modality_label": MODALITY_LABELS.get(modality, modality),
        "strategy_variant": strategy_variant,
        "strategy_label": STRATEGY_LABELS.get(strategy_variant, strategy_variant),
        "expected_models": expected_models,
        "coverage_by_model": coverage_by_model,
        "metrics": {
            "roc": {
                "source_scalar_metric": roc_source,
                **aggregate_scalar(roc_values_by_model, expected_models, std_mode),
            },
            "bwt": {
                "source_scalar_metric": "backward_transfer",
                **aggregate_scalar(bwt_values_by_model, expected_models, std_mode),
            },
        },
        "metric_matrix": aggregate_matrix(matrices_by_model, expected_models, concepts_by_model, std_mode),
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


def build_horizontal_latex_body(
    backbone: str,
    lookup: dict[tuple[str, str, str, str, str], dict[str, Any]],
    decimals: int,
    show_count: bool,
) -> str:
    lines: list[str] = []
    for dataset_index, dataset in enumerate(DATASETS):
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        for metric_key, metric_label in TABLE_METRICS.items():
            row_cells: list[str] = []
            for ordering in ORDERINGS:
                for modality in MODALITIES:
                    for strategy_variant in STRATEGY_VARIANTS:
                        cell = lookup[(dataset, backbone, ordering, modality, strategy_variant)]
                        row_cells.append(
                            render_metric_value(cell["metrics"][metric_key], decimals, show_count)
                        )

            if metric_key == "roc":
                prefix = f"\\multirow{{2}}{{*}}{{{dataset_label}}} & {metric_label}"
            else:
                prefix = f" & {metric_label}"
            line = prefix + " & " + " & ".join(row_cells) + " \\\\"
            if dataset_index < len(DATASETS) - 1 and metric_key == "bwt":
                line += ""
            lines.append(line)
        if dataset_index < len(DATASETS) - 1:
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_vertical_latex_body(
    backbone: str,
    lookup: dict[tuple[str, str, str, str, str], dict[str, Any]],
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
                image_naive = lookup[(dataset, backbone, ordering, "image", "naive")]
                image_continual = lookup[(dataset, backbone, ordering, "image", "continual")]
                pixel_naive = lookup[(dataset, backbone, ordering, "pixel", "naive")]
                pixel_continual = lookup[(dataset, backbone, ordering, "pixel", "continual")]

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
        "backbone",
        "ordering",
        "ordering_label",
        "modality",
        "modality_label",
        "strategy_variant",
        "strategy_label",
        "expected_count",
        "roc_count",
        "roc_mean",
        "roc_std",
        "roc_available_models",
        "roc_missing_models",
        "bwt_count",
        "bwt_mean",
        "bwt_std",
        "bwt_available_models",
        "bwt_missing_models",
        "matrix_count",
        "matrix_shape",
        "matrix_concepts_order_reference",
        "matrix_concepts_consistent_across_models",
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
                    "backbone": cell["backbone"],
                    "ordering": cell["ordering"],
                    "ordering_label": cell["ordering_label"],
                    "modality": cell["modality"],
                    "modality_label": cell["modality_label"],
                    "strategy_variant": cell["strategy_variant"],
                    "strategy_label": cell["strategy_label"],
                    "expected_count": len(cell["expected_models"]),
                    "roc_count": cell["metrics"]["roc"]["count"],
                    "roc_mean": cell["metrics"]["roc"]["mean"],
                    "roc_std": cell["metrics"]["roc"]["std"],
                    "roc_available_models": "|".join(cell["metrics"]["roc"]["available_models"]),
                    "roc_missing_models": "|".join(cell["metrics"]["roc"]["missing_models"]),
                    "bwt_count": cell["metrics"]["bwt"]["count"],
                    "bwt_mean": cell["metrics"]["bwt"]["mean"],
                    "bwt_std": cell["metrics"]["bwt"]["std"],
                    "bwt_available_models": "|".join(cell["metrics"]["bwt"]["available_models"]),
                    "bwt_missing_models": "|".join(cell["metrics"]["bwt"]["missing_models"]),
                    "matrix_count": matrix_payload.get("count"),
                    "matrix_shape": json.dumps(matrix_payload.get("shape")),
                    "matrix_concepts_order_reference": "|".join(
                        matrix_payload.get("concepts_order_reference", []) or []
                    ),
                    "matrix_concepts_consistent_across_models": matrix_payload.get(
                        "concepts_order_consistent_across_models"
                    ),
                }
            )


def write_tex(path: pathlib.Path, latex_bodies: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated table bodies for the continual summary tables.\n\n")
        for backbone in BACKBONES:
            handle.write(f"% Backbone: {backbone}\n")
            handle.write(latex_bodies[backbone])
            handle.write("\n")


def build_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    roc_count_distribution = Counter(cell["metrics"]["roc"]["count"] for cell in cells)
    bwt_count_distribution = Counter(cell["metrics"]["bwt"]["count"] for cell in cells)
    full_coverage_cells = sum(
        1
        for cell in cells
        if cell["metrics"]["roc"]["count"] == len(cell["expected_models"])
        and cell["metrics"]["bwt"]["count"] == len(cell["expected_models"])
    )
    expected_cells_total = (
        len(DATASETS) * len(BACKBONES) * len(ORDERINGS) * len(MODALITIES) * len(STRATEGY_VARIANTS)
    )
    return {
        "cells_total": len(cells),
        "records_total": len(cells),
        "expected_cells_total": expected_cells_total,
        "grid_matches_expected_total": len(cells) == expected_cells_total,
        "grid_dimensions": {
            "datasets": len(DATASETS),
            "backbones": len(BACKBONES),
            "orderings": len(ORDERINGS),
            "modalities": len(MODALITIES),
            "strategy_variants": len(STRATEGY_VARIANTS),
        },
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

    index = build_record_index(records)
    expected_models = [str(model) for model in args.models]
    cells: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            for ordering in ORDERINGS:
                for modality in MODALITIES:
                    for strategy_variant in STRATEGY_VARIANTS:
                        cells.append(
                            build_cell(
                                index=index,
                                dataset=dataset,
                                backbone=backbone,
                                ordering=ordering,
                                modality=modality,
                                strategy_variant=strategy_variant,
                                expected_models=expected_models,
                                roc_source=args.roc_source,
                                std_mode=args.std_mode,
                            )
                        )

    lookup = {
        (
            cell["dataset"],
            cell["backbone"],
            cell["ordering"],
            cell["modality"],
            cell["strategy_variant"],
        ): cell
        for cell in cells
    }
    latex_bodies_horizontal = {
        backbone: build_horizontal_latex_body(
            backbone=backbone,
            lookup=lookup,
            decimals=args.decimals,
            show_count=args.latex_show_count,
        )
        for backbone in BACKBONES
    }
    latex_bodies_vertical = {
        backbone: build_vertical_latex_body(
            backbone=backbone,
            lookup=lookup,
            decimals=args.decimals,
            show_count=args.latex_show_count,
        )
        for backbone in BACKBONES
    }
    latex_bodies = (
        latex_bodies_vertical
        if args.latex_layout == "vertical-orderings"
        else latex_bodies_horizontal
    )

    summary = build_summary(cells)
    output_payload = {
        "metadata": {
            "input_json": str(input_json),
            "output_json": str(output_json),
            "output_csv": str(output_csv),
            "output_tex": str(output_tex),
            "models": expected_models,
            "datasets": DATASETS,
            "dataset_labels": DATASET_LABELS,
            "backbones": BACKBONES,
            "orderings": ORDERINGS,
            "ordering_labels": ORDERING_LABELS,
            "modalities": MODALITIES,
            "strategy_variants": STRATEGY_VARIANTS,
            "strategy_labels": STRATEGY_LABELS,
            "roc_source": args.roc_source,
            "bwt_source": "backward_transfer",
            "std_mode": args.std_mode,
            "latex_layout": args.latex_layout,
            "table_note": (
                "Cell means/stds are aggregated across models, not across duplicate files "
                "for a single slot. metric_matrix.* aggregates the full continual matrix "
                "across models for forgetting analysis."
            ),
            "records_alias_note": (
                "Top-level 'records' is an alias of 'cells' so the aggregated JSON can be "
                "consumed with the same list-style access pattern as article_model_aggregates.json."
            ),
        },
        "summary": summary,
        "records": cells,
        "cells": cells,
        "latex_bodies": latex_bodies,
        "latex_bodies_horizontal": latex_bodies_horizontal,
        "latex_bodies_vertical": latex_bodies_vertical,
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    write_csv(output_csv, cells)
    write_tex(output_tex, latex_bodies)

    logger.info("Wrote JSON summary to %s", output_json)
    logger.info("Wrote CSV summary to %s", output_csv)
    logger.info("Wrote LaTeX fragments to %s", output_tex)
    logger.info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
