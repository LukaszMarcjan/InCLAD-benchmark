#!/usr/bin/env python3
"""Render article-style ordering-row PDFs for completed multidataset runs.

This script treats MULTIDATASET_SEQ_V1 as one benchmark and aggregates
completed runs in two stages:

1. average duplicate runs within the same method/backbone/ordering/variant
2. average across methods (cfa, padim, paste)

The output mirrors the compact `*_ordering_row.pdf` figures used in the
NeurIPS utilities, but only for the orderings available in the completed
multidataset results (Hard-to-Easy and Easy-to-Hard).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.visual_models.neurips_utilities.generate_article_overview_grids import (
    choose_cell_size,
    choose_ordering_font,
    choose_uniform_label_font,
    choose_value_font,
    compute_bottom_label_height,
    display_label,
    estimate_text_width,
    pdf_rect,
    render_panel,
    render_vertical_colorbar,
    write_simple_pdf,
)


DATASET_NAME = "MULTIDATASET_SEQ_V1-VisualBenchmark"
DATASET_SLUG = "multidataset"
DATASET_LABEL = "MultiDataset"
ORDERINGS = ["hard_to_easy", "easy_to_hard"]
ORDERING_LABELS = {
    "hard_to_easy": "Hard->Easy",
    "easy_to_hard": "Easy->Hard",
}
STRATEGY_FILE_LABELS = {
    "naive": "ncl",
    "continual": "cl",
}
MATRIX_KEYS = {
    "image": "concept_metric_callback_ROC-AUC",
    "pixel": "pixel_concept_metric_callback_Pixel-ROC-AUC",
}


@dataclass(frozen=True)
class AggregatedCell:
    backbone: str
    ordering: str
    strategy_variant: str
    modality: str
    concepts_order: list[str]
    mean_matrix: list[list[float]]
    std_matrix: list[list[float | None]]
    available_methods: list[str]
    sources_by_method: dict[str, list[str]]


@dataclass(frozen=True)
class RowFigure:
    dataset: str
    backbone: str
    strategy_variant: str
    modality: str
    cells: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-roots",
        nargs="+",
        type=pathlib.Path,
        default=[pathlib.Path("fin_res"), pathlib.Path("fin_resl")],
        help="Roots scanned recursively for completed multidataset JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("fin_fin/multidataset_ordering_rows_pdf"),
        help="Directory for generated ordering-row PDFs and manifest.",
    )
    parser.add_argument(
        "--matrix-stat",
        choices=["mean", "std"],
        default="mean",
        help="Which aggregate statistic to render in the PDFs.",
    )
    parser.add_argument(
        "--layout",
        choices=["combined", "split", "both"],
        default="combined",
        help="Whether to render one PDF with both orderings, separate PDFs per ordering, or both.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDFs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def std_sample(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = mean(values)
    assert center is not None
    return (sum((value - center) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def detect_method(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized == "paste":
        return "paste"
    if normalized == "padim":
        return "padim"
    if normalized == "cfa":
        return "cfa"
    return normalized


def detect_strategy_variant(strategy_name: str) -> str:
    return "naive" if strategy_name.strip().lower() == "naive" else "continual"


def detect_ordering(path: pathlib.Path) -> str | None:
    name = path.as_posix()
    if "hard-to-easy" in name:
        return "hard_to_easy"
    if "easy-to-hard" in name:
        return "easy_to_hard"
    return None


def ordered_matrix(matrix_mapping: dict[str, dict[str, float]], concepts_order: list[str]) -> list[list[float]]:
    return [
        [float(matrix_mapping[learned][evaluated]) for evaluated in concepts_order]
        for learned in concepts_order
    ]


def average_matrices(matrices: list[list[list[float]]]) -> list[list[float]]:
    row_count = len(matrices[0])
    col_count = len(matrices[0][0]) if row_count else 0
    out: list[list[float]] = []
    for row_index in range(row_count):
        row: list[float] = []
        for col_index in range(col_count):
            values = [matrix[row_index][col_index] for matrix in matrices]
            cell_mean = mean(values)
            assert cell_mean is not None
            row.append(cell_mean)
        out.append(row)
    return out


def std_matrices(matrices: list[list[list[float]]]) -> list[list[float | None]]:
    row_count = len(matrices[0])
    col_count = len(matrices[0][0]) if row_count else 0
    out: list[list[float | None]] = []
    for row_index in range(row_count):
        row: list[float | None] = []
        for col_index in range(col_count):
            values = [matrix[row_index][col_index] for matrix in matrices]
            row.append(std_sample(values))
        out.append(row)
    return out


def load_completed_runs(results_roots: list[pathlib.Path]) -> dict[tuple[str, str, str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for results_root in results_roots:
        for json_path in sorted(results_root.rglob("*_seed_*.json")):
            if "/fastflow/" in json_path.as_posix():
                continue

            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            if data.get("dataset", {}).get("name") != DATASET_NAME:
                continue

            run_status = data.get("run_status") or {}
            if run_status.get("state") != "completed":
                continue
            if run_status.get("completed_concepts_count") != run_status.get("total_concepts"):
                continue

            ordering = detect_ordering(json_path)
            if ordering is None:
                logging.warning("Skipping %s because ordering could not be inferred", json_path)
                continue

            model = data.get("model") or {}
            strategy = data.get("strategy") or {}
            method = detect_method(str(model.get("name", "")))
            backbone = str(model.get("backbone", "unknown"))
            strategy_variant = detect_strategy_variant(str(strategy.get("name", "")))

            for modality, key in MATRIX_KEYS.items():
                payload = data.get(key)
                if not isinstance(payload, dict):
                    continue
                matrix_mapping = payload.get("metric_matrix")
                concepts_order = payload.get("concepts_order")
                if not isinstance(matrix_mapping, dict) or not isinstance(concepts_order, list):
                    continue

                grouped[(backbone, ordering, strategy_variant, modality, method)].append(
                    {
                        "path": str(json_path),
                        "concepts_order": [str(item) for item in concepts_order],
                        "matrix": ordered_matrix(matrix_mapping, [str(item) for item in concepts_order]),
                    }
                )

    return grouped


def aggregate_cells(grouped_runs: dict[tuple[str, str, str, str, str], list[dict[str, Any]]]) -> list[AggregatedCell]:
    slot_to_methods: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)

    for (backbone, ordering, strategy_variant, modality, method), records in grouped_runs.items():
        reference_order = records[0]["concepts_order"]
        if any(record["concepts_order"] != reference_order for record in records[1:]):
            logging.warning(
                "Skipping %s/%s/%s/%s/%s because concept order differs across duplicate runs",
                backbone,
                ordering,
                strategy_variant,
                modality,
                method,
            )
            continue

        matrices = [record["matrix"] for record in records]
        slot_to_methods[(backbone, ordering, strategy_variant, modality)][method] = {
            "concepts_order": reference_order,
            "mean_matrix": average_matrices(matrices),
            "sources": [record["path"] for record in records],
        }

    aggregated: list[AggregatedCell] = []
    for (backbone, ordering, strategy_variant, modality), methods_payload in sorted(slot_to_methods.items()):
        methods = sorted(methods_payload)
        reference_order = methods_payload[methods[0]]["concepts_order"]
        if any(methods_payload[method]["concepts_order"] != reference_order for method in methods[1:]):
            logging.warning(
                "Skipping %s/%s/%s/%s because concept order differs across methods",
                backbone,
                ordering,
                strategy_variant,
                modality,
            )
            continue

        matrices = [methods_payload[method]["mean_matrix"] for method in methods]
        aggregated.append(
            AggregatedCell(
                backbone=backbone,
                ordering=ordering,
                strategy_variant=strategy_variant,
                modality=modality,
                concepts_order=reference_order,
                mean_matrix=average_matrices(matrices),
                std_matrix=std_matrices(matrices),
                available_methods=methods,
                sources_by_method={method: methods_payload[method]["sources"] for method in methods},
            )
        )

    return aggregated


def build_rows(cells: list[AggregatedCell]) -> list[RowFigure]:
    lookup = {
        (cell.backbone, cell.ordering, cell.strategy_variant, cell.modality): cell
        for cell in cells
    }

    rows: list[RowFigure] = []
    backbones = sorted({cell.backbone for cell in cells})
    for backbone in backbones:
        for strategy_variant in ["naive", "continual"]:
            for modality in ["image", "pixel"]:
                row_cells: dict[str, dict[str, Any]] = {}
                missing: list[str] = []
                for ordering in ORDERINGS:
                    cell = lookup.get((backbone, ordering, strategy_variant, modality))
                    if cell is None:
                        missing.append(ordering)
                        continue
                    row_cells[ordering] = {
                        "metric_matrix": {
                            "concepts_order_reference": cell.concepts_order,
                            "mean": cell.mean_matrix,
                            "std": cell.std_matrix,
                            "count": len(cell.available_methods),
                            "available_models": cell.available_methods,
                        }
                    }
                if missing:
                    logging.warning(
                        "Skipping row for %s/%s/%s because orderings are missing: %s",
                        backbone,
                        strategy_variant,
                        modality,
                        ", ".join(missing),
                    )
                    continue

                rows.append(
                    RowFigure(
                        dataset=DATASET_SLUG,
                        backbone=backbone,
                        strategy_variant=strategy_variant,
                        modality=modality,
                        cells=row_cells,
                    )
                )
    return rows


def output_path_for_row(output_dir: pathlib.Path, row: RowFigure, matrix_stat: str) -> pathlib.Path:
    strategy_slug = STRATEGY_FILE_LABELS[row.strategy_variant]
    stem = "_".join(
        [
            DATASET_SLUG,
            row.backbone,
            strategy_slug,
            row.modality,
            matrix_stat,
            "ordering_row",
        ]
    )
    return output_dir / stem


def output_path_for_split_row(
    output_dir: pathlib.Path,
    row: RowFigure,
    matrix_stat: str,
    ordering: str,
) -> pathlib.Path:
    strategy_slug = STRATEGY_FILE_LABELS[row.strategy_variant]
    stem = "_".join(
        [
            DATASET_SLUG,
            row.backbone,
            strategy_slug,
            row.modality,
            matrix_stat,
            ordering,
            "ordering_row",
        ]
    )
    return output_dir / stem


def write_row_pdf(
    row: RowFigure,
    output_path: pathlib.Path,
    matrix_stat: str,
    orderings: list[str] | None = None,
) -> None:
    render_orderings = orderings or ORDERINGS
    first_record = row.cells[render_orderings[0]]
    first_matrix_payload = first_record["metric_matrix"]
    concepts = [str(item) for item in first_matrix_payload["concepts_order_reference"]]
    display_labels = [display_label(concept) for concept in concepts]
    concept_count = len(concepts)
    cell_size = choose_cell_size(concept_count)
    label_font_size = choose_uniform_label_font(cell_size, display_labels, -75.0)
    value_font_size = choose_value_font(cell_size)
    ordering_font_size = choose_ordering_font(cell_size)
    label_width = max(estimate_text_width(label, label_font_size) for label in display_labels) + 14.0
    bottom_label_height = compute_bottom_label_height(display_labels, label_font_size, -75.0)
    matrix_size = concept_count * cell_size

    left_margin = 16.0
    right_margin = 16.0
    top_margin = 16.0
    panel_gap = max(22.0, round(cell_size * 0.85, 1))
    ordering_space = ordering_font_size + 10.0
    tick_font_size = max(9.0, min(14.0, label_font_size * 0.75))
    bar_width = max(10.0, round(cell_size * 0.18, 1))
    colorbar_section_width = bar_width + 4.0 + estimate_text_width("0.00", tick_font_size) + 10.0

    width = int(
        round(
            left_margin
            + colorbar_section_width
            + len(render_orderings) * (label_width + matrix_size)
            + (len(render_orderings) - 1) * panel_gap
            + right_margin
        )
    )
    height = int(round(top_margin + matrix_size + bottom_label_height + ordering_space + 14.0))

    commands: list[str] = []
    pdf_rect(
        commands,
        page_height=height,
        x=0,
        y_top=0,
        width=width,
        height=height,
        fill_rgb=(1.0, 1.0, 1.0),
        stroke_rgb=None,
    )

    vmin, vmax = (0.0, 1.0) if matrix_stat == "mean" else (0.0, 0.25)
    render_vertical_colorbar(
        commands,
        page_height=height,
        x=left_margin,
        y_top=top_margin,
        bar_width=bar_width,
        height=matrix_size,
        tick_font_size=tick_font_size,
        vmin=vmin,
        vmax=vmax,
    )

    cursor_x = left_margin + colorbar_section_width
    for ordering in render_orderings:
        record = row.cells[ordering]
        matrix_payload = record["metric_matrix"]
        ordered_concepts = [str(item) for item in matrix_payload["concepts_order_reference"]]
        matrix = matrix_payload[matrix_stat]
        cursor_right = render_panel(
            commands,
            page_height=height,
            matrix=matrix,
            concepts=ordered_concepts,
            x_offset=cursor_x,
            y_offset=top_margin,
            cell_size=cell_size,
            label_width=label_width,
            label_font_size=label_font_size,
            value_font_size=value_font_size,
            ordering_font_size=ordering_font_size,
            bottom_label_height=bottom_label_height,
            ordering_label=ORDERING_LABELS[ordering],
            matrix_stat=matrix_stat,
        )
        cursor_x = cursor_right + panel_gap

    write_simple_pdf(output_path.with_suffix(".pdf"), width, height, commands)


def write_manifest(output_dir: pathlib.Path, rows: list[RowFigure], matrix_stat: str) -> pathlib.Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "backbone",
                "strategy_variant",
                "modality",
                "matrix_stat",
                "relative_pdf_path",
                "orderings",
                "methods_hard_to_easy",
                "methods_easy_to_hard",
            ],
        )
        writer.writeheader()
        for row in rows:
            output_path = output_path_for_row(output_dir, row, matrix_stat).with_suffix(".pdf")
            writer.writerow(
                {
                    "dataset": DATASET_SLUG,
                    "backbone": row.backbone,
                    "strategy_variant": row.strategy_variant,
                    "modality": row.modality,
                    "matrix_stat": matrix_stat,
                    "relative_pdf_path": str(output_path.relative_to(output_dir)),
                    "orderings": "|".join(ORDERINGS),
                    "methods_hard_to_easy": "|".join(
                        row.cells["hard_to_easy"]["metric_matrix"]["available_models"]
                    ),
                    "methods_easy_to_hard": "|".join(
                        row.cells["easy_to_hard"]["metric_matrix"]["available_models"]
                    ),
                }
            )
    return manifest_path


def write_aggregate_json(output_dir: pathlib.Path, cells: list[AggregatedCell], matrix_stat: str) -> pathlib.Path:
    json_path = output_dir / f"{DATASET_SLUG}_{matrix_stat}_aggregate_cells.json"
    payload = {
        "dataset": DATASET_SLUG,
        "dataset_label": DATASET_LABEL,
        "orderings": ORDERINGS,
        "matrix_stat": matrix_stat,
        "cells": [
            {
                "backbone": cell.backbone,
                "ordering": cell.ordering,
                "strategy_variant": cell.strategy_variant,
                "modality": cell.modality,
                "available_methods": cell.available_methods,
                "sources_by_method": cell.sources_by_method,
                "metric_matrix": {
                    "concepts_order_reference": cell.concepts_order,
                    "mean": cell.mean_matrix,
                    "std": cell.std_matrix,
                    "count": len(cell.available_methods),
                },
            }
            for cell in cells
        ],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return json_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_roots = [path.expanduser().resolve() for path in args.results_roots]

    grouped_runs = load_completed_runs(results_roots)
    cells = aggregate_cells(grouped_runs)
    rows = build_rows(cells)

    generated = 0
    for row in rows:
        if args.layout in {"combined", "both"}:
            output_path = output_path_for_row(output_dir, row, args.matrix_stat).with_suffix(".pdf")
            if output_path.exists() and not args.overwrite:
                logging.info("Skipping existing %s", output_path)
            else:
                write_row_pdf(row, output_path.with_suffix(""), args.matrix_stat)
                generated += 1
                logging.info("Wrote %s", output_path)

        if args.layout in {"split", "both"}:
            for ordering in ORDERINGS:
                output_path = output_path_for_split_row(
                    output_dir,
                    row,
                    args.matrix_stat,
                    ordering,
                ).with_suffix(".pdf")
                if output_path.exists() and not args.overwrite:
                    logging.info("Skipping existing %s", output_path)
                    continue
                write_row_pdf(row, output_path.with_suffix(""), args.matrix_stat, [ordering])
                generated += 1
                logging.info("Wrote %s", output_path)

    manifest_path = write_manifest(output_dir, rows, args.matrix_stat)
    aggregate_json_path = write_aggregate_json(output_dir, cells, args.matrix_stat)
    logging.info("Generated %d PDF(s) in %s", generated, output_dir)
    logging.info("Manifest: %s", manifest_path)
    logging.info("Aggregate JSON: %s", aggregate_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
