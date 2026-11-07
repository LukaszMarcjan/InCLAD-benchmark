#!/usr/bin/env python3
"""Render compact one-row PDF heatmaps per dataset/backbone/strategy/modality.

Known limitation (Random-ordering panels): build_article_table_summary.py
averages the per-model matrices position by position and keeps a single
``concepts_order_reference`` (the concept order of the first available model,
CFA with the default model list). This script labels
every panel with that reference order and does not read
``concepts_order_consistent_across_models``. For ``easy_to_hard`` and
``hard_to_easy`` all models share one curriculum, so the labels are exact.
For ``random`` the models were run on different shuffled orders (the flag is
False for every Random panel of the paper's grid), so cell (i, j) of a Random
panel averages different concept pairs across models and the axis labels name
only the reference model's order. Read Random panels as an aggregate over
positions in the stream, not per concept. The rendering is left unchanged so
the paper's figures stay reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

# Make the repository root importable so "examples...." resolves without
# PYTHONPATH=. (the script lives three levels below the root).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.visual_models.neurips_utilities.generate_article_backbone_heatmaps import (  # noqa: E402
    estimate_text_width,
    finite_score,
    interpolate_color_rgb,
    pdf_line,
    pdf_rect,
    pdf_text,
    rgb_fraction,
    sanitize_slug,
    write_simple_pdf,
)


ORDERINGS = ["hard_to_easy", "easy_to_hard", "random"]
ORDERING_LABELS = {
    "hard_to_easy": "Hard->Easy",
    "easy_to_hard": "Easy->Hard",
    "random": "Random",
}
STRATEGY_FILE_LABELS = {
    "naive": "ncl",
    "continual": "cl",
}


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
        "--input-json",
        type=pathlib.Path,
        default=pathlib.Path("neurips_results/article_table_summary_across_models.json"),
        help="Across-model aggregate JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("neurips_results/article_dataset_ordering_rows_pdf"),
        help="Directory for generated PDFs.",
    )
    parser.add_argument(
        "--matrix-stat",
        choices=["mean", "std"],
        default="mean",
        help="Statistic to render from the aggregated matrix payload.",
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


def normalize_matrix(matrix: Any) -> list[list[float | None]]:
    if not isinstance(matrix, list):
        raise TypeError("Matrix payload must be a list.")
    normalized: list[list[float | None]] = []
    width: int | None = None
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("Matrix rows must be lists.")
        normalized_row: list[float | None] = []
        for value in row:
            if value is None:
                normalized_row.append(None)
            elif isinstance(value, bool):
                raise TypeError("Boolean values are not valid matrix entries.")
            elif isinstance(value, (int, float)):
                normalized_row.append(float(value))
            else:
                raise TypeError(f"Unsupported matrix value: {value!r}")
        if width is None:
            width = len(normalized_row)
        elif width != len(normalized_row):
            raise ValueError("Matrix rows must have equal length.")
        normalized.append(normalized_row)
    return normalized


def display_label(label: str) -> str:
    return label.replace("_", " ")


def choose_cell_size(concept_count: int) -> int:
    if concept_count <= 0:
        return 36
    target = int(round(390.0 / concept_count))
    return max(26, min(110, target))


def choose_label_font(cell_size: int) -> float:
    return max(10.0, min(32.0, round(cell_size * 0.62, 1)))


def choose_value_font(cell_size: int) -> float:
    return max(8.0, min(26.0, round(cell_size * 0.38, 1)))


def choose_ordering_font(cell_size: int) -> float:
    return max(12.0, min(28.0, round(cell_size * 0.62, 1)))


def choose_uniform_label_font(cell_size: int, labels: list[str], angle_deg: float) -> float:
    base_size = choose_label_font(cell_size)
    radians = math.radians(abs(angle_deg))
    longest_width_em = max((estimate_text_width(label, 1.0) for label in labels), default=1.0)
    horizontal_coeff = longest_width_em * math.cos(radians) + math.sin(radians)
    if horizontal_coeff <= 0.0:
        return base_size
    fitted_size = (cell_size * 0.9) / horizontal_coeff
    return max(8.5, min(base_size, round(fitted_size, 1)))


def matrix_value_range(matrix_stat: str) -> tuple[float, float]:
    if matrix_stat == "std":
        return (0.0, 0.25)
    return (0.0, 1.0)


def compute_bottom_label_height(labels: list[str], font_size: float, angle_deg: float) -> float:
    if abs(angle_deg) >= 89.0:
        longest = max((estimate_text_width(label, font_size) for label in labels), default=0.0)
        return longest + 16.0
    radians = math.radians(abs(angle_deg))
    max_projection = 0.0
    for label in labels:
        width = estimate_text_width(label, font_size)
        projection = width * math.sin(radians) + font_size * math.cos(radians)
        max_projection = max(max_projection, projection)
    return max_projection + 18.0


def build_rows(records: list[dict[str, Any]]) -> list[RowFigure]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["dataset"]),
            str(record["backbone"]),
            str(record["strategy_variant"]),
            str(record["modality"]),
        )
        grouped.setdefault(key, {})[str(record["ordering"])] = record

    rows: list[RowFigure] = []
    for key, cells in sorted(grouped.items()):
        missing = [ordering for ordering in ORDERINGS if ordering not in cells]
        if missing:
            logging.warning("Skipping %s because orderings are missing: %s", key, ", ".join(missing))
            continue
        rows.append(
            RowFigure(
                dataset=key[0],
                backbone=key[1],
                strategy_variant=key[2],
                modality=key[3],
                cells=cells,
            )
        )
    return rows


def output_path_for_row(output_dir: pathlib.Path, row: RowFigure, matrix_stat: str) -> pathlib.Path:
    strategy_slug = STRATEGY_FILE_LABELS.get(row.strategy_variant, sanitize_slug(row.strategy_variant))
    stem = "_".join(
        [
            sanitize_slug(row.dataset),
            sanitize_slug(row.backbone),
            strategy_slug,
            sanitize_slug(row.modality),
            sanitize_slug(matrix_stat),
            "ordering_row",
        ]
    )
    return output_dir / stem


def render_vertical_colorbar(
    commands: list[str],
    *,
    page_height: int,
    x: float,
    y_top: float,
    bar_width: float,
    height: float,
    tick_font_size: float,
    vmin: float,
    vmax: float,
) -> None:
    black = (0.0, 0.0, 0.0)
    steps = max(120, int(height / 1.2))
    step_h = height / steps
    for step in range(steps):
        # step=0 → vmin → bottom of bar; step=steps-1 → vmax → top
        ratio = step / max(steps - 1, 1)
        value = vmin + (vmax - vmin) * ratio
        fill_rgb = rgb_fraction(interpolate_color_rgb(value, vmin=vmin, vmax=vmax))
        rect_y_top = y_top + height - (step + 1) * step_h
        pdf_rect(
            commands,
            page_height=page_height,
            x=x,
            y_top=rect_y_top,
            width=bar_width,
            height=step_h + 0.5,
            fill_rgb=fill_rgb,
            stroke_rgb=None,
        )
    border_color = rgb_fraction((80, 80, 80))
    for x1, y1, x2, y2 in [
        (x, y_top, x + bar_width, y_top),
        (x, y_top + height, x + bar_width, y_top + height),
        (x, y_top, x, y_top + height),
        (x + bar_width, y_top, x + bar_width, y_top + height),
    ]:
        pdf_line(commands, page_height=page_height, x1=x1, y1=y1, x2=x2, y2=y2,
                 stroke_rgb=border_color, line_width=0.6)
    tick_x = x + bar_width + 4.0
    for tick_value in [vmin, (vmin + vmax) / 2.0, vmax]:
        ratio = 0.0 if vmax <= vmin else (tick_value - vmin) / (vmax - vmin)
        tick_y = y_top + height * (1.0 - ratio)
        pdf_line(
            commands,
            page_height=page_height,
            x1=x + bar_width,
            y1=tick_y,
            x2=tick_x - 1.0,
            y2=tick_y,
            stroke_rgb=rgb_fraction((60, 60, 60)),
            line_width=0.6,
        )
        pdf_text(
            commands,
            page_height=page_height,
            x=tick_x,
            y=tick_y + tick_font_size * 0.35,
            text=f"{tick_value:.2f}",
            font_size=tick_font_size,
            color=black,
            anchor="start",
        )


def render_panel(
    commands: list[str],
    *,
    page_height: int,
    matrix: list[list[float | None]],
    concepts: list[str],
    x_offset: float,
    y_offset: float,
    cell_size: int,
    label_width: float,
    label_font_size: float,
    value_font_size: float,
    ordering_font_size: float,
    bottom_label_height: float,
    ordering_label: str,
    matrix_stat: str,
) -> float:
    black = (0.0, 0.0, 0.0)
    matrix_size = len(concepts) * cell_size
    matrix_x = x_offset + label_width
    matrix_bottom_y = y_offset + matrix_size
    label_angle = -75.0
    vmin, vmax = matrix_value_range(matrix_stat)

    for row_index, concept in enumerate(concepts):
        y = y_offset + row_index * cell_size
        pdf_text(
            commands,
            page_height=page_height,
            x=x_offset + label_width - 6.0,
            y=y + cell_size / 2.0 + label_font_size * 0.28,
            text=display_label(concept),
            font_size=label_font_size,
            color=black,
            anchor="end",
        )

        for col_index, value in enumerate(matrix[row_index]):
            if col_index > row_index:
                continue
            x = matrix_x + col_index * cell_size
            fill_rgb = rgb_fraction(interpolate_color_rgb(value, vmin=vmin, vmax=vmax))
            pdf_rect(
                commands,
                page_height=page_height,
                x=x,
                y_top=y,
                width=cell_size,
                height=cell_size,
                fill_rgb=fill_rgb,
                stroke_rgb=None,
            )
            if finite_score(value):
                pdf_text(
                    commands,
                    page_height=page_height,
                    x=x + cell_size / 2.0,
                    y=y + cell_size / 2.0 + value_font_size * 0.28,
                    text=f"{value:.2f}",
                    font_size=value_font_size,
                    font_name="F2",
                    color=black,
                    anchor="middle",
                )

    for col_index, concept in enumerate(concepts):
        x = matrix_x + col_index * cell_size + cell_size / 2.0
        pdf_text(
            commands,
            page_height=page_height,
            x=x,
            y=matrix_bottom_y + 6.0,
            text=display_label(concept),
            font_size=label_font_size,
            color=black,
            rotate_deg=label_angle,
        )

    pdf_text(
        commands,
        page_height=page_height,
        x=matrix_x + matrix_size / 2.0,
        y=matrix_bottom_y + bottom_label_height + ordering_font_size + 2.0,
        text=ordering_label,
        font_size=ordering_font_size,
        font_name="F2",
        color=black,
        anchor="middle",
    )
    return x_offset + label_width + matrix_size


def write_row_pdf(row: RowFigure, output_path: pathlib.Path, matrix_stat: str) -> None:
    first_record = row.cells[ORDERINGS[0]]
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

    width = int(round(
        left_margin
        + colorbar_section_width
        + 3 * (label_width + matrix_size)
        + 2 * panel_gap
        + right_margin
    ))
    height = int(round(
        top_margin
        + matrix_size
        + bottom_label_height
        + ordering_space
        + 14.0
    ))

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

    vmin, vmax = matrix_value_range(matrix_stat)
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
    for ordering in ORDERINGS:
        record = row.cells[ordering]
        matrix_payload = record["metric_matrix"]
        ordered_concepts = [str(item) for item in matrix_payload["concepts_order_reference"]]
        matrix = normalize_matrix(matrix_payload[matrix_stat])
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


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    records = payload.get("records", payload.get("cells", []))
    if not isinstance(records, list):
        raise TypeError("Expected top-level 'records' or 'cells' list.")

    rows = build_rows(records)
    generated = 0
    for row in rows:
        output_path = output_path_for_row(args.output_dir, row, args.matrix_stat)
        write_row_pdf(row, output_path, args.matrix_stat)
        generated += 1
        logging.info("Wrote %s", output_path.with_suffix(".pdf"))
    logging.info("Generated %d PDFs in %s", generated, args.output_dir)


if __name__ == "__main__":
    main()
