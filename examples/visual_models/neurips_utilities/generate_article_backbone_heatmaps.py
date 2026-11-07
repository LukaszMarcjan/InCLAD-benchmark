"""
Generate article-ready heatmaps from the per-backbone aggregate summary JSON.

The input is expected to be the output of:

    examples/visual_models/build_article_table_summary.py

That JSON contains one aggregated cell for each:

    dataset x backbone x ordering x modality x strategy_variant

Each cell also stores `metric_matrix.mean/std`, so this script can render
heatmaps aggregated across models, separately for each backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
from dataclasses import dataclass
from html import escape
from typing import Any

logger = logging.getLogger(__name__)

BACKBONES = ["resnet18", "wide_resnet50_2"]
DATASETS = ["mvtec", "visa", "btech", "mpdd", "dagm"]
ORDERINGS = ["easy_to_hard", "hard_to_easy", "random"]
STRATEGY_VARIANTS = ["naive", "continual"]
MODALITIES = ["image", "pixel"]

DATASET_LABELS = {
    "mvtec": "MVTec",
    "visa": "VisA",
    "btech": "BTAD",
    "mpdd": "MPDD",
    "dagm": "DAGM",
}
ORDERING_LABELS = {
    "easy_to_hard": "Curriculum UP",
    "hard_to_easy": "Curriculum Down",
    "random": "Random",
}
STRATEGY_LABELS = {
    "naive": "NCL",
    "continual": "CL",
}
MODALITY_LABELS = {
    "image": "Image-level",
    "pixel": "Pixel-level",
}


@dataclass(slots=True)
class HeatmapRenderOptions:
    annotate: bool
    font_size: int
    title_font_size: int
    axis_font_size: int
    cell_font_size: int
    overwrite: bool
    dry_run: bool
    show_upper_diagonal: bool
    panel_layout: str
    show_title: bool
    show_panel_titles: bool
    show_x_axis_title: bool
    show_y_axis_title: bool
    output_format: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregated backbone heatmaps from article_table_summary_across_models.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-json",
        type=pathlib.Path,
        default=pathlib.Path("../../../ordering_files/neurips_results/article_table_summary_across_models.json"),
        help="JSON produced by build_article_table_summary.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("../../../ordering_files/neurips_results/article_heatmaps_by_backbone_combined"),
        help="Directory where SVG heatmaps and manifest CSV will be written.",
    )
    parser.add_argument(
        "--matrix-stat",
        choices=["mean", "std", "both"],
        default="mean",
        help="Whether to render mean heatmaps, std heatmaps, or both.",
    )
    parser.add_argument(
        "--modality",
        choices=["all", "image", "pixel", "separate"],
        default="all",
        help="Use 'all' for one combined image+pixel heatmap per slot, 'separate' for distinct image/pixel outputs, or choose one modality.",
    )
    parser.add_argument(
        "--output-format",
        choices=["svg", "pdf", "both", "json-only"],
        default="svg",
        help="Which heatmap file format to write. Use 'json-only' to export payload JSONs without rendering heatmaps.",
    )
    parser.add_argument(
        "--annotate",
        dest="annotate",
        action="store_true",
        help="Annotate heatmap cells with numeric values.",
    )
    parser.add_argument(
        "--no-annotate",
        dest="annotate",
        action="store_false",
        help="Disable numeric annotations.",
    )
    parser.set_defaults(annotate=True)
    parser.add_argument(
        "--show-upper-diagonal",
        action="store_true",
        help="Render the full matrix instead of muting cells above the diagonal.",
    )
    parser.add_argument(
        "--hide-title",
        dest="show_title",
        action="store_false",
        help="Hide the main heatmap title.",
    )
    parser.add_argument(
        "--show-title",
        dest="show_title",
        action="store_true",
        help="Show the main heatmap title.",
    )
    parser.set_defaults(show_title=True)
    parser.add_argument(
        "--hide-panel-titles",
        dest="show_panel_titles",
        action="store_false",
        help="Hide panel titles such as 'Image-level' and 'Pixel-level'.",
    )
    parser.add_argument(
        "--show-panel-titles",
        dest="show_panel_titles",
        action="store_true",
        help="Show panel titles such as 'Image-level' and 'Pixel-level'.",
    )
    parser.set_defaults(show_panel_titles=True)
    parser.add_argument(
        "--hide-x-axis-title",
        dest="show_x_axis_title",
        action="store_false",
        help="Hide the X-axis title.",
    )
    parser.add_argument(
        "--show-x-axis-title",
        dest="show_x_axis_title",
        action="store_true",
        help="Show the X-axis title.",
    )
    parser.set_defaults(show_x_axis_title=True)
    parser.add_argument(
        "--hide-y-axis-title",
        dest="show_y_axis_title",
        action="store_false",
        help="Hide the Y-axis title.",
    )
    parser.add_argument(
        "--show-y-axis-title",
        dest="show_y_axis_title",
        action="store_true",
        help="Show the Y-axis title.",
    )
    parser.set_defaults(show_y_axis_title=True)
    parser.add_argument(
        "--panel-layout",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="How to arrange image/pixel panels for combined heatmaps.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to all font sizes.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="Base font size for labels.",
    )
    parser.add_argument(
        "--title-font-size",
        type=int,
        default=16,
        help="Heatmap title font size.",
    )
    parser.add_argument(
        "--axis-font-size",
        type=int,
        default=13,
        help="Axis title font size.",
    )
    parser.add_argument(
        "--cell-font-size",
        type=int,
        default=11,
        help="Cell annotation font size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating files.",
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


def finite_score(score: float | None) -> bool:
    return score is not None and math.isfinite(score)


def interpolate_color_rgb(value: float | None, *, vmin: float, vmax: float) -> tuple[int, int, int]:
    if not finite_score(value):
        return (220, 220, 220)

    assert value is not None
    if vmax <= vmin:
        normalized = 0.0
    else:
        clamped = max(vmin, min(vmax, value))
        normalized = (clamped - vmin) / (vmax - vmin)

    stops = [
        (0.0, (13, 8, 135)),
        (0.5, (204, 71, 120)),
        (1.0, (240, 249, 33)),
    ]
    for index in range(len(stops) - 1):
        left_value, left_color = stops[index]
        right_value, right_color = stops[index + 1]
        if left_value <= normalized <= right_value:
            ratio = (normalized - left_value) / (right_value - left_value)
            color = tuple(
                round(left_channel + (right_channel - left_channel) * ratio)
                for left_channel, right_channel in zip(left_color, right_color)
            )
            return color  # type: ignore[return-value]
    color = stops[-1][1]
    return color


def interpolate_color(value: float | None, *, vmin: float, vmax: float) -> str:
    color = interpolate_color_rgb(value, vmin=vmin, vmax=vmax)
    return f"rgb({color[0]},{color[1]},{color[2]})"


def text_color_for_value(value: float | None, *, vmin: float, vmax: float) -> str:
    if not finite_score(value):
        return "black"
    assert value is not None
    if vmax <= vmin:
        normalized = 0.0
    else:
        clamped = max(vmin, min(vmax, value))
        normalized = (clamped - vmin) / (vmax - vmin)
    return "white" if normalized < 0.22 or normalized > 0.78 else "black"


def sanitize_slug(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
    )


def iter_output_formats(output_format: str) -> list[str]:
    if output_format == "both":
        return ["svg", "pdf"]
    if output_format == "json-only":
        return []
    return [output_format]


def pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def rgb_fraction(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)


_HELVETICA_CHAR_WIDTH = {
    ' ': 278, '!': 278, '"': 355, '#': 556, '$': 556, '%': 889, '&': 667,
    "'": 191, '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333,
    '.': 278, '/': 278, '0': 556, '1': 556, '2': 556, '3': 556, '4': 556,
    '5': 556, '6': 556, '7': 556, '8': 556, '9': 556, ':': 278, ';': 278,
    '<': 584, '=': 584, '>': 584, '?': 556, '@': 1015,
    'A': 667, 'B': 667, 'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778,
    'H': 722, 'I': 278, 'J': 500, 'K': 667, 'L': 556, 'M': 833, 'N': 722,
    'O': 778, 'P': 667, 'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722,
    'V': 667, 'W': 944, 'X': 667, 'Y': 667, 'Z': 611,
    '[': 278, '\\': 278, ']': 278, '^': 469, '_': 556, '`': 333,
    'a': 556, 'b': 556, 'c': 500, 'd': 556, 'e': 556, 'f': 278, 'g': 556,
    'h': 556, 'i': 222, 'j': 222, 'k': 500, 'l': 222, 'm': 833, 'n': 556,
    'o': 556, 'p': 556, 'q': 556, 'r': 333, 's': 500, 't': 278, 'u': 556,
    'v': 500, 'w': 722, 'x': 500, 'y': 500, 'z': 500,
}
_HELVETICA_DEFAULT_WIDTH = 556


def estimate_text_width(text: str, font_size: float) -> float:
    return sum(_HELVETICA_CHAR_WIDTH.get(c, _HELVETICA_DEFAULT_WIDTH) for c in text) * font_size / 1000.0


def slug_for_cell(cell: dict[str, Any], matrix_stat: str) -> str:
    return "_".join(
        [
            sanitize_slug(str(cell["dataset"])),
            sanitize_slug(str(cell["backbone"])),
            sanitize_slug(str(cell["ordering"])),
            sanitize_slug(str(cell["strategy_variant"])),
            sanitize_slug(str(cell["modality"])),
            sanitize_slug(matrix_stat),
        ]
    )


def output_path_for_cell(
    output_dir: pathlib.Path,
    cell: dict[str, Any],
    matrix_stat: str,
    output_format: str,
) -> pathlib.Path:
    suffix = ".svg" if output_format == "svg" else ".pdf"
    return (
        output_dir
        / str(cell["backbone"])
        / str(cell["dataset"])
        / str(cell["ordering"])
        / str(cell["strategy_variant"])
        / str(cell["modality"])
        / f"{slug_for_cell(cell, matrix_stat)}_heatmap{suffix}"
    )


def slug_for_combined(image_cell: dict[str, Any], matrix_stat: str) -> str:
    return "_".join(
        [
            sanitize_slug(str(image_cell["dataset"])),
            sanitize_slug(str(image_cell["backbone"])),
            sanitize_slug(str(image_cell["ordering"])),
            sanitize_slug(str(image_cell["strategy_variant"])),
            sanitize_slug(matrix_stat),
        ]
    )


def output_path_for_combined(
    output_dir: pathlib.Path,
    image_cell: dict[str, Any],
    matrix_stat: str,
    output_format: str,
) -> pathlib.Path:
    suffix = ".svg" if output_format == "svg" else ".pdf"
    return (
        output_dir
        / str(image_cell["backbone"])
        / str(image_cell["dataset"])
        / str(image_cell["ordering"])
        / str(image_cell["strategy_variant"])
        / f"{slug_for_combined(image_cell, matrix_stat)}_heatmap{suffix}"
    )


def json_path_for_output(output_path: pathlib.Path) -> pathlib.Path:
    return output_path.with_suffix(".json")


def normalize_matrix(matrix: Any) -> list[list[float | None]] | None:
    if not isinstance(matrix, list):
        return None
    normalized: list[list[float | None]] = []
    width: int | None = None
    for row in matrix:
        if not isinstance(row, list):
            return None
        normalized_row: list[float | None] = []
        for value in row:
            if value is None:
                normalized_row.append(None)
            elif isinstance(value, bool):
                return None
            elif isinstance(value, (int, float)):
                normalized_row.append(float(value))
            else:
                return None
        if width is None:
            width = len(normalized_row)
        elif width != len(normalized_row):
            return None
        normalized.append(normalized_row)
    return normalized


def matrix_value_range(matrix: list[list[float | None]], matrix_stat: str) -> tuple[float, float]:
    finite_values = [value for row in matrix for value in row if finite_score(value)]
    if not finite_values:
        return (0.0, 1.0)
    if matrix_stat == "std":
        return (0.0, max(finite_values))
    return (0.0, 1.0)


def render_value(value: float | None, matrix_stat: str) -> str:
    if not finite_score(value):
        return "NA"
    assert value is not None
    return f"{value:.3f}" if matrix_stat == "std" else f"{value:.2f}"


def build_title(cell: dict[str, Any], matrix_stat: str, matrix_count: int | None, expected_count: int | None) -> str:
    coverage = ""
    if matrix_count is not None and expected_count is not None:
        coverage = f" (n={matrix_count}/{expected_count})"
    return (
        f"{cell['dataset_label']} | {cell['backbone']} | {cell['ordering_label']} | "
        f"{cell['strategy_label']} | {cell['modality_label']} | {matrix_stat.upper()}{coverage}"
    )


def build_combined_title(
    image_cell: dict[str, Any],
    image_count: int | None,
    pixel_count: int | None,
    expected_count: int | None,
    matrix_stat: str,
) -> str:
    coverage_parts: list[str] = []
    if image_count is not None and expected_count is not None:
        coverage_parts.append(f"img n={image_count}/{expected_count}")
    if pixel_count is not None and expected_count is not None:
        coverage_parts.append(f"px n={pixel_count}/{expected_count}")
    coverage = f" ({', '.join(coverage_parts)})" if coverage_parts else ""
    return (
        f"{image_cell['dataset_label']} | {image_cell['backbone']} | {image_cell['ordering_label']} | "
        f"{image_cell['strategy_label']} | Image + Pixel | {matrix_stat.upper()}{coverage}"
    )


def compute_annotation_font_size(cell_size: int, requested_font_size: int) -> int:
    return max(requested_font_size, int(cell_size * 0.24))


def compute_panel_header_height(options: HeatmapRenderOptions, panel_title: str) -> int:
    height = 0
    if panel_title and options.show_panel_titles:
        height += 26
    if options.show_x_axis_title:
        height += 22
    return height


def compute_top_margin(options: HeatmapRenderOptions, panel_title: str) -> int:
    rotated_labels_clearance = max(86, int(options.font_size * 4.5))
    return compute_panel_header_height(options, panel_title) + rotated_labels_clearance


def append_svg_heatmap_panel(
    lines: list[str],
    *,
    matrix: list[list[float | None]],
    concepts_order: list[str],
    panel_title: str,
    x_offset: int,
    y_offset: int,
    top_margin: int,
    label_width: int,
    cell_size: int,
    colorbar_gap: int,
    colorbar_width: int,
    options: HeatmapRenderOptions,
    matrix_stat: str,
    gradient_id: str,
) -> int:
    matrix_width = len(concepts_order) * cell_size
    vmin, vmax = matrix_value_range(matrix, matrix_stat)
    panel_center_x = x_offset + label_width + matrix_width / 2
    header_y = y_offset + 24

    if panel_title and options.show_panel_titles:
        lines.append(
            f'<text class="subtitle" x="{panel_center_x:.1f}" y="{header_y}" text-anchor="middle">{escape(panel_title)}</text>'
        )
        header_y += 24
    if options.show_x_axis_title:
        lines.append(
            f'<text class="axis" x="{panel_center_x:.1f}" y="{header_y}" text-anchor="middle">Evaluating on concept</text>'
        )
    if options.show_y_axis_title:
        lines.append(
            f'<text class="axis" x="{x_offset + 18}" y="{y_offset + top_margin + len(concepts_order) * cell_size / 2:.1f}" '
            f'text-anchor="middle" transform="rotate(-90 {x_offset + 18} '
            f'{y_offset + top_margin + len(concepts_order) * cell_size / 2:.1f})">After learning concept</text>'
        )

    for col_index, concept in enumerate(concepts_order):
        x = x_offset + label_width + col_index * cell_size + cell_size / 2
        y = y_offset + top_margin - 10
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="start" '
            f'transform="rotate(-55 {x:.1f} {y:.1f})">{escape(concept)}</text>'
        )

    for row_index, learned_concept in enumerate(concepts_order):
        y = y_offset + top_margin + row_index * cell_size
        lines.append(
            f'<text x="{x_offset + label_width - 8}" y="{y + cell_size / 2 + 4:.1f}" text-anchor="end">{escape(learned_concept)}</text>'
        )
        for col_index in range(len(concepts_order)):
            x = x_offset + label_width + col_index * cell_size
            value = matrix[row_index][col_index]
            masked = (not options.show_upper_diagonal) and (col_index > row_index)
            fill = "#f5f5f5" if masked else interpolate_color(value, vmin=vmin, vmax=vmax)
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="white" stroke-width="1"/>'
            )
            if options.annotate and not masked:
                color = text_color_for_value(value, vmin=vmin, vmax=vmax)
                label = render_value(value, matrix_stat)
                lines.append(
                    f'<text class="cell-label" x="{x + cell_size / 2:.1f}" y="{y + cell_size / 2 + 3:.1f}" '
                    f'text-anchor="middle" fill="{color}">{label}</text>'
                )

    colorbar_x = x_offset + label_width + matrix_width + colorbar_gap
    colorbar_y = y_offset + top_margin
    colorbar_h = len(concepts_order) * cell_size
    lines.extend(
        [
            "<defs>",
            f'<linearGradient id="{gradient_id}" x1="0%" y1="100%" x2="0%" y2="0%">',
            f'<stop offset="0%" stop-color="{interpolate_color(vmin, vmin=vmin, vmax=vmax)}"/>',
            f'<stop offset="50%" stop-color="{interpolate_color((vmin + vmax) / 2.0, vmin=vmin, vmax=vmax)}"/>',
            f'<stop offset="100%" stop-color="{interpolate_color(vmax, vmin=vmin, vmax=vmax)}"/>',
            "</linearGradient>",
            "</defs>",
            f'<rect x="{colorbar_x}" y="{colorbar_y}" width="{colorbar_width}" height="{colorbar_h}" fill="url(#{gradient_id})" stroke="#444" stroke-width="0.8"/>',
        ]
    )
    tick_values = [vmin, (vmin + vmax) / 2.0, vmax]
    for tick_index, tick_value in enumerate(tick_values):
        ratio = 0.0 if len(tick_values) == 1 else tick_index / (len(tick_values) - 1)
        tick_y = colorbar_y + colorbar_h - ratio * colorbar_h
        lines.append(
            f'<line x1="{colorbar_x + colorbar_width}" y1="{tick_y:.1f}" x2="{colorbar_x + colorbar_width + 6}" y2="{tick_y:.1f}" stroke="#333" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{colorbar_x + colorbar_width + 10}" y="{tick_y + 4:.1f}" text-anchor="start">{tick_value:.2f}</text>'
        )

    return colorbar_x + colorbar_width + 60


def write_svg_heatmap(
    matrix: list[list[float | None]],
    concepts_order: list[str],
    title: str,
    output_path: pathlib.Path,
    options: HeatmapRenderOptions,
    matrix_stat: str,
) -> None:
    max_label_length = max((len(concept) for concept in concepts_order), default=8)
    cell_size = max(48, int(options.cell_font_size * 4.2))
    annotation_font_size = compute_annotation_font_size(cell_size, options.cell_font_size)
    label_width = max(120, int(max_label_length * max(options.font_size * 0.72, 6)) + 24)
    top_margin = compute_top_margin(options, "")
    panel_y_offset = 40 if options.show_title else 0
    right_margin = 60
    bottom_margin = 36
    colorbar_width = 24
    colorbar_gap = 16
    panel_width = label_width + len(concepts_order) * cell_size + colorbar_gap + colorbar_width + 60
    width = panel_width + right_margin
    height = panel_y_offset + top_margin + len(concepts_order) * cell_size + bottom_margin

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        f"text {{ font-family: Helvetica, Arial, sans-serif; font-size: {options.font_size}px; fill: black; }}",
        f".title {{ font-size: {options.title_font_size}px; font-weight: 700; fill: black; }}",
        f".subtitle {{ font-size: {max(options.axis_font_size + 1, 12)}px; font-weight: 700; fill: black; }}",
        f".axis {{ font-size: {options.axis_font_size}px; font-weight: 600; fill: black; }}",
        f".cell-label {{ font-size: {annotation_font_size}px; font-weight: 600; }}",
        f".muted {{ fill: #f3f3f3; }}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    if options.show_title:
        lines.append(f'<text class="title" x="{width / 2:.1f}" y="24" text-anchor="middle">{escape(title)}</text>')
    append_svg_heatmap_panel(
        lines,
        matrix=matrix,
        concepts_order=concepts_order,
        panel_title="",
        x_offset=0,
        y_offset=panel_y_offset,
        top_margin=top_margin,
        label_width=label_width,
        cell_size=cell_size,
        colorbar_gap=colorbar_gap,
        colorbar_width=colorbar_width,
        options=options,
        matrix_stat=matrix_stat,
        gradient_id=sanitize_slug(output_path.stem) + "_gradient_single",
    )
    # Replace the empty subtitle introduced by the shared panel renderer.
    lines = [line for line in lines if 'class="subtitle"' not in line or "></text>" not in line]
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_dual_heatmap(
    *,
    image_matrix: list[list[float | None]],
    pixel_matrix: list[list[float | None]],
    concepts_order: list[str],
    title: str,
    output_path: pathlib.Path,
    options: HeatmapRenderOptions,
    matrix_stat: str,
) -> None:
    max_label_length = max((len(concept) for concept in concepts_order), default=8)
    cell_size = max(48, int(options.cell_font_size * 4.2))
    annotation_font_size = compute_annotation_font_size(cell_size, options.cell_font_size)
    label_width = max(120, int(max_label_length * max(options.font_size * 0.72, 6)) + 24)
    top_margin = compute_top_margin(options, "Image-level")
    global_title_offset = 40 if options.show_title else 0
    right_margin = 60
    bottom_margin = 36
    colorbar_width = 24
    colorbar_gap = 16
    panel_gap = 40
    panel_width = label_width + len(concepts_order) * cell_size + colorbar_gap + colorbar_width + 60
    panel_height = top_margin + len(concepts_order) * cell_size + bottom_margin
    if options.panel_layout == "horizontal":
        width = panel_width * 2 + panel_gap + right_margin
        height = global_title_offset + panel_height
        image_x_offset = 0
        image_y_offset = global_title_offset
        pixel_x_offset = panel_width + panel_gap
        pixel_y_offset = global_title_offset
    else:
        width = panel_width + right_margin
        height = global_title_offset + panel_height * 2 + panel_gap
        image_x_offset = 0
        image_y_offset = global_title_offset
        pixel_x_offset = 0
        pixel_y_offset = global_title_offset + panel_height + panel_gap

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        f"text {{ font-family: Helvetica, Arial, sans-serif; font-size: {options.font_size}px; fill: black; }}",
        f".title {{ font-size: {options.title_font_size}px; font-weight: 700; fill: black; }}",
        f".subtitle {{ font-size: {max(options.axis_font_size + 1, 12)}px; font-weight: 700; fill: black; }}",
        f".axis {{ font-size: {options.axis_font_size}px; font-weight: 600; fill: black; }}",
        f".cell-label {{ font-size: {annotation_font_size}px; font-weight: 600; }}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    if options.show_title:
        lines.append(f'<text class="title" x="{width / 2:.1f}" y="24" text-anchor="middle">{escape(title)}</text>')

    append_svg_heatmap_panel(
        lines,
        matrix=image_matrix,
        concepts_order=concepts_order,
        panel_title="Image-level",
        x_offset=image_x_offset,
        y_offset=image_y_offset,
        top_margin=top_margin,
        label_width=label_width,
        cell_size=cell_size,
        colorbar_gap=colorbar_gap,
        colorbar_width=colorbar_width,
        options=options,
        matrix_stat=matrix_stat,
        gradient_id=sanitize_slug(output_path.stem) + "_gradient_image",
    )
    append_svg_heatmap_panel(
        lines,
        matrix=pixel_matrix,
        concepts_order=concepts_order,
        panel_title="Pixel-level",
        x_offset=pixel_x_offset,
        y_offset=pixel_y_offset,
        top_margin=top_margin,
        label_width=label_width,
        cell_size=cell_size,
        colorbar_gap=colorbar_gap,
        colorbar_width=colorbar_width,
        options=options,
        matrix_stat=matrix_stat,
        gradient_id=sanitize_slug(output_path.stem) + "_gradient_pixel",
    )

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def pdf_text(
    commands: list[str],
    *,
    page_height: float,
    x: float,
    y: float,
    text: str,
    font_size: float,
    font_name: str = "F1",
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    anchor: str = "start",
    rotate_deg: float | None = None,
) -> None:
    if not text:
        return
    width = estimate_text_width(text, font_size)
    if anchor == "middle":
        offset = width / 2.0
    elif anchor == "end":
        offset = width
    else:
        offset = 0.0

    if offset > 0.0:
        if rotate_deg is not None:
            rad = math.radians(rotate_deg)
            x -= offset * math.cos(rad)
            y += offset * math.sin(rad)
        else:
            x -= offset

    baseline_y = page_height - y
    safe_text = pdf_escape(text)
    commands.append("BT")
    commands.append(f"/{font_name} {font_size:.2f} Tf")
    commands.append(f"{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} rg")
    if rotate_deg is None:
        commands.append(f"1 0 0 1 {x:.2f} {baseline_y:.2f} Tm")
    else:
        radians = math.radians(rotate_deg)
        cos_v = math.cos(radians)
        sin_v = math.sin(radians)
        commands.append(
            f"{cos_v:.6f} {sin_v:.6f} {-sin_v:.6f} {cos_v:.6f} {x:.2f} {baseline_y:.2f} Tm"
        )
    commands.append(f"({safe_text}) Tj")
    commands.append("ET")


def pdf_rect(
    commands: list[str],
    *,
    page_height: float,
    x: float,
    y_top: float,
    width: float,
    height: float,
    fill_rgb: tuple[float, float, float],
    stroke_rgb: tuple[float, float, float] | None = None,
    line_width: float = 1.0,
) -> None:
    bottom_y = page_height - y_top - height
    commands.append("q")
    commands.append(f"{line_width:.2f} w")
    commands.append(f"{fill_rgb[0]:.4f} {fill_rgb[1]:.4f} {fill_rgb[2]:.4f} rg")
    if stroke_rgb is not None:
        commands.append(f"{stroke_rgb[0]:.4f} {stroke_rgb[1]:.4f} {stroke_rgb[2]:.4f} RG")
        commands.append(f"{x:.2f} {bottom_y:.2f} {width:.2f} {height:.2f} re B")
    else:
        commands.append(f"{x:.2f} {bottom_y:.2f} {width:.2f} {height:.2f} re f")
    commands.append("Q")


def pdf_line(
    commands: list[str],
    *,
    page_height: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke_rgb: tuple[float, float, float],
    line_width: float = 1.0,
) -> None:
    commands.append("q")
    commands.append(f"{line_width:.2f} w")
    commands.append(f"{stroke_rgb[0]:.4f} {stroke_rgb[1]:.4f} {stroke_rgb[2]:.4f} RG")
    commands.append(f"{x1:.2f} {page_height - y1:.2f} m {x2:.2f} {page_height - y2:.2f} l S")
    commands.append("Q")


def write_simple_pdf(output_path: pathlib.Path, width: int, height: int, commands: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream_data = "\n".join(commands).encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>"
        ).encode("latin-1"),
        b"<< /Length " + str(len(stream_data)).encode("latin-1") + b" >>\nstream\n" + stream_data + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
    )
    output_path.write_bytes(pdf)


def append_pdf_heatmap_panel(
    commands: list[str],
    *,
    page_width: int,
    page_height: int,
    matrix: list[list[float | None]],
    concepts_order: list[str],
    panel_title: str,
    x_offset: int,
    y_offset: int,
    top_margin: int,
    label_width: int,
    cell_size: int,
    colorbar_gap: int,
    colorbar_width: int,
    options: HeatmapRenderOptions,
    matrix_stat: str,
) -> None:
    del page_width
    matrix_width = len(concepts_order) * cell_size
    vmin, vmax = matrix_value_range(matrix, matrix_stat)
    panel_center_x = x_offset + label_width + matrix_width / 2
    header_y = y_offset + 24
    black = (0.0, 0.0, 0.0)

    if panel_title and options.show_panel_titles:
        pdf_text(
            commands,
            page_height=page_height,
            x=panel_center_x,
            y=header_y,
            text=panel_title,
            font_size=max(options.axis_font_size + 1, 12),
            font_name="F2",
            color=black,
            anchor="middle",
        )
        header_y += 24
    if options.show_x_axis_title:
        pdf_text(
            commands,
            page_height=page_height,
            x=panel_center_x,
            y=header_y,
            text="Evaluating on concept",
            font_size=options.axis_font_size,
            font_name="F2",
            color=black,
            anchor="middle",
        )
    if options.show_y_axis_title:
        pdf_text(
            commands,
            page_height=page_height,
            x=x_offset + 18,
            y=y_offset + top_margin + len(concepts_order) * cell_size / 2,
            text="After learning concept",
            font_size=options.axis_font_size,
            font_name="F2",
            color=black,
            anchor="middle",
            rotate_deg=90.0,
        )

    for col_index, concept in enumerate(concepts_order):
        x = x_offset + label_width + col_index * cell_size + cell_size / 2
        y = y_offset + top_margin - 10
        pdf_text(
            commands,
            page_height=page_height,
            x=x,
            y=y,
            text=concept,
            font_size=options.font_size,
            color=black,
            rotate_deg=-55.0,
        )

    annotation_font_size = compute_annotation_font_size(cell_size, options.cell_font_size)
    for row_index, learned_concept in enumerate(concepts_order):
        y = y_offset + top_margin + row_index * cell_size
        pdf_text(
            commands,
            page_height=page_height,
            x=x_offset + label_width - 8,
            y=y + cell_size / 2 + 4,
            text=learned_concept,
            font_size=options.font_size,
            color=black,
            anchor="end",
        )
        for col_index in range(len(concepts_order)):
            x = x_offset + label_width + col_index * cell_size
            value = matrix[row_index][col_index]
            masked = (not options.show_upper_diagonal) and (col_index > row_index)
            fill_rgb = (245, 245, 245) if masked else interpolate_color_rgb(value, vmin=vmin, vmax=vmax)
            pdf_rect(
                commands,
                page_height=page_height,
                x=x,
                y_top=y,
                width=cell_size,
                height=cell_size,
                fill_rgb=rgb_fraction(fill_rgb),
                stroke_rgb=(1.0, 1.0, 1.0),
            )
            if options.annotate and not masked:
                text_color = rgb_fraction((255, 255, 255)) if text_color_for_value(value, vmin=vmin, vmax=vmax) == "white" else black
                pdf_text(
                    commands,
                    page_height=page_height,
                    x=x + cell_size / 2,
                    y=y + cell_size / 2 + annotation_font_size * 0.32,
                    text=render_value(value, matrix_stat),
                    font_size=annotation_font_size,
                    font_name="F2",
                    color=text_color,
                    anchor="middle",
                )

    colorbar_x = x_offset + label_width + matrix_width + colorbar_gap
    colorbar_y = y_offset + top_margin
    colorbar_h = len(concepts_order) * cell_size
    steps = max(60, int(colorbar_h / 3))
    for step in range(steps):
        ratio = step / max(steps - 1, 1)
        value = vmin + (vmax - vmin) * ratio
        fill_rgb = rgb_fraction(interpolate_color_rgb(value, vmin=vmin, vmax=vmax))
        rect_y = colorbar_y + colorbar_h - (step + 1) * colorbar_h / steps
        pdf_rect(
            commands,
            page_height=page_height,
            x=colorbar_x,
            y_top=rect_y,
            width=colorbar_width,
            height=colorbar_h / steps + 0.5,
            fill_rgb=fill_rgb,
            stroke_rgb=None,
        )
    pdf_rect(
        commands,
        page_height=page_height,
        x=colorbar_x,
        y_top=colorbar_y,
        width=colorbar_width,
        height=colorbar_h,
        fill_rgb=(0.0, 0.0, 0.0),
        stroke_rgb=rgb_fraction((68, 68, 68)),
        line_width=0.8,
    )
    tick_values = [vmin, (vmin + vmax) / 2.0, vmax]
    for tick_index, tick_value in enumerate(tick_values):
        ratio = 0.0 if len(tick_values) == 1 else tick_index / (len(tick_values) - 1)
        tick_y = colorbar_y + colorbar_h - ratio * colorbar_h
        pdf_line(
            commands,
            page_height=page_height,
            x1=colorbar_x + colorbar_width,
            y1=tick_y,
            x2=colorbar_x + colorbar_width + 6,
            y2=tick_y,
            stroke_rgb=rgb_fraction((51, 51, 51)),
        )
        pdf_text(
            commands,
            page_height=page_height,
            x=colorbar_x + colorbar_width + 10,
            y=tick_y + 4,
            text=f"{tick_value:.2f}",
            font_size=options.font_size,
            color=black,
            anchor="start",
        )


def write_pdf_heatmap(
    matrix: list[list[float | None]],
    concepts_order: list[str],
    title: str,
    output_path: pathlib.Path,
    options: HeatmapRenderOptions,
    matrix_stat: str,
) -> None:
    max_label_length = max((len(concept) for concept in concepts_order), default=8)
    cell_size = max(48, int(options.cell_font_size * 4.2))
    label_width = max(120, int(max_label_length * max(options.font_size * 0.72, 6)) + 24)
    top_margin = compute_top_margin(options, "")
    panel_y_offset = 40 if options.show_title else 0
    right_margin = 60
    bottom_margin = 36
    colorbar_width = 24
    colorbar_gap = 16
    panel_width = label_width + len(concepts_order) * cell_size + colorbar_gap + colorbar_width + 60
    width = panel_width + right_margin
    height = panel_y_offset + top_margin + len(concepts_order) * cell_size + bottom_margin

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
    if options.show_title:
        pdf_text(
            commands,
            page_height=height,
            x=width / 2,
            y=24,
            text=title,
            font_size=options.title_font_size,
            font_name="F2",
            anchor="middle",
        )
    append_pdf_heatmap_panel(
        commands,
        page_width=width,
        page_height=height,
        matrix=matrix,
        concepts_order=concepts_order,
        panel_title="",
        x_offset=0,
        y_offset=panel_y_offset,
        top_margin=top_margin,
        label_width=label_width,
        cell_size=cell_size,
        colorbar_gap=colorbar_gap,
        colorbar_width=colorbar_width,
        options=options,
        matrix_stat=matrix_stat,
    )
    write_simple_pdf(output_path, width, height, commands)


def write_pdf_dual_heatmap(
    *,
    image_matrix: list[list[float | None]],
    pixel_matrix: list[list[float | None]],
    concepts_order: list[str],
    title: str,
    output_path: pathlib.Path,
    options: HeatmapRenderOptions,
    matrix_stat: str,
) -> None:
    max_label_length = max((len(concept) for concept in concepts_order), default=8)
    cell_size = max(48, int(options.cell_font_size * 4.2))
    label_width = max(120, int(max_label_length * max(options.font_size * 0.72, 6)) + 24)
    top_margin = compute_top_margin(options, "Image-level")
    global_title_offset = 40 if options.show_title else 0
    right_margin = 60
    bottom_margin = 36
    colorbar_width = 24
    colorbar_gap = 16
    panel_gap = 40
    panel_width = label_width + len(concepts_order) * cell_size + colorbar_gap + colorbar_width + 60
    panel_height = top_margin + len(concepts_order) * cell_size + bottom_margin
    if options.panel_layout == "horizontal":
        width = panel_width * 2 + panel_gap + right_margin
        height = global_title_offset + panel_height
        image_x_offset = 0
        image_y_offset = global_title_offset
        pixel_x_offset = panel_width + panel_gap
        pixel_y_offset = global_title_offset
    else:
        width = panel_width + right_margin
        height = global_title_offset + panel_height * 2 + panel_gap
        image_x_offset = 0
        image_y_offset = global_title_offset
        pixel_x_offset = 0
        pixel_y_offset = global_title_offset + panel_height + panel_gap

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
    if options.show_title:
        pdf_text(
            commands,
            page_height=height,
            x=width / 2,
            y=24,
            text=title,
            font_size=options.title_font_size,
            font_name="F2",
            anchor="middle",
        )
    append_pdf_heatmap_panel(
        commands,
        page_width=width,
        page_height=height,
        matrix=image_matrix,
        concepts_order=concepts_order,
        panel_title="Image-level",
        x_offset=image_x_offset,
        y_offset=image_y_offset,
        top_margin=top_margin,
        label_width=label_width,
        cell_size=cell_size,
        colorbar_gap=colorbar_gap,
        colorbar_width=colorbar_width,
        options=options,
        matrix_stat=matrix_stat,
    )
    append_pdf_heatmap_panel(
        commands,
        page_width=width,
        page_height=height,
        matrix=pixel_matrix,
        concepts_order=concepts_order,
        panel_title="Pixel-level",
        x_offset=pixel_x_offset,
        y_offset=pixel_y_offset,
        top_margin=top_margin,
        label_width=label_width,
        cell_size=cell_size,
        colorbar_gap=colorbar_gap,
        colorbar_width=colorbar_width,
        options=options,
        matrix_stat=matrix_stat,
    )
    write_simple_pdf(output_path, width, height, commands)


def load_cells(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"Expected 'cells' list in {path}")
    return cells


def should_include_modality(cell: dict[str, Any], modality_filter: str) -> bool:
    return modality_filter == "all" or str(cell["modality"]) == modality_filter


def iter_matrix_stats(matrix_stat: str) -> list[str]:
    if matrix_stat == "both":
        return ["mean", "std"]
    return [matrix_stat]


def write_manifest(output_dir: pathlib.Path, rows: list[dict[str, Any]]) -> pathlib.Path:
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "dataset",
        "backbone",
        "ordering",
        "strategy_variant",
        "modality",
        "matrix_stat",
        "output_format",
        "image_matrix_count",
        "pixel_matrix_count",
        "expected_count",
        "relative_output_path",
        "relative_json_path",
        "title",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return manifest_path


def matrix_to_concept_mapping(
    matrix: list[list[float | None]],
    concepts_order: list[str],
) -> dict[str, dict[str, float | None]]:
    return {
        learned_concept: {
            evaluated_concept: matrix[row_index][col_index]
            for col_index, evaluated_concept in enumerate(concepts_order)
        }
        for row_index, learned_concept in enumerate(concepts_order)
    }


def diagonal_mapping(
    matrix: list[list[float | None]],
    concepts_order: list[str],
) -> dict[str, float | None]:
    return {
        concept: matrix[index][index]
        for index, concept in enumerate(concepts_order)
    }


def build_concept_only_payload_combined(
    *,
    image_cell: dict[str, Any],
    pixel_cell: dict[str, Any],
    image_payload: dict[str, Any],
    pixel_payload: dict[str, Any],
    image_matrix: list[list[float | None]],
    pixel_matrix: list[list[float | None]],
    concepts_order: list[str],
    matrix_stat: str,
) -> dict[str, Any]:
    return {
        "dataset": image_cell["dataset"],
        "dataset_label": image_cell["dataset_label"],
        "backbone": image_cell["backbone"],
        "ordering": image_cell["ordering"],
        "ordering_label": image_cell["ordering_label"],
        "strategy_variant": image_cell["strategy_variant"],
        "strategy_label": image_cell["strategy_label"],
        "matrix_stat": matrix_stat,
        "concepts": {
            "order": concepts_order,
            "count": len(concepts_order),
        },
        "image": {
            "count": image_payload.get("count"),
            "expected_count": image_payload.get("expected_count"),
            "available_models": image_payload.get("available_models", []),
            "missing_models": image_payload.get("missing_models", []),
            "matrix": image_matrix,
            "matrix_by_concept": matrix_to_concept_mapping(image_matrix, concepts_order),
            "diagonal_by_concept": diagonal_mapping(image_matrix, concepts_order),
            "lower_triangle": image_payload.get("lower_triangle_mean" if matrix_stat == "mean" else "lower_triangle_std"),
            "lower_or_equal_triangle": image_payload.get(
                "lower_or_equal_triangle_mean" if matrix_stat == "mean" else "lower_or_equal_triangle_std"
            ),
        },
        "pixel": {
            "count": pixel_payload.get("count"),
            "expected_count": pixel_payload.get("expected_count"),
            "available_models": pixel_payload.get("available_models", []),
            "missing_models": pixel_payload.get("missing_models", []),
            "matrix": pixel_matrix,
            "matrix_by_concept": matrix_to_concept_mapping(pixel_matrix, concepts_order),
            "diagonal_by_concept": diagonal_mapping(pixel_matrix, concepts_order),
            "lower_triangle": pixel_payload.get("lower_triangle_mean" if matrix_stat == "mean" else "lower_triangle_std"),
            "lower_or_equal_triangle": pixel_payload.get(
                "lower_or_equal_triangle_mean" if matrix_stat == "mean" else "lower_or_equal_triangle_std"
            ),
        },
    }


def build_concept_only_payload_single(
    *,
    cell: dict[str, Any],
    matrix_payload: dict[str, Any],
    matrix: list[list[float | None]],
    concepts_order: list[str],
    matrix_stat: str,
) -> dict[str, Any]:
    return {
        "dataset": cell["dataset"],
        "dataset_label": cell["dataset_label"],
        "backbone": cell["backbone"],
        "ordering": cell["ordering"],
        "ordering_label": cell["ordering_label"],
        "strategy_variant": cell["strategy_variant"],
        "strategy_label": cell["strategy_label"],
        "modality": cell["modality"],
        "modality_label": cell["modality_label"],
        "matrix_stat": matrix_stat,
        "concepts": {
            "order": concepts_order,
            "count": len(concepts_order),
        },
        "matrix_info": {
            "count": matrix_payload.get("count"),
            "expected_count": matrix_payload.get("expected_count"),
            "available_models": matrix_payload.get("available_models", []),
            "missing_models": matrix_payload.get("missing_models", []),
            "matrix": matrix,
            "matrix_by_concept": matrix_to_concept_mapping(matrix, concepts_order),
            "diagonal_by_concept": diagonal_mapping(matrix, concepts_order),
            "lower_triangle": matrix_payload.get("lower_triangle_mean" if matrix_stat == "mean" else "lower_triangle_std"),
            "lower_or_equal_triangle": matrix_payload.get(
                "lower_or_equal_triangle_mean" if matrix_stat == "mean" else "lower_or_equal_triangle_std"
            ),
        },
    }


def write_json_payload(output_path: pathlib.Path, payload: dict[str, Any], options: HeatmapRenderOptions) -> pathlib.Path:
    json_path = json_path_for_output(output_path)
    if json_path.exists() and not options.overwrite and not options.dry_run:
        return json_path
    if options.dry_run:
        return json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return json_path


def generate_heatmaps(
    input_json: pathlib.Path,
    output_dir: pathlib.Path,
    matrix_stat: str,
    modality_filter: str,
    options: HeatmapRenderOptions,
) -> tuple[int, pathlib.Path]:
    cells = load_cells(input_json)
    manifest_rows: list[dict[str, Any]] = []
    written = 0
    cell_lookup = {
        (
            str(cell["backbone"]),
            str(cell["dataset"]),
            str(cell["ordering"]),
            str(cell["strategy_variant"]),
            str(cell["modality"]),
        ): cell
        for cell in cells
    }

    for backbone in BACKBONES:
        for dataset in DATASETS:
            for ordering in ORDERINGS:
                for strategy_variant in STRATEGY_VARIANTS:
                    if modality_filter == "all":
                        image_cell = cell_lookup.get((backbone, dataset, ordering, strategy_variant, "image"))
                        pixel_cell = cell_lookup.get((backbone, dataset, ordering, strategy_variant, "pixel"))
                        if image_cell is None or pixel_cell is None:
                            logger.warning(
                                "Missing image/pixel pair for %s / %s / %s / %s",
                                backbone,
                                dataset,
                                ordering,
                                strategy_variant,
                            )
                            continue

                        image_payload = image_cell.get("metric_matrix")
                        pixel_payload = pixel_cell.get("metric_matrix")
                        if not isinstance(image_payload, dict) or not isinstance(pixel_payload, dict):
                            continue

                        image_order = image_payload.get("concepts_order_reference", [])
                        pixel_order = pixel_payload.get("concepts_order_reference", [])
                        if not isinstance(image_order, list) or not isinstance(pixel_order, list):
                            continue
                        if [str(item) for item in image_order] != [str(item) for item in pixel_order]:
                            logger.warning(
                                "Concept order mismatch for %s / %s / %s / %s; skipping combined heatmap",
                                backbone,
                                dataset,
                                ordering,
                                strategy_variant,
                            )
                            continue

                        for stat_name in iter_matrix_stats(matrix_stat):
                            image_matrix = normalize_matrix(image_payload.get(stat_name))
                            pixel_matrix = normalize_matrix(pixel_payload.get(stat_name))
                            if image_matrix is None or pixel_matrix is None:
                                continue

                            json_base_path = output_path_for_combined(output_dir, image_cell, stat_name, "svg")
                            title = build_combined_title(
                                image_cell=image_cell,
                                image_count=image_payload.get("count"),
                                pixel_count=pixel_payload.get("count"),
                                expected_count=image_payload.get("expected_count"),
                                matrix_stat=stat_name,
                            )
                        json_payload = build_concept_only_payload_combined(
                            image_cell=image_cell,
                            pixel_cell=pixel_cell,
                            image_payload=image_payload,
                                pixel_payload=pixel_payload,
                                image_matrix=image_matrix,
                                pixel_matrix=pixel_matrix,
                                concepts_order=[str(item) for item in image_order],
                                matrix_stat=stat_name,
                            )
                        json_path = write_json_payload(json_base_path, json_payload, options)
                        output_formats = iter_output_formats(options.output_format)
                        if not output_formats:
                            manifest_rows.append(
                                {
                                    "dataset": dataset,
                                    "backbone": backbone,
                                    "ordering": ordering,
                                    "strategy_variant": strategy_variant,
                                    "modality": "combined",
                                    "matrix_stat": stat_name,
                                    "output_format": "json-only",
                                    "image_matrix_count": image_payload.get("count"),
                                    "pixel_matrix_count": pixel_payload.get("count"),
                                    "expected_count": image_payload.get("expected_count"),
                                    "relative_output_path": "",
                                    "relative_json_path": str(json_path.relative_to(output_dir)),
                                    "title": title,
                                }
                            )
                        for output_format_name in output_formats:
                            out_path = output_path_for_combined(output_dir, image_cell, stat_name, output_format_name)
                            manifest_rows.append(
                                {
                                    "dataset": dataset,
                                    "backbone": backbone,
                                    "ordering": ordering,
                                    "strategy_variant": strategy_variant,
                                    "modality": "combined",
                                    "matrix_stat": stat_name,
                                    "output_format": output_format_name,
                                    "image_matrix_count": image_payload.get("count"),
                                    "pixel_matrix_count": pixel_payload.get("count"),
                                    "expected_count": image_payload.get("expected_count"),
                                    "relative_output_path": str(out_path.relative_to(output_dir)),
                                    "relative_json_path": str(json_path.relative_to(output_dir)),
                                    "title": title,
                                }
                            )
                            if out_path.exists() and not options.overwrite:
                                logger.info("  [skip] %s already exists", out_path.relative_to(output_dir))
                                continue
                            if options.dry_run:
                                logger.info("  [dry-run] would write %s", out_path.relative_to(output_dir))
                                continue

                            if output_format_name == "svg":
                                write_svg_dual_heatmap(
                                    image_matrix=image_matrix,
                                    pixel_matrix=pixel_matrix,
                                    concepts_order=[str(item) for item in image_order],
                                    title=title,
                                    output_path=out_path,
                                    options=options,
                                    matrix_stat=stat_name,
                                )
                            else:
                                write_pdf_dual_heatmap(
                                    image_matrix=image_matrix,
                                    pixel_matrix=pixel_matrix,
                                    concepts_order=[str(item) for item in image_order],
                                    title=title,
                                    output_path=out_path,
                                    options=options,
                                    matrix_stat=stat_name,
                                )
                            logger.info("  heatmap: %s", out_path.relative_to(output_dir))
                            written += 1
                        continue

                    requested_modalities = MODALITIES if modality_filter == "separate" else [modality_filter]
                    for modality in requested_modalities:
                        cell = cell_lookup.get((backbone, dataset, ordering, strategy_variant, modality))
                        if cell is None:
                            logger.warning(
                                "Missing summary cell for %s / %s / %s / %s / %s",
                                backbone,
                                dataset,
                                ordering,
                                strategy_variant,
                                modality,
                            )
                            continue

                        matrix_payload = cell.get("metric_matrix")
                        if not isinstance(matrix_payload, dict):
                            continue
                        concepts_order = matrix_payload.get("concepts_order_reference", [])
                        if not isinstance(concepts_order, list) or not concepts_order:
                            logger.debug("Skipping cell without concepts_order: %s", cell)
                            continue

                        for stat_name in iter_matrix_stats(matrix_stat):
                            matrix = normalize_matrix(matrix_payload.get(stat_name))
                            if matrix is None:
                                logger.debug("Skipping %s heatmap without matrix: %s", stat_name, cell)
                                continue

                            json_base_path = output_path_for_cell(output_dir, cell, stat_name, "svg")
                            title = build_title(
                                cell=cell,
                                matrix_stat=stat_name,
                                matrix_count=matrix_payload.get("count"),
                                expected_count=matrix_payload.get("expected_count"),
                            )
                            json_payload = build_concept_only_payload_single(
                                cell=cell,
                                matrix_payload=matrix_payload,
                                matrix=matrix,
                                concepts_order=[str(item) for item in concepts_order],
                                matrix_stat=stat_name,
                            )
                            json_path = write_json_payload(json_base_path, json_payload, options)
                            output_formats = iter_output_formats(options.output_format)
                            if not output_formats:
                                manifest_rows.append(
                                    {
                                        "dataset": dataset,
                                        "backbone": backbone,
                                        "ordering": ordering,
                                        "strategy_variant": strategy_variant,
                                        "modality": modality,
                                        "matrix_stat": stat_name,
                                        "output_format": "json-only",
                                        "image_matrix_count": matrix_payload.get("count") if modality == "image" else "",
                                        "pixel_matrix_count": matrix_payload.get("count") if modality == "pixel" else "",
                                        "expected_count": matrix_payload.get("expected_count"),
                                        "relative_output_path": "",
                                        "relative_json_path": str(json_path.relative_to(output_dir)),
                                        "title": title,
                                    }
                                )
                            for output_format_name in output_formats:
                                out_path = output_path_for_cell(output_dir, cell, stat_name, output_format_name)
                                manifest_rows.append(
                                    {
                                        "dataset": dataset,
                                        "backbone": backbone,
                                        "ordering": ordering,
                                        "strategy_variant": strategy_variant,
                                        "modality": modality,
                                        "matrix_stat": stat_name,
                                        "output_format": output_format_name,
                                        "image_matrix_count": matrix_payload.get("count") if modality == "image" else "",
                                        "pixel_matrix_count": matrix_payload.get("count") if modality == "pixel" else "",
                                        "expected_count": matrix_payload.get("expected_count"),
                                        "relative_output_path": str(out_path.relative_to(output_dir)),
                                        "relative_json_path": str(json_path.relative_to(output_dir)),
                                        "title": title,
                                    }
                                )
                                if out_path.exists() and not options.overwrite:
                                    logger.info("  [skip] %s already exists", out_path.relative_to(output_dir))
                                    continue
                                if options.dry_run:
                                    logger.info("  [dry-run] would write %s", out_path.relative_to(output_dir))
                                    continue

                                if output_format_name == "svg":
                                    write_svg_heatmap(
                                        matrix=matrix,
                                        concepts_order=[str(item) for item in concepts_order],
                                        title=title,
                                        output_path=out_path,
                                        options=options,
                                        matrix_stat=stat_name,
                                    )
                                else:
                                    write_pdf_heatmap(
                                        matrix=matrix,
                                        concepts_order=[str(item) for item in concepts_order],
                                        title=title,
                                        output_path=out_path,
                                        options=options,
                                        matrix_stat=stat_name,
                                    )
                                logger.info("  heatmap: %s", out_path.relative_to(output_dir))
                                written += 1

    manifest_path = write_manifest(output_dir, manifest_rows)
    return written, manifest_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    input_json = args.input_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.font_scale <= 0:
        raise ValueError("--font-scale must be positive")

    options = HeatmapRenderOptions(
        annotate=args.annotate,
        font_size=max(1, int(round(args.font_size * args.font_scale))),
        title_font_size=max(1, int(round(args.title_font_size * args.font_scale))),
        axis_font_size=max(1, int(round(args.axis_font_size * args.font_scale))),
        cell_font_size=max(1, int(round(args.cell_font_size * args.font_scale))),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        show_upper_diagonal=args.show_upper_diagonal,
        panel_layout=args.panel_layout,
        show_title=args.show_title,
        show_panel_titles=args.show_panel_titles,
        show_x_axis_title=args.show_x_axis_title,
        show_y_axis_title=args.show_y_axis_title,
        output_format=args.output_format,
    )

    written, manifest_path = generate_heatmaps(
        input_json=input_json,
        output_dir=output_dir,
        matrix_stat=args.matrix_stat,
        modality_filter=args.modality,
        options=options,
    )
    if args.output_format == "json-only":
        logger.info("Wrote JSON payloads only; no heatmap files requested")
    else:
        logger.info("Wrote %d heatmap(s)", written)
    logger.info("Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
