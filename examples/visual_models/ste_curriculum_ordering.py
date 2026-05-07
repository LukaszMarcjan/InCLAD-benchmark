"""
Analyze STE visual-model results and derive curriculum concept orderings.

The script expects a results root produced by run_all_experiments.py, for example:
    examples/visual_models/results/

It scans recursively for *_seed_*.json files, reads concept metric matrices,
uses each diagonal value as a single-task concept score, and writes:
    - per-result diagonal scores
    - per-result easy-to-hard and hard-to-easy orderings
    - final benchmark-level orderings averaged across models/seeds
    - optional heatmaps in original, easy-to-hard, and hard-to-easy orders

Usage:
    python3 examples/visual_models/ste_curriculum_ordering.py \
        examples/visual_models/results

    python3 examples/visual_models/ste_curriculum_ordering.py \
        examples/visual_models/results --metric ROC-AUC

    python3 examples/visual_models/ste_curriculum_ordering.py \
        examples/visual_models/results --all-metrics --no-heatmaps
"""

import argparse
import csv
import json
import logging
import math
import pathlib
import re
from html import escape
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterable

logger = logging.getLogger(__name__)

METRIC_KEY_PREFIX = "concept_metric_callback_"
SEED_RESULT_PATTERN = re.compile(r".+_seed_(\d+)$")

PLOTTING_DEPENDENCIES = None
PLOTTING_IMPORT_FAILED = False


@dataclass(frozen=True)
class MetricMatrixRecord:
    json_path: pathlib.Path
    model: str
    benchmark: str
    seed: str
    metric_name: str
    concepts_order: list[str]
    matrix: dict[str, dict[str, float]]


def metric_slug(metric_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", metric_name.lower()).strip("-")


def normalize_metric_name(metric_name: str) -> str:
    return metric_slug(metric_name)


def infer_model_and_benchmark(json_path: pathlib.Path, results_root: pathlib.Path, data: dict[str, Any]) -> tuple[str, str]:
    try:
        parts = json_path.relative_to(results_root).parts
    except ValueError:
        parts = json_path.parts

    if len(parts) >= 3:
        return parts[0], parts[1]

    model_name = str(data.get("model", {}).get("name", json_path.parent.parent.name)).lower()
    benchmark_name = json_path.parent.name.lower()
    return model_name, benchmark_name


def infer_seed(json_path: pathlib.Path) -> str:
    match = SEED_RESULT_PATTERN.match(json_path.stem)
    return match.group(1) if match else ""


def is_ste_result(data: dict[str, Any]) -> bool:
    return str(data.get("strategy", {}).get("name", "")).lower() == "ste"


def find_metric_matrices(data: dict[str, Any]) -> list[tuple[str, dict[str, dict[str, float]], list[str]]]:
    matrices = []
    for key, value in data.items():
        if not key.startswith(METRIC_KEY_PREFIX) or not isinstance(value, dict):
            continue

        metric_matrix = value.get("metric_matrix")
        concepts_order = value.get("concepts_order")
        if not isinstance(metric_matrix, dict) or not isinstance(concepts_order, list):
            continue

        metric_name = str(value.get("base_metric_name", key[len(METRIC_KEY_PREFIX):]))
        matrices.append((metric_name, metric_matrix, [str(concept) for concept in concepts_order]))

    return matrices


def load_records(args: argparse.Namespace) -> list[MetricMatrixRecord]:
    records: list[MetricMatrixRecord] = []
    requested_metric = normalize_metric_name(args.metric)

    json_paths = sorted(args.results_root.rglob("*_seed_*.json"))
    if not json_paths:
        logger.warning(f"No '*_seed_*.json' files found under {args.results_root}")
        return records

    for json_path in json_paths:
        try:
            with json_path.open(encoding="utf-8") as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Skipping unreadable result file {json_path}: {exc}")
            continue

        if not args.include_non_ste and not is_ste_result(data):
            logger.info(f"Skipping non-STE result: {json_path.relative_to(args.results_root)}")
            continue

        model, benchmark = infer_model_and_benchmark(json_path, args.results_root, data)
        seed = infer_seed(json_path)

        for metric_name, matrix, concepts_order in find_metric_matrices(data):
            if not args.all_metrics and normalize_metric_name(metric_name) != requested_metric:
                continue

            records.append(
                MetricMatrixRecord(
                    json_path=json_path,
                    model=model,
                    benchmark=benchmark,
                    seed=seed,
                    metric_name=metric_name,
                    concepts_order=concepts_order,
                    matrix=matrix,
                )
            )

    return records


def diagonal_scores(record: MetricMatrixRecord) -> dict[str, float]:
    scores = {}
    for concept in record.concepts_order:
        try:
            scores[concept] = float(record.matrix[concept][concept])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Missing diagonal score for concept '%s' in %s (%s)",
                concept,
                record.json_path,
                exc,
            )
            scores[concept] = math.nan
    return scores


def finite_score(score: float) -> bool:
    return math.isfinite(score)


def easy_to_hard_order(scores: dict[str, float], higher_is_better: bool) -> list[str]:
    def sort_key(concept: str):
        score = scores[concept]
        if not finite_score(score):
            return (1, 0.0, concept)
        return (0, -score if higher_is_better else score, concept)

    return sorted(scores, key=sort_key)


def hard_to_easy_order(scores: dict[str, float], higher_is_better: bool) -> list[str]:
    def sort_key(concept: str):
        score = scores[concept]
        if not finite_score(score):
            return (1, 0.0, concept)
        return (0, score if higher_is_better else -score, concept)

    return sorted(scores, key=sort_key)


def hardness(score: float, higher_is_better: bool) -> float:
    if not finite_score(score):
        return math.nan
    return 1.0 - score if higher_is_better else score


def json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def json_scores(scores: dict[str, float]) -> str:
    return json.dumps(scores, ensure_ascii=False, sort_keys=True)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str], dry_run: bool) -> None:
    if dry_run:
        logger.info(f"[dry-run] would write {path} ({len(rows)} rows)")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {path} ({len(rows)} rows)")


def build_per_result_rows(
    records: list[MetricMatrixRecord],
    higher_is_better: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []

    for record in records:
        scores = diagonal_scores(record)
        easy_order = easy_to_hard_order(scores, higher_is_better)
        hard_order = hard_to_easy_order(scores, higher_is_better)
        easy_ranks = {concept: rank for rank, concept in enumerate(easy_order, start=1)}
        hard_ranks = {concept: rank for rank, concept in enumerate(hard_order, start=1)}

        for concept in record.concepts_order:
            score = scores[concept]
            score_rows.append(
                {
                    "model": record.model,
                    "benchmark": record.benchmark,
                    "seed": record.seed,
                    "metric": record.metric_name,
                    "concept": concept,
                    "score": score,
                    "hardness": hardness(score, higher_is_better),
                    "easy_rank": easy_ranks[concept],
                    "hard_rank": hard_ranks[concept],
                    "json_path": str(record.json_path),
                }
            )

        ordering_rows.append(
            {
                "model": record.model,
                "benchmark": record.benchmark,
                "seed": record.seed,
                "metric": record.metric_name,
                "easy_to_hard": json_list(easy_order),
                "hard_to_easy": json_list(hard_order),
                "scores": json_scores(scores),
                "json_path": str(record.json_path),
            }
        )

    return score_rows, ordering_rows


def build_final_rows(
    score_rows: list[dict[str, Any]],
    higher_is_better: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    grouped_models: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    grouped_seeds: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for row in score_rows:
        key = (row["benchmark"], row["metric"], row["concept"])
        score = float(row["score"])
        if finite_score(score):
            grouped_scores[key].append(score)
        grouped_models[key].add(str(row["model"]))
        grouped_seeds[key].add(str(row["seed"]))

    concept_rows: list[dict[str, Any]] = []
    scores_by_group: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)

    for (benchmark, metric_name, concept), values in sorted(grouped_scores.items()):
        mean_score = mean(values) if values else math.nan
        std_score = pstdev(values) if len(values) > 1 else 0.0
        scores_by_group[(benchmark, metric_name)][concept] = mean_score
        concept_rows.append(
            {
                "benchmark": benchmark,
                "metric": metric_name,
                "concept": concept,
                "mean_score": mean_score,
                "std_score": std_score,
                "hardness": hardness(mean_score, higher_is_better),
                "n": len(values),
                "models": json_list(sorted(grouped_models[(benchmark, metric_name, concept)])),
                "seeds": json_list(sorted(grouped_seeds[(benchmark, metric_name, concept)])),
            }
        )

    final_ordering_rows: list[dict[str, Any]] = []
    for (benchmark, metric_name), scores in sorted(scores_by_group.items()):
        easy_order = easy_to_hard_order(scores, higher_is_better)
        hard_order = hard_to_easy_order(scores, higher_is_better)
        final_ordering_rows.append(
            {
                "benchmark": benchmark,
                "metric": metric_name,
                "easy_to_hard": json_list(easy_order),
                "hard_to_easy": json_list(hard_order),
                "mean_scores": json_scores(scores),
            }
        )

    return concept_rows, final_ordering_rows


def aggregate_matrices(records: list[MetricMatrixRecord]) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    grouped_cells: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for record in records:
        group_key = (record.benchmark, record.metric_name)
        for learned_concept in record.concepts_order:
            for evaluated_concept in record.concepts_order:
                try:
                    value = float(record.matrix[learned_concept][evaluated_concept])
                except (KeyError, TypeError, ValueError):
                    continue
                if finite_score(value):
                    grouped_cells[group_key][learned_concept][evaluated_concept].append(value)

    aggregated = {}
    for group_key, learned_map in grouped_cells.items():
        aggregated[group_key] = {}
        for learned_concept, evaluated_map in learned_map.items():
            aggregated[group_key][learned_concept] = {
                evaluated_concept: mean(values)
                for evaluated_concept, values in evaluated_map.items()
                if values
            }

    return aggregated


def get_plotting_dependencies():
    global PLOTTING_DEPENDENCIES, PLOTTING_IMPORT_FAILED

    if PLOTTING_IMPORT_FAILED:
        return None

    if PLOTTING_DEPENDENCIES is None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from pyclad.analysis.scenario_heatmap import plot_metric_heatmap
        except Exception as exc:
            PLOTTING_IMPORT_FAILED = True
            logger.error(f"Cannot import plotting dependencies, heatmaps will be skipped: {exc}")
            return None

        PLOTTING_DEPENDENCIES = (plt, plot_metric_heatmap)

    return PLOTTING_DEPENDENCIES


def interpolate_color(value: float) -> str:
    if not finite_score(value):
        return "rgb(220,220,220)"

    value = max(0.0, min(1.0, value))
    stops = [
        (0.0, (13, 8, 135)),
        (0.5, (204, 71, 120)),
        (1.0, (240, 249, 33)),
    ]

    for index in range(len(stops) - 1):
        left_value, left_color = stops[index]
        right_value, right_color = stops[index + 1]
        if left_value <= value <= right_value:
            ratio = (value - left_value) / (right_value - left_value)
            color = tuple(
                round(left_channel + (right_channel - left_channel) * ratio)
                for left_channel, right_channel in zip(left_color, right_color)
            )
            return f"rgb({color[0]},{color[1]},{color[2]})"

    color = stops[-1][1]
    return f"rgb({color[0]},{color[1]},{color[2]})"


def write_svg_heatmap(
    matrix: dict[str, dict[str, float]],
    concepts_order: list[str],
    title: str,
    output_path: pathlib.Path,
    annotate: bool,
    font_size: int,
    title_font_size: int,
    axis_font_size: int,
    cell_font_size: int,
) -> None:
    max_label_length = max((len(concept) for concept in concepts_order), default=8)
    cell_size = max(34, int(cell_font_size * 2.8))
    label_width = max(120, int(max_label_length * max(font_size * 0.72, 6)) + 20)
    top_margin = max(170, int(font_size * 10 + 60))
    right_margin = 40
    bottom_margin = 40
    width = label_width + len(concepts_order) * cell_size + right_margin
    height = top_margin + len(concepts_order) * cell_size + bottom_margin

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        f"text {{ font-family: Helvetica, Arial, sans-serif; font-size: {font_size}px; fill: black; }}",
        f".title {{ font-size: {title_font_size}px; font-weight: 700; fill: black; }}",
        f".axis {{ font-size: {axis_font_size}px; font-weight: 600; fill: black; }}",
        f".cell-label {{ font-size: {cell_font_size}px; font-weight: 600; fill: black; }}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{width / 2:.1f}" y="24" text-anchor="middle">{escape(title)}</text>',
        f'<text class="axis" x="{label_width + len(concepts_order) * cell_size / 2:.1f}" y="48" text-anchor="middle">Evaluating on concept</text>',
        f'<text class="axis" x="18" y="{top_margin + len(concepts_order) * cell_size / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top_margin + len(concepts_order) * cell_size / 2:.1f})">After learning concept</text>',
    ]

    for col_index, concept in enumerate(concepts_order):
        x = label_width + col_index * cell_size + cell_size / 2
        y = top_margin - 10
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="start" transform="rotate(-55 {x:.1f} {y:.1f})">{escape(concept)}</text>'
        )

    for row_index, learned_concept in enumerate(concepts_order):
        y = top_margin + row_index * cell_size
        lines.append(
            f'<text x="{label_width - 8}" y="{y + cell_size / 2 + 4:.1f}" text-anchor="end">{escape(learned_concept)}</text>'
        )

        for col_index, evaluated_concept in enumerate(concepts_order):
            x = label_width + col_index * cell_size
            try:
                value = float(matrix[learned_concept][evaluated_concept])
            except (KeyError, TypeError, ValueError):
                value = math.nan

            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{interpolate_color(value)}" stroke="white" stroke-width="1"/>'
            )
            if annotate:
                label = "NA" if not finite_score(value) else f"{value:.2f}"
                lines.append(
                    f'<text class="cell-label" x="{x + cell_size / 2:.1f}" y="{y + cell_size / 2 + 3:.1f}" text-anchor="middle">{label}</text>'
                )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_heatmap(
    matrix: dict[str, dict[str, float]],
    concepts_order: list[str],
    title: str,
    output_path: pathlib.Path,
    args: argparse.Namespace,
) -> bool:
    if output_path.exists() and not args.overwrite:
        logger.info(f"  [skip] {output_path.name} already exists (use --overwrite to regenerate)")
        return False

    if args.dry_run:
        logger.info(f"  [dry-run] would write {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.heatmap_format == "svg":
        write_svg_heatmap(
            matrix=matrix,
            concepts_order=concepts_order,
            title=title,
            output_path=output_path,
            annotate=args.annotate,
            font_size=args.font_size,
            title_font_size=args.title_font_size,
            axis_font_size=args.axis_font_size,
            cell_font_size=args.cell_font_size,
        )
        logger.info(f"  heatmap: {output_path}")
        return True

    plotting_dependencies = get_plotting_dependencies()
    if plotting_dependencies is None:
        return False

    plt, plot_metric_heatmap = plotting_dependencies

    try:
        plt.close("all")
        plot_metric_heatmap(
            matrix=matrix,
            concepts_order=concepts_order,
            title=title,
            annotate=args.annotate,
            ignore_upper_diagonal=True,
            figsize=tuple(args.figsize),
            output_path=output_path,
        )
        axis = plt.gca()
        axis.set_title(title, fontsize=args.title_font_size, color="black")
        axis.set_xlabel(axis.get_xlabel(), fontsize=args.axis_font_size, color="black")
        axis.set_ylabel(axis.get_ylabel(), fontsize=args.axis_font_size, color="black")
        axis.tick_params(axis="both", colors="black", labelsize=args.font_size)
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            tick_label.set_color("black")
            tick_label.set_fontsize(args.font_size)
        for text in axis.texts:
            text.set_color("black")
            text.set_fontsize(args.cell_font_size)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close("all")
        logger.info(f"  heatmap: {output_path}")
        return True
    except Exception as exc:
        logger.error(f"Failed to generate heatmap {output_path}: {exc}")
        return False


def write_per_result_heatmaps(
    records: list[MetricMatrixRecord],
    output_dir: pathlib.Path,
    args: argparse.Namespace,
) -> int:
    written = 0
    for record in records:
        scores = diagonal_scores(record)
        easy_order = easy_to_hard_order(scores, args.higher_is_better)
        hard_order = hard_to_easy_order(scores, args.higher_is_better)
        metric_name = metric_slug(record.metric_name)
        heatmap_dir = output_dir / "heatmaps" / "per_result" / record.model / record.benchmark
        prefix = f"{record.json_path.stem}_{metric_name}"

        variants = [
            ("original", record.concepts_order),
            ("easy_to_hard", easy_order),
            ("hard_to_easy", hard_order),
        ]
        for variant_name, order in variants:
            out_path = heatmap_dir / f"{prefix}_{variant_name}.{args.heatmap_format}"
            title = (
                f"{record.metric_name} - {record.model}/{record.benchmark} "
                f"seed={record.seed} - {variant_name.replace('_', ' ')}"
            )
            if write_heatmap(record.matrix, order, title, out_path, args):
                written += 1

    return written


def write_final_heatmaps(
    records: list[MetricMatrixRecord],
    final_ordering_rows: list[dict[str, Any]],
    output_dir: pathlib.Path,
    args: argparse.Namespace,
) -> int:
    final_order_by_group = {
        (row["benchmark"], row["metric"]): {
            "easy_to_hard": json.loads(row["easy_to_hard"]),
            "hard_to_easy": json.loads(row["hard_to_easy"]),
        }
        for row in final_ordering_rows
    }
    aggregated_matrices = aggregate_matrices(records)

    written = 0
    for (benchmark, metric_name), matrix in sorted(aggregated_matrices.items()):
        orders = final_order_by_group.get((benchmark, metric_name))
        if orders is None:
            continue

        metric_name_slug = metric_slug(metric_name)
        heatmap_dir = output_dir / "heatmaps" / "final" / benchmark
        for variant_name, order in orders.items():
            out_path = heatmap_dir / f"final_{metric_name_slug}_{variant_name}.{args.heatmap_format}"
            title = f"Final mean {metric_name} - {benchmark} - {variant_name.replace('_', ' ')}"
            if write_heatmap(matrix, order, title, out_path, args):
                written += 1

    return written


def write_markdown_summary(final_ordering_rows: list[dict[str, Any]], output_path: pathlib.Path, dry_run: bool) -> None:
    lines = [
        "# STE Curriculum Ordering Summary",
        "",
        "Higher metric values are treated as easier concepts by default.",
        "",
    ]

    for row in final_ordering_rows:
        lines.extend(
            [
                f"## {row['benchmark']} - {row['metric']}",
                "",
                f"Easy to hard: `{row['easy_to_hard']}`",
                "",
                f"Hard to easy: `{row['hard_to_easy']}`",
                "",
            ]
        )

    if dry_run:
        logger.info(f"[dry-run] would write {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive easy-to-hard and hard-to-easy curriculum orderings from STE result JSONs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_root", type=pathlib.Path, help="Root directory with visual model result JSONs")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for ordering reports and heatmaps (default: RESULTS_ROOT/ste_curriculum_ordering)",
    )
    parser.add_argument("--metric", type=str, default="ROC-AUC", help="Metric to analyze when --all-metrics is not set")
    parser.add_argument("--all-metrics", action="store_true", help="Analyze every concept metric matrix in each JSON")
    parser.add_argument(
        "--include-non-ste",
        action="store_true",
        help="Also analyze result files whose strategy name is not STE",
    )
    parser.add_argument(
        "--lower-is-better",
        dest="higher_is_better",
        action="store_false",
        help="Treat lower metric values as easier instead of higher values",
    )
    parser.set_defaults(higher_is_better=True, heatmaps=True, annotate=True)
    parser.add_argument("--heatmaps", dest="heatmaps", action="store_true", help="Generate heatmaps")
    parser.add_argument("--no-heatmaps", dest="heatmaps", action="store_false", help="Skip heatmap generation")
    parser.add_argument("--heatmap-format", type=str, default="svg", choices=["pdf", "png", "svg"], help="Heatmap file format")
    parser.add_argument("--font-size", type=int, default=11, help="Base font size for heatmap labels")
    parser.add_argument("--title-font-size", type=int, default=16, help="Heatmap title font size")
    parser.add_argument("--axis-font-size", type=int, default=12, help="Heatmap axis title font size")
    parser.add_argument("--cell-font-size", type=int, default=9, help="Heatmap cell annotation font size")
    parser.add_argument("--annotate", dest="annotate", action="store_true", help="Annotate heatmap cells")
    parser.add_argument("--no-annotate", dest="annotate", action="store_false", help="Disable heatmap cell annotations")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 10], metavar=("W", "H"), help="Heatmap size in inches")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs that already exist")
    parser.add_argument("--dry-run", action="store_true", help="Show planned outputs without writing files")

    args = parser.parse_args()
    args.results_root = args.results_root.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.results_root / "ste_curriculum_ordering"
    )

    if not args.results_root.is_dir():
        parser.error(f"results root does not exist: {args.results_root}")

    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    records = load_records(args)
    if not records:
        logger.warning("No matching metric matrices found")
        return 0

    logger.info(f"Loaded {len(records)} metric matrix record(s)")

    score_rows, ordering_rows = build_per_result_rows(records, args.higher_is_better)
    final_concept_rows, final_ordering_rows = build_final_rows(score_rows, args.higher_is_better)

    write_csv(
        args.output_dir / "diagonal_scores.csv",
        score_rows,
        ["model", "benchmark", "seed", "metric", "concept", "score", "hardness", "easy_rank", "hard_rank", "json_path"],
        args.dry_run,
    )
    write_csv(
        args.output_dir / "per_result_orderings.csv",
        ordering_rows,
        ["model", "benchmark", "seed", "metric", "easy_to_hard", "hard_to_easy", "scores", "json_path"],
        args.dry_run,
    )
    write_csv(
        args.output_dir / "final_concept_scores.csv",
        final_concept_rows,
        ["benchmark", "metric", "concept", "mean_score", "std_score", "hardness", "n", "models", "seeds"],
        args.dry_run,
    )
    write_csv(
        args.output_dir / "final_orderings.csv",
        final_ordering_rows,
        ["benchmark", "metric", "easy_to_hard", "hard_to_easy", "mean_scores"],
        args.dry_run,
    )
    write_markdown_summary(final_ordering_rows, args.output_dir / "summary.md", args.dry_run)

    if args.heatmaps:
        heatmaps_written = write_per_result_heatmaps(records, args.output_dir, args)
        heatmaps_written += write_final_heatmaps(records, final_ordering_rows, args.output_dir, args)
        logger.info(f"Heatmap generation complete: {heatmaps_written} file(s) written")
    else:
        logger.info("Heatmap generation skipped")

    logger.info(f"Done. Outputs under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
