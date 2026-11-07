"""
Compute per-concept difficulty ranking from STE results across multiple models.

For each concept, extracts the diagonal score (model trained on that concept,
tested on that concept) from the metric_matrix. Ranks are averaged across
models (rank aggregation) to produce a single hard-to-easy and easy-to-hard
ordering per dataset.

The curricula shipped in examples/visual_models/ordering/*.csv (the ones used
by the paper's runs and the HuggingFace manifests) are authoritative. Several
concepts share the same average rank, and the original script broke those ties
by Python set iteration order, which changes with PYTHONHASHSEED. Use
``--tie-break`` to choose a deterministic rule:

    score      (default) ties ordered by ascending average score, then name.
               Reproduces the shipped BTech, DAGM, MPDD and MVTec files, but
               not VisA (three tied pairs come out swapped).
    name       ties ordered by concept name.
    set-order  the original behaviour (depends on PYTHONHASHSEED).

No single rule reproduces all five shipped files. To regenerate them exactly,
also pass ``--reference-dir examples/visual_models/ordering``: ties are then
broken by each concept's position in the matching shipped hard_to_easy CSV.
Per-model ranks (rank_concepts) give equal scores distinct ranks in input
order; this is unchanged, and the shipped files were produced that way.

Usage:
    python examples/visual_models/neurips_utilities/concept_ordering.py \
        --model patchcore:<results_root>/patchcore \
        --model padim:<results_root>/padim \
        --model cfa:<results_root>/cfa \
        --output-dir <results_root>/ordering

    # Regenerate the shipped curricula byte-for-byte from the STE results:
    python examples/visual_models/neurips_utilities/concept_ordering.py \
        --model patchcore:<ste_root>/patchcore \
        --model padim:<ste_root>/padim \
        --model cfa:<ste_root>/cfa \
        --reference-dir examples/visual_models/ordering \
        --output-dir <results_root>/ordering

    # With metric and dataset filter:
    python examples/visual_models/neurips_utilities/concept_ordering.py \
        --model patchcore:<results_root>/patchcore \
        --model padim:<results_root>/padim \
        --metric F1-Score \
        --datasets mvtec btech \
        --font-scale 1.4 \
        --heatmap-fontsize 12 \
        --heatmap-text-color black \
        --output-dir <results_root>/ordering
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEED_PATTERN = re.compile(r".+_seed_(\d+)$")
METRIC_KEY_PREFIX = "concept_metric_callback_"
KNOWN_BENCHMARKS = ("mvtec", "btech", "visa", "mpdd", "dagm")


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class ConceptScore:
    concept: str
    score: float  # diagonal AUC/F1 from STE run


@dataclass
class SeedResult:
    model: str
    dataset: str
    seed: int
    concept_scores: dict[str, float]  # concept → diagonal score


@dataclass
class ConceptRanking:
    """Per-concept aggregated ranking across models."""
    concept: str
    scores: dict[str, float] = field(default_factory=dict)   # model → avg diagonal score
    ranks: dict[str, float] = field(default_factory=dict)    # model → rank (1=hardest)
    avg_rank: float = 0.0
    avg_score: float = 0.0


def scaled_fontsize(size: float, font_scale: float) -> float:
    return max(1.0, size * font_scale)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def infer_dataset(json_path: Path, root_dir: Path) -> str:
    try:
        parts = json_path.relative_to(root_dir).parts[:-1]
    except ValueError:
        parts = json_path.parts[:-1]
    for part in parts:
        if part.lower() in KNOWN_BENCHMARKS:
            return part.lower()
    return json_path.parent.name.lower()


def extract_seed(json_path: Path) -> Optional[int]:
    m = SEED_PATTERN.match(json_path.stem)
    return int(m.group(1)) if m else None


def find_metric_payload(data: dict, metric_name: str) -> Optional[dict]:
    normalized = metric_name.lower()
    for key, value in data.items():
        if not key.startswith(METRIC_KEY_PREFIX) or not isinstance(value, dict):
            continue
        if str(value.get("base_metric_name", "")).lower() == normalized:
            return value
    return None


def extract_diagonal_scores(metric_matrix: dict, concepts: list[str]) -> dict[str, float]:
    scores = {}
    for concept in concepts:
        row = metric_matrix.get(concept, {})
        if concept in row:
            scores[concept] = float(row[concept])
    return scores


def load_model_results(
    model_name: str,
    model_dir: Path,
    metric_name: str,
    datasets_filter: Optional[set[str]],
) -> list[SeedResult]:
    results = []

    for json_path in sorted(model_dir.rglob("*.json")):
        seed = extract_seed(json_path)
        if seed is None:
            continue

        dataset = infer_dataset(json_path, model_dir)
        if datasets_filter is not None and dataset not in datasets_filter:
            continue

        try:
            with json_path.open(encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Skipping {json_path}: {exc}")
            continue

        payload = find_metric_payload(data, metric_name)
        if payload is None:
            logger.warning(f"Metric '{metric_name}' not found in {json_path}")
            continue

        concepts = payload.get("concepts_order", [])
        matrix = payload.get("metric_matrix", {})
        if not concepts or not matrix:
            logger.warning(f"Missing concepts_order or metric_matrix in {json_path}")
            continue

        diagonal = extract_diagonal_scores(matrix, concepts)
        if not diagonal:
            logger.warning(f"Empty diagonal in {json_path}")
            continue

        results.append(SeedResult(
            model=model_name,
            dataset=dataset,
            seed=seed,
            concept_scores=diagonal,
        ))
        logger.debug(f"  Loaded {model_name} / {dataset} / seed={seed}: {len(diagonal)} concepts")

    return results


# ── aggregation ───────────────────────────────────────────────────────────────

def average_across_seeds(results: list[SeedResult]) -> dict[tuple[str, str], dict[str, float]]:
    """Average diagonal scores across seeds per (model, dataset)."""
    accumulated: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r.model, r.dataset)
        for concept, score in r.concept_scores.items():
            accumulated[key][concept].append(score)

    averaged: dict[tuple[str, str], dict[str, float]] = {}
    for key, concept_scores in accumulated.items():
        averaged[key] = {
            concept: sum(scores) / len(scores)
            for concept, scores in concept_scores.items()
        }
    return averaged


def rank_concepts(scores: dict[str, float]) -> dict[str, int]:
    """Assign ranks: rank 1 = hardest (lowest score), rank N = easiest (highest score)."""
    sorted_concepts = sorted(scores, key=lambda c: scores[c])
    return {concept: rank + 1 for rank, concept in enumerate(sorted_concepts)}


def spearman_correlation(ranks_a: dict[str, float], ranks_b: dict[str, float]) -> float:
    shared = sorted(set(ranks_a) & set(ranks_b))
    if len(shared) < 2:
        return float("nan")
    n = len(shared)
    d_sq = sum((ranks_a[c] - ranks_b[c]) ** 2 for c in shared)
    return 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))


TIE_BREAK_RULES = ("score", "name", "set-order")


def load_reference_order(path: Path, concept_column: str = "concept") -> list[str]:
    """Concept order of an existing ordering CSV (used only to break ties)."""
    with path.open(newline="", encoding="utf-8") as fp:
        return [row[concept_column].strip() for row in csv.DictReader(fp) if row.get(concept_column, "").strip()]


def build_concept_rankings(
    averaged: dict[tuple[str, str], dict[str, float]],
    models: list[str],
    dataset: str,
    tie_break: str = "score",
    reference_order: Optional[list[str]] = None,
) -> list[ConceptRanking]:
    """Aggregate per-model ranks; return concepts hardest first.

    ``tie_break`` orders concepts with equal average rank (see TIE_BREAK_RULES).
    ``reference_order``, when given, takes precedence for tied concepts: they
    keep their relative position in that list (concepts missing from it go last,
    ordered by ``tie_break``).
    """
    if tie_break not in TIE_BREAK_RULES:
        raise ValueError(f"Unknown tie_break {tie_break!r}; expected one of {TIE_BREAK_RULES}")

    all_concepts: set[str] = set()
    for (model, ds), scores in averaged.items():
        if ds == dataset:
            all_concepts.update(scores)

    # Iterating a set gives a PYTHONHASHSEED-dependent order; only the legacy
    # "set-order" rule keeps it.
    concept_iteration = list(all_concepts) if tie_break == "set-order" else sorted(all_concepts)
    rankings: dict[str, ConceptRanking] = {c: ConceptRanking(concept=c) for c in concept_iteration}

    for model in models:
        key = (model, dataset)
        if key not in averaged:
            logger.warning(f"No results for model='{model}' dataset='{dataset}'")
            continue

        scores = averaged[key]
        ranks = rank_concepts(scores)

        for concept, score in scores.items():
            if concept in rankings:
                rankings[concept].scores[model] = score
                rankings[concept].ranks[model] = ranks[concept]

    for ranking in rankings.values():
        if ranking.ranks:
            ranking.avg_rank = sum(ranking.ranks.values()) / len(ranking.ranks)
        if ranking.scores:
            finite = [s for s in ranking.scores.values() if not math.isnan(s)]
            ranking.avg_score = sum(finite) / len(finite) if finite else float("nan")

    # Original rule: sort by average rank only; ties keep set iteration order.
    # return sorted(rankings.values(), key=lambda r: r.avg_rank)
    reference_position = {c: i for i, c in enumerate(reference_order or [])}
    missing_position = len(reference_position)

    def sort_key(r: ConceptRanking):
        key: list = [r.avg_rank]
        if reference_order is not None:
            key.append(reference_position.get(r.concept, missing_position))
        if tie_break == "score":
            score = r.avg_score if not math.isnan(r.avg_score) else float("inf")
            key.extend([score, r.concept])
        elif tie_break == "name":
            key.append(r.concept)
        return tuple(key)

    return sorted(rankings.values(), key=sort_key)


# ── output ────────────────────────────────────────────────────────────────────

def plot_diagonal_heatmaps(
    rankings: list[ConceptRanking],
    models: list[str],
    dataset: str,
    metric: str,
    output_path: Path,
    heatmap_fontsize: float = 7.0,
    heatmap_text_color: str = "auto",
    font_scale: float = 1.0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # concepts ordered hard-to-easy (hardest at top)
    concepts = [r.concept for r in rankings]
    n = len(concepts)
    ranking_by_concept = {r.concept: r for r in rankings}

    n_cols = len(models)
    panel_size = min(5.0, max(3.0, n * 0.28))
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_size * n_cols, panel_size))
    if n_cols == 1:
        axes = [axes]

    masked = np.ma.array(np.zeros((n, n)), mask=True)

    for ax, model in zip(axes, models):
        data = masked.copy()
        for i, concept in enumerate(concepts):
            score = ranking_by_concept[concept].scores.get(model, float("nan"))
            if not math.isnan(score):
                data[i, i] = score
                data.mask[i, i] = False

        ax.imshow(np.ones((n, n)), aspect="equal", cmap="Greys", vmin=0, vmax=1, alpha=0.08)
        im = ax.imshow(data, aspect="equal", cmap="plasma", vmin=0.0, vmax=1.0)

        for i in range(n):
            val = ranking_by_concept[concepts[i]].scores.get(model, float("nan"))
            if not math.isnan(val):
                if heatmap_text_color == "auto":
                    color = "white" if val < 0.5 else "black"
                else:
                    color = heatmap_text_color
                ax.text(
                    i,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=scaled_fontsize(heatmap_fontsize, font_scale),
                    color=color,
                )

        ax.set_xticks(range(n))
        ax.set_xticklabels(
            concepts,
            rotation=45,
            ha="right",
            fontsize=scaled_fontsize(heatmap_fontsize, font_scale),
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels(concepts, fontsize=scaled_fontsize(heatmap_fontsize, font_scale))
        ax.set_title(model, fontsize=scaled_fontsize(heatmap_fontsize + 2, font_scale), fontweight="bold")
        ax.set_xlabel("concept", fontsize=scaled_fontsize(heatmap_fontsize, font_scale))
        ax.set_ylabel("concept", fontsize=scaled_fontsize(heatmap_fontsize, font_scale))
        ax.tick_params(axis="both", labelsize=scaled_fontsize(heatmap_fontsize, font_scale))

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(color="white", linewidth=0.5)

        colorbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=scaled_fontsize(heatmap_fontsize, font_scale))

    fig.suptitle(
        f"{dataset} — {metric} diagonal (STE)",
        fontsize=scaled_fontsize(heatmap_fontsize + 4, font_scale),
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ordering_heatmap(
    rankings: list[ConceptRanking],
    models: list[str],
    dataset: str,
    metric: str,
    output_path: Path,
    heatmap_fontsize: float = 9.0,
    heatmap_text_color: str = "auto",
    font_scale: float = 1.0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    concepts = [r.concept for r in reversed(rankings)]  # easiest at top
    cols = models + ["avg"]
    ranking_by_concept = {r.concept: r for r in rankings}

    data = np.full((len(concepts), len(cols)), float("nan"))
    for row_i, concept in enumerate(concepts):
        r = ranking_by_concept[concept]
        for col_j, model in enumerate(models):
            data[row_i, col_j] = r.scores.get(model, float("nan"))
        data[row_i, len(models)] = r.avg_score

    fig_h = max(3, len(concepts) * 0.45 + 1.5)
    fig_w = max(4, len(cols) * 1.2 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", cmap="plasma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        cols,
        rotation=30,
        ha="right",
        fontsize=scaled_fontsize(heatmap_fontsize, font_scale),
    )
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=scaled_fontsize(heatmap_fontsize, font_scale))
    ax.tick_params(axis="both", labelsize=scaled_fontsize(heatmap_fontsize, font_scale))

    for row_i in range(len(concepts)):
        for col_j in range(len(cols)):
            val = data[row_i, col_j]
            if not math.isnan(val):
                if heatmap_text_color == "auto":
                    color = "black" if 0.3 < val < 0.75 else "white"
                else:
                    color = heatmap_text_color
                ax.text(
                    col_j,
                    row_i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=scaled_fontsize(heatmap_fontsize, font_scale),
                    color=color,
                )

    colorbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    colorbar.ax.tick_params(labelsize=scaled_fontsize(heatmap_fontsize, font_scale))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_bar_subplot(
    ax,
    concepts: list[str],
    scores: list[float],
    title: str,
    color: str,
    metric: str,
    font_scale: float = 1.0,
) -> None:
    import numpy as np
    y = np.arange(len(concepts))
    bars = ax.barh(y, scores, height=0.5, color=color, alpha=0.85)
    for bar, score in zip(bars, scores):
        if not math.isnan(score):
            ax.text(
                min(score + 0.01, 1.08),
                bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}",
                va="center",
                fontsize=scaled_fontsize(5, font_scale),
            )
    ax.set_yticks(y)
    ax.set_yticklabels(concepts, fontsize=scaled_fontsize(6, font_scale))
    ax.set_xlim(0.0, 1.15)
    ax.set_xlabel(metric, fontsize=scaled_fontsize(6, font_scale))
    ax.set_title(title, fontsize=scaled_fontsize(7, font_scale), fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.tick_params(axis="x", labelsize=scaled_fontsize(6, font_scale))
    ax.tick_params(axis="y", labelsize=scaled_fontsize(6, font_scale))


def plot_ordering_chart(
    rankings: list[ConceptRanking],
    models: list[str],
    dataset: str,
    metric: str,
    output_path: Path,
    font_scale: float = 1.0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # easiest at top, hardest at bottom
    concepts = [r.concept for r in reversed(rankings)]
    ranking_by_concept = {r.concept: r for r in rankings}
    colors = plt.cm.tab10.colors

    panels = [(m, colors[i % len(colors)]) for i, m in enumerate(models)]
    panels.append(("avg", "steelblue"))
    n_panels = len(panels)

    n_cols = 2
    n_rows = math.ceil(n_panels / n_cols)
    fig_w = 3.5 * n_cols
    fig_h = max(3, len(concepts) * 0.2 + 1.0) * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=True)
    axes = np.array(axes).flatten()

    for ax, (label, color) in zip(axes, panels):
        if label == "avg":
            scores = [ranking_by_concept[c].avg_score for c in concepts]
        else:
            scores = [ranking_by_concept[c].scores.get(label, float("nan")) for c in concepts]
        _draw_bar_subplot(ax, concepts, scores, label, color, metric, font_scale=font_scale)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Spearman footer
    corr_lines = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            ranks1 = {r.concept: r.ranks[m1] for r in rankings if m1 in r.ranks}
            ranks2 = {r.concept: r.ranks[m2] for r in rankings if m2 in r.ranks}
            rho = spearman_correlation(ranks1, ranks2)
            corr_lines.append(f"{m1} vs {m2}: ρ = {rho:.3f}")

    footer = "Spearman:  " + "   |   ".join(corr_lines) if corr_lines else ""
    fig.suptitle(f"{dataset} — concept difficulty ({metric})  |  ← harder / easier →",
                 fontsize=scaled_fontsize(11, font_scale), y=1.01)
    if footer:
        fig.text(
            0.5,
            -0.01,
            footer,
            ha="center",
            va="top",
            fontsize=scaled_fontsize(8, font_scale),
            color="dimgray",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_ordering_csv(
    path: Path,
    rankings: list[ConceptRanking],
    models: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_cols = [f"{m}_score" for m in models]
    rank_cols = [f"{m}_rank" for m in models]
    fieldnames = ["rank", "concept", "avg_score", "avg_rank"] + score_cols + rank_cols

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(rankings, start=1):
            row: dict[str, str] = {
                "rank": str(rank),
                "concept": r.concept,
                "avg_score": f"{r.avg_score:.6f}",
                "avg_rank": f"{r.avg_rank:.2f}",
            }
            for model in models:
                row[f"{model}_score"] = f"{r.scores.get(model, float('nan')):.6f}"
                row[f"{model}_rank"] = str(r.ranks.get(model, ""))
            writer.writerow(row)


def print_ordering(rankings: list[ConceptRanking], models: list[str], title: str) -> None:
    score_w = max(len(m) for m in models) + 2
    header = f"{'Rank':>4}  {'Concept':<20}  {'Avg score':>9}  {'Avg rank':>8}"
    for m in models:
        header += f"  {m:>{score_w}}"
    logger.info(title)
    logger.info("─" * len(header))
    logger.info(header)
    logger.info("─" * len(header))
    for rank, r in enumerate(rankings, start=1):
        line = f"{rank:>4}  {r.concept:<20}  {r.avg_score:>9.4f}  {r.avg_rank:>8.2f}"
        for m in models:
            score = r.scores.get(m, float("nan"))
            line += f"  {score:>{score_w}.4f}"
        logger.info(line)
    logger.info("")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-concept difficulty rankings from STE results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        metavar="NAME:PATH",
        required=True,
        help="Model name and results directory, e.g. patchcore:results/patchcore. Repeat for each model.",
    )
    parser.add_argument("--metric", type=str, default="ROC-AUC", help="Metric to use for ranking")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Dataset filter (default: all)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="concept_ordering",
        help="Directory for output CSV and heatmap files",
    )
    parser.add_argument(
        "--tie-break",
        type=str,
        choices=TIE_BREAK_RULES,
        default="score",
        help="How to order concepts with equal average rank: ascending average score then name, "
        "concept name, or the original set iteration order (depends on PYTHONHASHSEED)",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Directory with existing <dataset>_hard_to_easy_<metric>.csv files; tied concepts keep "
        "their order from these files (examples/visual_models/ordering reproduces the shipped curricula)",
    )
    parser.add_argument("--heatmaps", action="store_true", help="Generate heatmap PDFs")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Global multiplier for all plot text sizes",
    )
    parser.add_argument(
        "--heatmap-fontsize",
        type=float,
        default=9.0,
        help="Base font size used for heatmap labels and cell annotations",
    )
    parser.add_argument(
        "--heatmap-text-color",
        type=str,
        choices=("auto", "black", "white"),
        default="black",
        help="Text color for heatmap cell annotations",
    )
    return parser.parse_args()


def parse_model_args(model_args: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for arg in model_args:
        if ":" not in arg:
            raise SystemExit(f"--model must be NAME:PATH, got: '{arg}'")
        name, _, path_str = arg.partition(":")
        path = Path(path_str).expanduser()
        if not path.is_dir():
            raise SystemExit(f"Directory does not exist for model '{name}': {path}")
        parsed.append((name, path))
    return parsed


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    model_specs = parse_model_args(args.models)
    model_names = [name for name, _ in model_specs]
    datasets_filter = None if args.datasets is None else {d.lower() for d in args.datasets}
    output_dir = Path(args.output_dir).expanduser()

    # Load all results
    all_results: list[SeedResult] = []
    for model_name, model_dir in model_specs:
        results = load_model_results(model_name, model_dir, args.metric, datasets_filter)
        logger.info(f"Loaded {len(results)} seed result(s) for model '{model_name}'")
        all_results.extend(results)

    if not all_results:
        raise SystemExit("No results loaded. Check --model paths and --metric name.")

    averaged = average_across_seeds(all_results)

    datasets = sorted({ds for (_, ds) in averaged})
    logger.info(f"Datasets found: {datasets}")
    logger.info("")

    for dataset in datasets:
        logger.info(f"{'═' * 60}")
        logger.info(f"Dataset: {dataset}  |  Metric: {args.metric}")
        logger.info("")

        slug = dataset.replace(" ", "_")
        metric_slug = args.metric.lower().replace(" ", "_").replace("/", "-")

        reference_order = None
        if args.reference_dir is not None:
            reference_path = Path(args.reference_dir).expanduser() / f"{slug}_hard_to_easy_{metric_slug}.csv"
            if reference_path.is_file():
                reference_order = load_reference_order(reference_path)
                logger.info(f"Breaking ties with reference order: {reference_path}")
            else:
                logger.warning(f"No reference ordering for '{dataset}' at {reference_path}; using --tie-break")

        rankings = build_concept_rankings(
            averaged,
            model_names,
            dataset,
            tie_break=args.tie_break,
            reference_order=reference_order,
        )

        if not rankings:
            logger.warning(f"No concepts found for dataset '{dataset}'")
            continue

        hard_to_easy = list(rankings)                  # rank 1 = hardest
        easy_to_hard = list(reversed(rankings))        # rank 1 = easiest

        print_ordering(hard_to_easy, model_names, "Hard → Easy (rank 1 = hardest)")
        print_ordering(easy_to_hard, model_names, "Easy → Hard (rank 1 = easiest)")

        # Spearman correlations between models
        if len(model_names) >= 2:
            logger.info("Spearman rank correlations:")
            for i, m1 in enumerate(model_names):
                for m2 in model_names[i + 1:]:
                    ranks1 = {r.concept: r.ranks[m1] for r in rankings if m1 in r.ranks}
                    ranks2 = {r.concept: r.ranks[m2] for r in rankings if m2 in r.ranks}
                    rho = spearman_correlation(ranks1, ranks2)
                    logger.info(f"  {m1} vs {m2}: ρ = {rho:.3f}")
            logger.info("")

        h2e_path = output_dir / f"{slug}_hard_to_easy_{metric_slug}.csv"
        e2h_path = output_dir / f"{slug}_easy_to_hard_{metric_slug}.csv"
        write_ordering_csv(h2e_path, hard_to_easy, model_names)
        write_ordering_csv(e2h_path, easy_to_hard, model_names)
        logger.info(f"Wrote: {h2e_path}")
        logger.info(f"Wrote: {e2h_path}")

        if args.heatmaps:
            chart_path = output_dir / f"{slug}_difficulty_{metric_slug}_bars.pdf"
            plot_ordering_chart(
                rankings=hard_to_easy,
                models=model_names,
                dataset=dataset,
                metric=args.metric,
                output_path=chart_path,
                font_scale=args.font_scale,
            )
            logger.info(f"Wrote: {chart_path}")

            heatmap_path = output_dir / f"{slug}_difficulty_{metric_slug}_heatmap.pdf"
            plot_ordering_heatmap(
                rankings=hard_to_easy,
                models=model_names,
                dataset=dataset,
                metric=args.metric,
                output_path=heatmap_path,
                heatmap_fontsize=args.heatmap_fontsize,
                heatmap_text_color=args.heatmap_text_color,
                font_scale=args.font_scale,
            )
            logger.info(f"Wrote: {heatmap_path}")

            diagonal_path = output_dir / f"{slug}_difficulty_{metric_slug}_diagonal.pdf"
            plot_diagonal_heatmaps(
                rankings=hard_to_easy,
                models=model_names,
                dataset=dataset,
                metric=args.metric,
                output_path=diagonal_path,
                heatmap_fontsize=args.heatmap_fontsize,
                heatmap_text_color=args.heatmap_text_color,
                font_scale=args.font_scale,
            )
            logger.info(f"Wrote: {diagonal_path}")

        logger.info("")


if __name__ == "__main__":
    main()
