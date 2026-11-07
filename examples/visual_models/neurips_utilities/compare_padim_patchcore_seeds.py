"""
Compare PaDiM and PatchCore runs across shared datasets and shared seeds.

The script expects two directories containing per-seed JSON results:
    - one for PaDiM
    - one for PatchCore

For every dataset present in both directories, and for every seed present in
both methods, it extracts the selected metric's `DiagonalAverage`, keeps the
shared concept ordering, assigns a stable `order_id` to each unique concept
order within a dataset, and writes five CSV reports:

1. detailed ranking per `(dataset, seed, method)`
   columns: dataset, seed, method, average_diagonal, rank, order_id, concepts_order

2. aggregated difficulty per `(dataset, seed)` with both methods side by side
   columns: dataset, seed, padim_average_diagonal, padim_rank,
            patchcore_average_diagonal, patchcore_rank, mean_average_diagonal,
            difficulty_rank, methods_present, order_id, concepts_order

3. easy/hard seed summary per dataset
   columns: dataset, easy_seed, easy_mean_average_diagonal, easy_order_id,
            easy_concepts_order, hard_seed, hard_mean_average_diagonal,
            hard_order_id, hard_concepts_order

4. order mapping per dataset
   columns: dataset, order_id, concepts_order, seeds, methods

5. order summary per dataset
   columns: dataset, order_id, order_rank, mean_average_diagonal,
            mean_padim_average_diagonal, mean_patchcore_average_diagonal,
            seed_count, seeds, concepts_order

By default the script compares `ROC-AUC`, but you can switch to `F1-Score`
with `--metric`.

Usage:
    PYTHONPATH=src python3 examples/visual_models/compare_padim_patchcore_seeds.py \
        --padim-dir examples/visual_models/results_padim \
        --patchcore-dir examples/visual_models/results_patchcore

    PYTHONPATH=src python3 examples/visual_models/compare_padim_patchcore_seeds.py \
        --padim-dir examples/visual_models/results_mvtec/padim \
        --patchcore-dir examples/visual_models/results_patchcore \
        --metric F1-Score --output-dir /tmp/seed_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

logger = logging.getLogger(__name__)

SEED_RESULT_PATTERN = re.compile(r".+_seed_(\d+)$")
METRIC_KEY_PREFIX = "concept_metric_callback_"
KNOWN_BENCHMARKS = (
    "mvtec",
    "btech",
    "visa",
    "mpdd",
    "dagm",
    "ksdd2",
    "magnetic_tile",
)


@dataclass(frozen=True)
class MethodResult:
    dataset: str
    seed: int
    method: str
    average_diagonal: float
    concepts_order: tuple[str, ...]
    json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PaDiM and PatchCore results for shared datasets and seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--padim-dir",
        type=str,
        required=True,
        help="Directory containing PaDiM result JSON files",
    )
    parser.add_argument(
        "--patchcore-dir",
        type=str,
        required=True,
        help="Directory containing PatchCore result JSON files",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ROC-AUC",
        help="Metric name to compare, e.g. 'ROC-AUC' or 'F1-Score'",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset filter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/visual_models/seed_comparison",
        help="Directory where CSV reports will be written",
    )
    return parser.parse_args()


def metric_slug(metric_name: str) -> str:
    return metric_name.lower().replace(" ", "_").replace("/", "-")


def format_score(score: float) -> str:
    if isinstance(score, float) and math.isnan(score):
        return "nan"
    return f"{score:.10f}"


def format_optional_score(score: float | None) -> str:
    if score is None:
        return ""
    return format_score(score)


def format_optional_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def concepts_order_to_str(concepts_order: tuple[str, ...]) -> str:
    return json.dumps(list(concepts_order), ensure_ascii=False)


def infer_dataset_name(json_path: Path, root_dir: Path) -> str:
    try:
        relative_parts = json_path.relative_to(root_dir).parts[:-1]
    except ValueError:
        relative_parts = json_path.parts[:-1]

    for part in relative_parts:
        normalized = part.lower()
        if normalized in KNOWN_BENCHMARKS:
            return normalized

    return json_path.parent.name.lower()


def extract_seed(json_path: Path) -> int | None:
    match = SEED_RESULT_PATTERN.match(json_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def find_metric_payload(data: dict, requested_metric: str) -> dict | None:
    requested_normalized = requested_metric.lower()

    for key, value in data.items():
        if not key.startswith(METRIC_KEY_PREFIX):
            continue
        if not isinstance(value, dict):
            continue

        base_metric_name = str(value.get("base_metric_name", key[len(METRIC_KEY_PREFIX):]))
        if base_metric_name.lower() == requested_normalized:
            return value

    return None


def parse_average_diagonal(metric_payload: dict, json_path: Path, metric_name: str) -> float:
    metrics = metric_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing 'metrics' section for '{metric_name}' in {json_path}")

    if "DiagonalAverage" not in metrics:
        raise ValueError(f"Missing 'DiagonalAverage' for '{metric_name}' in {json_path}")

    value = metrics["DiagonalAverage"]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Unexpected DiagonalAverage type in {json_path}: {type(value)!r}")

    return float(value)


def discover_method_results(
    results_dir: Path,
    method: str,
    requested_metric: str,
    datasets_filter: set[str] | None,
) -> dict[tuple[str, int], MethodResult]:
    found: dict[tuple[str, int], MethodResult] = {}

    for json_path in sorted(results_dir.rglob(f"{method}_seed_*.json")):
        dataset = infer_dataset_name(json_path, results_dir)
        if datasets_filter is not None and dataset not in datasets_filter:
            continue

        seed = extract_seed(json_path)
        if seed is None:
            logger.warning(f"Skipping file with unrecognized seed pattern: {json_path}")
            continue

        try:
            with json_path.open(encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Skipping unreadable file {json_path}: {exc}")
            continue

        metric_payload = find_metric_payload(data, requested_metric)
        if metric_payload is None:
            logger.warning(f"Skipping {json_path}: metric '{requested_metric}' not found")
            continue

        concepts_order = metric_payload.get("concepts_order")
        if not isinstance(concepts_order, list) or not concepts_order:
            logger.warning(f"Skipping {json_path}: missing concepts_order for metric '{requested_metric}'")
            continue

        try:
            average_diagonal = parse_average_diagonal(metric_payload, json_path, requested_metric)
        except ValueError as exc:
            logger.warning(str(exc))
            continue

        key = (dataset, seed)
        found[key] = MethodResult(
            dataset=dataset,
            seed=seed,
            method=method,
            average_diagonal=average_diagonal,
            concepts_order=tuple(str(name) for name in concepts_order),
            json_path=json_path,
        )

    return found


def shared_results(
    padim_results: dict[tuple[str, int], MethodResult],
    patchcore_results: dict[tuple[str, int], MethodResult],
) -> dict[str, list[MethodResult]]:
    shared_keys = sorted(set(padim_results) & set(patchcore_results))
    grouped: dict[str, list[MethodResult]] = defaultdict(list)

    for key in shared_keys:
        padim_result = padim_results[key]
        patchcore_result = patchcore_results[key]

        if padim_result.concepts_order != patchcore_result.concepts_order:
            logger.warning(
                "Concept ordering mismatch for dataset=%s seed=%s between %s and %s",
                padim_result.dataset,
                padim_result.seed,
                padim_result.json_path,
                patchcore_result.json_path,
            )

        grouped[padim_result.dataset].append(padim_result)
        grouped[patchcore_result.dataset].append(patchcore_result)

    return dict(sorted(grouped.items()))


def sort_key_for_descending_score(score: float) -> tuple[int, float]:
    if math.isnan(score):
        return (1, 0.0)
    return (0, -score)


def mean_of_scores(scores: list[float]) -> float:
    finite_scores = [score for score in scores if not math.isnan(score)]
    if not finite_scores:
        return float("nan")
    return sum(finite_scores) / len(finite_scores)


def build_order_id_lookup(grouped_results: dict[str, list[MethodResult]]) -> dict[str, dict[tuple[str, ...], str]]:
    lookup: dict[str, dict[tuple[str, ...], str]] = {}

    for dataset, results in grouped_results.items():
        unique_orders = sorted(
            {result.concepts_order for result in results},
            key=concepts_order_to_str,
        )
        lookup[dataset] = {
            concepts_order: f"order_{index}"
            for index, concepts_order in enumerate(unique_orders)
        }

    return lookup


def build_method_rank_lookup(grouped_results: dict[str, list[MethodResult]]) -> dict[str, dict[str, dict[int, int]]]:
    lookup: dict[str, dict[str, dict[int, int]]] = {}

    for dataset, results in grouped_results.items():
        by_method: dict[str, list[MethodResult]] = defaultdict(list)
        for result in results:
            by_method[result.method].append(result)

        dataset_lookup: dict[str, dict[int, int]] = {}
        for method, method_results in by_method.items():
            ordered_results = sorted(
                method_results,
                key=lambda result: (sort_key_for_descending_score(result.average_diagonal), result.seed),
            )
            dataset_lookup[method] = {
                result.seed: rank
                for rank, result in enumerate(ordered_results)
            }

        lookup[dataset] = dataset_lookup

    return lookup


def build_detailed_rows(
    grouped_results: dict[str, list[MethodResult]],
    order_id_lookup: dict[str, dict[tuple[str, ...], str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for dataset, results in grouped_results.items():
        by_method: dict[str, list[MethodResult]] = defaultdict(list)
        for result in results:
            by_method[result.method].append(result)

        for method, method_results in sorted(by_method.items()):
            ordered_results = sorted(
                method_results,
                key=lambda result: (sort_key_for_descending_score(result.average_diagonal), result.seed),
            )

            for rank, result in enumerate(ordered_results):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": str(result.seed),
                        "method": method,
                        "average_diagonal": format_score(result.average_diagonal),
                        "rank": str(rank),
                        "order_id": order_id_lookup[dataset][result.concepts_order],
                        "concepts_order": concepts_order_to_str(result.concepts_order),
                    }
                )

    return rows


def build_seed_summary_rows(
    grouped_results: dict[str, list[MethodResult]],
    order_id_lookup: dict[str, dict[tuple[str, ...], str]],
    method_rank_lookup: dict[str, dict[str, dict[int, int]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for dataset, results in grouped_results.items():
        by_seed: dict[int, list[MethodResult]] = defaultdict(list)
        for result in results:
            by_seed[result.seed].append(result)

        aggregated: list[tuple[int, float, tuple[str, ...], list[str]]] = []
        for seed, seed_results in by_seed.items():
            mean_score = mean_of_scores([result.average_diagonal for result in seed_results])
            reference_order = seed_results[0].concepts_order
            methods_present = sorted(result.method for result in seed_results)
            aggregated.append((seed, mean_score, reference_order, methods_present))

        ordered_aggregated = sorted(
            aggregated,
            key=lambda item: (sort_key_for_descending_score(item[1]), item[0]),
        )

        for difficulty_rank, (seed, mean_score, concepts_order, methods_present) in enumerate(ordered_aggregated):
            result_by_method = {
                result.method: result
                for result in by_seed[seed]
            }
            padim_result = result_by_method.get("padim")
            patchcore_result = result_by_method.get("patchcore")
            rows.append(
                {
                    "dataset": dataset,
                    "seed": str(seed),
                    "padim_average_diagonal": format_optional_score(
                        None if padim_result is None else padim_result.average_diagonal
                    ),
                    "padim_rank": format_optional_int(
                        method_rank_lookup.get(dataset, {}).get("padim", {}).get(seed)
                    ),
                    "patchcore_average_diagonal": format_optional_score(
                        None if patchcore_result is None else patchcore_result.average_diagonal
                    ),
                    "patchcore_rank": format_optional_int(
                        method_rank_lookup.get(dataset, {}).get("patchcore", {}).get(seed)
                    ),
                    "mean_average_diagonal": format_score(mean_score),
                    "difficulty_rank": str(difficulty_rank),
                    "methods_present": ",".join(methods_present),
                    "order_id": order_id_lookup[dataset][concepts_order],
                    "concepts_order": concepts_order_to_str(concepts_order),
                }
            )

    return rows


def build_easy_hard_rows(
    grouped_results: dict[str, list[MethodResult]],
    order_id_lookup: dict[str, dict[tuple[str, ...], str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for dataset, results in grouped_results.items():
        by_seed: dict[int, list[MethodResult]] = defaultdict(list)
        for result in results:
            by_seed[result.seed].append(result)

        aggregated: list[tuple[int, float, tuple[str, ...]]] = []
        for seed, seed_results in by_seed.items():
            mean_score = mean_of_scores([result.average_diagonal for result in seed_results])
            aggregated.append((seed, mean_score, seed_results[0].concepts_order))

        aggregated = sorted(
            aggregated,
            key=lambda item: (sort_key_for_descending_score(item[1]), item[0]),
        )

        finite_aggregated = [item for item in aggregated if not math.isnan(item[1])]
        if not finite_aggregated:
            logger.warning(f"Dataset '{dataset}' has only NaN scores, skipping easy/hard summary")
            continue

        easy_seed, easy_score, easy_order = finite_aggregated[0]
        hard_seed, hard_score, hard_order = finite_aggregated[-1]
        rows.append(
            {
                "dataset": dataset,
                "easy_seed": str(easy_seed),
                "easy_mean_average_diagonal": format_score(easy_score),
                "easy_order_id": order_id_lookup[dataset][easy_order],
                "easy_concepts_order": concepts_order_to_str(easy_order),
                "hard_seed": str(hard_seed),
                "hard_mean_average_diagonal": format_score(hard_score),
                "hard_order_id": order_id_lookup[dataset][hard_order],
                "hard_concepts_order": concepts_order_to_str(hard_order),
            }
        )

    return rows


def build_order_mapping_rows(
    grouped_results: dict[str, list[MethodResult]],
    order_id_lookup: dict[str, dict[tuple[str, ...], str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for dataset, results in grouped_results.items():
        by_order: dict[tuple[str, ...], list[MethodResult]] = defaultdict(list)
        for result in results:
            by_order[result.concepts_order].append(result)

        unique_orders = sorted(by_order.items(), key=lambda item: concepts_order_to_str(item[0]))
        for concepts_order, order_results in unique_orders:
            seeds = sorted({result.seed for result in order_results})
            methods = sorted({result.method for result in order_results})
            rows.append(
                {
                    "dataset": dataset,
                    "order_id": order_id_lookup[dataset][concepts_order],
                    "concepts_order": concepts_order_to_str(concepts_order),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "methods": ",".join(methods),
                }
            )

    return rows


def build_order_summary_rows(
    grouped_results: dict[str, list[MethodResult]],
    order_id_lookup: dict[str, dict[tuple[str, ...], str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for dataset, results in grouped_results.items():
        by_order: dict[tuple[str, ...], list[MethodResult]] = defaultdict(list)
        for result in results:
            by_order[result.concepts_order].append(result)

        aggregated_orders: list[tuple[str, float, float, float, list[int], tuple[str, ...]]] = []
        for concepts_order, order_results in by_order.items():
            padim_scores = [
                result.average_diagonal
                for result in order_results
                if result.method == "padim"
            ]
            patchcore_scores = [
                result.average_diagonal
                for result in order_results
                if result.method == "patchcore"
            ]
            mean_padim = mean_of_scores(padim_scores)
            mean_patchcore = mean_of_scores(patchcore_scores)
            mean_score = mean_of_scores(padim_scores + patchcore_scores)
            seeds = sorted({result.seed for result in order_results})
            aggregated_orders.append(
                (
                    order_id_lookup[dataset][concepts_order],
                    mean_score,
                    mean_padim,
                    mean_patchcore,
                    seeds,
                    concepts_order,
                )
            )

        ordered_aggregated_orders = sorted(
            aggregated_orders,
            key=lambda item: (sort_key_for_descending_score(item[1]), item[0]),
        )

        for order_rank, (order_id, mean_score, mean_padim, mean_patchcore, seeds, concepts_order) in enumerate(ordered_aggregated_orders):
            rows.append(
                {
                    "dataset": dataset,
                    "order_id": order_id,
                    "order_rank": str(order_rank),
                    "mean_average_diagonal": format_score(mean_score),
                    "mean_padim_average_diagonal": format_score(mean_padim),
                    "mean_patchcore_average_diagonal": format_score(mean_patchcore),
                    "seed_count": str(len(seeds)),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "concepts_order": concepts_order_to_str(concepts_order),
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    padim_dir = Path(args.padim_dir).expanduser()
    patchcore_dir = Path(args.patchcore_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    datasets_filter = None if args.datasets is None else {dataset.lower() for dataset in args.datasets}

    if not padim_dir.is_dir():
        raise SystemExit(f"PaDiM directory does not exist: {padim_dir}")
    if not patchcore_dir.is_dir():
        raise SystemExit(f"PatchCore directory does not exist: {patchcore_dir}")

    padim_results = discover_method_results(
        results_dir=padim_dir,
        method="padim",
        requested_metric=args.metric,
        datasets_filter=datasets_filter,
    )
    patchcore_results = discover_method_results(
        results_dir=patchcore_dir,
        method="patchcore",
        requested_metric=args.metric,
        datasets_filter=datasets_filter,
    )

    logger.info(f"Loaded {len(padim_results)} PaDiM result(s)")
    logger.info(f"Loaded {len(patchcore_results)} PatchCore result(s)")

    grouped_results = shared_results(padim_results, patchcore_results)
    if not grouped_results:
        raise SystemExit("No shared dataset/seed pairs found between PaDiM and PatchCore")

    shared_pair_count = sum(len(results) for results in grouped_results.values()) // 2
    logger.info(
        f"Found {shared_pair_count} shared dataset/seed pair(s) across {len(grouped_results)} dataset(s)"
    )

    order_id_lookup = build_order_id_lookup(grouped_results)
    method_rank_lookup = build_method_rank_lookup(grouped_results)
    detailed_rows = build_detailed_rows(grouped_results, order_id_lookup)
    seed_summary_rows = build_seed_summary_rows(grouped_results, order_id_lookup, method_rank_lookup)
    easy_hard_rows = build_easy_hard_rows(grouped_results, order_id_lookup)
    order_mapping_rows = build_order_mapping_rows(grouped_results, order_id_lookup)
    order_summary_rows = build_order_summary_rows(grouped_results, order_id_lookup)

    metric_name_slug = metric_slug(args.metric)
    detailed_path = output_dir / f"detailed_ranking_{metric_name_slug}.csv"
    seed_summary_path = output_dir / f"seed_summary_{metric_name_slug}.csv"
    easy_hard_path = output_dir / f"easy_hard_{metric_name_slug}.csv"
    order_mapping_path = output_dir / f"order_mapping_{metric_name_slug}.csv"
    order_summary_path = output_dir / f"order_summary_{metric_name_slug}.csv"

    write_csv(
        detailed_path,
        detailed_rows,
        ["dataset", "seed", "method", "average_diagonal", "rank", "order_id", "concepts_order"],
    )
    write_csv(
        seed_summary_path,
        seed_summary_rows,
        [
            "dataset",
            "seed",
            "padim_average_diagonal",
            "padim_rank",
            "patchcore_average_diagonal",
            "patchcore_rank",
            "mean_average_diagonal",
            "difficulty_rank",
            "methods_present",
            "order_id",
            "concepts_order",
        ],
    )
    write_csv(
        easy_hard_path,
        easy_hard_rows,
        [
            "dataset",
            "easy_seed",
            "easy_mean_average_diagonal",
            "easy_order_id",
            "easy_concepts_order",
            "hard_seed",
            "hard_mean_average_diagonal",
            "hard_order_id",
            "hard_concepts_order",
        ],
    )
    write_csv(
        order_mapping_path,
        order_mapping_rows,
        ["dataset", "order_id", "concepts_order", "seeds", "methods"],
    )
    write_csv(
        order_summary_path,
        order_summary_rows,
        [
            "dataset",
            "order_id",
            "order_rank",
            "mean_average_diagonal",
            "mean_padim_average_diagonal",
            "mean_patchcore_average_diagonal",
            "seed_count",
            "seeds",
            "concepts_order",
        ],
    )

    logger.info("")
    for row in easy_hard_rows:
        logger.info(
            "Dataset=%s  easy_seed=%s  hard_seed=%s",
            row["dataset"],
            row["easy_seed"],
            row["hard_seed"],
        )

    logger.info("")
    logger.info(f"Wrote detailed ranking to: {detailed_path}")
    logger.info(f"Wrote seed summary to: {seed_summary_path}")
    logger.info(f"Wrote easy/hard summary to: {easy_hard_path}")
    logger.info(f"Wrote order mapping to: {order_mapping_path}")
    logger.info(f"Wrote order summary to: {order_summary_path}")


if __name__ == "__main__":
    main()
