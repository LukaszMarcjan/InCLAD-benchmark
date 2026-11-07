"""
Analyze result JSON files stored under a NeurIPS-style results directory.

The script scans recursively for `*_seed_*.json` files, extracts shared
metadata and aggregate continual-learning metrics from both image-level and
pixel-level result payloads, and writes CSV reports that are easy to inspect
in a spreadsheet or import into pandas/R later.

Reports written:
1. `detailed_metrics.csv`
   One row per `(result JSON, metric payload)`.

2. `group_summary.csv`
   Mean/std/min/max summaries grouped by benchmark, model, strategy,
   metric kind/name, and ordering metadata.

3. `top_runs.csv`
   Best individual runs ranked by a chosen score column.

Usage:
    python3 examples/visual_models/analyze_neurips_results.py neurips_results

    python3 examples/visual_models/analyze_neurips_results.py \
        neurips_results \
        --benchmark visa \
        --strategy replay \
        --metric-kind pixel \
        --metric Pixel-ROC-AUC

    python3 examples/visual_models/analyze_neurips_results.py \
        neurips_results \
        --output-dir neurips_results/analysis \
        --top-k 100 \
        --sort-by diagonal_average
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterable

logger = logging.getLogger(__name__)

IMAGE_METRIC_KEY_PREFIX = "concept_metric_callback_"
PIXEL_METRIC_KEY_PREFIX = "pixel_concept_metric_callback_"
METRIC_KEY_PREFIXES = {
    "image": IMAGE_METRIC_KEY_PREFIX,
    "pixel": PIXEL_METRIC_KEY_PREFIX,
}
SEED_RESULT_PATTERN = re.compile(r".+_seed_(\d+)$")
KNOWN_BENCHMARKS = {
    "mvtec",
    "btech",
    "visa",
    "mpdd",
    "dagm",
    "ksdd2",
    "magnetic_tile",
}
TOP_RUN_FIELDS = [
    "source_group",
    "run_folder",
    "benchmark",
    "model",
    "strategy",
    "metric_kind",
    "metric_name",
    "ordering_mode",
    "ordering_metric",
    "seed",
    "sort_score",
    "continual_average",
    "diagonal_average",
    "backward_transfer",
    "forward_transfer",
    "train_time_total",
    "eval_time_total",
    "peak_memory_max",
    "json_path",
]


@dataclass(frozen=True)
class MetricRecord:
    json_path: pathlib.Path
    source_group: str
    run_folder: str
    benchmark: str
    dataset_name: str
    model: str
    strategy: str
    strategy_name: str
    seed: int | None
    metric_kind: str
    metric_name: str
    eval_level: str
    continual_average: float
    diagonal_average: float
    backward_transfer: float
    forward_transfer: float
    concept_count: int
    concepts_order: tuple[str, ...]
    ordering_mode: str
    ordering_metric: str
    ordering_source: str
    ordering_file: str
    replay_buffer_fraction: float | None
    train_time_total: float | None
    eval_time_total: float | None
    peak_memory_max: float | None
    current_memory_max: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze JSON results stored under neurips_results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_root",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("neurips_results"),
        help="Root directory with result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Where CSV reports should be written (default: RESULTS_ROOT/analysis)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="*",
        default=None,
        help="Optional benchmark filter, e.g. visa btech dagm",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=None,
        help="Optional model filter, e.g. padim patchcore rd4ad",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        nargs="*",
        default=None,
        help="Optional strategy filter, e.g. naive cl replay",
    )
    parser.add_argument(
        "--metric-kind",
        type=str,
        default="all",
        choices=["all", "image", "pixel"],
        help="Keep only image metrics, pixel metrics, or both",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="*",
        default=None,
        help="Optional metric-name filter, e.g. ROC-AUC Pixel-ROC-AUC F1-Score",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="diagonal_average",
        choices=[
            "diagonal_average",
            "continual_average",
            "backward_transfer",
            "forward_transfer",
            "train_time_total",
            "eval_time_total",
            "peak_memory_max",
        ],
        help="Column used to rank top individual runs",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Rank smaller values first (useful for time or memory columns)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many rows to keep in top_runs.csv",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep metric payloads with empty concepts_order",
    )
    return parser.parse_args()


def normalize_label(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def normalized_filter(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {normalize_label(value) for value in values if str(value).strip()}
    return normalized or None


def json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.10f}"


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def max_numeric(mapping: Any) -> float | None:
    if not isinstance(mapping, dict):
        return None
    values = [numeric_or_none(value) for value in mapping.values()]
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    return max(finite_values) if finite_values else None


def infer_seed(json_path: pathlib.Path, data: dict[str, Any]) -> int | None:
    run_config = data.get("run_config")
    if isinstance(run_config, dict):
        seed_value = run_config.get("seed")
        if isinstance(seed_value, int):
            return seed_value

    match = SEED_RESULT_PATTERN.match(json_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def infer_source_group(json_path: pathlib.Path, results_root: pathlib.Path) -> str:
    try:
        parts = json_path.relative_to(results_root).parts
    except ValueError:
        parts = json_path.parts
    return parts[0] if parts else ""


def infer_run_folder(json_path: pathlib.Path, results_root: pathlib.Path) -> str:
    try:
        parts = json_path.relative_to(results_root).parts
    except ValueError:
        parts = json_path.parts
    return parts[1] if len(parts) >= 2 else json_path.parent.name


def infer_benchmark(json_path: pathlib.Path, results_root: pathlib.Path, data: dict[str, Any]) -> str:
    run_config = data.get("run_config")
    if isinstance(run_config, dict):
        benchmark = run_config.get("benchmark")
        if benchmark:
            return normalize_label(str(benchmark))

    try:
        relative_parts = json_path.relative_to(results_root).parts[:-1]
    except ValueError:
        relative_parts = json_path.parts[:-1]

    for part in relative_parts:
        normalized = normalize_label(part)
        if normalized in KNOWN_BENCHMARKS:
            return normalized

    dataset_payload = data.get("dataset")
    if isinstance(dataset_payload, dict):
        dataset_name = str(dataset_payload.get("name", ""))
        if dataset_name:
            lowered = dataset_name.lower()
            for benchmark_name in KNOWN_BENCHMARKS:
                if benchmark_name in lowered:
                    return benchmark_name

    return normalize_label(json_path.parent.name)


def infer_model(data: dict[str, Any]) -> str:
    run_config = data.get("run_config")
    if isinstance(run_config, dict):
        model = run_config.get("model")
        if model:
            return normalize_label(str(model))

    model_payload = data.get("model")
    if isinstance(model_payload, dict):
        model_name = model_payload.get("name")
        if model_name:
            return normalize_label(str(model_name))

    return ""


def infer_strategy(data: dict[str, Any]) -> tuple[str, str]:
    run_config = data.get("run_config")
    if isinstance(run_config, dict):
        strategy = run_config.get("strategy")
        if strategy:
            strategy_value = normalize_label(str(strategy))
            strategy_payload = data.get("strategy")
            strategy_name = ""
            if isinstance(strategy_payload, dict):
                strategy_name = str(strategy_payload.get("name", ""))
            return strategy_value, strategy_name

    strategy_payload = data.get("strategy")
    if isinstance(strategy_payload, dict):
        raw_name = str(strategy_payload.get("name", ""))
        normalized_name = normalize_label(raw_name)
        if "replay" in normalized_name:
            return "replay", raw_name
        if normalized_name.endswith("cl") or "continual" in normalized_name:
            return "cl", raw_name
        if normalized_name == "naive":
            return "naive", raw_name
        return normalized_name, raw_name

    return "", ""


def detect_metric_kind(key: str, payload: dict[str, Any]) -> str | None:
    evaluation_level = normalize_label(str(payload.get("evaluation_level", "")))
    if evaluation_level in {"image", "pixel"}:
        return evaluation_level

    for metric_kind, prefix in METRIC_KEY_PREFIXES.items():
        if key.startswith(prefix):
            return metric_kind

    return None


def metric_name_from_key(key: str, payload: dict[str, Any], metric_kind: str) -> str:
    prefix = METRIC_KEY_PREFIXES.get(metric_kind, "")
    fallback = key[len(prefix):] if key.startswith(prefix) else key
    return str(payload.get("base_metric_name", fallback))


def metric_value(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def extract_ordering_metadata(data: dict[str, Any]) -> tuple[str, str, str, str]:
    run_config = data.get("run_config")
    if not isinstance(run_config, dict):
        return "", "", "", ""

    ordering = run_config.get("ordering_metadata")
    if not isinstance(ordering, dict):
        return "", "", "", ""

    return (
        str(ordering.get("ordering_mode", "")),
        str(ordering.get("ordering_metric", "")),
        str(ordering.get("source", "")),
        str(ordering.get("ordering_file", "")),
    )


def extract_time_totals(data: dict[str, Any]) -> tuple[float | None, float | None]:
    payload = data.get("time_evaluation_callback")
    if not isinstance(payload, dict):
        return None, None
    return numeric_or_none(payload.get("train_time_total")), numeric_or_none(payload.get("eval_time_total"))


def extract_memory_peaks(data: dict[str, Any]) -> tuple[float | None, float | None]:
    payload = data.get("memory_usage_callback")
    if not isinstance(payload, dict):
        return None, None
    return max_numeric(payload.get("peak_memory_usage")), max_numeric(payload.get("current_memory_usage"))


def extract_replay_buffer_fraction(data: dict[str, Any]) -> float | None:
    run_config = data.get("run_config")
    if isinstance(run_config, dict):
        replay_buffer_fraction = numeric_or_none(run_config.get("replay_buffer_fraction"))
        if replay_buffer_fraction is not None:
            return replay_buffer_fraction

        cli_args = run_config.get("cli_args")
        if isinstance(cli_args, dict):
            replay_buffer_fraction = numeric_or_none(cli_args.get("replay_buffer_fraction"))
            if replay_buffer_fraction is not None:
                return replay_buffer_fraction

    return None


def discover_records(results_root: pathlib.Path) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    json_paths = sorted(results_root.rglob("*_seed_*.json"))

    if not json_paths:
        logger.warning("No '*_seed_*.json' files found under %s", results_root)
        return records

    logger.info("Found %d candidate result JSON file(s)", len(json_paths))

    for index, json_path in enumerate(json_paths, start=1):
        if index % 500 == 0:
            logger.info("Processed %d / %d JSON files", index, len(json_paths))

        try:
            with json_path.open(encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable JSON %s: %s", json_path, exc)
            continue

        source_group = infer_source_group(json_path, results_root)
        run_folder = infer_run_folder(json_path, results_root)
        benchmark = infer_benchmark(json_path, results_root, data)
        dataset_name = str(data.get("dataset", {}).get("name", "")) if isinstance(data.get("dataset"), dict) else ""
        model = infer_model(data)
        strategy, strategy_name = infer_strategy(data)
        seed = infer_seed(json_path, data)
        ordering_mode, ordering_metric, ordering_source, ordering_file = extract_ordering_metadata(data)
        replay_buffer_fraction = extract_replay_buffer_fraction(data)
        train_time_total, eval_time_total = extract_time_totals(data)
        peak_memory_max, current_memory_max = extract_memory_peaks(data)
        eval_level = str(data.get("run_config", {}).get("eval_level", "")) if isinstance(data.get("run_config"), dict) else ""

        for key, payload in data.items():
            if not isinstance(payload, dict):
                continue

            metric_kind = detect_metric_kind(key, payload)
            if metric_kind is None:
                continue

            metrics = payload.get("metrics")
            concepts_order = payload.get("concepts_order")
            if not isinstance(metrics, dict) or not isinstance(concepts_order, list):
                logger.debug("Skipping malformed metric payload in %s: %s", json_path, key)
                continue

            records.append(
                MetricRecord(
                    json_path=json_path,
                    source_group=source_group,
                    run_folder=run_folder,
                    benchmark=benchmark,
                    dataset_name=dataset_name,
                    model=model,
                    strategy=strategy,
                    strategy_name=strategy_name,
                    seed=seed,
                    metric_kind=metric_kind,
                    metric_name=metric_name_from_key(key, payload, metric_kind),
                    eval_level=eval_level,
                    continual_average=metric_value(metrics, "ContinualAverage"),
                    diagonal_average=metric_value(metrics, "DiagonalAverage"),
                    backward_transfer=metric_value(metrics, "BackwardTransfer"),
                    forward_transfer=metric_value(metrics, "ForwardTransfer"),
                    concept_count=len(concepts_order),
                    concepts_order=tuple(str(concept) for concept in concepts_order),
                    ordering_mode=ordering_mode,
                    ordering_metric=ordering_metric,
                    ordering_source=ordering_source,
                    ordering_file=ordering_file,
                    replay_buffer_fraction=replay_buffer_fraction,
                    train_time_total=train_time_total,
                    eval_time_total=eval_time_total,
                    peak_memory_max=peak_memory_max,
                    current_memory_max=current_memory_max,
                )
            )

    return records


def record_matches_filters(
    record: MetricRecord,
    benchmark_filter: set[str] | None,
    model_filter: set[str] | None,
    strategy_filter: set[str] | None,
    metric_kind: str,
    metric_filter: set[str] | None,
) -> bool:
    if benchmark_filter is not None and normalize_label(record.benchmark) not in benchmark_filter:
        return False
    if model_filter is not None and normalize_label(record.model) not in model_filter:
        return False
    if strategy_filter is not None and normalize_label(record.strategy) not in strategy_filter:
        return False
    if metric_kind != "all" and record.metric_kind != metric_kind:
        return False
    if metric_filter is not None and normalize_label(record.metric_name) not in metric_filter:
        return False
    return True


def build_detailed_rows(records: list[MetricRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "source_group": record.source_group,
                "run_folder": record.run_folder,
                "benchmark": record.benchmark,
                "dataset_name": record.dataset_name,
                "model": record.model,
                "strategy": record.strategy,
                "strategy_name": record.strategy_name,
                "seed": "" if record.seed is None else str(record.seed),
                "metric_kind": record.metric_kind,
                "metric_name": record.metric_name,
                "eval_level": record.eval_level,
                "concept_count": str(record.concept_count),
                "continual_average": format_float(record.continual_average),
                "diagonal_average": format_float(record.diagonal_average),
                "backward_transfer": format_float(record.backward_transfer),
                "forward_transfer": format_float(record.forward_transfer),
                "train_time_total": format_float(record.train_time_total),
                "eval_time_total": format_float(record.eval_time_total),
                "peak_memory_max": format_float(record.peak_memory_max),
                "current_memory_max": format_float(record.current_memory_max),
                "replay_buffer_fraction": format_float(record.replay_buffer_fraction),
                "ordering_mode": record.ordering_mode,
                "ordering_metric": record.ordering_metric,
                "ordering_source": record.ordering_source,
                "ordering_file": record.ordering_file,
                "concepts_order": json_list(record.concepts_order),
                "json_path": str(record.json_path),
            }
        )
    return rows


def summarize_values(values: list[float | None]) -> tuple[int, float | None, float | None, float | None, float | None]:
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    if not finite_values:
        return 0, None, None, None, None
    if len(finite_values) == 1:
        return 1, finite_values[0], 0.0, finite_values[0], finite_values[0]
    return (
        len(finite_values),
        mean(finite_values),
        pstdev(finite_values),
        min(finite_values),
        max(finite_values),
    )


def build_group_summary_rows(records: list[MetricRecord]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[MetricRecord]] = defaultdict(list)
    group_fields = (
        "source_group",
        "benchmark",
        "model",
        "strategy",
        "metric_kind",
        "metric_name",
        "ordering_mode",
        "ordering_metric",
    )

    for record in records:
        key = (
            record.source_group,
            record.benchmark,
            record.model,
            record.strategy,
            record.metric_kind,
            record.metric_name,
            record.ordering_mode,
            record.ordering_metric,
        )
        grouped[key].append(record)

    rows: list[dict[str, str]] = []
    for key in sorted(grouped):
        group_records = grouped[key]
        row = {field: value for field, value in zip(group_fields, key)}
        row["runs"] = str(len(group_records))
        row["unique_seeds"] = str(len({record.seed for record in group_records}))
        row["concept_count_min"] = str(min(record.concept_count for record in group_records))
        row["concept_count_max"] = str(max(record.concept_count for record in group_records))
        row["strategy_names"] = json_list(sorted({record.strategy_name for record in group_records if record.strategy_name}))
        row["run_folders"] = json_list(sorted({record.run_folder for record in group_records}))

        metrics_to_summarize = {
            "continual_average": [record.continual_average for record in group_records],
            "diagonal_average": [record.diagonal_average for record in group_records],
            "backward_transfer": [record.backward_transfer for record in group_records],
            "forward_transfer": [record.forward_transfer for record in group_records],
            "train_time_total": [record.train_time_total for record in group_records],
            "eval_time_total": [record.eval_time_total for record in group_records],
            "peak_memory_max": [record.peak_memory_max for record in group_records],
            "current_memory_max": [record.current_memory_max for record in group_records],
        }

        for metric_name, values in metrics_to_summarize.items():
            count, avg, std, min_value, max_value = summarize_values(values)
            row[f"{metric_name}_count"] = str(count)
            row[f"{metric_name}_mean"] = format_float(avg)
            row[f"{metric_name}_std"] = format_float(std)
            row[f"{metric_name}_min"] = format_float(min_value)
            row[f"{metric_name}_max"] = format_float(max_value)

        rows.append(row)

    return rows


def sort_key_for_record(record: MetricRecord, sort_by: str, ascending: bool) -> tuple[int, float]:
    value = getattr(record, sort_by)
    if value is None or not math.isfinite(value):
        return (1, 0.0)
    return (0, value if ascending else -value)


def build_top_rows(records: list[MetricRecord], sort_by: str, ascending: bool, top_k: int) -> list[dict[str, str]]:
    ordered = sorted(records, key=lambda record: (sort_key_for_record(record, sort_by, ascending), str(record.json_path)))
    top_records = ordered[: max(top_k, 0)]

    rows: list[dict[str, str]] = []
    for record in top_records:
        sort_score = getattr(record, sort_by)
        rows.append(
            {
                "source_group": record.source_group,
                "run_folder": record.run_folder,
                "benchmark": record.benchmark,
                "model": record.model,
                "strategy": record.strategy,
                "metric_kind": record.metric_kind,
                "metric_name": record.metric_name,
                "ordering_mode": record.ordering_mode,
                "ordering_metric": record.ordering_metric,
                "seed": "" if record.seed is None else str(record.seed),
                "sort_score": format_float(sort_score),
                "continual_average": format_float(record.continual_average),
                "diagonal_average": format_float(record.diagonal_average),
                "backward_transfer": format_float(record.backward_transfer),
                "forward_transfer": format_float(record.forward_transfer),
                "train_time_total": format_float(record.train_time_total),
                "eval_time_total": format_float(record.eval_time_total),
                "peak_memory_max": format_float(record.peak_memory_max),
                "json_path": str(record.json_path),
            }
        )
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []

    with path.open("w", newline="", encoding="utf-8") as fp:
        if fieldnames:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            fp.write("")

    logger.info("Wrote %s (%d rows)", path, len(rows))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results root does not exist: {results_root}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_root / "analysis"
    )

    benchmark_filter = normalized_filter(args.benchmark)
    model_filter = normalized_filter(args.model)
    strategy_filter = normalized_filter(args.strategy)
    metric_filter = normalized_filter(args.metric)

    records = discover_records(results_root)
    if not args.include_empty:
        non_empty_records = [record for record in records if record.concept_count > 0]
        skipped_empty = len(records) - len(non_empty_records)
        if skipped_empty > 0:
            logger.info("Skipping %d empty metric payload(s) with concept_count = 0", skipped_empty)
        records = non_empty_records

    filtered_records = [
        record
        for record in records
        if record_matches_filters(
            record=record,
            benchmark_filter=benchmark_filter,
            model_filter=model_filter,
            strategy_filter=strategy_filter,
            metric_kind=args.metric_kind,
            metric_filter=metric_filter,
        )
    ]

    logger.info("Loaded %d metric payload(s), kept %d after filtering", len(records), len(filtered_records))
    if not filtered_records:
        logger.warning("No records matched the selected filters")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "detailed_metrics.csv", [])
        write_csv(output_dir / "group_summary.csv", [])
        write_csv(output_dir / "top_runs.csv", [])
        return 0

    detailed_rows = build_detailed_rows(filtered_records)
    summary_rows = build_group_summary_rows(filtered_records)
    top_rows = build_top_rows(filtered_records, args.sort_by, args.ascending, args.top_k)

    write_csv(output_dir / "detailed_metrics.csv", detailed_rows)
    write_csv(output_dir / "group_summary.csv", summary_rows)
    write_csv(output_dir / "top_runs.csv", top_rows)

    logger.info("Analysis complete")
    logger.info("Benchmarks: %s", ", ".join(sorted({record.benchmark for record in filtered_records})))
    logger.info("Models: %s", ", ".join(sorted({record.model for record in filtered_records})))
    logger.info("Strategies: %s", ", ".join(sorted({record.strategy for record in filtered_records})))
    logger.info("Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
