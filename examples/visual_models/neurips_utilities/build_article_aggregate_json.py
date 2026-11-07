"""
Build an article-friendly aggregate JSON from the curated coverage CSV.

The output keeps one record per logical slot:

    dataset x model x setting x strategy x ordering

For each slot, the script aggregates image-level and pixel-level ROC-AUC
payloads, computes mean/std for the scalar continual metrics, and also
computes mean/std for the full metric matrix so the lower triangle can be
inspected for forgetting.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any

# parents[2] is examples/; the repository root (which holds the examples
# package) is parents[3].
# REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.visual_models.neurips_utilities.audit_neurips_experiment_grid import canonical_hash  # noqa: E402

logger = logging.getLogger(__name__)

IMAGE_PAYLOAD_KEY = "concept_metric_callback_ROC-AUC"
PIXEL_PAYLOAD_KEY = "pixel_concept_metric_callback_Pixel-ROC-AUC"
SCALAR_METRIC_KEYS = {
    "continual_average": "ContinualAverage",
    "diagonal_average": "DiagonalAverage",
    "backward_transfer": "BackwardTransfer",
    "forward_transfer": "ForwardTransfer",
}
ORDERING_LABELS = {
    "easy_to_hard": "curriculum_up",
    "hard_to_easy": "curriculum_down",
    "random": "random",
}
STRATEGY_VARIANTS = {
    "naive": "naive",
    "cl": "continual",
    "replay": "continual",
}
COMPLETE_STATUSES = {"COMPLETE_BOTH_JSON", "COMPLETE_SPLIT_JSONS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an aggregate JSON for article tables from coverage_480_experiments.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coverage-csv",
        type=pathlib.Path,
        default=pathlib.Path("../../../ordering_files/neurips_results/coverage_480_experiments.csv"),
        help="Coverage CSV produced from the curated NeurIPS results view.",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=pathlib.Path("../../../ordering_files/neurips_results/article_model_aggregates.json"),
        help="Where the aggregate JSON should be written.",
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


def split_path_field(value: str) -> list[pathlib.Path]:
    return [pathlib.Path(part.strip()) for part in value.split("|") if part.strip()]


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return (sum((value - center) ** 2 for value in values) / len(values)) ** 0.5


def lower_triangle_only(matrix: list[list[float]], *, strict: bool) -> list[list[float | None]]:
    lowered: list[list[float | None]] = []
    for row_index, row in enumerate(matrix):
        lowered_row: list[float | None] = []
        for col_index, value in enumerate(row):
            keep = row_index > col_index if strict else row_index >= col_index
            lowered_row.append(value if keep else None)
        lowered.append(lowered_row)
    return lowered


def normalize_matrix(matrix: Any, concepts_order: list[str]) -> list[list[float]] | None:
    if isinstance(matrix, dict):
        if not concepts_order:
            concepts_order = [str(key) for key in matrix.keys()]
        normalized_from_mapping: list[list[float]] = []
        for row_key in concepts_order:
            row_mapping = matrix.get(row_key)
            if not isinstance(row_mapping, dict):
                return None
            normalized_row: list[float] = []
            for col_key in concepts_order:
                numeric = numeric_or_none(row_mapping.get(col_key))
                if numeric is None:
                    return None
                normalized_row.append(numeric)
            normalized_from_mapping.append(normalized_row)
        return normalized_from_mapping

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


def aggregate_scalar(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "std": population_std(values) if values else None,
        "values": values,
    }


def aggregate_matrix(matrices: list[list[list[float]]]) -> dict[str, Any] | None:
    if not matrices:
        return None
    n_rows = len(matrices[0])
    n_cols = len(matrices[0][0]) if matrices[0] else 0
    if any(len(matrix) != n_rows for matrix in matrices):
        return None
    if any(any(len(row) != n_cols for row in matrix) for matrix in matrices):
        return None

    mean_matrix: list[list[float]] = []
    std_matrix: list[list[float]] = []
    for row_index in range(n_rows):
        mean_row: list[float] = []
        std_row: list[float] = []
        for col_index in range(n_cols):
            values = [matrix[row_index][col_index] for matrix in matrices]
            mean_row.append(mean(values))
            std_row.append(population_std(values))
        mean_matrix.append(mean_row)
        std_matrix.append(std_row)

    return {
        "count": len(matrices),
        "shape": [n_rows, n_cols],
        "mean": mean_matrix,
        "std": std_matrix,
        "lower_triangle_mean": lower_triangle_only(mean_matrix, strict=True),
        "lower_triangle_std": lower_triangle_only(std_matrix, strict=True),
        "lower_or_equal_triangle_mean": lower_triangle_only(mean_matrix, strict=False),
        "lower_or_equal_triangle_std": lower_triangle_only(std_matrix, strict=False),
    }


def dedupe_paths_by_payload(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    unique_by_hash: dict[str, pathlib.Path] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable JSON %s: %s", path, exc)
            continue
        payload_hash = canonical_hash(data)
        unique_by_hash.setdefault(payload_hash, path)
    return sorted(unique_by_hash.values())


def load_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping unreadable JSON %s: %s", path, exc)
        return None


def aggregate_modality(paths: list[pathlib.Path], payload_key: str) -> dict[str, Any]:
    unique_paths = dedupe_paths_by_payload(paths)
    source_files: list[str] = []
    concepts_orders: list[list[str]] = []
    matrices: list[list[list[float]]] = []
    scalar_values: dict[str, list[float]] = {name: [] for name in SCALAR_METRIC_KEYS}

    for path in unique_paths:
        data = load_json(path)
        if data is None:
            continue
        payload = data.get(payload_key)
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        concepts_order = [str(item) for item in payload.get("concepts_order", [])]
        matrix = normalize_matrix(payload.get("metric_matrix"), concepts_order)
        if matrix is None:
            continue

        source_files.append(str(path))
        concepts_orders.append(concepts_order)
        matrices.append(matrix)
        for name, raw_key in SCALAR_METRIC_KEYS.items():
            value = numeric_or_none(metrics.get(raw_key))
            if value is not None:
                scalar_values[name].append(value)

    reference_order = concepts_orders[0] if concepts_orders else []
    concepts_consistent = all(order == reference_order for order in concepts_orders)

    return {
        "available": bool(source_files),
        "source_files": source_files,
        "count": len(source_files),
        "concepts_order": reference_order,
        "concepts_order_consistent": concepts_consistent,
        "scalar_metrics": {
            name: aggregate_scalar(values)
            for name, values in scalar_values.items()
        },
        "metric_matrix": aggregate_matrix(matrices),
    }


def source_lists_from_row(row: dict[str, str]) -> dict[str, list[pathlib.Path]]:
    return {
        "both": split_path_field(row.get("source_both_json", "")),
        "image": split_path_field(row.get("source_image_json", "")),
        "pixel": split_path_field(row.get("source_pixel_json", "")),
        "incomplete": split_path_field(row.get("found_but_incomplete_json", "")),
    }


def article_variant(strategy: str) -> str:
    return STRATEGY_VARIANTS.get(strategy, "unknown")


def build_record(row: dict[str, str]) -> dict[str, Any]:
    sources = source_lists_from_row(row)
    status = row["status"]

    if status == "COMPLETE_BOTH_JSON":
        image_paths = sources["both"]
        pixel_paths = sources["both"]
    elif status == "COMPLETE_SPLIT_JSONS":
        image_paths = sources["image"]
        pixel_paths = sources["pixel"]
    else:
        image_paths = sources["image"] + sources["both"] + sources["incomplete"]
        pixel_paths = sources["pixel"] + sources["both"] + sources["incomplete"]

    image = aggregate_modality(image_paths, IMAGE_PAYLOAD_KEY)
    pixel = aggregate_modality(pixel_paths, PIXEL_PAYLOAD_KEY)

    return {
        "dataset": row["dataset"],
        "model": row["model"],
        "setting": row["setting"],
        "backbone_or_setting": row["setting"],
        "strategy": row["strategy"],
        "strategy_variant": article_variant(row["strategy"]),
        "ordering": row["ordering"],
        "ordering_label": ORDERING_LABELS.get(row["ordering"], row["ordering"]),
        "status": status.lower(),
        "article_usable": row["status"] in COMPLETE_STATUSES,
        "coverage_status": row["status"],
        "coverage_notes": row.get("notes", ""),
        "source_files": {
            "both": [str(path) for path in dedupe_paths_by_payload(sources["both"])],
            "image": [str(path) for path in dedupe_paths_by_payload(sources["image"])],
            "pixel": [str(path) for path in dedupe_paths_by_payload(sources["pixel"])],
            "incomplete": [str(path) for path in dedupe_paths_by_payload(sources["incomplete"])],
        },
        "image": image,
        "pixel": pixel,
    }


def load_coverage_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_by_field(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        grouped[str(record[field])][record["coverage_status"]] += 1
    return {
        key: dict(sorted(counter.items()))
        for key, counter in sorted(grouped.items())
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    coverage_csv = args.coverage_csv.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = load_coverage_rows(coverage_csv)
    logger.info("Loaded %d coverage rows from %s", len(rows), coverage_csv)

    records = [build_record(row) for row in rows]
    records.sort(
        key=lambda record: (
            record["dataset"],
            record["model"],
            record["setting"],
            record["strategy"],
            record["ordering"],
        )
    )

    status_counts = Counter(record["coverage_status"] for record in records)
    payload = {
        "metadata": {
            "coverage_csv": str(coverage_csv),
            "output_json": str(output_json),
            "std_mode": "population",
            "strategy_variants": STRATEGY_VARIANTS,
            "ordering_labels": ORDERING_LABELS,
            "scalar_metrics": SCALAR_METRIC_KEYS,
            "matrix_note": (
                "metric_matrix.mean/std keep the full matrix; "
                "lower_triangle_* keeps only cells strictly below the diagonal."
            ),
        },
        "summary": {
            "slots_total": len(records),
            "coverage_status_counts": dict(sorted(status_counts.items())),
            "by_dataset": summary_by_field(records, "dataset"),
            "by_model": summary_by_field(records, "model"),
            "by_setting": summary_by_field(records, "setting"),
            "by_strategy_variant": summary_by_field(records, "strategy_variant"),
        },
        "records": records,
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    logger.info("Wrote aggregate JSON to %s", output_json)
    logger.info("Coverage counts: %s", dict(sorted(status_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
