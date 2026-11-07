"""
Copy unique result JSONs into a review-friendly folder tree.

Input:
    A manifest produced by `group_results_by_dataset.py`.

Output layout:
    <output_dir>/<dataset>/<model>/<ordering>/<strategy>/<copied-json>

Only rows with `status=linked` are copied, so mirrored duplicates already
filtered out by the manifest do not get copied again.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import re
import shutil
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

KNOWN_DATASETS = (
    "btech",
    "dagm",
    "ksdd2",
    "magnetic_tile",
    "mpdd",
    "mvtec",
    "visa",
)

KNOWN_MODELS = (
    "cfa",
    "draem",
    "efficientad",
    "fastflow",
    "ganomaly",
    "padim",
    "paste",
    "patchcore",
    "rd4ad",
    "stfpm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy unique result JSONs into dataset/model/ordering/strategy folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("neurips_results/by_dataset_links/manifest.csv"),
        help="Manifest CSV produced by group_results_by_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("neurips_results/review_grouped_copies"),
        help="Where copied files should be written",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def infer_dataset(data: dict[str, Any], source_path: pathlib.Path) -> str:
    dataset_name = normalize_text((data.get("dataset") or {}).get("name"))
    for dataset in KNOWN_DATASETS:
        if dataset in dataset_name:
            return dataset

    path_text = normalize_text(source_path.as_posix())
    for dataset in KNOWN_DATASETS:
        if f"/{dataset}/" in path_text or f"order-{dataset}-" in path_text:
            return dataset

    return "unknown_dataset"


def infer_model(data: dict[str, Any], source_path: pathlib.Path) -> str:
    path_text = normalize_text(source_path.as_posix())
    for model in KNOWN_MODELS:
        if f"/{model}/" in path_text or f"results_{model}_" in path_text or source_path.name.lower().startswith(f"{model}_"):
            return model

    model_name = normalize_text((data.get("model") or {}).get("name"))
    for model in KNOWN_MODELS:
        if model in model_name:
            return model

    return "unknown_model"


def infer_strategy(data: dict[str, Any], source_path: pathlib.Path) -> str:
    path_text = normalize_text(source_path.as_posix())
    if "_naive_" in path_text or "/naive/" in path_text:
        return "naive"
    if "_replay_" in path_text or "/replay/" in path_text:
        return "replay"
    if "_cl_" in path_text or "/cl/" in path_text:
        return "cl"

    strategy_name = normalize_text((data.get("strategy") or {}).get("name"))
    if "naive" in strategy_name:
        return "naive"
    if "replay" in strategy_name:
        return "replay"
    if strategy_name == "cl" or "continual" in strategy_name:
        return "cl"

    return "unknown_strategy"


def infer_ordering(data: dict[str, Any], source_path: pathlib.Path) -> str:
    run_config = data.get("run_config") or {}
    ordering_metadata = run_config.get("ordering_metadata") or {}

    ordering_file = ordering_metadata.get("ordering_file")
    if ordering_file:
        file_name = pathlib.Path(str(ordering_file)).name.lower()
        if "easy_to_hard" in file_name:
            return "easy_to_hard"
        if "hard_to_easy" in file_name:
            return "hard_to_easy"

    ordering_mode = normalize_text(ordering_metadata.get("ordering_mode"))
    if ordering_mode in {"easy_to_hard", "hard_to_easy", "random"}:
        return ordering_mode

    cli_args = run_config.get("cli_args") or {}
    cli_ordering_file = cli_args.get("ordering_file") or run_config.get("ordering_file")
    if cli_ordering_file:
        file_name = pathlib.Path(str(cli_ordering_file)).name.lower()
        if "easy_to_hard" in file_name:
            return "easy_to_hard"
        if "hard_to_easy" in file_name:
            return "hard_to_easy"

    cli_ordering_mode = normalize_text(cli_args.get("ordering_mode"))
    if cli_ordering_mode in {"easy_to_hard", "hard_to_easy", "random"}:
        return cli_ordering_mode

    if cli_args.get("shuffle_concepts"):
        return "random"

    file_name = source_path.name.lower()
    if "_order-shuffle_" in file_name or "_order-random_" in file_name:
        return "random"
    if re.search(r"_order-(?:[^_]+-)?easy-to-hard(?:-roc-auc)?(?:_buf-frac[^_]+)?_seed_", file_name):
        return "easy_to_hard"
    if re.search(r"_order-(?:[^_]+-)?hard-to-easy(?:-roc-auc)?(?:_buf-frac[^_]+)?_seed_", file_name):
        return "hard_to_easy"
    if "_order-" not in file_name:
        return "random"

    return "unknown_ordering"


def sanitize_fragment(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value or "value"


def build_destination_name(source_path: pathlib.Path, manifest_row: dict[str, str]) -> str:
    parts = source_path.parts
    source_group = parts[-5] if len(parts) >= 5 else "source"
    run_folder = parts[-4] if len(parts) >= 4 else "run"

    # Prefer top-level results buckets like 73_results / hpc_results / res2504
    for index, part in enumerate(parts):
        if part == "neurips_results" and index + 2 < len(parts):
            source_group = parts[index + 1]
            run_folder = parts[index + 2]
            break

    base = source_path.stem
    suffix = source_path.suffix
    hash_prefix = manifest_row.get("canonical_hash", "")[:8]
    return f"{sanitize_fragment(source_group)}__{sanitize_fragment(run_folder)}__{sanitize_fragment(base)}__{hash_prefix}{suffix}"


def copy_unique_results(manifest_path: pathlib.Path, output_dir: pathlib.Path) -> tuple[int, Counter[str]]:
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    linked_rows = [row for row in rows if row.get("status") == "linked"]

    manifest_out_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    copied = 0

    for row in linked_rows:
        source_path = pathlib.Path(row["source_path"]).expanduser().resolve()
        try:
            with source_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable JSON %s: %s", source_path, exc)
            continue

        dataset = infer_dataset(data, source_path)
        model = infer_model(data, source_path)
        ordering = infer_ordering(data, source_path)
        strategy = infer_strategy(data, source_path)

        dest_dir = output_dir / dataset / model / ordering / strategy
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_name = build_destination_name(source_path, row)
        dest_path = dest_dir / dest_name
        shutil.copy2(source_path, dest_path)

        manifest_out_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "ordering": ordering,
                "strategy": strategy,
                "source_path": str(source_path),
                "copied_path": str(dest_path),
                "canonical_hash": row.get("canonical_hash", ""),
            }
        )
        counts[dataset] += 1
        copied += 1

    manifest_out_path = output_dir / "copied_manifest.csv"
    with manifest_out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "model",
                "ordering",
                "strategy",
                "source_path",
                "copied_path",
                "canonical_hash",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_out_rows)

    summary_out_path = output_dir / "copied_summary.csv"
    with summary_out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "copied_files"])
        for dataset in sorted(counts):
            writer.writerow([dataset, counts[dataset]])

    logger.info("Copied %d JSON files into %s", copied, output_dir)
    logger.info("Manifest: %s", manifest_out_path)
    logger.info("Summary: %s", summary_out_path)
    return copied, counts


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_unique_results(manifest_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
