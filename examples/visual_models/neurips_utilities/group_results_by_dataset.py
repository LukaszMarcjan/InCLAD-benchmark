"""
Group result JSON files under per-dataset folders without copying data.

The script scans recursively under a results root, infers the dataset for each
JSON result, deduplicates mirrored runs by a canonical JSON hash, and creates
symbolic links under:

    <results_root>/by_dataset_links/<dataset>/<original-relative-path>

This keeps the source files in place while giving a dataset-centric view.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pathlib
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

KNOWN_DATASETS = (
    "btech",
    "dagm",
    "mpdd",
    "mvtec",
    "visa",
    "inclad-md",
    "inclad_md",
    "inclad-bench",
    "inclad_bench",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group result JSON files by dataset using symlinks.",
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
        help="Output directory for grouped symlinks (default: RESULTS_ROOT/by_dataset_links)",
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


def infer_dataset(data: dict[str, Any], json_path: pathlib.Path) -> str | None:
    dataset_name = normalize_text((data.get("dataset") or {}).get("name"))
    for dataset in KNOWN_DATASETS:
        if dataset in dataset_name:
            return dataset

    path_text = normalize_text(json_path.as_posix())
    for dataset in KNOWN_DATASETS:
        if f"/{dataset}/" in path_text or f"order-{dataset}-" in path_text:
            return dataset

    return None


def strip_path_like_keys(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"output_path", "log_path", "registry_path"}:
                continue
            cleaned[key] = strip_path_like_keys(item)
        return cleaned
    if isinstance(value, list):
        return [strip_path_like_keys(item) for item in value]
    return value


def canonical_hash(data: dict[str, Any]) -> str:
    normalized = strip_path_like_keys(data)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_unlink(path: pathlib.Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def write_manifest(output_dir: pathlib.Path, rows: list[dict[str, str]]) -> pathlib.Path:
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "dataset",
        "status",
        "source_path",
        "symlink_path",
        "duplicate_of",
        "canonical_hash",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_summary(output_dir: pathlib.Path, counts: Counter[str], duplicates: Counter[str], skipped_unknown: int) -> pathlib.Path:
    summary_path = output_dir / "summary.csv"
    datasets = sorted(set(counts) | set(duplicates))
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "linked_files", "deduplicated_files"])
        for dataset in datasets:
            writer.writerow([dataset, counts[dataset], duplicates[dataset]])
        writer.writerow(["__unknown__", skipped_unknown, ""])
    return summary_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    results_root = args.results_root.expanduser().resolve()
    output_dir = (args.output_dir or (results_root / "by_dataset_links")).expanduser().resolve()

    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[tuple[str, str], pathlib.Path] = {}
    manifest_rows: list[dict[str, str]] = []
    linked_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    skipped_unknown = 0

    json_paths = sorted(results_root.rglob("*.json"))
    logger.info("Scanning %d JSON files under %s", len(json_paths), results_root)

    for json_path in json_paths:
        if output_dir in json_path.parents:
            continue

        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable JSON %s: %s", json_path, exc)
            continue

        dataset = infer_dataset(data, json_path)
        if dataset is None:
            skipped_unknown += 1
            manifest_rows.append(
                {
                    "dataset": "",
                    "status": "unknown_dataset",
                    "source_path": str(json_path),
                    "symlink_path": "",
                    "duplicate_of": "",
                    "canonical_hash": "",
                }
            )
            continue

        digest = canonical_hash(data)
        duplicate_key = (dataset, digest)
        original_source = seen_hashes.get(duplicate_key)
        if original_source is not None:
            duplicate_counts[dataset] += 1
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "status": "duplicate_skipped",
                    "source_path": str(json_path),
                    "symlink_path": "",
                    "duplicate_of": str(original_source),
                    "canonical_hash": digest,
                }
            )
            continue

        seen_hashes[duplicate_key] = json_path

        relative_source = json_path.relative_to(results_root)
        symlink_path = output_dir / dataset / relative_source
        symlink_path.parent.mkdir(parents=True, exist_ok=True)

        if symlink_path.exists() or symlink_path.is_symlink():
            safe_unlink(symlink_path)

        relative_target = pathlib.Path(os.path.relpath(json_path, symlink_path.parent))
        symlink_path.symlink_to(relative_target)

        linked_counts[dataset] += 1
        manifest_rows.append(
            {
                "dataset": dataset,
                "status": "linked",
                "source_path": str(json_path),
                "symlink_path": str(symlink_path),
                "duplicate_of": "",
                "canonical_hash": digest,
            }
        )

    manifest_path = write_manifest(output_dir, manifest_rows)
    summary_path = write_summary(output_dir, linked_counts, duplicate_counts, skipped_unknown)

    logger.info("Created %d dataset symlinks in %s", sum(linked_counts.values()), output_dir)
    for dataset in sorted(linked_counts):
        logger.info(
            "  %s: %d linked, %d duplicates skipped",
            dataset,
            linked_counts[dataset],
            duplicate_counts[dataset],
        )
    logger.info("  unknown_dataset: %d skipped", skipped_unknown)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
