"""
Build an updated, dataset-grouped view of result JSON files with readable names.

The script scans multiple result roots, deduplicates mirrored runs by canonical
JSON content, and creates renamed symbolic links under:

    neurips_results/<dataset>/<renamed-result>.json

The renamed file indicates:
- model
- strategy
- backbone / setting
- ordering
- evaluation level
- source bucket

It also writes a manifest and a summary CSV for auditing.
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
import sys
from collections import Counter, defaultdict
from typing import Any

# parents[2] is examples/; the repository root (which holds the examples
# package) is parents[3].
# REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.visual_models.neurips_utilities.audit_neurips_experiment_grid import (  # noqa: E402
    canonical_hash,
    full_categories_selected,
    full_train_test,
    has_image_metrics,
    has_pixel_metrics,
    infer_dataset,
    infer_model,
    infer_ordering,
    infer_setting,
    infer_strategy,
    load_expected_orders,
    modality_is_full,
    slot_is_expected,
    SlotKey,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_ROOTS = (
    "r_res0505",
    "results_20260505",
    "neurips_results",
)
SKIP_DIR_NAMES = {
    "analysis",
    "by_dataset_links",
    "review_grouped_copies",
    "neurips_results",
    "__pycache__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the updated neurips_results view with readable names.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-roots",
        type=pathlib.Path,
        nargs="+",
        default=[pathlib.Path(root) for root in DEFAULT_INPUT_ROOTS],
        help="Result roots to scan, in priority order",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("../../../ordering_files/neurips_results"),
        help="Where the renamed dataset-grouped view should be created",
    )
    parser.add_argument(
        "--ordering-dir",
        type=pathlib.Path,
        default=pathlib.Path("examples/visual_models/ordering"),
        help="Directory with ordering CSV files used to infer legacy orderings",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Whether to create symbolic links or physical copies",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and rebuild OUTPUT_DIR if it already exists",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def sanitize_fragment(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "unknown"


def normalize_ordering_label(ordering: str) -> str:
    mapping = {
        "easy_to_hard": "easy-to-hard",
        "hard_to_easy": "hard-to-easy",
        "random": "random",
        "unknown": "unknown",
    }
    return mapping.get(ordering, ordering.replace("_", "-"))


def resolve_display_ordering(
    data: dict[str, Any],
    source_path: pathlib.Path,
    dataset: str,
    expected_orders: dict[str, dict[str, list[str]]],
) -> tuple[str, str]:
    rc = data.get("run_config") or {}
    cli_args = rc.get("cli_args") or {}
    ordering_metadata = rc.get("ordering_metadata") or {}

    seed = rc.get("seed")
    if seed is None:
        seed = data.get("seed")
    if seed is None:
        seed = cli_args.get("seed")
    seed_text = "" if seed is None else str(seed)

    ordering_file = (
        ordering_metadata.get("ordering_file")
        or cli_args.get("ordering_file")
        or rc.get("ordering_file")
    )
    ordering_file_name = pathlib.Path(str(ordering_file)).name.lower() if ordering_file else ""

    # User-verified naming rule for the main NeurIPS datasets:
    # - seed=191664963 + explicit ordering file => ordered
    # - seed=33158374 => random
    if dataset in {"btech", "dagm", "mpdd", "mvtec", "visa"}:
        if seed_text == "33158374":
            return "random", "btech_seed_rule"
        if seed_text == "191664963" and ordering_file_name == f"{dataset}_easy_to_hard_roc-auc.csv":
            return "easy-to-hard", "btech_seed_rule"
        if seed_text == "191664963" and ordering_file_name == f"{dataset}_hard_to_easy_roc-auc.csv":
            return "hard-to-easy", "btech_seed_rule"

    ordering_key, ordering_source = infer_ordering(data, source_path, dataset, expected_orders)
    return normalize_ordering_label(ordering_key), ordering_source


def detect_eval_level(data: dict[str, Any]) -> str:
    if has_image_metrics(data) and has_pixel_metrics(data):
        return "both"
    if has_image_metrics(data):
        return "image"
    if has_pixel_metrics(data):
        return "pixel"
    return "unknown"


def detect_completeness(
    data: dict[str, Any],
    dataset: str,
    model: str,
    setting: str,
    strategy: str,
    ordering: str,
    expected_orders: dict[str, dict[str, list[str]]],
) -> str:
    slot = SlotKey(dataset=dataset, model=model, setting=setting, strategy=strategy, ordering=ordering)
    if not slot_is_expected(slot):
        return "out_of_scope"

    if not full_categories_selected(data, dataset, expected_orders):
        return "partial_categories"
    if not full_train_test(data):
        return "subsampled"

    image_present = has_image_metrics(data)
    pixel_present = has_pixel_metrics(data)
    image_full = image_present and modality_is_full(data, dataset, expected_orders, "image")
    pixel_full = pixel_present and modality_is_full(data, dataset, expected_orders, "pixel")

    if image_full and pixel_full:
        return "full"
    if image_full and not pixel_present:
        return "image_only"
    if pixel_full and not image_present:
        return "pixel_only"
    if image_present and pixel_present:
        return "incomplete_both"
    if image_present:
        return "incomplete_image"
    if pixel_present:
        return "incomplete_pixel"
    return "no_metrics"


def detect_run_state(data: dict[str, Any]) -> str:
    return sanitize_fragment(str((data.get("run_status") or {}).get("state") or "unknown").lower())


def iter_json_paths(root: pathlib.Path, output_dir: pathlib.Path) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for json_path in sorted(root.rglob("*.json")):
        if output_dir in json_path.parents:
            continue
        if any(part in SKIP_DIR_NAMES for part in json_path.parts):
            continue
        paths.append(json_path)
    return paths


def source_labels(source_path: pathlib.Path, scan_root: pathlib.Path) -> tuple[str, str]:
    relative = source_path.relative_to(scan_root)
    parts = relative.parts

    if scan_root.name == "neurips_results":
        source_group = parts[0] if len(parts) >= 1 else scan_root.name
        run_folder = parts[1] if len(parts) >= 2 else "root"
    else:
        source_group = scan_root.name
        run_folder = parts[0] if len(parts) >= 1 else "root"

    return sanitize_fragment(source_group), sanitize_fragment(run_folder)


def build_link_name(
    *,
    model: str,
    strategy: str,
    setting: str,
    ordering: str,
    eval_level: str,
    completeness: str,
    run_state: str,
    source_group: str,
    run_folder: str,
    canonical_hash_prefix: str,
) -> str:
    parts = [
        f"model-{sanitize_fragment(model)}",
        f"strategy-{sanitize_fragment(strategy)}",
        f"setting-{sanitize_fragment(setting)}",
        f"ordering-{sanitize_fragment(ordering)}",
        f"eval-{sanitize_fragment(eval_level)}",
        f"completeness-{sanitize_fragment(completeness)}",
        f"state-{sanitize_fragment(run_state)}",
        f"source-{source_group}",
        f"run-{run_folder}",
        canonical_hash_prefix,
    ]
    return "__".join(parts) + ".json"


def link_or_copy(source_path: pathlib.Path, dest_path: pathlib.Path, mode: str) -> None:
    if dest_path.exists() or dest_path.is_symlink():
        dest_path.unlink()

    if mode == "copy":
        shutil.copy2(source_path, dest_path)
        return

    relative_target = pathlib.Path(os.path.relpath(source_path, dest_path.parent))
    dest_path.symlink_to(relative_target)


def write_manifest(output_dir: pathlib.Path, rows: list[dict[str, str]]) -> pathlib.Path:
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "dataset",
        "model",
        "strategy",
        "setting",
        "ordering",
        "eval_level",
        "completeness",
        "run_state",
        "source_root",
        "source_group",
        "run_folder",
        "source_path",
        "target_path",
        "canonical_hash",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_summary(output_dir: pathlib.Path, rows: list[dict[str, str]]) -> pathlib.Path:
    summary_path = output_dir / "summary.csv"
    per_dataset: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        dataset = row["dataset"]
        per_dataset[dataset]["files"] += 1
        per_dataset[dataset][f"eval::{row['eval_level']}"] += 1
        per_dataset[dataset][f"completeness::{row['completeness']}"] += 1

    fieldnames = [
        "dataset",
        "files",
        "eval_both",
        "eval_image",
        "eval_pixel",
        "eval_unknown",
        "full",
        "image_only",
        "pixel_only",
        "incomplete_both",
        "incomplete_image",
        "incomplete_pixel",
        "partial_categories",
        "subsampled",
        "out_of_scope",
        "no_metrics",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in sorted(per_dataset):
            counts = per_dataset[dataset]
            writer.writerow(
                {
                    "dataset": dataset,
                    "files": counts["files"],
                    "eval_both": counts["eval::both"],
                    "eval_image": counts["eval::image"],
                    "eval_pixel": counts["eval::pixel"],
                    "eval_unknown": counts["eval::unknown"],
                    "full": counts["completeness::full"],
                    "image_only": counts["completeness::image_only"],
                    "pixel_only": counts["completeness::pixel_only"],
                    "incomplete_both": counts["completeness::incomplete_both"],
                    "incomplete_image": counts["completeness::incomplete_image"],
                    "incomplete_pixel": counts["completeness::incomplete_pixel"],
                    "partial_categories": counts["completeness::partial_categories"],
                    "subsampled": counts["completeness::subsampled"],
                    "out_of_scope": counts["completeness::out_of_scope"],
                    "no_metrics": counts["completeness::no_metrics"],
                }
            )

    return summary_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    input_roots = [root.expanduser().resolve() for root in args.input_roots]
    output_dir = args.output_dir.expanduser().resolve()
    ordering_dir = args.ordering_dir.expanduser().resolve()
    expected_orders = load_expected_orders(ordering_dir)

    for root in input_roots:
        if not root.exists():
            raise FileNotFoundError(f"Input root does not exist: {root}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to rebuild it.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    seen_hashes: dict[str, pathlib.Path] = {}
    dataset_counts: Counter[str] = Counter()
    duplicate_count = 0

    for scan_root in input_roots:
        json_paths = iter_json_paths(scan_root, output_dir)
        logger.info("Scanning %d JSON files under %s", len(json_paths), scan_root)

        for source_path in json_paths:
            try:
                with source_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping unreadable JSON %s: %s", source_path, exc)
                continue

            digest = canonical_hash(data)
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes[digest] = source_path

            dataset = infer_dataset(data, source_path)
            model = infer_model(data, source_path)
            strategy = infer_strategy(data, source_path)
            setting = infer_setting(data, model)
            ordering_key, _ = infer_ordering(data, source_path, dataset, expected_orders)
            ordering, _ = resolve_display_ordering(data, source_path, dataset, expected_orders)
            eval_level = detect_eval_level(data)
            completeness = detect_completeness(
                data,
                dataset,
                model,
                setting,
                strategy,
                ordering_key,
                expected_orders,
            )
            run_state = detect_run_state(data)

            source_group, run_folder = source_labels(source_path, scan_root)
            link_name = build_link_name(
                model=model,
                strategy=strategy,
                setting=setting,
                ordering=ordering,
                eval_level=eval_level,
                completeness=completeness,
                run_state=run_state,
                source_group=source_group,
                run_folder=run_folder,
                canonical_hash_prefix=digest[:8],
            )

            dataset_folder = sanitize_fragment(dataset if dataset else "unknown_dataset")
            dest_dir = output_dir / dataset_folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / link_name
            link_or_copy(source_path, dest_path, args.mode)

            manifest_rows.append(
                {
                    "dataset": dataset_folder,
                    "model": model,
                    "strategy": strategy,
                    "setting": setting,
                    "ordering": ordering,
                    "eval_level": eval_level,
                    "completeness": completeness,
                    "run_state": run_state,
                    "source_root": scan_root.name,
                    "source_group": source_group,
                    "run_folder": run_folder,
                    "source_path": str(source_path),
                    "target_path": str(dest_path),
                    "canonical_hash": digest,
                }
            )
            dataset_counts[dataset_folder] += 1

    manifest_path = write_manifest(output_dir, manifest_rows)
    summary_path = write_summary(output_dir, manifest_rows)

    logger.info("Created %d renamed %s entries in %s", len(manifest_rows), args.mode, output_dir)
    logger.info("Skipped %d mirrored duplicates", duplicate_count)
    for dataset, count in sorted(dataset_counts.items()):
        logger.info("  %s: %d", dataset, count)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
