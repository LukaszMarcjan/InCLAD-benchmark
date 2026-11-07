"""
Independently audit the NeurIPS visual AD experiment grid from raw JSON results.

The script scans result JSON files directly under ``neurips_results``, deduplicates
mirrored runs, infers the logical experiment slot

    dataset x model x setting x strategy x ordering

and then checks which expected slots are complete.

Coverage is computed from the raw metric payload, not just the filename:
- ordering can come from ``run_config.ordering_metadata.ordering_file``
- ordered runs are cross-checked against local ``ordering`` CSVs
- split ``image`` / ``pixel`` runs are merged into one logical slot
- a slot counts as complete only if it has full image and full pixel results
- full runs must use all concepts and full train/test data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DATASETS = ("btech", "dagm", "mpdd", "mvtec", "visa")
ORDERINGS = ("easy_to_hard", "hard_to_easy", "random")

MODEL_CONFIG: dict[str, dict[str, list[str]]] = {
    "cfa": {"settings": ["resnet18", "wide_resnet50_2"], "strategies": ["naive", "cl"]},
    "efficientad": {"settings": ["small"], "strategies": ["naive", "replay"]},
    "fastflow": {"settings": ["resnet18", "wide_resnet50_2"], "strategies": ["naive", "replay"]},
    "ganomaly": {"settings": ["default"], "strategies": ["naive", "replay"]},
    "padim": {"settings": ["resnet18", "wide_resnet50_2"], "strategies": ["naive", "cl"]},
    "paste": {"settings": ["resnet18", "wide_resnet50_2"], "strategies": ["naive", "replay"]},
    "patchcore": {"settings": ["resnet18", "wide_resnet50_2"], "strategies": ["naive", "cl"]},
    "rd4ad": {"settings": ["resnet18", "wide_resnet50_2"], "strategies": ["naive", "replay"]},
    "stfpm": {"settings": ["resnet18", "wide_resnet50_2"], "strategies": ["naive", "replay"]},
}

NAME_TO_MODEL = {
    "cfa": "cfa",
    "efficientad": "efficientad",
    "fastflow": "fastflow",
    "ganomaly": "ganomaly",
    "padim": "padim",
    "patchcore": "patchcore",
    "paste": "paste",
    "rd4ad": "rd4ad",
    "stfpm": "stfpm",
}

SKIP_DIR_NAMES = {"analysis", "by_dataset_links", "review_grouped_copies"}


@dataclass(frozen=True)
class SlotKey:
    dataset: str
    model: str
    setting: str
    strategy: str
    ordering: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit logical experiment coverage from raw NeurIPS result JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_root",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("neurips_results"),
        help="Root directory containing raw result JSON files",
    )
    parser.add_argument(
        "--ordering-dir",
        type=pathlib.Path,
        default=pathlib.Path("examples/visual_models/ordering"),
        help="Directory with easy/hard ordering CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("neurips_results/analysis"),
        help="Where audit CSVs should be written",
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


def normalize_sequence(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def load_expected_orders(ordering_dir: pathlib.Path) -> dict[str, dict[str, list[str]]]:
    expected: dict[str, dict[str, list[str]]] = {}
    for dataset in DATASETS:
        expected[dataset] = {}
        for ordering in ("easy_to_hard", "hard_to_easy"):
            csv_path = ordering_dir / f"{dataset}_{ordering}_roc-auc.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            expected[dataset][ordering] = [row["concept"].strip() for row in rows if row.get("concept")]
    return expected


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


def infer_dataset(data: dict[str, Any], json_path: pathlib.Path) -> str:
    dataset_name = normalize_text((data.get("dataset") or {}).get("name"))
    rc = data.get("run_config") or {}
    benchmark = normalize_text(rc.get("benchmark") or ((rc.get("cli_args") or {}).get("benchmark")))
    path_text = normalize_text(json_path.as_posix())

    for dataset in DATASETS:
        if dataset in dataset_name or benchmark == dataset:
            return dataset
        if f"/{dataset}/" in path_text or f"order-{dataset}-" in path_text:
            return dataset

    return "unknown"


def infer_model(data: dict[str, Any], json_path: pathlib.Path) -> str:
    path_text = normalize_text(json_path.as_posix())
    for fragment, model in NAME_TO_MODEL.items():
        if f"/{fragment}/" in path_text or json_path.name.lower().startswith(f"{fragment}_"):
            return model

    model_name = normalize_text((data.get("model") or {}).get("name"))
    for fragment, model in NAME_TO_MODEL.items():
        if fragment == model_name or fragment in model_name:
            return model

    return "unknown"


def infer_strategy(data: dict[str, Any], json_path: pathlib.Path) -> str:
    path_text = normalize_text(json_path.as_posix())
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

    rc = data.get("run_config") or {}
    config_strategy = normalize_text(rc.get("strategy") or ((rc.get("cli_args") or {}).get("strategy")))
    if config_strategy in {"naive", "replay", "cl"}:
        return config_strategy

    return "unknown"


def normalize_backbone(value: Any) -> str:
    text = normalize_text(value)
    if "wide" in text and "50" in text:
        return "wide_resnet50_2"
    if "resnet18" in text:
        return "resnet18"
    return text


def infer_setting(data: dict[str, Any], model: str) -> str:
    rc = data.get("run_config") or {}
    cli_args = rc.get("cli_args") or {}
    model_payload = data.get("model") or {}
    model_config = rc.get("model_config") or {}

    if model == "efficientad":
        for candidate in (
            model_payload.get("model_size"),
            cli_args.get("model_size"),
            model_config.get("model_size"),
        ):
            text = normalize_text(candidate)
            if text:
                return text
        return "small"

    if model == "ganomaly":
        return "default"

    for candidate in (
        model_payload.get("backbone"),
        model_payload.get("backbone_name"),
        rc.get("backbone"),
        rc.get("backbone_name"),
        cli_args.get("backbone"),
        model_config.get("backbone"),
        model_config.get("backbone_name"),
    ):
        backbone = normalize_backbone(candidate)
        if backbone in {"resnet18", "wide_resnet50_2"}:
            return backbone

    return "unknown"


def extract_image_order(data: dict[str, Any]) -> list[str]:
    callback = data.get("concept_metric_callback_ROC-AUC") or {}
    if isinstance(callback, dict):
        seq = normalize_sequence(callback.get("concepts_order"))
        if seq:
            return seq
    callback = data.get("concept_metric_callback_F1-Score") or {}
    if isinstance(callback, dict):
        seq = normalize_sequence(callback.get("concepts_order"))
        if seq:
            return seq
    return []


def extract_pixel_order(data: dict[str, Any]) -> list[str]:
    callback = data.get("pixel_concept_metric_callback_Pixel-ROC-AUC") or {}
    if isinstance(callback, dict):
        seq = normalize_sequence(callback.get("concepts_order"))
        if seq:
            return seq
    callback = data.get("pixel_concept_metric_callback_Pixel-F1-Score") or {}
    if isinstance(callback, dict):
        seq = normalize_sequence(callback.get("concepts_order"))
        if seq:
            return seq
    return []


def extract_run_order(data: dict[str, Any]) -> list[str]:
    rc = data.get("run_config") or {}
    seq = normalize_sequence(rc.get("concept_order"))
    if seq:
        return seq
    ordering_metadata = rc.get("ordering_metadata") or {}
    seq = normalize_sequence(ordering_metadata.get("dataset_concepts"))
    return seq


def infer_ordering(
    data: dict[str, Any],
    json_path: pathlib.Path,
    dataset: str,
    expected_orders: dict[str, dict[str, list[str]]],
) -> tuple[str, str]:
    rc = data.get("run_config") or {}
    cli_args = rc.get("cli_args") or {}
    ordering_metadata = rc.get("ordering_metadata") or {}

    def from_name(text: str) -> str:
        lowered = text.lower()
        if "easy_to_hard" in lowered or "easy-to-hard" in lowered:
            return "easy_to_hard"
        if "hard_to_easy" in lowered or "hard-to-easy" in lowered:
            return "hard_to_easy"
        return ""

    for candidate, source in (
        (ordering_metadata.get("ordering_file"), "ordering_file"),
        (ordering_metadata.get("ordering_mode"), "ordering_mode"),
        (cli_args.get("ordering_file"), "cli_ordering_file"),
        (cli_args.get("ordering_mode"), "cli_ordering_mode"),
        (rc.get("ordering_file"), "run_config_ordering_file"),
    ):
        text = normalize_text(candidate)
        match = from_name(text)
        if match:
            return match, source

    expected = expected_orders.get(dataset, {})
    easy = expected.get("easy_to_hard", [])
    hard = expected.get("hard_to_easy", [])
    for sequence, source in (
        (extract_run_order(data), "run_config.concept_order"),
        (extract_image_order(data), "image.concepts_order"),
        (extract_pixel_order(data), "pixel.concepts_order"),
    ):
        if sequence and sequence == easy:
            return "easy_to_hard", source
        if sequence and sequence == hard:
            return "hard_to_easy", source

    file_name = json_path.name.lower()
    if re.search(r"_order-(?:[^_]+-)?easy-to-hard(?:-roc-auc)?(?:_buf-frac[^_]+)?_seed_", file_name):
        return "easy_to_hard", "filename"
    if re.search(r"_order-(?:[^_]+-)?hard-to-easy(?:-roc-auc)?(?:_buf-frac[^_]+)?_seed_", file_name):
        return "hard_to_easy", "filename"
    if "_order-shuffle_" in file_name or "_order-random_" in file_name:
        return "random", "filename"

    shuffle_concepts = bool(cli_args.get("shuffle_concepts") or rc.get("shuffle_concepts"))
    if shuffle_concepts:
        return "random", "shuffle_concepts"

    for sequence, source in (
        (extract_run_order(data), "run_config.concept_order"),
        (extract_image_order(data), "image.concepts_order"),
        (extract_pixel_order(data), "pixel.concepts_order"),
    ):
        if sequence and sequence != easy and sequence != hard:
            return "random", source

    if normalize_text(ordering_metadata.get("ordering_file")) in {"", "none"} and "_order-" not in file_name:
        return "random", "fallback_no_order"

    if "_order-" not in file_name and not normalize_text(ordering_metadata.get("ordering_file")):
        return "random", "fallback_no_order"

    return "unknown", "unresolved"


def expected_concept_count(dataset: str, expected_orders: dict[str, dict[str, list[str]]]) -> int:
    return len(expected_orders[dataset]["easy_to_hard"])


def full_categories_selected(data: dict[str, Any], dataset: str, expected_orders: dict[str, dict[str, list[str]]]) -> bool:
    rc = data.get("run_config") or {}
    cli_args = rc.get("cli_args") or {}
    categories = rc.get("categories")
    if categories in (None, [], ""):
        categories = cli_args.get("categories")
    if categories in (None, [], ""):
        return True
    selected = set(normalize_sequence(categories))
    full = set(expected_orders[dataset]["easy_to_hard"])
    return selected == full


def full_train_test(data: dict[str, Any]) -> bool:
    rc = data.get("run_config") or {}
    cli_args = rc.get("cli_args") or {}
    for key in ("max_train", "max_test"):
        for container in (rc, cli_args):
            value = container.get(key)
            if value not in (None, "", 0):
                return False
    return True


def has_image_metrics(data: dict[str, Any]) -> bool:
    return isinstance(data.get("concept_metric_callback_ROC-AUC"), dict)


def has_pixel_metrics(data: dict[str, Any]) -> bool:
    return isinstance(data.get("pixel_concept_metric_callback_Pixel-ROC-AUC"), dict)


def modality_is_full(data: dict[str, Any], dataset: str, expected_orders: dict[str, dict[str, list[str]]], modality: str) -> bool:
    expected_count = expected_concept_count(dataset, expected_orders)
    if modality == "image":
        sequence = extract_image_order(data) or extract_run_order(data)
    else:
        sequence = extract_pixel_order(data) or extract_run_order(data)
    if len(sequence) != expected_count:
        return False
    dataset_payload = data.get("dataset") or {}
    train_count = dataset_payload.get("tran_concepts_no")
    test_count = dataset_payload.get("test_concepts_no")
    if train_count not in (None, expected_count):
        return False
    if test_count not in (None, expected_count):
        return False
    return True


def slot_is_expected(slot: SlotKey) -> bool:
    model_config = MODEL_CONFIG.get(slot.model)
    if model_config is None:
        return False
    return (
        slot.dataset in DATASETS
        and slot.setting in model_config["settings"]
        and slot.strategy in model_config["strategies"]
        and slot.ordering in ORDERINGS
    )


def iter_json_paths(results_root: pathlib.Path) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for json_path in sorted(results_root.rglob("*.json")):
        if any(part in SKIP_DIR_NAMES for part in json_path.parts):
            continue
        paths.append(json_path)
    return paths


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    results_root = args.results_root.expanduser().resolve()
    ordering_dir = args.ordering_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_orders = load_expected_orders(ordering_dir)
    raw_paths = iter_json_paths(results_root)
    logger.info("Scanning %d raw JSON files under %s", len(raw_paths), results_root)

    seen_hashes: dict[str, pathlib.Path] = {}
    audit_rows: list[dict[str, Any]] = []
    slot_records: dict[SlotKey, dict[str, Any]] = {}
    duplicates = 0

    for json_path in raw_paths:
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable JSON %s: %s", json_path, exc)
            continue

        digest = canonical_hash(data)
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes[digest] = json_path

        dataset = infer_dataset(data, json_path)
        model = infer_model(data, json_path)
        strategy = infer_strategy(data, json_path)
        setting = infer_setting(data, model)
        ordering, ordering_source = infer_ordering(data, json_path, dataset, expected_orders)

        slot = SlotKey(dataset=dataset, model=model, setting=setting, strategy=strategy, ordering=ordering)
        expected_slot = slot_is_expected(slot)

        image_present = has_image_metrics(data)
        pixel_present = has_pixel_metrics(data)
        image_full = expected_slot and image_present and modality_is_full(data, dataset, expected_orders, "image")
        pixel_full = expected_slot and pixel_present and modality_is_full(data, dataset, expected_orders, "pixel")
        category_full = expected_slot and full_categories_selected(data, dataset, expected_orders)
        train_test_full = full_train_test(data)

        reason_parts: list[str] = []
        if dataset == "unknown":
            reason_parts.append("unknown_dataset")
        if model == "unknown":
            reason_parts.append("unknown_model")
        if strategy == "unknown":
            reason_parts.append("unknown_strategy")
        if setting == "unknown":
            reason_parts.append("unknown_setting")
        if ordering == "unknown":
            reason_parts.append("unknown_ordering")
        if not category_full and expected_slot:
            reason_parts.append("partial_categories")
        if not train_test_full:
            reason_parts.append("subsampled_train_or_test")
        if expected_slot and image_present and not image_full:
            reason_parts.append("incomplete_image")
        if expected_slot and pixel_present and not pixel_full:
            reason_parts.append("incomplete_pixel")
        if not expected_slot and dataset != "unknown" and model in MODEL_CONFIG:
            reason_parts.append("out_of_scope_setting_or_strategy")

        audit_rows.append(
            {
                "source_path": str(json_path),
                "canonical_hash": digest,
                "dataset": dataset,
                "model": model,
                "setting": setting,
                "strategy": strategy,
                "ordering": ordering,
                "ordering_source": ordering_source,
                "image_present": int(image_present),
                "pixel_present": int(pixel_present),
                "image_full": int(image_full),
                "pixel_full": int(pixel_full),
                "full_categories": int(category_full),
                "full_train_test": int(train_test_full),
                "expected_slot": int(expected_slot),
                "reasons": ";".join(reason_parts),
            }
        )

        if not expected_slot or not category_full or not train_test_full:
            continue

        record = slot_records.setdefault(
            slot,
            {
                "dataset": dataset,
                "model": model,
                "setting": setting,
                "strategy": strategy,
                "ordering": ordering,
                "image_files": 0,
                "pixel_files": 0,
                "example_image": "",
                "example_pixel": "",
                "sources": [],
            },
        )
        if image_full:
            record["image_files"] += 1
            record["example_image"] = record["example_image"] or str(json_path)
        if pixel_full:
            record["pixel_files"] += 1
            record["example_pixel"] = record["example_pixel"] or str(json_path)
        record["sources"].append(str(json_path))

    present_rows: list[dict[str, Any]] = []
    observed_complete: set[SlotKey] = set()
    for slot, record in sorted(
        slot_records.items(),
        key=lambda item: (item[0].dataset, item[0].model, item[0].setting, item[0].strategy, item[0].ordering),
    ):
        present_rows.append(
            {
                "benchmark": record["dataset"],
                "model": record["model"],
                "setting": record["setting"],
                "strategy": record["strategy"],
                "ordering": record["ordering"],
                "image_files": record["image_files"],
                "pixel_files": record["pixel_files"],
                "example_image": record["example_image"],
                "example_pixel": record["example_pixel"],
            }
        )
        if record["image_files"] > 0 and record["pixel_files"] > 0:
            observed_complete.add(slot)

    expected_slots = [
        SlotKey(dataset, model, setting, strategy, ordering)
        for dataset in DATASETS
        for model, config in MODEL_CONFIG.items()
        for setting in config["settings"]
        for strategy in config["strategies"]
        for ordering in ORDERINGS
    ]
    missing_rows = [
        {
            "benchmark": slot.dataset,
            "model": slot.model,
            "setting": slot.setting,
            "strategy": slot.strategy,
            "ordering": slot.ordering,
        }
        for slot in expected_slots
        if slot not in observed_complete
    ]

    summary_rows: list[dict[str, Any]] = []
    counts_by_dataset_model: Counter[tuple[str, str]] = Counter()
    for slot in observed_complete:
        counts_by_dataset_model[(slot.dataset, slot.model)] += 1
    for dataset in DATASETS:
        for model, config in MODEL_CONFIG.items():
            expected_count = len(config["settings"]) * len(config["strategies"]) * len(ORDERINGS)
            observed_count = counts_by_dataset_model[(dataset, model)]
            summary_rows.append(
                {
                    "benchmark": dataset,
                    "model": model,
                    "expected_slots": expected_count,
                    "observed_complete_slots": observed_count,
                    "missing_slots": expected_count - observed_count,
                }
            )

    ordering_summary: Counter[tuple[str, str, str]] = Counter()
    seed_summary: Counter[tuple[str, str, str, str]] = Counter()
    for row in audit_rows:
        if not int(row["expected_slot"]):
            continue
        ordering_summary[(row["ordering"], row["ordering_source"], row["dataset"])] += 1
        try:
            with pathlib.Path(row["source_path"]).open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:  # noqa: BLE001
            continue
        rc = data.get("run_config") or {}
        cli_args = rc.get("cli_args") or {}
        seed = rc.get("seed")
        master_seed = cli_args.get("master_seed")
        seed_summary[(row["ordering"], str(seed), str(master_seed), row["dataset"])] += 1

    audit_path = output_dir / "nine_models_independent_audit.csv"
    present_path = output_dir / "nine_models_independent_present_slots.csv"
    missing_path = output_dir / "nine_models_independent_missing_from_480.csv"
    summary_path = output_dir / "nine_models_independent_coverage_summary.csv"
    ordering_summary_path = output_dir / "nine_models_independent_ordering_summary.csv"
    seed_summary_path = output_dir / "nine_models_independent_seed_summary.csv"

    write_csv(
        audit_path,
        [
            "source_path",
            "canonical_hash",
            "dataset",
            "model",
            "setting",
            "strategy",
            "ordering",
            "ordering_source",
            "image_present",
            "pixel_present",
            "image_full",
            "pixel_full",
            "full_categories",
            "full_train_test",
            "expected_slot",
            "reasons",
        ],
        audit_rows,
    )
    write_csv(
        present_path,
        [
            "benchmark",
            "model",
            "setting",
            "strategy",
            "ordering",
            "image_files",
            "pixel_files",
            "example_image",
            "example_pixel",
        ],
        present_rows,
    )
    write_csv(missing_path, ["benchmark", "model", "setting", "strategy", "ordering"], missing_rows)
    write_csv(summary_path, ["benchmark", "model", "expected_slots", "observed_complete_slots", "missing_slots"], summary_rows)
    write_csv(
        ordering_summary_path,
        ["ordering", "ordering_source", "dataset", "row_count"],
        [
            {
                "ordering": ordering,
                "ordering_source": source,
                "dataset": dataset,
                "row_count": count,
            }
            for (ordering, source, dataset), count in sorted(ordering_summary.items())
        ],
    )
    write_csv(
        seed_summary_path,
        ["ordering", "seed", "master_seed", "dataset", "row_count"],
        [
            {
                "ordering": ordering,
                "seed": seed,
                "master_seed": master_seed,
                "dataset": dataset,
                "row_count": count,
            }
            for (ordering, seed, master_seed, dataset), count in sorted(seed_summary.items())
        ],
    )

    logger.info("Unique raw files audited: %d", len(audit_rows))
    logger.info("Skipped mirrored duplicates: %d", duplicates)
    logger.info("Observed complete logical slots: %d / %d", len(observed_complete), len(expected_slots))
    logger.info("Summary: %s", summary_path)
    logger.info("Missing: %s", missing_path)
    logger.info("Present: %s", present_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
