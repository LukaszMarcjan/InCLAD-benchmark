"""Build the multidataset_seq(_v1) CSV benchmark from locally available source datasets.

The paper's InCLAD-MD rows were run on this CSV benchmark under the name
``multidataset_seq_v1`` with the curricula in ``multidataset_ordering/``.
The released equivalent is ``InCLADMDDataset`` (``--benchmark inclad-md``),
which yields the same 27-concept easy_to_hard / hard_to_easy orders; this
script is only needed to rebuild the original CSV benchmark.

Example (run from the repository root):

    PYTHONPATH=src python examples/visual_models/neurips_utilities/build_multidataset_seq.py \\
        --benchmark-name multidataset_seq_v1 \\
        --registry-path /path/to/registry.json \\
        --output-root /path/to/visual_datasets/multidataset_seq_v1

Source dataset roots are resolved like every other visual benchmark
(``--source-root benchmark=/path``, registry JSON, PYCLAD_* environment variables).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from pyclad.data.readers.visual_benchmark_reader import CsvBenchmarkSpec, available_visual_benchmarks
from pyclad.data.readers.visual_dataset_ordering import load_concept_order_from_file
from pyclad.data.readers.visual_dataset_registry import (
    index_registered_visual_benchmark,
    read_registered_visual_benchmark_dataset,
    resolve_visual_benchmark_root,
)

SCRIPT_DIR = Path(__file__).resolve().parent
# The script lives in examples/visual_models/neurips_utilities/, so the
# repository root is three levels up (parents[2]); parents[1] is examples/.
# PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[2]
# DEFAULT_ORDERING_DIR = SCRIPT_DIR / "multidataset_ordering"
DEFAULT_ORDERING_DIR = SCRIPT_DIR / "multidataset_ordering"
DEFAULT_SELECTED_CONCEPTS_FILE = DEFAULT_ORDERING_DIR / "multidataset_seq_selected_concepts.csv"
DEFAULT_CATEGORY_ORDER_FILE = DEFAULT_ORDERING_DIR / "multidataset_seq_easy_to_hard_blocks_roc-auc.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "src" / "pyclad" / "data" / "datasets" / "visual_datasets" / "multidataset_seq"
DEFAULT_OUTPUT_FILENAME = "multidataset_seq.csv"
DEFAULT_BENCHMARK_NAME = "multidataset_seq"
# Name recorded by the paper's InCLAD-MD runs (cli_args.benchmark).
PAPER_BENCHMARK_NAME = "multidataset_seq_v1"
BENCHMARK_NAME_CHOICES = (DEFAULT_BENCHMARK_NAME, PAPER_BENCHMARK_NAME)
DEFAULT_FIELDNAMES = (
    "sample_id",
    "source_dataset",
    "source_category",
    "category",
    "category_order",
    "split",
    "image",
    "label",
    "image_label",
    "mask",
    "defect_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the multidataset_seq synthetic CSV benchmark from the registered visual datasets."
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default=DEFAULT_BENCHMARK_NAME,
        choices=list(BENCHMARK_NAME_CHOICES),
        help="Synthetic benchmark name written into the loader-facing CSV "
        f"('{PAPER_BENCHMARK_NAME}' is the name used by the paper's runs).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory that will contain the generated CSV benchmark.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help="Name of the generated split CSV inside output-root.",
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        default=None,
        help="Optional visual dataset registry JSON path used to resolve source dataset roots.",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=None,
        help="Optional benchmark=/path override for a source dataset root. Repeatable.",
    )
    parser.add_argument(
        "--selected-concepts-file",
        type=str,
        default=str(DEFAULT_SELECTED_CONCEPTS_FILE),
        help="CSV that lists the selected source concepts and their synthetic benchmark names.",
    )
    parser.add_argument(
        "--category-order-file",
        type=str,
        default=str(DEFAULT_CATEGORY_ORDER_FILE),
        help="Ordering CSV whose concept column becomes the benchmark default category_order.",
    )
    parser.add_argument(
        "--ordering-column",
        type=str,
        default="concept",
        help="Column name used to read concept names from category-order-file.",
    )
    parser.add_argument(
        "--path-mode",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
        help="Write image and mask paths relative to output-root or as absolute paths.",
    )
    parser.add_argument(
        "--verify-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the generated benchmark back through the standard loader and print a short summary.",
    )
    return parser.parse_args()


def parse_mapping_args(items: list[str] | None) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    if items is None:
        return resolved

    for item in items:
        key, sep, value = str(item).partition("=")
        if not sep:
            raise ValueError(
                f"Expected benchmark=/path for --source-root, got {item!r}. "
                f"Available benchmarks: {', '.join(available_visual_benchmarks())}"
            )
        benchmark = key.strip().lower()
        if not benchmark:
            raise ValueError(f"Missing benchmark name in --source-root={item!r}")
        if benchmark in resolved:
            raise ValueError(f"Duplicate --source-root override for benchmark '{benchmark}'")
        resolved[benchmark] = Path(value.strip()).expanduser().resolve()
    return resolved


def load_selected_concepts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Selected concepts CSV has no header: {path}")
        rows = list(reader)

    selected: list[dict[str, str]] = []
    seen_categories: set[str] = set()
    for row in rows:
        source_dataset = row.get("dataset", "").strip().lower()
        source_category = row.get("source_concept", "").strip()
        benchmark_category = row.get("benchmark_concept", "").strip() or row.get("concept", "").strip()
        if not source_dataset or not source_category or not benchmark_category:
            raise ValueError(
                f"Each selected concept row must define dataset, source_concept, and benchmark_concept/concept. "
                f"Problematic row: {row}"
            )
        if benchmark_category in seen_categories:
            raise ValueError(f"Duplicate synthetic category in selected concepts CSV: {benchmark_category}")
        seen_categories.add(benchmark_category)
        selected.append(
            {
                "source_dataset": source_dataset,
                "source_category": source_category,
                "category": benchmark_category,
            }
        )
    return selected


def validate_category_order(category_order: list[str], selected_categories: list[str]) -> None:
    duplicates = find_duplicates(category_order)
    if duplicates:
        raise ValueError(f"Category order contains duplicates: {duplicates}")

    missing = [category for category in selected_categories if category not in category_order]
    if missing:
        raise ValueError(f"Category order is missing selected categories: {missing}")

    extra = [category for category in category_order if category not in selected_categories]
    if extra:
        raise ValueError(f"Category order contains categories outside the selected concept list: {extra}")


def find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def serialize_path(path: Path | None, output_root: Path, path_mode: str) -> str:
    if path is None:
        return ""

    resolved_path = Path(path).expanduser().resolve()
    if path_mode == "absolute":
        return str(resolved_path)
    return os.path.relpath(resolved_path, start=output_root)


def build_rows(
    selected_concepts: list[dict[str, str]],
    category_order: list[str],
    output_root: Path,
    path_mode: str,
    registry_path: str | None,
    source_roots: dict[str, Path],
) -> list[dict[str, str]]:
    category_positions = {category: index + 1 for index, category in enumerate(category_order)}
    rows: list[dict[str, str]] = []
    sample_index = 0

    for selected in selected_concepts:
        source_dataset = selected["source_dataset"]
        source_category = selected["source_category"]
        benchmark_category = selected["category"]
        dataset_root = source_roots.get(source_dataset)
        if dataset_root is None:
            dataset_root = resolve_visual_benchmark_root(
                benchmark=source_dataset,
                registry_path=registry_path,
            )

        samples = index_registered_visual_benchmark(
            benchmark=source_dataset,
            root=dataset_root,
            registry_path=registry_path,
            categories=[source_category],
        )
        if not samples:
            raise ValueError(
                f"No samples were indexed for {source_dataset}:{source_category}. "
                f"Resolved source root: {dataset_root}"
            )

        for sample in samples:
            label = "normal" if int(sample.image_label) == 0 else (sample.defect_type or "anomaly")
            rows.append(
                {
                    "sample_id": f"{benchmark_category}:{sample_index:06d}",
                    "source_dataset": source_dataset,
                    "source_category": source_category,
                    "category": benchmark_category,
                    "category_order": str(category_positions[benchmark_category]),
                    "split": sample.split,
                    "image": serialize_path(sample.image_path, output_root=output_root, path_mode=path_mode),
                    "label": label,
                    "image_label": str(int(sample.image_label)),
                    "mask": serialize_path(sample.mask_path, output_root=output_root, path_mode=path_mode),
                    "defect_type": sample.defect_type or "",
                }
            )
            sample_index += 1

    rows.sort(
        key=lambda row: (
            int(row["category_order"]),
            0 if row["split"] == "train" else 1,
            row["image"],
        )
    )
    return rows


def write_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def verify_generated_benchmark(benchmark_name: str, output_root: Path, output_filename: str = DEFAULT_OUTPUT_FILENAME) -> None:
    # The benchmark name alone is not a reader preset, so loading by name raises
    # "Unsupported visual benchmark"; load through an explicit CsvBenchmarkSpec.
    # dataset = read_registered_visual_benchmark_dataset(
    #     benchmark=benchmark_name,
    #     root=output_root,
    #     data_mode="paths",
    # )
    spec = CsvBenchmarkSpec(
        name=benchmark_name,
        csv_path=output_filename,
        category_column="category",
        category_order_column="category_order",
        split_column="split",
        train_split_value="train",
        test_split_value="test",
        image_column="image",
        label_column="label",
        normal_label_value="normal",
        mask_column="mask",
        defect_type_column="defect_type",
    )
    dataset = read_registered_visual_benchmark_dataset(
        benchmark=spec,
        root=output_root,
        data_mode="paths",
    )
    order = [concept.name for concept in dataset.train_concepts()]
    train_count = sum(len(concept.data) for concept in dataset.train_concepts())
    test_count = sum(len(concept.data) for concept in dataset.test_concepts())
    print(
        f"verified benchmark={benchmark_name} concepts={len(order)} train_samples={train_count} "
        f"test_samples={test_count}"
    )
    print(f"order={order}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_path = output_root / args.output_filename
    selected_concepts_path = Path(args.selected_concepts_file).expanduser().resolve()
    category_order_path = Path(args.category_order_file).expanduser().resolve()
    source_roots = parse_mapping_args(args.source_root)

    selected_concepts = load_selected_concepts(selected_concepts_path)
    selected_categories = [row["category"] for row in selected_concepts]
    category_order = load_concept_order_from_file(
        ordering_path=category_order_path,
        benchmark=args.benchmark_name,
        ordering_column=args.ordering_column,
    )
    validate_category_order(category_order, selected_categories)

    rows = build_rows(
        selected_concepts=selected_concepts,
        category_order=category_order,
        output_root=output_root,
        path_mode=args.path_mode,
        registry_path=args.registry_path,
        source_roots=source_roots,
    )
    write_rows(output_path=output_path, rows=rows)

    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"selected_concepts={len(selected_categories)}")

    if args.verify_load:
        verify_generated_benchmark(
            benchmark_name=args.benchmark_name,
            output_root=output_root,
            output_filename=args.output_filename,
        )


if __name__ == "__main__":
    main()
