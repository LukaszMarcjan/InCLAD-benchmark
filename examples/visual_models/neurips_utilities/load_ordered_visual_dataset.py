"""
Load a registered visual benchmark dataset using the concept order from an ordering file.

Examples
--------
python3 examples/visual_models/load_ordered_visual_dataset.py \
    examples/visual_models/ordering/btech_easy_to_hard_roc-auc.csv

python3 examples/visual_models/load_ordered_visual_dataset.py \
    examples/visual_models/results/ste_curriculum_ordering/final_orderings.csv \
    --benchmark btech \
    --ordering-mode easy_to_hard
"""

from __future__ import annotations

import argparse

from pyclad.data.readers.visual_dataset_ordering import (
    infer_visual_benchmark_from_ordering_path,
    load_concept_order_from_file,
    read_ordered_registered_visual_benchmark_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a visual benchmark dataset in the concept order defined by an ordering file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ordering_path", type=str, help="Path to the ordering CSV/JSON file")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark name; if omitted, the script infers it from the ordering filename",
    )
    parser.add_argument("--root", type=str, default=None, help="Explicit dataset root directory")
    parser.add_argument("--registry-path", type=str, default=None, help="Path to visual dataset registry JSON")
    parser.add_argument("--ordering-column", type=str, default="concept", help="Column that stores concept names")
    parser.add_argument(
        "--ordering-mode",
        type=str,
        default=None,
        choices=["easy_to_hard", "hard_to_easy"],
        help="Ordering column to use for summary CSV/JSON files",
    )
    parser.add_argument("--dataset-name", type=str, default=None, help="Override dataset name")
    parser.add_argument(
        "--data-mode",
        type=str,
        default="numpy",
        choices=["numpy", "paths"],
        help="Whether to materialize images or return file paths",
    )
    parser.add_argument(
        "--color-mode",
        type=str,
        default="rgb",
        choices=["rgb", "grayscale"],
        help="Color mode used when data-mode=numpy",
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize images to a square resolution")
    parser.add_argument("--max-train", type=int, default=None, help="Max training samples per concept")
    parser.add_argument("--max-test", type=int, default=None, help="Max test samples per concept")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resize_to = (args.resize, args.resize) if args.resize is not None else None
    benchmark = args.benchmark or infer_visual_benchmark_from_ordering_path(args.ordering_path)
    concept_order = load_concept_order_from_file(
        ordering_path=args.ordering_path,
        benchmark=benchmark,
        ordering_column=args.ordering_column,
        ordering_mode=args.ordering_mode,
    )
    dataset = read_ordered_registered_visual_benchmark_dataset(
        ordering_path=args.ordering_path,
        benchmark=benchmark,
        root=args.root,
        registry_path=args.registry_path,
        dataset_name=args.dataset_name,
        ordering_column=args.ordering_column,
        ordering_mode=args.ordering_mode,
        data_mode=args.data_mode,
        resize_to=resize_to,
        color_mode=args.color_mode,
        max_train_samples_per_category=args.max_train,
        max_test_samples_per_category=args.max_test,
    )

    print(f"Loaded dataset: {dataset.name()}")
    print(f"Benchmark: {benchmark}")
    print(f"Concept order: {concept_order}")

    test_by_name = {concept.name: concept for concept in dataset.test_concepts()}
    for index, train_concept in enumerate(dataset.train_concepts(), start=1):
        test_concept = test_by_name.get(train_concept.name)
        train_count = len(train_concept.data)
        test_count = 0 if test_concept is None else len(test_concept.data)
        print(f"{index:02d}. {train_concept.name}: train={train_count}, test={test_count}")


if __name__ == "__main__":
    main()
