from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from pyclad.data.readers.visual_benchmark_manifest import (
    DEFAULT_VISUAL_BENCHMARK_MANIFESTS_DIR,
    DEFAULT_VISUAL_BENCHMARK_ORDERINGS_DIR,
    load_visual_benchmark_manifest_ordering,
    manifest_output_filename,
    read_visual_benchmark_manifest_dataset,
)
from pyclad.data.readers.visual_dataset_registry import (
    VISUAL_BENCHMARK_DIRECTORY_CANDIDATES,
    read_registered_visual_benchmark_dataset,
)

DEFAULT_SHARED_ROOT = DEFAULT_VISUAL_BENCHMARK_MANIFESTS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a manifest-loaded dataset is identical to the dataset used by the experiment loader."
    )
    parser.add_argument("--benchmark", required=True, help="Benchmark name, e.g. mvtec, btech, mpdd, dagm, visa")
    parser.add_argument(
        "--ordering-name",
        choices=["dataset", "easy_to_hard", "hard_to_easy", "random"],
        default="dataset",
        help="Concept ordering to compare",
    )
    parser.add_argument(
        "--shared-root",
        type=str,
        default=str(DEFAULT_SHARED_ROOT),
        help="Directory that contains the local benchmark folders",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=str(DEFAULT_VISUAL_BENCHMARK_MANIFESTS_DIR),
        help="Directory that contains generated manifest CSV files",
    )
    parser.add_argument(
        "--ordering-dir",
        type=str,
        default=str(DEFAULT_VISUAL_BENCHMARK_ORDERINGS_DIR),
        help="Directory with easy-to-hard and hard-to-easy ordering CSV files",
    )
    parser.add_argument("--easy-master-seed", type=int, default=42, help="Master seed metadata for fixed orderings")
    parser.add_argument("--random-master-seed", type=int, default=123, help="Master seed for random ordering")
    parser.add_argument("--resize", type=int, default=None, help="Optional square resize for numpy comparison")
    parser.add_argument(
        "--color-mode",
        choices=["rgb", "grayscale"],
        default="rgb",
        help="Color mode for numpy comparison",
    )
    parser.add_argument(
        "--check-numpy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also load both datasets in numpy mode and compare arrays 1:1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_root = Path(args.shared_root).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    ordering_dir = Path(args.ordering_dir).expanduser().resolve()
    benchmark_root = resolve_benchmark_root_from_shared_root(args.benchmark, shared_root)

    master_seed = args.random_master_seed if args.ordering_name == "random" else (
        args.easy_master_seed if args.ordering_name in {"easy_to_hard", "hard_to_easy"} else None
    )
    ordering = load_visual_benchmark_manifest_ordering(
        benchmark=args.benchmark,
        root=benchmark_root,
        ordering_name=args.ordering_name,
        ordering_dir=ordering_dir,
        master_seed=master_seed,
    )
    manifest_path = resolve_manifest_path(
        manifest_dir=manifest_dir,
        benchmark=args.benchmark,
        ordering_name=ordering.name,
        master_seed=ordering.master_seed,
        seed=ordering.seed,
    )

    experiment_dataset = read_registered_visual_benchmark_dataset(
        benchmark=args.benchmark,
        root=benchmark_root,
        categories=ordering.category_order,
        data_mode="paths",
    )
    manifest_dataset = read_visual_benchmark_manifest_dataset(
        root=benchmark_root,
        manifest_path=manifest_path,
        benchmark=args.benchmark,
        data_mode="paths",
    )

    compare_datasets(
        left=experiment_dataset,
        right=manifest_dataset,
        mode_label="paths",
    )
    print("paths: OK")

    if args.check_numpy:
        resize_to: Optional[tuple[int, int]] = None if args.resize is None else (args.resize, args.resize)
        experiment_numpy = read_registered_visual_benchmark_dataset(
            benchmark=args.benchmark,
            root=benchmark_root,
            categories=ordering.category_order,
            data_mode="numpy",
            resize_to=resize_to,
            color_mode=args.color_mode,
        )
        manifest_numpy = read_visual_benchmark_manifest_dataset(
            root=benchmark_root,
            manifest_path=manifest_path,
            benchmark=args.benchmark,
            data_mode="numpy",
            resize_to=resize_to,
            color_mode=args.color_mode,
        )
        compare_datasets(
            left=experiment_numpy,
            right=manifest_numpy,
            mode_label=f"numpy resize={resize_to} color={args.color_mode}",
        )
        print("numpy: OK")

    print("equivalence: OK")
    print(f"benchmark={args.benchmark}")
    print(f"ordering_name={ordering.name}")
    print(f"ordering_master_seed={ordering.master_seed}")
    print(f"ordering_seed={ordering.seed}")
    print(f"manifest={manifest_path.name}")
    print(f"concept_order={[concept.name for concept in manifest_dataset.train_concepts()]}")


def compare_datasets(left, right, mode_label: str) -> None:
    left_train = left.train_concepts()
    right_train = right.train_concepts()
    left_test = left.test_concepts()
    right_test = right.test_concepts()

    assert [concept.name for concept in left_train] == [concept.name for concept in right_train], (
        f"{mode_label}: train concept order mismatch"
    )
    assert [concept.name for concept in left_test] == [concept.name for concept in right_test], (
        f"{mode_label}: test concept order mismatch"
    )

    for left_concept, right_concept in zip(left_train, right_train):
        compare_concepts(left_concept, right_concept, f"{mode_label}:train:{left_concept.name}")

    for left_concept, right_concept in zip(left_test, right_test):
        compare_concepts(left_concept, right_concept, f"{mode_label}:test:{left_concept.name}")


def compare_concepts(left_concept, right_concept, label: str) -> None:
    assert left_concept.name == right_concept.name, f"{label}: concept name mismatch"
    assert len(left_concept.data) == len(right_concept.data), f"{label}: concept length mismatch"

    left_data = np.asarray(left_concept.data, dtype=object) if getattr(left_concept.data, "dtype", None) == object else left_concept.data
    right_data = np.asarray(right_concept.data, dtype=object) if getattr(right_concept.data, "dtype", None) == object else right_concept.data

    if getattr(left_concept.data, "dtype", None) == object or getattr(right_concept.data, "dtype", None) == object:
        assert left_concept.data.dtype == right_concept.data.dtype, f"{label}: dtype mismatch"
        assert np.array_equal(left_data, right_data), f"{label}: path payload mismatch"
    else:
        assert left_concept.data.dtype == right_concept.data.dtype, f"{label}: dtype mismatch"
        assert left_concept.data.shape == right_concept.data.shape, f"{label}: data shape mismatch"
        assert np.array_equal(left_concept.data, right_concept.data), f"{label}: numpy payload mismatch"

    if left_concept.labels is None or right_concept.labels is None:
        assert left_concept.labels is None and right_concept.labels is None, f"{label}: label presence mismatch"
    else:
        assert left_concept.labels.shape == right_concept.labels.shape, f"{label}: label shape mismatch"
        assert np.array_equal(left_concept.labels, right_concept.labels), f"{label}: labels mismatch"


def resolve_benchmark_root_from_shared_root(benchmark: str, shared_root: Path) -> Path:
    candidates = VISUAL_BENCHMARK_DIRECTORY_CANDIDATES.get(benchmark, (benchmark,))
    for dirname in candidates:
        candidate = shared_root / dirname
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find benchmark '{benchmark}' inside {shared_root}. Tried: {list(candidates)}"
    )


def resolve_manifest_path(
    manifest_dir: Path,
    benchmark: str,
    ordering_name: str,
    master_seed: Optional[int],
    seed: Optional[int],
) -> Path:
    manifest_name = manifest_output_filename(
        benchmark=benchmark,
        ordering_name=ordering_name,
        ordering_master_seed=master_seed,
        ordering_seed=seed,
    )
    manifest_path = manifest_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    return manifest_path.resolve()


if __name__ == "__main__":
    main()
