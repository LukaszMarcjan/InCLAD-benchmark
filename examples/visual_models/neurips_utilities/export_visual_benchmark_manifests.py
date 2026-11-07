from __future__ import annotations

import argparse
from pathlib import Path

from pyclad.data.readers.visual_benchmark_manifest import (
    VISUAL_BENCHMARK_SOURCES,
    DEFAULT_VISUAL_BENCHMARK_MANIFESTS_DIR,
    DEFAULT_VISUAL_BENCHMARK_ORDERINGS_DIR,
    load_visual_benchmark_manifest_ordering,
    read_visual_benchmark_manifest_dataset,
    write_registered_visual_benchmark_manifest,
)
from pyclad.data.readers.visual_dataset_registry import VISUAL_BENCHMARK_DIRECTORY_CANDIDATES

DEFAULT_BENCHMARKS = ("mvtec", "btech", "mpdd", "dagm", "visa")
DEFAULT_SHARED_ROOT = DEFAULT_VISUAL_BENCHMARK_MANIFESTS_DIR.parent

# BTech (BTAD) license as stated in the paper. The published manifests carry
# "CC BY-SA" without the version; pass --btech-license "CC BY-SA" to match them.
DEFAULT_BTECH_LICENSE = "CC BY-SA 4.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export standardized CSV manifests for the visual benchmarks used in this repository."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to export",
    )
    parser.add_argument(
        "--shared-root",
        type=str,
        default=str(DEFAULT_SHARED_ROOT),
        help="Directory that contains the local benchmark folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_VISUAL_BENCHMARK_MANIFESTS_DIR),
        help="Directory where the CSV manifests will be written",
    )
    parser.add_argument(
        "--ordering-dir",
        type=str,
        default=str(DEFAULT_VISUAL_BENCHMARK_ORDERINGS_DIR),
        help="Directory that contains easy-to-hard and hard-to-easy ordering CSV files",
    )
    parser.add_argument(
        "--ordering-metric",
        type=str,
        default="roc-auc",
        help="Metric slug used in ordering filenames",
    )
    parser.add_argument(
        "--easy-master-seed",
        type=int,
        default=42,
        help="Master seed recorded in easy-to-hard and hard-to-easy manifests",
    )
    parser.add_argument(
        "--random-master-seed",
        type=int,
        default=123,
        help="Master seed used to generate random concept orders",
    )
    parser.add_argument(
        "--btech-license",
        type=str,
        default=DEFAULT_BTECH_LICENSE,
        help="source_license written for BTech rows (the published manifests use 'CC BY-SA')",
    )
    parser.add_argument(
        "--verify-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load each generated manifest back through the CSV loader and print concept counts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Rows take source_license from VISUAL_BENCHMARK_SOURCES; set the BTech entry
    # for this process only.
    VISUAL_BENCHMARK_SOURCES["btech"] = {**VISUAL_BENCHMARK_SOURCES["btech"], "license": args.btech_license}
    shared_root = Path(args.shared_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ordering_dir = Path(args.ordering_dir).expanduser().resolve()

    roots = {benchmark: resolve_benchmark_root_from_shared_root(benchmark, shared_root) for benchmark in args.benchmarks}

    for benchmark in args.benchmarks:
        benchmark_root = roots[benchmark]
        orderings = [
            load_visual_benchmark_manifest_ordering(benchmark=benchmark, root=benchmark_root, ordering_name="dataset"),
            load_visual_benchmark_manifest_ordering(
                benchmark=benchmark,
                root=benchmark_root,
                ordering_name="easy_to_hard",
                ordering_dir=ordering_dir,
                ordering_metric=args.ordering_metric,
                master_seed=args.easy_master_seed,
            ),
            load_visual_benchmark_manifest_ordering(
                benchmark=benchmark,
                root=benchmark_root,
                ordering_name="hard_to_easy",
                ordering_dir=ordering_dir,
                ordering_metric=args.ordering_metric,
                master_seed=args.easy_master_seed,
            ),
            load_visual_benchmark_manifest_ordering(
                benchmark=benchmark,
                root=benchmark_root,
                ordering_name="random",
                master_seed=args.random_master_seed,
            ),
        ]

        for ordering in orderings:
            manifest_path = write_registered_visual_benchmark_manifest(
                benchmark=benchmark,
                root=benchmark_root,
                output_dir=output_dir,
                category_order=ordering.category_order,
                ordering_name=ordering.name,
                ordering_master_seed=ordering.master_seed,
                ordering_seed=ordering.seed,
            )
            print(
                f"{benchmark} [{ordering.name}] master_seed={ordering.master_seed} "
                f"seed={ordering.seed}: {manifest_path.name}"
            )
            if not args.verify_load:
                continue

            dataset = read_visual_benchmark_manifest_dataset(
                root=benchmark_root,
                manifest_path=manifest_path,
                benchmark=benchmark,
                data_mode="paths",
            )
            print(
                f"  concepts={len(dataset.train_concepts())} "
                f"train={sum(len(concept.data) for concept in dataset.train_concepts())} "
                f"test={sum(len(concept.data) for concept in dataset.test_concepts())} "
                f"order={[concept.name for concept in dataset.train_concepts()]}"
            )


def resolve_benchmark_root_from_shared_root(benchmark: str, shared_root: Path) -> Path:
    candidates = VISUAL_BENCHMARK_DIRECTORY_CANDIDATES.get(benchmark, (benchmark,))
    for dirname in candidates:
        candidate = shared_root / dirname
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find benchmark '{benchmark}' inside {shared_root}. Tried: {list(candidates)}"
    )


if __name__ == "__main__":
    main()
