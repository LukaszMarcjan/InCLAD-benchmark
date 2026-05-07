"""
Generate heatmaps for all benchmark datasets and seeds saved under a results directory.

Pass any directory that contains per-seed result JSON files, for example:
    <results_root>/
    <results_root>/padim/
    <results_root>/padim/mvtec/
    <results_root>/padim/visa/replay/

The script scans that directory recursively for JSON files matching
`*_seed_*.json`, groups them by benchmark dataset, and for every image-level
and pixel-level metric matrix found in each file saves a PDF heatmap next to
the JSON.

Output naming:
    padim_seed_<seed>.json
    padim_seed_<seed>_roc-auc_heatmap.pdf
    padim_seed_<seed>_f1-score_heatmap.pdf

Usage:
    python3 examples/visual_models/generate_heatmaps.py <results_root>
    python3 examples/visual_models/generate_heatmaps.py --results-dir <results_root>/patchcore
    python3 examples/visual_models/generate_heatmaps.py <results_root>/padim/mvtec --dry-run
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import pathlib
import re
from typing import Literal

logger = logging.getLogger(__name__)

# Keys in the JSON that hold concept metric matrices
IMAGE_METRIC_KEY_PREFIX = "concept_metric_callback_"
PIXEL_METRIC_KEY_PREFIX = "pixel_concept_metric_callback_"
METRIC_KEY_PREFIXES = {
    "image": IMAGE_METRIC_KEY_PREFIX,
    "pixel": PIXEL_METRIC_KEY_PREFIX,
}
SEED_RESULT_PATTERN = re.compile(r".+_seed_(\d+)$")
KNOWN_BENCHMARKS = (
    "mvtec",
    "btech",
    "visa",
    "mpdd",
    "dagm",
    "inclad-md",
    "inclad_md",
    "inclad-bench",
    "inclad_bench",
)

PLOTTING_DEPENDENCIES = None


@dataclass(slots=True)
class HeatmapGenerationOptions:
    annotate: bool = True
    figsize: tuple[int, int] = (10, 10)
    ignore_upper_diagonal: bool = True
    overwrite: bool = False
    dry_run: bool = False
    metric_kind: Literal["all", "image", "pixel"] = "all"


def _detect_metric_kind(key: str, value: dict) -> str | None:
    evaluation_level = str(value.get("evaluation_level", "")).strip().lower()
    if evaluation_level in {"image", "pixel"}:
        return evaluation_level

    for metric_kind, prefix in METRIC_KEY_PREFIXES.items():
        if key.startswith(prefix):
            return metric_kind

    return None


def find_metric_matrices(data: dict) -> list[tuple[str, str, dict, list[str]]]:
    """
    Return all (metric_kind, metric_name, matrix, concepts_order) tuples found
    in a result dict, regardless of how many metrics were recorded.
    """
    found = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        metric_kind = _detect_metric_kind(key, value)
        if metric_kind is None:
            continue
        matrix = value.get("metric_matrix")
        concepts_order = value.get("concepts_order")
        if matrix is None or concepts_order is None:
            logger.debug(f"  Skipping key '{key}': missing matrix or concepts_order")
            continue
        prefix = METRIC_KEY_PREFIXES.get(metric_kind, "")
        metric_name = value.get("base_metric_name", key[len(prefix):])
        found.append((metric_kind, metric_name, matrix, concepts_order))
    return found


def heatmap_path(json_path: pathlib.Path, metric_name: str) -> pathlib.Path:
    safe_name = metric_name.lower().replace(" ", "_").replace("/", "-")
    return json_path.with_name(f"{json_path.stem}_{safe_name}_heatmap.pdf")


def infer_benchmark_name(json_path: pathlib.Path, results_dir: pathlib.Path) -> str:
    """Infer benchmark name from the path relative to the provided results directory."""
    try:
        relative_parts = json_path.relative_to(results_dir).parts[:-1]
    except ValueError:
        relative_parts = json_path.parts[:-1]

    for part in relative_parts:
        normalized = part.lower()
        if normalized in KNOWN_BENCHMARKS:
            return normalized

    for part in json_path.parts[:-1]:
        normalized = part.lower()
        if normalized in KNOWN_BENCHMARKS:
            return normalized

    return json_path.parent.name.lower()


def seed_result_sort_key(json_path: pathlib.Path) -> tuple[str, int, str]:
    """Sort result files by benchmark folder and numeric seed when possible."""
    match = SEED_RESULT_PATTERN.match(json_path.stem)
    seed = int(match.group(1)) if match else -1
    return (json_path.parent.as_posix(), seed, json_path.name)


def get_plotting_dependencies():
    """Import plotting stack lazily so dry-run does not require matplotlib."""
    global PLOTTING_DEPENDENCIES

    if PLOTTING_DEPENDENCIES is None:
        import matplotlib

        matplotlib.use("Agg")  # no display needed; works on headless SSH servers

        import matplotlib.pyplot as plt

        from pyclad.analysis.scenario_heatmap import plot_metric_heatmap

        PLOTTING_DEPENDENCIES = (plt, plot_metric_heatmap)

    return PLOTTING_DEPENDENCIES


def discover_seed_result_files(results_dir: pathlib.Path) -> list[tuple[str, list[pathlib.Path]]]:
    """Group all per-seed result JSON files by benchmark name."""
    benchmark_to_files: dict[str, list[pathlib.Path]] = defaultdict(list)

    for json_path in sorted(results_dir.rglob("*_seed_*.json"), key=seed_result_sort_key):
        benchmark_name = infer_benchmark_name(json_path, results_dir)
        benchmark_to_files[benchmark_name].append(json_path)

    return [
        (benchmark_name, files)
        for benchmark_name, files in sorted(benchmark_to_files.items())
    ]


def process_file(
    json_path: pathlib.Path,
    benchmark_name: str,
    options: HeatmapGenerationOptions,
) -> int:
    """Generate heatmaps for one JSON file. Returns number of heatmaps written."""
    try:
        with json_path.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Cannot read {json_path}: {exc}")
        return 0

    matrices = find_metric_matrices(data)
    if not matrices:
        logger.debug(f"  No metric matrices in {json_path.name}, skipping")
        return 0

    written = 0
    for metric_kind, metric_name, matrix, concepts_order in matrices:
        if options.metric_kind != "all" and metric_kind != options.metric_kind:
            logger.debug(f"  Skipping '{metric_name}' in {json_path.name}: metric kind '{metric_kind}' filtered out")
            continue

        out_path = heatmap_path(json_path, metric_name)

        if out_path.exists() and not options.overwrite:
            logger.info(f"  [skip] {out_path.name} already exists (use --overwrite to regenerate)")
            continue

        if options.dry_run:
            logger.info(
                f"  [dry-run] would write {out_path.name}  "
                f"({metric_kind}, {metric_name}, {len(concepts_order)} concepts)"
            )
            continue

        try:
            plt, plot_metric_heatmap = get_plotting_dependencies()
            plt.close("all")

            plot_metric_heatmap(
                matrix=matrix,
                concepts_order=concepts_order,
                title=f"{metric_name} - {benchmark_name} - {json_path.stem}",
                annotate=options.annotate,
                ignore_upper_diagonal=not options.ignore_upper_diagonal,
                figsize=options.figsize,
                output_path=out_path,
            )
            plt.close("all")
            logger.info(f"  ✓ {out_path.name}")
            written += 1
        except Exception as exc:
            logger.error(f"  ✗ Failed to generate heatmap for '{metric_name}' in {json_path.name}: {exc}")

    return written


def build_options_from_args(args: argparse.Namespace) -> HeatmapGenerationOptions:
    return HeatmapGenerationOptions(
        annotate=args.annotate,
        figsize=tuple(args.figsize),
        ignore_upper_diagonal=args.ignore_upper_diagonal,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        metric_kind=args.metric_kind,
    )


def generate_heatmaps_for_results_dir(
    results_root: pathlib.Path,
    options: HeatmapGenerationOptions,
) -> int:
    benchmark_groups = discover_seed_result_files(results_root)
    if not benchmark_groups:
        logger.warning(f"No per-seed JSON files matching '*_seed_*.json' found under {results_root}")
        return 0

    total_json_files = sum(len(files) for _, files in benchmark_groups)
    logger.info(
        f"Found {total_json_files} seed result JSON file(s) across "
        f"{len(benchmark_groups)} benchmark(s) under {results_root}"
    )

    total_written = 0
    for benchmark_name, json_files in benchmark_groups:
        logger.info("")
        logger.info(f"Benchmark: {benchmark_name} ({len(json_files)} seed result file(s))")
        benchmark_written = 0

        for json_path in json_files:
            logger.info(f"Processing: {json_path.relative_to(results_root)}")
            benchmark_written += process_file(json_path, benchmark_name, options)

        total_written += benchmark_written
        logger.info(f"Benchmark '{benchmark_name}' complete: {benchmark_written} heatmap(s)")

    logger.info("")
    if options.dry_run:
        logger.info("Dry-run complete - no files written")
    else:
        logger.info(f"Done - {total_written} heatmap(s) generated")

    return total_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heatmaps for all benchmark datasets and seeds in a results directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "results_dir", nargs="?",
        help="Path to a results directory (whole batch root or any nested model/benchmark/strategy subtree)",
    )
    parser.add_argument(
        "--results-dir", dest="results_dir_flag", type=str,
        help="Path to a results directory (whole batch root or any nested model/benchmark/strategy subtree)",
    )
    parser.add_argument("--results-root", dest="results_dir_flag", type=str, help=argparse.SUPPRESS)
    parser.set_defaults(annotate=True, ignore_upper_diagonal=True)
    parser.add_argument(
        "--annotate", dest="annotate", action="store_true",
        help="Annotate heatmap cells with numeric values",
    )
    parser.add_argument(
        "--no-annotate", dest="annotate", action="store_false",
        help="Disable numeric annotations in heatmap cells",
    )
    parser.add_argument(
        "--figsize", type=int, nargs=2, default=[10, 10], metavar=("W", "H"),
        help="Figure size in inches (default: 10 10)",
    )
    parser.add_argument(
        "--ignore-upper-diagonal", dest="ignore_upper_diagonal", action="store_true",
        help="Mask the upper diagonal (forward predictions not yet made)",
    )
    parser.add_argument(
        "--no-ignore-upper-diagonal", dest="ignore_upper_diagonal", action="store_false",
        help="Show the full matrix including upper diagonal",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate heatmaps even if they already exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without writing any files",
    )
    parser.add_argument(
        "--metric-kind",
        type=str,
        choices=["all", "image", "pixel"],
        default="all",
        help="Choose whether to generate image-level heatmaps, pixel-level heatmaps, or both",
    )
    args = parser.parse_args()
    args.results_dir = args.results_dir_flag or args.results_dir

    if args.results_dir is None:
        parser.error("provide a results directory as a positional argument or via --results-dir")

    return args


if __name__ == "__main__":
    args = parse_args()
    options = build_options_from_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    results_root = pathlib.Path(args.results_dir).expanduser()
    if not results_root.is_dir():
        logger.error(f"Results root does not exist: {results_root}")
        raise SystemExit(1)
    generate_heatmaps_for_results_dir(results_root, options)
