"""
Batch runner for continual visual anomaly-detection experiments using the add-on
CL visual strategies.

This script mirrors `run_all_experiments.py`, but points to
`examples/clvad/run_continual_visual_ad_levels_cl.py` and defaults to the
memory-bank visual models introduced for continual learning:

- PaDiM
- PatchCore
- CFA
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_base_batch_runner():
    module_path = Path(__file__).with_name("run_all_experiments.py")
    spec = importlib.util.spec_from_file_location("pyclad_visual_batch_base_cl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base batch runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_batch_runner()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

_BASE.CLVAD_RUNNER = PROJECT_ROOT / "examples" / "clvad" / "run_continual_visual_ad_levels_cl.py"
_BASE.SUPPORTED_MODELS = ("cfa", "padim", "patchcore")
_BASE.SUPPORTED_STRATEGIES = ("naive", "replay", "cumulative", "ste", "cl")
_BASE.DEFAULT_STRATEGIES = ("naive", "cumulative", "cl")
_BASE.PIXEL_LEVEL_MODELS = ("cfa", "padim", "patchcore")

BATCH_LOG_FILENAME = _BASE.BATCH_LOG_FILENAME


def parse_args() -> argparse.Namespace:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--continual-memory-bank-size",
        type=int,
        default=None,
        help="Explicit PatchCore CL memory-bank budget. Defaults to replay-buffer-derived size.",
    )
    extra_args, remaining = extra_parser.parse_known_args()

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _BASE.parse_args()
    finally:
        sys.argv = original_argv

    if extra_args.continual_memory_bank_size is not None and extra_args.continual_memory_bank_size <= 0:
        raise SystemExit(
            f"--continual-memory-bank-size must be positive, got {extra_args.continual_memory_bank_size}"
        )

    args.continual_memory_bank_size = extra_args.continual_memory_bank_size
    return args


def build_command(
    model: str,
    strategy: str,
    benchmark: str,
    dataset_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = _BASE.build_command(model, strategy, benchmark, dataset_path, args)
    if args.continual_memory_bank_size is not None and model == "patchcore":
        command.extend(["--continual-memory-bank-size", str(args.continual_memory_bank_size)])
    return command


def main() -> None:
    try:
        args = parse_args()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    results_root = Path(args.output_root).expanduser().resolve()
    args.output_root = str(results_root)
    batch_log_path = results_root / BATCH_LOG_FILENAME
    _BASE.configure_batch_logging(batch_log_path)
    logger.info("%s", "═" * 72)
    logger.info("Batch log: %s", batch_log_path)

    if not _BASE.CLVAD_RUNNER.is_file():
        logger.error("Continual visual AD CL runner not found: %s", _BASE.CLVAD_RUNNER)
        sys.exit(1)

    datasets_root = Path(args.datasets_root).expanduser().resolve()
    if not datasets_root.is_dir():
        logger.error("Datasets root does not exist: %s", datasets_root)
        sys.exit(1)

    ordering_dir = Path(args.ordering_dir).expanduser().resolve()
    if args.ordering_mode is not None and not ordering_dir.is_dir():
        logger.error("Ordering directory does not exist: %s", ordering_dir)
        sys.exit(1)

    if args.benchmarks is None:
        benchmarks = _BASE.discover_benchmarks(datasets_root)
        logger.info("Auto-discovered %d benchmark(s): %s", len(benchmarks), benchmarks)
    else:
        benchmarks = []
        for benchmark in args.benchmarks:
            dataset_dir = _BASE.resolve_benchmark_dir(datasets_root, benchmark)
            if dataset_dir is None:
                logger.warning("Benchmark '%s' not found under %s, skipping", benchmark, datasets_root)
                continue
            benchmarks.append(benchmark)

    if not benchmarks:
        logger.error("No benchmarks found. Check --datasets-root or --benchmarks.")
        sys.exit(1)

    unused_ordering_files = sorted(set(args.ordering_files) - set(benchmarks))
    if unused_ordering_files:
        logger.warning("Ordering files were provided for benchmarks not being run: %s", unused_ordering_files)

    jobs: list[tuple[str, str, str, list[str]]] = []
    for model in args.models:
        for benchmark in benchmarks:
            dataset_path = _BASE.resolve_benchmark_dir(datasets_root, benchmark)
            if dataset_path is None:
                logger.warning("Benchmark '%s' disappeared before scheduling, skipping", benchmark)
                continue
            for strategy in args.strategies:
                command = build_command(model, strategy, benchmark, dataset_path, args)
                jobs.append((model, benchmark, strategy, command))

    total = len(jobs)
    logger.info(
        "Total jobs: %d  (%d models x %d benchmarks x %d strategies x %d seeds each)",
        total,
        len(args.models),
        len(benchmarks),
        len(args.strategies),
        args.n_runs,
    )
    logger.info("Evaluation level: %s", args.eval_level)
    if args.eval_level in {"pixel", "both"}:
        logger.info("Pixel threshold mode: %s", args.pixel_threshold_mode)
        logger.info("Pixel threshold: %s", args.pixel_threshold)
        logger.info("Pixel threshold quantile: %s", args.pixel_threshold_quantile)
        logger.info("Skip anomalous samples without masks: %s", args.pixel_skip_missing_masks)
    logger.info("Results root: %s", results_root)
    logger.info("")

    if args.dry_run:
        for model, benchmark, strategy, command in jobs:
            logger.info("[DRY RUN] %s / %s / %s", model, benchmark, strategy)
            logger.info("  %s", " ".join(command))
            logger.info("")
        return

    failed: list[tuple[str, str, str, int]] = []
    for job_index, (model, benchmark, strategy, command) in enumerate(jobs, start=1):
        logger.info("%s", "─" * 72)
        logger.info("Job %d/%d: %s / %s / %s", job_index, total, model, benchmark, strategy)
        logger.info("Command: %s", " ".join(command))
        logger.info("")

        started_at = time.perf_counter()
        result = subprocess.run(command)
        elapsed = time.perf_counter() - started_at

        if result.returncode != 0:
            logger.error("  FAILED (exit code %d) after %.1fs", result.returncode, elapsed)
            failed.append((model, benchmark, strategy, result.returncode))
        else:
            logger.info("  done in %.1fs", elapsed)

        logger.info("")

    if args.generate_heatmaps:
        logger.info("%s", "═" * 72)
        if failed:
            logger.info("Some jobs failed; generating heatmaps for any results that were written.")
        logger.info("Generating heatmaps under: %s", results_root)
        heatmap_options = _BASE.HeatmapGenerationOptions(
            annotate=args.heatmap_annotate,
            figsize=tuple(args.heatmap_figsize),
            ignore_upper_diagonal=not args.heatmap_show_upper_diagonal,
            overwrite=args.overwrite_heatmaps,
            dry_run=False,
        )
        heatmaps_written = _BASE.generate_heatmaps_for_results_dir(results_root, heatmap_options)
        logger.info("Heatmap generation complete: %d file(s) written", heatmaps_written)
        logger.info("")

    logger.info("%s", "═" * 72)
    logger.info("Completed %d/%d jobs successfully", total - len(failed), total)
    if failed:
        logger.warning("Failed jobs:")
        for model, benchmark, strategy, return_code in failed:
            logger.warning("  %s / %s / %s  (exit code %d)", model, benchmark, strategy, return_code)
        sys.exit(1)

    logger.info("All results saved under: %s", results_root)


if __name__ == "__main__":
    main()
