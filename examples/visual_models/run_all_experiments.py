"""
Batch runner for continual visual anomaly-detection experiments.

This script orchestrates `examples/clvad/run_continual_visual_ad_levels.py`
across selected models, strategies, and visual benchmarks. Each
`(model x benchmark x strategy)` combination runs in a separate subprocess, so
memory is fully released between jobs.

Examples
--------
Run all supported models on all discovered datasets with the three continual
strategies requested most often for benchmarking:

    python examples/visual_models/run_all_experiments.py \
        --n-runs 3 \
        --master-seed 42

Limit to a couple of models, a couple of datasets, and only naive/replay:

    python examples/visual_models/run_all_experiments.py \
        --models patchcore padim \
        --strategies naive replay \
        --benchmarks mvtec btech

Use an explicit curriculum file from ordering:

    python examples/visual_models/run_all_experiments.py \
        --models cfa draem efficientad \
        --benchmarks mvtec \
        --ordering-file examples/visual_models/ordering/mvtec_easy_to_hard_roc-auc.csv

Use ordering-mode files from ordering for every selected benchmark:

    python examples/visual_models/run_all_experiments.py \
        --benchmarks mvtec visa \
        --ordering-mode hard_to_easy

Run both image-level and pixel-level evaluation for pixel-capable models:

    python examples/visual_models/run_all_experiments.py \
        --models patchcore padim rd4ad \
        --benchmarks mvtec \
        --eval-level both \
        --pixel-threshold-mode train-quantile \
        --pixel-threshold-quantile 0.995

Skip automatic heatmap generation after the batch run:

    python examples/visual_models/run_all_experiments.py \
        --benchmarks visa \
        --no-generate-heatmaps

The batch runner appends its own log stream to
`<output-root>/run_all_experiments.log`. Per-seed training logs remain managed
by the underlying continual-learning runner.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from datetime import datetime

from generate_heatmaps import HeatmapGenerationOptions, generate_heatmaps_for_results_dir
from pyclad.data.readers.visual_benchmark_reader import (
    PREDEFINED_BENCHMARK_ALIASES,
    available_visual_benchmarks,
)
from pyclad.data.readers.visual_dataset_ordering import infer_visual_benchmark_from_ordering_path
from pyclad.data.readers.visual_dataset_registry import VISUAL_BENCHMARK_DIRECTORY_CANDIDATES


def default_results_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"results_{timestamp}")

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CLVAD_RUNNER = PROJECT_ROOT / "examples" / "clvad" / "run_continual_visual_ad_levels.py"
CLVAD_CL_RUNNER = PROJECT_ROOT / "examples" / "clvad" / "run_continual_visual_ad_cl.py"
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "src" / "pyclad" / "data" / "datasets" / "visual_datasets"
DEFAULT_ORDERING_DIR = SCRIPT_DIR / "ordering"
BATCH_LOG_FILENAME = "run_all_experiments.log"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"

SUPPORTED_MODELS = (
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
    "unet",
)
SUPPORTED_STRATEGIES = ("naive", "replay", "cumulative", "ste", "cl")
DEFAULT_STRATEGIES = ("naive", "replay", "cumulative")
CL_STRATEGY_MODELS = ("cfa", "padim", "patchcore")
SUPPORTED_BENCHMARKS = tuple(available_visual_benchmarks()) + ("inclad-md",)
PIXEL_LEVEL_MODELS = (
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
        description="Run continual visual AD experiments across models, datasets, and strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--datasets-root",
        type=str,
        default=str(DEFAULT_DATASETS_ROOT),
        help="Parent directory containing all benchmark dataset folders",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(SUPPORTED_MODELS),
        help=f"Models to run. Available: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help=f"Strategies to run. Available: {', '.join(SUPPORTED_STRATEGIES)}",
    )
    parser.add_argument(
        "--strategy",
        dest="strategy_aliases",
        action="append",
        default=None,
        help="Backwards-compatible alias for a single strategy; repeatable",
    )
    parser.add_argument(
        "--benchmarks",
        "--datasets",
        dest="benchmarks",
        type=str,
        nargs="*",
        default=None,
        help=f"Benchmarks to use (default: auto-discover). Available: {', '.join(SUPPORTED_BENCHMARKS)}",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(default_results_root()),
        help="Root directory for all results",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")

    parser.add_argument("--n-runs", type=int, default=1, help="Number of seeded runs per job")
    parser.add_argument("--master-seed", type=int, default=42, help="Master seed")

    ds = parser.add_argument_group("dataset")
    ds.add_argument("--registry-path", type=str, default=None, help="Optional visual dataset registry JSON path")
    ds.add_argument(
        "--inclad-bench",
        action="store_true",
        default=False,
        help="Load manifests from the InCLAD-Bench HuggingFace repository (anonmllab/inclad-bench) "
        "instead of discovering local folder structure. Images must still be available locally.",
    )
    ds.add_argument("--categories", type=str, nargs="*", default=None, help="Categories to use (default: all)")
    ds.add_argument("--resize", type=int, default=None, help="Resize images to this square size")
    ds.add_argument("--max-train", type=int, default=None, help="Max training samples per category")
    ds.add_argument("--max-test", type=int, default=None, help="Max test samples per category")
    ds.add_argument(
        "--concept-order",
        type=str,
        nargs="*",
        default=None,
        help="Explicit concept order to use for every selected benchmark",
    )
    ds.add_argument(
        "--ordering-file",
        type=str,
        action="append",
        default=None,
        help=(
            "Ordering file to use for a benchmark. Pass either /path/to/file.csv and let the script infer the "
            "benchmark from the filename, or benchmark=/path/to/file.csv. Repeat to provide one file per benchmark."
        ),
    )
    ds.add_argument(
        "--ordering-dir",
        type=str,
        default=str(DEFAULT_ORDERING_DIR),
        help="Directory containing benchmark ordering files for --ordering-mode",
    )
    ds.add_argument(
        "--ordering-mode",
        type=str,
        default=None,
        choices=["easy_to_hard", "hard_to_easy", "random"],
        help="Use benchmark ordering files from ordering-dir",
    )
    ds.add_argument(
        "--ordering-metric",
        type=str,
        default="roc-auc",
        help="Metric slug used in ordering filenames for --ordering-mode",
    )
    ds.add_argument(
        "--ordering-column",
        type=str,
        default="concept",
        help="CSV column containing concept names when loading an ordering file",
    )
    ds.add_argument(
        "--shuffle-concepts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle concept order independently for each seed",
    )

    md = parser.add_argument_group("model")
    md.add_argument("--backbone", type=str, default=None, help="Backbone override when supported")
    md.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable pretrained teacher/backbone/encoder weights when supported",
    )
    md.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze/unfreeze teacher/backbone/encoder when supported",
    )
    md.add_argument("--batch-size", type=int, default=None, help="Training batch size override")
    md.add_argument("--epochs", type=int, default=None, help="Training epochs override")
    md.add_argument("--device", type=str, default=None, help="Torch device override")
    md.add_argument("--threshold-quantile", type=float, default=None, help="Threshold quantile override")
    md.add_argument(
        "--show-training-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable model-native progress bars when supported",
    )
    md.add_argument("--variant", type=str, default=None, help="Variant override for models such as DRAEM")
    md.add_argument("--model-size", type=str, default=None, help="Model size override for EfficientAD")
    md.add_argument("--n-features", type=int, default=None, help="Feature count override when supported")
    md.add_argument("--coreset-ratio", type=float, default=None, help="PatchCore coreset ratio override")
    md.add_argument("--n-neighbors", type=int, default=None, help="Nearest-neighbor count override")
    md.add_argument("--ad-layers", type=int, nargs="*", default=None, help="PaSTe anomaly-detection layers")
    md.add_argument(
        "--student-bootstrap-layer",
        type=int,
        default=None,
        help="PaSTe student bootstrap layer; pass -1 to disable bootstrap",
    )
    md.add_argument("--teacher-weights-path", type=str, default=None, help="EfficientAD teacher checkpoint path")
    md.add_argument("--penalty-source-root", type=str, default=None, help="EfficientAD penalty-image directory")
    md.add_argument("--anomaly-source-root", type=str, default=None, help="DRAEM anomaly source directory")
    md.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="JSON file with raw model-config overrides",
    )
    md.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Inline JSON object with raw model-config overrides",
    )

    ex = parser.add_argument_group("experiment")
    ex.add_argument(
        "--replay-buffer-fraction",
        type=float,
        default=0.2,
        help="Replay buffer size as a fraction of the average concept size (e.g. 0.20 = 20%%)",
    )
    ex.add_argument(
        "--eval-level",
        type=str,
        choices=["image", "pixel", "both"],
        default="image",
        help="Evaluation granularity for visual anomaly detection",
    )
    ex.add_argument(
        "--pixel-threshold",
        type=float,
        default=0.5,
        help="Threshold for pixel F1, Dice, and IoU when pixel-level evaluation is enabled",
    )
    ex.add_argument(
        "--pixel-threshold-mode",
        type=str,
        choices=["fixed", "train-quantile"],
        default="fixed",
        help="Threshold strategy for thresholded pixel metrics",
    )
    ex.add_argument(
        "--pixel-threshold-quantile",
        type=float,
        default=0.995,
        help="Quantile used when --pixel-threshold-mode=train-quantile",
    )
    ex.add_argument(
        "--pixel-skip-missing-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip anomalous test samples without masks when pixel-level evaluation is enabled",
    )

    reports = parser.add_argument_group("reporting")
    reports.add_argument(
        "--generate-heatmaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate per-result heatmap PDFs under the output root after the batch run",
    )
    reports.add_argument(
        "--heatmap-annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate generated heatmap cells with numeric values",
    )
    reports.add_argument(
        "--heatmap-figsize",
        type=int,
        nargs=2,
        default=[10, 10],
        metavar=("W", "H"),
        help="Figure size in inches for generated heatmaps",
    )
    reports.add_argument(
        "--heatmap-show-upper-diagonal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the full heatmap matrix instead of masking the upper diagonal",
    )
    reports.add_argument(
        "--overwrite-heatmaps",
        action="store_true",
        help="Regenerate heatmaps even when a PDF already exists",
    )

    args = parser.parse_args()

    if args.strategies is not None and args.strategy_aliases is not None:
        parser.error("Use either --strategies or --strategy, not both.")

    args.models = _normalize_models(args.models, parser)
    args.strategies = _normalize_strategies(args.strategies or args.strategy_aliases or DEFAULT_STRATEGIES, parser)
    args.benchmarks = _normalize_benchmarks(args.benchmarks or None, parser)
    args.ordering_files = resolve_ordering_files(args.ordering_file or [])
    if not 0.0 < float(args.pixel_threshold_quantile) < 1.0:
        parser.error(f"--pixel-threshold-quantile must be in (0, 1), got {args.pixel_threshold_quantile}")
    _validate_eval_level_models(args, parser)
    _validate_cl_strategy_models(args, parser)

    if args.shuffle_concepts and (args.concept_order or args.ordering_files or args.ordering_mode is not None):
        parser.error("Fixed ordering cannot be combined with --shuffle-concepts.")

    return args


def _normalize_models(models: list[str], parser: argparse.ArgumentParser) -> list[str]:
    normalized = []
    seen = set()

    for model in models:
        key = str(model).strip().lower()
        if key not in SUPPORTED_MODELS:
            parser.error(f"Unknown model '{model}'. Available: {', '.join(SUPPORTED_MODELS)}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)

    return normalized


def _normalize_strategies(strategies: list[str], parser: argparse.ArgumentParser) -> list[str]:
    normalized = []
    seen = set()

    for strategy in strategies:
        key = str(strategy).strip().lower()
        if key not in SUPPORTED_STRATEGIES:
            parser.error(f"Unknown strategy '{strategy}'. Available: {', '.join(SUPPORTED_STRATEGIES)}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)

    return normalized


def _normalize_benchmarks(
    benchmarks: list[str] | None,
    parser: argparse.ArgumentParser,
) -> list[str] | None:
    if benchmarks is None:
        return None

    normalized = []
    seen = set()

    for benchmark in benchmarks:
        key = normalize_benchmark_name(benchmark)
        if key not in SUPPORTED_BENCHMARKS:
            parser.error(f"Unknown benchmark '{benchmark}'. Available: {', '.join(SUPPORTED_BENCHMARKS)}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)

    return normalized


def _validate_eval_level_models(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.eval_level not in {"pixel", "both"}:
        return

    unsupported = [model for model in args.models if model not in PIXEL_LEVEL_MODELS]
    if not unsupported:
        return

    supported = ", ".join(PIXEL_LEVEL_MODELS)
    parser.error(
        "Pixel-level evaluation is only available for these visual models: "
        f"{supported}. Unsupported selection: {', '.join(unsupported)}"
    )


def _validate_cl_strategy_models(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if "cl" not in args.strategies:
        return

    unsupported = [model for model in args.models if model not in CL_STRATEGY_MODELS]
    if not unsupported:
        return

    supported = ", ".join(CL_STRATEGY_MODELS)
    parser.error(
        "Strategy 'cl' is only available for these visual models: "
        f"{supported}. Unsupported selection: {', '.join(unsupported)}"
    )


def normalize_benchmark_name(benchmark: str) -> str:
    key = benchmark.strip().lower()
    return PREDEFINED_BENCHMARK_ALIASES.get(key, key)


def resolve_ordering_files(ordering_specs: list[str]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}

    for spec in ordering_specs:
        benchmark_name, raw_path = _split_ordering_spec(spec)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Ordering file does not exist: {path}")

        benchmark = normalize_benchmark_name(benchmark_name) if benchmark_name is not None else None
        if benchmark is None:
            try:
                benchmark = infer_visual_benchmark_from_ordering_path(path)
            except ValueError as exc:
                raise ValueError(
                    f"Could not infer benchmark from ordering file '{path.name}'. "
                    "Pass benchmark=/path/to/file.csv instead."
                ) from exc

        if benchmark in resolved:
            raise ValueError(f"Duplicate ordering file provided for benchmark '{benchmark}'.")
        resolved[benchmark] = path

    return resolved


def _split_ordering_spec(spec: str) -> tuple[str | None, str]:
    left, sep, right = spec.partition("=")
    if not sep:
        return None, spec
    return left.strip(), right.strip()


def resolve_benchmark_dir(datasets_root: Path, benchmark: str) -> Path | None:
    for dirname in VISUAL_BENCHMARK_DIRECTORY_CANDIDATES.get(benchmark, (benchmark,)):
        candidate = datasets_root / dirname
        if candidate.is_dir():
            return candidate
    return None


def discover_benchmarks(datasets_root: Path) -> list[str]:
    discovered = []
    for benchmark in SUPPORTED_BENCHMARKS:
        if resolve_benchmark_dir(datasets_root, benchmark) is not None:
            discovered.append(benchmark)
    return discovered


def build_command(
    model: str,
    strategy: str,
    benchmark: str,
    dataset_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    runner_path = CLVAD_CL_RUNNER if strategy == "cl" else CLVAD_RUNNER
    output_dir = Path(args.output_root) / model / benchmark / strategy
    command = [
        sys.executable,
        str(runner_path),
        "--model",
        model,
        "--strategy",
        strategy,
        "--benchmark",
        benchmark,
        "--root",
        str(dataset_path),
        "--n-runs",
        str(args.n_runs),
        "--master-seed",
        str(args.master_seed),
        "--output-dir",
        str(output_dir),
        "--replay-buffer-fraction",
        str(args.replay_buffer_fraction),
        "--eval-level",
        args.eval_level,
    ]

    _append_optional(command, "--registry-path", args.registry_path)
    if getattr(args, "inclad_bench", False):
        command.append("--inclad-bench")
    _append_optional(command, "--resize", args.resize)
    _append_optional(command, "--max-train", args.max_train)
    _append_optional(command, "--max-test", args.max_test)
    _append_optional(command, "--backbone", args.backbone)
    _append_optional(command, "--batch-size", args.batch_size)
    _append_optional(command, "--epochs", args.epochs)
    _append_optional(command, "--device", args.device)
    _append_optional(command, "--threshold-quantile", args.threshold_quantile)
    _append_optional(command, "--variant", args.variant)
    _append_optional(command, "--model-size", args.model_size)
    _append_optional(command, "--n-features", args.n_features)
    _append_optional(command, "--coreset-ratio", args.coreset_ratio)
    _append_optional(command, "--n-neighbors", args.n_neighbors)
    _append_optional(command, "--student-bootstrap-layer", args.student_bootstrap_layer)
    _append_optional(command, "--teacher-weights-path", args.teacher_weights_path)
    _append_optional(command, "--penalty-source-root", args.penalty_source_root)
    _append_optional(command, "--anomaly-source-root", args.anomaly_source_root)
    _append_optional(command, "--config-file", args.config_file)
    _append_optional(command, "--config-json", args.config_json)
    _append_boolean_optional(command, "--pretrained", args.pretrained)
    _append_boolean_optional(command, "--freeze-backbone", args.freeze_backbone)
    _append_boolean_optional(command, "--show-training-progress", args.show_training_progress)
    _append_list(command, "--categories", args.categories)
    _append_list(command, "--ad-layers", args.ad_layers)

    if args.eval_level in {"pixel", "both"}:
        _append_optional(command, "--pixel-threshold", args.pixel_threshold)
        _append_optional(command, "--pixel-threshold-mode", args.pixel_threshold_mode)
        _append_optional(command, "--pixel-threshold-quantile", args.pixel_threshold_quantile)
        _append_boolean_optional(command, "--pixel-skip-missing-masks", args.pixel_skip_missing_masks)

    if args.concept_order:
        _append_list(command, "--concept-order", args.concept_order)
    elif benchmark in args.ordering_files:
        command.extend(["--ordering-file", str(args.ordering_files[benchmark])])
        _append_optional(command, "--ordering-column", args.ordering_column)
    elif args.ordering_mode is not None:
        command.extend(["--ordering-mode", args.ordering_mode])
        if not getattr(args, "inclad_bench", False):
            command.extend(
                [
                    "--ordering-dir",
                    str(Path(args.ordering_dir).expanduser().resolve()),
                    "--ordering-metric",
                    args.ordering_metric,
                    "--ordering-column",
                    args.ordering_column,
                ]
            )
    elif args.shuffle_concepts:
        command.append("--shuffle-concepts")

    return command


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _append_list(command: list[str], flag: str, values: list[object] | None) -> None:
    if not values:
        return
    command.extend([flag, *[str(value) for value in values]])


def _append_boolean_optional(command: list[str], flag: str, value: bool | None) -> None:
    if value is None:
        return
    flag_name = flag.lstrip("-")
    command.append(flag if value else f"--no-{flag_name}")


def configure_batch_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


if __name__ == "__main__":
    try:
        args = parse_args()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    results_root = Path(args.output_root).expanduser().resolve()
    args.output_root = str(results_root)
    batch_log_path = results_root / BATCH_LOG_FILENAME
    configure_batch_logging(batch_log_path)
    logger.info("%s", "═" * 72)
    logger.info("Batch log: %s", batch_log_path)

    required_runners = {CLVAD_CL_RUNNER if strategy == "cl" else CLVAD_RUNNER for strategy in args.strategies}
    missing_runners = [runner for runner in sorted(required_runners) if not runner.is_file()]
    if missing_runners:
        for runner in missing_runners:
            logger.error("Continual visual AD runner not found: %s", runner)
        sys.exit(1)

    datasets_root = Path(args.datasets_root).expanduser().resolve()
    if not datasets_root.is_dir():
        logger.error("Datasets root does not exist: %s", datasets_root)
        sys.exit(1)

    ordering_dir = Path(args.ordering_dir).expanduser().resolve()
    if args.ordering_mode is not None and not getattr(args, "inclad_bench", False) and not ordering_dir.is_dir():
        logger.error("Ordering directory does not exist: %s", ordering_dir)
        sys.exit(1)

    inclad_bench = getattr(args, "inclad_bench", False)

    if args.benchmarks is None:
        benchmarks = discover_benchmarks(datasets_root)
        logger.info("Auto-discovered %d benchmark(s): %s", len(benchmarks), benchmarks)
    else:
        benchmarks = []
        for benchmark in args.benchmarks:
            if benchmark == "inclad-md" and inclad_bench:
                benchmarks.append(benchmark)
                continue
            dataset_dir = resolve_benchmark_dir(datasets_root, benchmark)
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
            if benchmark == "inclad-md" and inclad_bench:
                dataset_path = datasets_root
            else:
                dataset_path = resolve_benchmark_dir(datasets_root, benchmark)
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
        sys.exit(0)

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
            logger.error("  ✗ FAILED (exit code %d) after %.1fs", result.returncode, elapsed)
            failed.append((model, benchmark, strategy, result.returncode))
        else:
            logger.info("  ✓ done in %.1fs", elapsed)

        logger.info("")

    if args.generate_heatmaps:
        logger.info("%s", "═" * 72)
        if failed:
            logger.info("Some jobs failed; generating heatmaps for any results that were written.")
        logger.info("Generating heatmaps under: %s", results_root)
        heatmap_options = HeatmapGenerationOptions(
            annotate=args.heatmap_annotate,
            figsize=tuple(args.heatmap_figsize),
            ignore_upper_diagonal=not args.heatmap_show_upper_diagonal,
            overwrite=args.overwrite_heatmaps,
            dry_run=False,
        )
        heatmaps_written = generate_heatmaps_for_results_dir(results_root, heatmap_options)
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
