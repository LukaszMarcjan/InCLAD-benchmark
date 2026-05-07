"""
Continual visual anomaly detection runner with add-on CL strategies for visual models.

This runner keeps the existing pyCLAD core APIs intact and adds a separate path for
visual continual-learning methods such as PaDiM-CL, PatchCore-CL, and CFA with
hybrid replay.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import pathlib
import sys
import time
from datetime import datetime

from pyclad.callbacks.callback import Callback
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.models.model import Model
from pyclad.output.output_writer import InfoProvider
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario

logger = logging.getLogger(__name__)


def _load_local_module(filename: str, module_name: str):
    module_path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_local_module("run_continual_visual_ad.py", "pyclad_visual_runner_base_cl")
_LEVELS = _load_local_module("run_continual_visual_ad_levels.py", "pyclad_visual_runner_levels_cl")
_CL_FACTORY = _load_local_module("cl_factory.py", "pyclad_visual_runner_cl_factory")

build_cl_model = _CL_FACTORY.build_model
build_cl_strategy = _CL_FACTORY.build_strategy
resolve_continual_memory_bank_size = _CL_FACTORY.resolve_continual_memory_bank_size
validate_strategy_choice = _CL_FACTORY.validate_strategy_choice
build_eval_callbacks = _LEVELS.build_callbacks
pixel_output_suffix = _LEVELS._pixel_output_suffix
validate_eval_level = _LEVELS._validate_eval_level

MODEL_SPECS = _BASE.MODEL_SPECS
ModelSpec = _BASE.ModelSpec
RunConfigProvider = _BASE.RunConfigProvider
RunStatusProvider = _BASE.RunStatusProvider
JsonCheckpointCallback = _BASE.JsonCheckpointCallback
ScenarioProgressCallback = _BASE.ScenarioProgressCallback
collect_result_providers = _BASE.collect_result_providers
write_result_snapshot = _BASE.write_result_snapshot
set_global_seeds = _BASE.set_global_seeds
seed_log_handler = _BASE.seed_log_handler
resolve_fixed_concept_order = _BASE.resolve_fixed_concept_order
reorder_dataset = _BASE.reorder_dataset
shuffle_dataset = _BASE.shuffle_dataset
create_run_config_payload = _BASE.create_run_config_payload
generate_seeds = _BASE.generate_seeds
log_dataset_overview = _BASE.log_dataset_overview
read_registered_visual_benchmark_dataset = _BASE.read_registered_visual_benchmark_dataset
resolve_visual_benchmark_root = _BASE.resolve_visual_benchmark_root
replay_buffer_file_suffix = _BASE.replay_buffer_file_suffix
ordering_output_suffix = _BASE.ordering_output_suffix
resolve_replay_buffer_size = _BASE.resolve_replay_buffer_size
traceback = _BASE.traceback


def default_results_root(model: str, strategy: str) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return pathlib.Path(f"results_{model}_{strategy}_{timestamp}")


def resolve_output_dir(args: argparse.Namespace) -> pathlib.Path:
    if args.output_dir is not None:
        return pathlib.Path(args.output_dir)
    return default_results_root(args.model, args.strategy)


def parse_args() -> argparse.Namespace:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--strategy",
        type=str,
        choices=["naive", "cumulative", "replay", "ste", "cl"],
        default=None,
        help="Continual-learning strategy",
    )
    extra_parser.add_argument(
        "--continual-memory-bank-size",
        type=int,
        default=None,
        help="Explicit PatchCore CL memory-bank budget. Defaults to the replay-buffer-derived size.",
    )
    extra_parser.add_argument(
        "--eval-level",
        type=str,
        choices=["image", "pixel", "both"],
        default="image",
        help="Evaluation level: 'image' (ROC-AUC + F1 image-level, default), "
             "'pixel' (pixel ROC-AUC + pixel F1 only), or 'both'.",
    )
    extra_parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=0.5,
        help="Fixed threshold applied to anomaly maps for pixel F1, Dice, and IoU (default: 0.5).",
    )
    extra_parser.add_argument(
        "--pixel-threshold-mode",
        type=str,
        choices=["fixed", "train-quantile"],
        default="fixed",
        help="How to determine the pixel-level threshold (default: fixed).",
    )
    extra_parser.add_argument(
        "--pixel-threshold-quantile",
        type=float,
        default=0.995,
        help="Quantile for pixel threshold when --pixel-threshold-mode=train-quantile (default: 0.995).",
    )
    extra_parser.add_argument(
        "--pixel-skip-missing-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip anomalous test samples that do not provide a pixel-level mask (default: True).",
    )
    extra_args, remaining = extra_parser.parse_known_args()

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _BASE.parse_args()
    finally:
        sys.argv = original_argv

    if extra_args.strategy is not None:
        args.strategy = extra_args.strategy
    args.continual_memory_bank_size = extra_args.continual_memory_bank_size
    args.eval_level = extra_args.eval_level
    args.pixel_threshold = extra_args.pixel_threshold
    args.pixel_threshold_mode = extra_args.pixel_threshold_mode
    args.pixel_threshold_quantile = extra_args.pixel_threshold_quantile
    args.pixel_skip_missing_masks = extra_args.pixel_skip_missing_masks
    validate_strategy_choice(args.model, args.strategy)
    if not 0.0 < float(args.pixel_threshold_quantile) < 1.0:
        raise SystemExit(
            f"--pixel-threshold-quantile must be in (0, 1), got {args.pixel_threshold_quantile}"
        )
    return args


def build_model(args: argparse.Namespace, seed: int) -> tuple[Model, object]:
    return build_cl_model(_BASE, args, seed)


def build_strategy(
    args: argparse.Namespace,
    model: Model,
    model_creation_fn,
    replay_buffer_size: int,
    run_dataset: ConceptsDataset,
):
    return build_cl_strategy(_BASE, args, model, model_creation_fn, replay_buffer_size, run_dataset)


def run_single(
    args: argparse.Namespace,
    seed: int,
    run_index: int,
    total_runs: int,
    dataset: ConceptsDataset,
    output_dir: pathlib.Path,
    dataset_root: pathlib.Path | None = None,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordering_suffix = ordering_output_suffix(args)
    replay_suffix = replay_buffer_file_suffix(args)
    pixel_suffix = pixel_output_suffix(args)
    output_path = (
        output_dir
        / f"{args.model}_{args.strategy}_{args.eval_level}{pixel_suffix}{ordering_suffix}{replay_suffix}_seed_{seed}.json"
    )
    log_path = (
        output_dir
        / f"{args.model}_{args.strategy}_{args.eval_level}{pixel_suffix}{ordering_suffix}{replay_suffix}_seed_{seed}.log"
    )

    with seed_log_handler(log_path):
        run_status = RunStatusProvider(output_path=output_path, log_path=log_path)
        run_payload = create_run_config_payload(
            args=args,
            seed=seed,
            output_path=output_path,
            log_path=log_path,
        )
        run_payload["eval_level"] = args.eval_level
        run_payload["pixel_skip_missing_masks"] = args.pixel_skip_missing_masks
        run_payload["pixel_threshold"] = args.pixel_threshold
        run_payload["pixel_threshold_mode"] = args.pixel_threshold_mode
        run_payload["pixel_threshold_quantile"] = args.pixel_threshold_quantile
        if args.continual_memory_bank_size is not None:
            run_payload["continual_memory_bank_size"] = int(args.continual_memory_bank_size)
        run_provider = RunConfigProvider(run_payload)

        model: Model | None = None
        config = None
        run_dataset: ConceptsDataset | None = None
        strategy = None
        callbacks: list[Callback] = []

        def write_snapshot() -> None:
            write_result_snapshot(
                output_path=output_path,
                providers=collect_result_providers(
                    model=model,
                    run_dataset=run_dataset,
                    strategy=strategy,
                    callbacks=callbacks,
                    extra_providers=[run_provider, run_status],
                ),
                completed_concepts=run_status.completed_concepts(),
            )

        logger.info("═══ Run %d/%d  seed=%d ═══", run_index + 1, total_runs, seed)
        set_global_seeds(seed)

        try:
            logger.info("  [1/5] Resolving concept order and run dataset")
            fixed_order, ordering_metadata = resolve_fixed_concept_order(args, dataset)
            if fixed_order is not None:
                run_dataset = reorder_dataset(dataset, fixed_order)
                logger.info(
                    "  Concept order source: %s (%s)",
                    ordering_metadata["source"],
                    ordering_metadata.get("ordering_file", "inline"),
                )
            elif args.shuffle_concepts:
                run_dataset = shuffle_dataset(dataset, seed)
                ordering_metadata = {"source": "shuffle", "seed": seed}
                logger.info("  Concept order source: shuffle (seed=%d)", seed)
            else:
                run_dataset = dataset
                ordering_metadata = {"source": "dataset"}
                logger.info("  Concept order source: dataset")

            run_status.set_total_concepts(len(run_dataset.train_concepts()))

            if ordering_metadata.get("skipped_concepts"):
                logger.warning("  Skipped concepts not present in dataset: %s", ordering_metadata["skipped_concepts"])
            logger.info("  Concept order: %s", [concept.name for concept in run_dataset.train_concepts()])
            log_dataset_overview(run_dataset, "  Run dataset order:")

            logger.info("  [2/5] Building model and effective configuration")
            model_factory = lambda: build_model(args, seed)[0]
            model, config = build_model(args, seed)
            validate_eval_level(args, model)
            run_payload.update(
                create_run_config_payload(
                    args=args,
                    seed=seed,
                    output_path=output_path,
                    log_path=log_path,
                    config=config,
                    run_dataset=run_dataset,
                    ordering_metadata=ordering_metadata,
                )
            )
            if dataset_root is not None:
                run_payload["resolved_root"] = str(dataset_root)
            logger.info("  Model device: %s", getattr(model, "_device", "unknown"))
            if hasattr(config, "show_training_progress"):
                logger.info("  Model-native training progress bars: %s", config.show_training_progress)
            else:
                logger.info("  Model-native training progress bars: not supported by this model")
            if hasattr(config, "batch_size"):
                logger.info("  Effective batch size: %s", getattr(config, "batch_size"))
            if hasattr(config, "epochs"):
                logger.info("  Effective epochs: %s", getattr(config, "epochs"))

            if args.strategy == "cl" and args.model == "patchcore":
                run_payload["patchcore_previous_ratio"] = float(getattr(args, "patchcore_previous_ratio", 0.25))
                logger.info("  PatchCore CL previous_ratio: %s", run_payload["patchcore_previous_ratio"])

            logger.info("  [3/5] Building strategy and callbacks")
            replay_buffer_size = 0
            if args.strategy == "replay" or (args.model == "cfa" and args.strategy == "cl"):
                replay_buffer_size = resolve_replay_buffer_size(args, run_dataset)
                run_payload["replay_buffer_fraction"] = args.replay_buffer_fraction
                run_payload["replay_buffer_size"] = replay_buffer_size

            strategy = build_strategy(
                args=args,
                model=model,
                model_creation_fn=model_factory,
                replay_buffer_size=replay_buffer_size,
                run_dataset=run_dataset,
            )
            callbacks = build_eval_callbacks(
                args=args,
                dataset=run_dataset,
                strategy=strategy,
                dataset_root=dataset_root,
            )

            callbacks.append(
                JsonCheckpointCallback(
                    output_path=output_path,
                    run_status_provider=run_status,
                    write_snapshot=write_snapshot,
                )
            )

            scenario = ConceptIncrementalScenario(
                dataset=run_dataset,
                strategy=strategy,
                callbacks=callbacks,
            )

            logger.info("  [4/5] Running scenario")
            started_at = time.perf_counter()
            scenario.run()
            elapsed = time.perf_counter() - started_at
        except BaseException as exc:
            run_status.mark_failed(exc, traceback.format_exc())
            logger.exception("  Run did not finish cleanly; attempting to save partial results to %s", output_path)
            try:
                write_snapshot()
            except Exception:
                logger.exception("  Failed to save partial results to %s", output_path)
            raise

        logger.info("  [5/5] Saving results")
        run_status.mark_completed(elapsed_seconds=elapsed)
        skipped_callbacks = [type(callback).__name__ for callback in callbacks if not isinstance(callback, InfoProvider)]
        if skipped_callbacks:
            logger.info("    Skipping non-serializable callbacks in JSON output: %s", skipped_callbacks)
        write_snapshot()

        logger.info("    ✓ finished in %.1fs → %s", elapsed, output_path)

    return output_path


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("pyclad.scenarios.concept_incremental").setLevel(logging.WARNING)

    if args.strategy == "ste" and args.n_runs > 1:
        logger.warning(
            "Strategy 'ste' resets the model for each concept, so extra seeds usually only change initialization "
            "and optionally concept order. For a single curriculum pass you will typically want --n-runs 1."
        )

    output_dir = resolve_output_dir(args)
    seeds = generate_seeds(args.master_seed, args.n_runs)
    logger.info("Master seed: %d  →  per-run seeds: %s", args.master_seed, seeds)
    logger.info("Results directory: %s", output_dir)
    logger.info("[1/2] Loading dataset")

    spec = MODEL_SPECS[args.model]
    resize = int(args.resize) if args.resize is not None else spec.default_resize
    resize_to = (resize, resize)
    logger.info(
        "Dataset options: categories=%s, resize=%s, max_train=%s, max_test=%s, eval_level=%s",
        args.categories if args.categories is not None else "all",
        resize_to,
        args.max_train,
        args.max_test,
        args.eval_level,
    )
    if args.strategy == "replay" or (args.model == "cfa" and args.strategy == "cl"):
        logger.info("Replay buffer fraction: %.2f", args.replay_buffer_fraction)
    if args.strategy == "cl" and args.model == "patchcore" and args.continual_memory_bank_size is not None:
        logger.info("PatchCore CL memory budget: %d", args.continual_memory_bank_size)
    if args.eval_level in {"pixel", "both"}:
        logger.info(
            "Pixel-level options: threshold_mode=%s, threshold=%.3f, quantile=%.3f, skip_missing_masks=%s",
            args.pixel_threshold_mode,
            args.pixel_threshold,
            args.pixel_threshold_quantile,
            args.pixel_skip_missing_masks,
        )

    dataset_load_started_at = time.perf_counter()

    if getattr(args, "inclad_bench", False):
        from pyclad.data.datasets.inclad_bench_dataset import InCLADBenchDataset

        inclad_ordering = args.ordering_mode if args.ordering_mode else "easy_to_hard"
        logger.info(
            "Loading InCLAD-Bench manifest: benchmark=%s, ordering=%s",
            args.benchmark,
            inclad_ordering,
        )
        dataset = InCLADBenchDataset(
            benchmark=args.benchmark,
            ordering=inclad_ordering,
            root=args.root,
            registry_path=args.registry_path,
            categories=args.categories,
            data_mode="numpy",
            resize_to=resize_to,
            color_mode="rgb",
            max_train_samples_per_category=args.max_train,
            max_test_samples_per_category=args.max_test,
        )
    else:
        dataset_root = resolve_visual_benchmark_root(
            benchmark=args.benchmark,
            root=args.root,
            registry_path=args.registry_path,
        )
        logger.info("Dataset root: %s", dataset_root)
        dataset = read_registered_visual_benchmark_dataset(
            benchmark=args.benchmark,
            root=dataset_root,
            registry_path=args.registry_path,
            categories=args.categories,
            data_mode="numpy",
            resize_to=resize_to,
            color_mode="rgb",
            max_train_samples_per_category=args.max_train,
            max_test_samples_per_category=args.max_test,
        )
    dataset_load_elapsed = time.perf_counter() - dataset_load_started_at
    logger.info(
        "Dataset loaded in %.1fs: %s (%d train concepts, %d test concepts)",
        dataset_load_elapsed,
        dataset.name(),
        len(dataset.train_concepts()),
        len(dataset.test_concepts()),
    )
    log_dataset_overview(dataset, "Loaded dataset order:")

    logger.info("[2/2] Starting experiment runs")
    saved_paths: list[pathlib.Path] = []
    for run_index, seed in enumerate(seeds):
        path = run_single(
            args=args,
            seed=seed,
            run_index=run_index,
            total_runs=len(seeds),
            dataset=dataset,
            output_dir=output_dir,
            dataset_root=dataset_root,
        )
        saved_paths.append(path)

    logger.info("All %d runs complete.", len(saved_paths))
    for path in saved_paths:
        logger.info("  %s", path)


if __name__ == "__main__":
    main()
