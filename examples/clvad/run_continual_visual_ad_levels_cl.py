"""
Visual continual anomaly-detection runner with image-level and pixel-level evaluation
for the add-on CL visual strategies.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import pathlib
import sys
import time
from typing import Any

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.callbacks.evaluation.visual_pixel_concept_metric_callback import (
    VisualPixelConceptMetricCallback,
)
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.pixel_dice_score import PixelDiceScore
from pyclad.metrics.base.pixel_f1_score import PixelF1Score
from pyclad.metrics.base.pixel_iou import PixelIoU
from pyclad.metrics.base.pixel_roc_auc import PixelRocAuc
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.diagonal_average import DiagonalAverage
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.output.output_writer import InfoProvider
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario

logger = logging.getLogger(__name__)

PIXEL_LEVEL_MODELS = {
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
}


def _load_base_runner():
    module_path = pathlib.Path(__file__).with_name("run_continual_visual_ad_cl.py")
    spec = importlib.util.spec_from_file_location("pyclad_visual_runner_levels_cl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CL visual runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_runner()


def parse_args() -> argparse.Namespace:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--eval-level",
        choices=["image", "pixel", "both"],
        default="image",
        help="Evaluation granularity for visual anomaly detection",
    )
    extra_parser.add_argument(
        "--pixel-skip-missing-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip anomalous test samples that do not provide a pixel-level mask",
    )
    extra_parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to anomaly maps for pixel F1, Dice, and IoU",
    )
    extra_parser.add_argument(
        "--pixel-threshold-mode",
        choices=["fixed", "train-quantile"],
        default="fixed",
        help="Threshold selection strategy for thresholded pixel metrics",
    )
    extra_parser.add_argument(
        "--pixel-threshold-quantile",
        type=float,
        default=0.995,
        help="Quantile of seen normal train pixel scores used when --pixel-threshold-mode=train-quantile",
    )

    extra_args, remaining = extra_parser.parse_known_args()
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _BASE.parse_args()
    finally:
        sys.argv = original_argv

    args.eval_level = extra_args.eval_level
    args.pixel_skip_missing_masks = extra_args.pixel_skip_missing_masks
    args.pixel_threshold = extra_args.pixel_threshold
    args.pixel_threshold_mode = extra_args.pixel_threshold_mode
    args.pixel_threshold_quantile = extra_args.pixel_threshold_quantile
    if not 0.0 < float(args.pixel_threshold_quantile) < 1.0:
        raise SystemExit(
            f"--pixel-threshold-quantile must be in (0, 1), got {args.pixel_threshold_quantile}"
        )
    return args


def build_callbacks(
    args: argparse.Namespace,
    dataset,
    strategy,
    dataset_root: pathlib.Path,
) -> list[Callback]:
    continual_metrics = [ContinualAverage(), BackwardTransfer(), ForwardTransfer(), DiagonalAverage()]
    callbacks: list[Callback] = [_BASE.ScenarioProgressCallback(dataset)]

    if args.eval_level in {"image", "both"}:
        callbacks.extend(
            [
                ConceptMetricCallback(base_metric=RocAuc(), metrics=continual_metrics),
                ConceptMetricCallback(base_metric=F1Score(), metrics=continual_metrics),
            ]
        )

    if args.eval_level in {"pixel", "both"}:
        callbacks.append(
            VisualPixelConceptMetricCallback(
                strategy=strategy,
                benchmark=args.benchmark,
                root=dataset_root,
                registry_path=args.registry_path,
                categories=args.categories,
                max_test_samples_per_category=args.max_test,
                base_metrics=[
                    PixelRocAuc(),
                    PixelF1Score(threshold=args.pixel_threshold),
                    PixelDiceScore(threshold=args.pixel_threshold),
                    PixelIoU(threshold=args.pixel_threshold),
                ],
                metrics=continual_metrics,
                skip_missing_anomaly_masks=args.pixel_skip_missing_masks,
                threshold_mode=args.pixel_threshold_mode,
                fixed_threshold=args.pixel_threshold,
                threshold_quantile=args.pixel_threshold_quantile,
            )
        )

    callbacks.extend([TimeEvaluationCallback(), MemoryUsageCallback()])
    return callbacks


def _validate_eval_level(args: argparse.Namespace, model: Any) -> None:
    if args.eval_level not in {"pixel", "both"}:
        return

    score_maps_fn = getattr(model, "score_maps", None)
    if args.model not in PIXEL_LEVEL_MODELS or not callable(score_maps_fn):
        supported = ", ".join(sorted(PIXEL_LEVEL_MODELS))
        raise ValueError(
            f"Model '{args.model}' does not support pixel-level evaluation in this runner. "
            f"Supported models: {supported}"
        )


def _file_safe_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _pixel_output_suffix(args: argparse.Namespace) -> str:
    if args.eval_level not in {"pixel", "both"}:
        return ""

    if args.pixel_threshold_mode == "fixed":
        return f"_{args.pixel_threshold_mode}-{_file_safe_float(args.pixel_threshold)}"
    return f"_{args.pixel_threshold_mode}-{_file_safe_float(args.pixel_threshold_quantile)}"


def run_single(
    args: argparse.Namespace,
    seed: int,
    run_index: int,
    total_runs: int,
    dataset,
    output_dir: pathlib.Path,
    dataset_root: pathlib.Path,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordering_suffix = _BASE.ordering_output_suffix(args)
    replay_suffix = _BASE.replay_buffer_file_suffix(args)
    pixel_suffix = _pixel_output_suffix(args)
    output_path = (
        output_dir
        / f"{args.model}_{args.strategy}_{args.eval_level}{pixel_suffix}{ordering_suffix}{replay_suffix}_seed_{seed}.json"
    )
    log_path = (
        output_dir
        / f"{args.model}_{args.strategy}_{args.eval_level}{pixel_suffix}{ordering_suffix}{replay_suffix}_seed_{seed}.log"
    )

    with _BASE.seed_log_handler(log_path):
        run_status = _BASE.RunStatusProvider(output_path=output_path, log_path=log_path)
        run_payload = _BASE.create_run_config_payload(
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
        run_provider = _BASE.RunConfigProvider(run_payload)

        model = None
        config = None
        run_dataset = None
        strategy = None
        callbacks: list[Callback] = []

        def write_snapshot() -> None:
            _BASE.write_result_snapshot(
                output_path=output_path,
                providers=_BASE.collect_result_providers(
                    model=model,
                    run_dataset=run_dataset,
                    strategy=strategy,
                    callbacks=callbacks,
                    extra_providers=[run_provider, run_status],
                ),
                completed_concepts=run_status.completed_concepts(),
            )

        logger.info("═══ Run %d/%d  seed=%d ═══", run_index + 1, total_runs, seed)
        _BASE.set_global_seeds(seed)

        try:
            logger.info("  [1/5] Resolving concept order and run dataset")
            fixed_order, ordering_metadata = _BASE.resolve_fixed_concept_order(args, dataset)
            if fixed_order is not None:
                run_dataset = _BASE.reorder_dataset(dataset, fixed_order)
                logger.info(
                    "  Concept order source: %s (%s)",
                    ordering_metadata["source"],
                    ordering_metadata.get("ordering_file", "inline"),
                )
            elif args.shuffle_concepts:
                run_dataset = _BASE.shuffle_dataset(dataset, seed)
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
            _BASE.log_dataset_overview(run_dataset, "  Run dataset order:")

            logger.info("  [2/5] Building model and effective configuration")
            model_factory = lambda: _BASE.build_model(args, seed)[0]
            model, config = _BASE.build_model(args, seed)
            _validate_eval_level(args, model)
            run_payload.update(
                _BASE.create_run_config_payload(
                    args=args,
                    seed=seed,
                    output_path=output_path,
                    log_path=log_path,
                    config=config,
                    run_dataset=run_dataset,
                    ordering_metadata=ordering_metadata,
                )
            )
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

            logger.info("  [3/5] Building strategy and callbacks")
            replay_buffer_size = 0
            if args.strategy == "replay" or (args.model == "cfa" and args.strategy == "cl"):
                replay_buffer_size = _BASE.resolve_replay_buffer_size(args, run_dataset)
                run_payload["replay_buffer_fraction"] = args.replay_buffer_fraction
                run_payload["replay_buffer_size"] = replay_buffer_size
            strategy = _BASE.build_strategy(
                args=args,
                model=model,
                model_creation_fn=model_factory,
                replay_buffer_size=replay_buffer_size,
                run_dataset=run_dataset,
            )
            callbacks = build_callbacks(args=args, dataset=run_dataset, strategy=strategy, dataset_root=dataset_root)
            callbacks.append(
                _BASE.JsonCheckpointCallback(
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
            run_status.mark_failed(exc, _BASE.traceback.format_exc())
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

    output_dir = _BASE.resolve_output_dir(args)
    seeds = _BASE.generate_seeds(args.master_seed, args.n_runs)
    logger.info("Master seed: %d  →  per-run seeds: %s", args.master_seed, seeds)
    logger.info("Results directory: %s", output_dir)
    logger.info("[1/2] Loading dataset")

    spec = _BASE.MODEL_SPECS[args.model]
    resize = int(args.resize) if args.resize is not None else spec.default_resize
    resize_to = (resize, resize)
    dataset_root = _BASE.resolve_visual_benchmark_root(
        benchmark=args.benchmark,
        root=args.root,
        registry_path=args.registry_path,
    )
    logger.info("Dataset root: %s", dataset_root)
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
    if args.eval_level in {"pixel", "both"}:
        logger.info(
            "Pixel-level options: threshold_mode=%s, threshold=%s, threshold_quantile=%s, skip_missing_masks=%s",
            args.pixel_threshold_mode,
            args.pixel_threshold,
            args.pixel_threshold_quantile,
            args.pixel_skip_missing_masks,
        )

    dataset_load_started_at = time.perf_counter()
    if getattr(args, "inclad_bench", False):
        inclad_ordering = args.ordering_mode if args.ordering_mode else "easy_to_hard"
        if args.benchmark == "inclad-md":
            import os
            from pyclad.data.datasets.inclad_md_dataset import InCLADMDDataset
            logger.info("Loading InCLAD-MD manifest: ordering=%s", inclad_ordering)
            os.environ["PYCLAD_VISUAL_DATASETS_ROOT"] = str(dataset_root)
            dataset = InCLADMDDataset(
                ordering=inclad_ordering,
                registry_path=args.registry_path,
                categories=args.categories,
                data_mode="numpy",
                resize_to=resize_to,
                color_mode="rgb",
                max_train_samples_per_category=args.max_train,
                max_test_samples_per_category=args.max_test,
            )
        else:
            from pyclad.data.datasets.inclad_bench_dataset import InCLADBenchDataset
            logger.info(
                "Loading InCLAD-Bench manifest: benchmark=%s, ordering=%s",
                args.benchmark,
                inclad_ordering,
            )
            dataset = InCLADBenchDataset(
                benchmark=args.benchmark,
                ordering=inclad_ordering,
                root=dataset_root,
                registry_path=args.registry_path,
                categories=args.categories,
                data_mode="numpy",
                resize_to=resize_to,
                color_mode="rgb",
                max_train_samples_per_category=args.max_train,
                max_test_samples_per_category=args.max_test,
            )
    else:
        dataset = _BASE.read_registered_visual_benchmark_dataset(
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
    _BASE.log_dataset_overview(dataset, "Loaded dataset order:")

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
