"""Step-scheduled visual continual AD runner with schedule-aware metrics.

Runs grouped-component scenarios, in which one training step can introduce
several categories, with schedule-aware callbacks and metrics:

* RectangularConceptMetricCallbackWithMask (image-level)
* RectangularVisualPixelConceptMetricCallbackWithMask (pixel-level)
* RectangularForgettingMeasure / ForwardTransfer / NewTaskAcquisition

These metrics know which training step first introduced each test
category and avoid the bias the naive ``ForgettingMeasure`` exhibits on
rectangular T x N matrices (T < N), where unseen-future columns dilute
the forgetting average.

Example
-------
MVTec 14-1 with PaDiM naive (small smoke run)::

    python examples/clvad/run_continual_visual_ad_step_schedule.py \\
        --model padim --strategy naive \\
        --benchmark mvtec --root /path/to/mvtec_ad \\
        --step-schedule 14-1 --eval-level both \\
        --max-train 5 --max-test 3 --resize 64
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
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.rectangular_concept_metric_with_mask import (
    RectangularConceptMetricCallbackWithMask,
)
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.visual_step_schedule import (
    apply_step_schedule,
    compute_first_seen_step,
    parse_step_schedule,
)
from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.final_step_average import FinalStepAverage
from pyclad.metrics.continual.rectangular_forgetting_measure import (
    RectangularForgettingMeasure,
)
from pyclad.metrics.continual.rectangular_forward_transfer import (
    RectangularForwardTransfer,
)
from pyclad.metrics.continual.rectangular_new_task_acquisition import (
    RectangularNewTaskAcquisition,
)
from pyclad.output.output_writer import InfoProvider
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario

logger = logging.getLogger(__name__)


def _load_module(filename: str, alias: str):
    module_path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_LEVELS = _load_module("run_continual_visual_ad_levels.py", "pyclad_visual_levels_runner_for_step_schedule")
_BASE = _LEVELS._BASE
_CL_FACTORY = _load_module("cl_factory.py", "pyclad_visual_step_schedule_cl_factory")


def parse_args() -> argparse.Namespace:
    help_requested = any(argument in {"-h", "--help"} for argument in sys.argv[1:])
    extra_parser = argparse.ArgumentParser(add_help=False)
    # Intercept --strategy with an extended set of choices (including 'cl')
    # because the underlying _LEVELS / _BASE parsers only know
    # {naive, cumulative, replay, ste}. Without this, `--strategy cl` would
    # fail argparse validation before our parse_args() finishes.
    extra_parser.add_argument(
        "--strategy",
        type=str,
        choices=["naive", "cumulative", "replay", "ste", "cl", "ucad"],
        default=None,
        help="Continual-learning strategy (extends the base choices with 'cl').",
    )
    extra_parser.add_argument(
        "--step-schedule",
        type=str,
        required=not help_requested,
        help="CDAD-style step schedule, e.g. '14-1', '10-5', '3x5', '10-1x5', '8-1x4'.",
    )
    extra_parser.add_argument(
        "--group-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Group test concepts using the same schedule as training "
        "(default: keep per-category test for CDAD-style A-AUROC reporting).",
    )
    extra_parser.add_argument(
        "--continual-memory-bank-size",
        type=int,
        default=None,
        help="Recorded in run_config only; it does not change PatchCore CL, whose history budget is "
        "previous_ratio x |current-task coreset| with previous_ratio fixed at 0.25.",
    )

    extra_args, remaining = extra_parser.parse_known_args()
    if help_requested:
        print("Step-schedule options:")
        print("  --strategy {naive,cumulative,replay,ste,cl,ucad}  CL strategy (adds 'cl' and UCAD)")
        print("  --step-schedule SPEC              CDAD-style schedule string (e.g. '10-1x5')")
        print("  --[no-]group-test                 Whether to also group test concepts per schedule")
        print()

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _LEVELS.parse_args()
    finally:
        sys.argv = original_argv

    if extra_args.step_schedule is None:
        raise SystemExit("--step-schedule is required (e.g. '14-1', '10-1x5', '8-1x4').")

    if extra_args.strategy is not None:
        args.strategy = extra_args.strategy

    args.step_schedule = extra_args.step_schedule
    args.group_test = bool(extra_args.group_test)
    args.continual_memory_bank_size = extra_args.continual_memory_bank_size

    if args.group_test:
        raise SystemExit(
            "This runner only supports per-category test evaluation "
            "(rectangular T x N)."
        )

    try:
        _CL_FACTORY.validate_strategy_choice(args.model, args.strategy)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        parse_step_schedule(args.step_schedule)
    except ValueError as exc:
        raise SystemExit(f"Invalid --step-schedule {args.step_schedule!r}: {exc}") from exc

    return args


def build_callbacks(
    args: argparse.Namespace,
    dataset: ConceptsDataset,
    strategy,
    dataset_root: pathlib.Path,
    first_seen_step: dict,
) -> list[Callback]:
    legacy_metrics = [FinalStepAverage()]
    schedule_aware_metrics = [
        RectangularForgettingMeasure(),
        RectangularForwardTransfer(),
        RectangularNewTaskAcquisition(),
    ]

    callbacks: list[Callback] = [_BASE.ScenarioProgressCallback(dataset)]

    for base_metric in [RocAuc(), F1Score(), AveragePrecision()]:
        callbacks.append(
            RectangularConceptMetricCallbackWithMask(
                base_metric=base_metric,
                metrics=legacy_metrics,
                schedule_aware_metrics=schedule_aware_metrics,
                first_seen_step=first_seen_step,
            )
        )

    if args.eval_level in {"pixel", "both"}:
        from pyclad.callbacks.evaluation.rectangular_visual_pixel_concept_metric_callback_with_mask import (
            RectangularVisualPixelConceptMetricCallbackWithMask,
        )
        from pyclad.metrics.base.pixel_aupro import PixelAUPRO
        from pyclad.metrics.base.pixel_average_precision import PixelAveragePrecision
        from pyclad.metrics.base.pixel_dice_score import PixelDiceScore
        from pyclad.metrics.base.pixel_f1_score import PixelF1Score
        from pyclad.metrics.base.pixel_iou import PixelIoU
        from pyclad.metrics.base.pixel_roc_auc import PixelRocAuc

        callbacks.append(
            RectangularVisualPixelConceptMetricCallbackWithMask(
                strategy=strategy,
                benchmark=args.benchmark,
                root=dataset_root,
                registry_path=args.registry_path,
                categories=args.categories,
                max_test_samples_per_category=args.max_test,
                base_metrics=[
                    PixelRocAuc(),
                    PixelAveragePrecision(),
                    PixelAUPRO(),
                    PixelF1Score(threshold=args.pixel_threshold),
                    PixelDiceScore(threshold=args.pixel_threshold),
                    PixelIoU(threshold=args.pixel_threshold),
                ],
                metrics=legacy_metrics,
                schedule_aware_metrics=schedule_aware_metrics,
                first_seen_step=first_seen_step,
                skip_missing_anomaly_masks=args.pixel_skip_missing_masks,
                threshold_mode=args.pixel_threshold_mode,
                fixed_threshold=args.pixel_threshold,
                threshold_quantile=args.pixel_threshold_quantile,
            )
        )

    callbacks.extend([TimeEvaluationCallback(), MemoryUsageCallback()])
    return callbacks


def _schedule_output_suffix(args: argparse.Namespace) -> str:
    safe = args.step_schedule.replace("×", "x").replace("*", "x")
    return f"_schd-{safe}-perCat"


def run_single(
    args: argparse.Namespace,
    seed: int,
    run_index: int,
    total_runs: int,
    dataset: ConceptsDataset,
    output_dir: pathlib.Path,
    dataset_root: pathlib.Path,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordering_suffix = _BASE.ordering_output_suffix(args)
    replay_suffix = _BASE.replay_buffer_file_suffix(args)
    pixel_suffix = _LEVELS._pixel_output_suffix(args)
    schedule_suffix = _schedule_output_suffix(args)
    output_path = (
        output_dir
        / f"{args.model}_{args.strategy}_{args.eval_level}{pixel_suffix}{ordering_suffix}{replay_suffix}{schedule_suffix}_seed_{seed}.json"
    )
    log_path = (
        output_dir
        / f"{args.model}_{args.strategy}_{args.eval_level}{pixel_suffix}{ordering_suffix}{replay_suffix}{schedule_suffix}_seed_{seed}.log"
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
        run_payload["step_schedule"] = args.step_schedule
        run_payload["step_schedule_parsed"] = parse_step_schedule(args.step_schedule)
        run_payload["group_test"] = args.group_test
        run_payload["runner_variant"] = "step_schedule"
        run_provider = _BASE.RunConfigProvider(run_payload)

        model: Any = None
        config: Any = None
        run_dataset: ConceptsDataset | None = None
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
            logger.info("  [1/5] Resolving concept order and step schedule")
            fixed_order, ordering_metadata = _BASE.resolve_fixed_concept_order(args, dataset)
            if fixed_order is not None:
                ordered = _BASE.reorder_dataset(dataset, fixed_order)
                logger.info(
                    "  Concept order source: %s (%s)",
                    ordering_metadata["source"],
                    ordering_metadata.get("ordering_file", "inline"),
                )
            elif args.shuffle_concepts:
                ordered = _BASE.shuffle_dataset(dataset, seed)
                ordering_metadata = {"source": "shuffle", "seed": seed}
                logger.info("  Concept order source: shuffle (seed=%d)", seed)
            else:
                ordered = dataset
                ordering_metadata = {"source": "dataset"}
                logger.info("  Concept order source: dataset")

            ordered_train_names = [c.name for c in ordered.train_concepts()]
            logger.info("  Concept order (before grouping): %s", ordered_train_names)

            run_dataset = apply_step_schedule(
                dataset=ordered,
                schedule=args.step_schedule,
                group_test=False,
                name_mode="indexed",
            )
            first_seen_step = dict(
                zip(
                    ordered_train_names,
                    compute_first_seen_step(ordered_train_names, args.step_schedule),
                )
            )

            schedule_metadata = {
                "step_schedule": args.step_schedule,
                "step_schedule_parsed": parse_step_schedule(args.step_schedule),
                "group_test": False,
                "train_steps": [c.name for c in run_dataset.train_concepts()],
                "test_concepts": [c.name for c in run_dataset.test_concepts()],
                "first_seen_step": first_seen_step,
            }
            run_status.set_total_concepts(len(run_dataset.train_concepts()))

            if ordering_metadata.get("skipped_concepts"):
                logger.warning(
                    "  Skipped concepts not present in dataset: %s",
                    ordering_metadata["skipped_concepts"],
                )
            logger.info(
                "  Train steps after schedule: %d  (sizes: %s)",
                len(run_dataset.train_concepts()),
                schedule_metadata["step_schedule_parsed"],
            )
            logger.info(
                "  Test concepts after schedule: %d  (per-category eval)",
                len(run_dataset.test_concepts()),
            )
            logger.info("  first_seen_step: %s", first_seen_step)
            _BASE.log_dataset_overview(run_dataset, "  Run dataset order:")

            logger.info("  [2/5] Building model and effective configuration")
            model_factory = lambda: _CL_FACTORY.build_model(_BASE, args, seed)[0]
            model, config = _CL_FACTORY.build_model(_BASE, args, seed)
            _LEVELS._validate_eval_level(args, model)
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
            run_payload["schedule_metadata"] = schedule_metadata

            logger.info("  Model device: %s", getattr(model, "_device", "unknown"))

            logger.info("  [3/5] Building strategy and callbacks")
            replay_buffer_size = 0
            if args.strategy == "replay" or (args.model == "cfa" and args.strategy == "cl"):
                replay_buffer_size = _BASE.resolve_replay_buffer_size(args, run_dataset)
                run_payload["replay_buffer_fraction"] = args.replay_buffer_fraction
                run_payload["replay_buffer_size"] = replay_buffer_size
            if args.model == "patchcore" and args.strategy == "cl":
                # Provenance only: PatchCoreCLStrategy ignores this value and budgets
                # history as previous_ratio (0.25) x |current-task coreset|.
                memory_bank_size = _CL_FACTORY.resolve_continual_memory_bank_size(_BASE, args, run_dataset)
                run_payload["continual_memory_bank_size"] = memory_bank_size
                args.continual_memory_bank_size = memory_bank_size
            strategy = _CL_FACTORY.build_strategy(
                _BASE,
                args,
                model,
                model_factory,
                replay_buffer_size,
                run_dataset,
            )
            callbacks = build_callbacks(
                args=args,
                dataset=run_dataset,
                strategy=strategy,
                dataset_root=dataset_root,
                first_seen_step=first_seen_step,
            )
            callbacks.append(
                _BASE.JsonCheckpointCallback(
                    output_path=output_path,
                    run_status_provider=run_status,
                    write_snapshot=write_snapshot,
                )
            )

            scenario = _BASE.build_scenario(
                strategy_name=args.strategy,
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
            logger.exception(
                "  Run did not finish cleanly; attempting to save partial results to %s",
                output_path,
            )
            try:
                write_snapshot()
            except Exception:
                logger.exception("  Failed to save partial results to %s", output_path)
            raise

        logger.info("  [5/5] Saving results")
        run_status.mark_completed(elapsed_seconds=elapsed)
        skipped_callbacks = [
            type(callback).__name__ for callback in callbacks if not isinstance(callback, InfoProvider)
        ]
        if skipped_callbacks:
            logger.info(
                "    Skipping non-serializable callbacks in JSON output: %s",
                skipped_callbacks,
            )
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

    output_dir = _BASE.resolve_output_dir(args)
    seeds = _BASE.generate_seeds(args.master_seed, args.n_runs)
    logger.info("Master seed: %d  →  per-run seeds: %s", args.master_seed, seeds)
    logger.info("Results directory: %s", output_dir)
    logger.info("Step schedule: %s", args.step_schedule)
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
