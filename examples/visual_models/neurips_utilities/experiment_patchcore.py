"""
PatchCore experiment runner with multiple pseudo-random seeds.

Generates N reproducible seeds from a master seed and runs a full
continual-learning scenario for each one, saving per-seed results.

Usage:
    python examples/visual_models/experiment_patchcore.py \
        --root /path/to/mvtec_ad --benchmark mvtec \
        --n-runs 10 --master-seed 42 --pretrained

Each run produces a JSON file whose name contains the seed, e.g.
    results_20260420_153000/patchcore/patchcore_seed_58712.json
"""

import argparse
import contextlib
import logging
import pathlib
import random
import time

import numpy as np
import torch
from results_paths import default_results_root

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.readers.visual_benchmark_reader import read_visual_benchmark_dataset
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.diagonal_average import DiagonalAverage
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.patchcore.config import PatchCoreConfig
from pyclad.models.patchcore.patchcore import PatchCore
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.baselines.ste import SingleTaskExpertStrategy
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def seed_log_handler(log_path: pathlib.Path):
    """Temporarily add a FileHandler for a single seed run, then remove it."""
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def generate_seeds(master_seed: int, n: int) -> list[int]:
    """Generate *n* deterministic pseudo-random seeds from *master_seed*."""
    rng = np.random.default_rng(master_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


def set_global_seeds(seed: int) -> None:
    """Pin every source of randomness so a run is fully reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shuffle_dataset(dataset: ConceptsDataset, seed: int) -> ConceptsDataset:
    """Return a new dataset with concept ordering shuffled according to *seed*.

    Train and test concepts are shuffled together (same permutation) so each
    train concept stays paired with its corresponding test concept.
    """
    rng = np.random.default_rng(seed)
    pairs = list(zip(dataset.train_concepts(), dataset.test_concepts()))
    indices = rng.permutation(len(pairs))
    shuffled = [pairs[i] for i in indices]
    train_concepts, test_concepts = zip(*shuffled)
    concept_names = [c.name for c in train_concepts]
    logger.info(f"  Concept order (seed={seed}): {concept_names}")
    return ConceptsDataset(
        name=dataset.name(),
        train_concepts=list(train_concepts),
        test_concepts=list(test_concepts),
    )


def build_callbacks():
    return [
        ConceptMetricCallback(
            base_metric=RocAuc(),
            metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer(), DiagonalAverage()],
        ),
        ConceptMetricCallback(
            base_metric=F1Score(),
            metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer(), DiagonalAverage()],
        ),
        TimeEvaluationCallback(),
        MemoryUsageCallback(),
    ]


def build_model(args: argparse.Namespace, seed: int) -> PatchCore:
    return PatchCore(
        PatchCoreConfig(
            backbone_name=args.backbone,
            pretrained_backbone=args.pretrained,
            input_size=(224, 224),
            batch_size=args.batch_size,
            coreset_sampling_ratio=args.coreset_ratio,
            n_neighbors=args.n_neighbors,
            threshold_quantile=0.99,
            random_seed=seed,
            device=args.device,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PatchCore experiments with multiple seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── dataset ──────────────────────────────────────────────────────
    ds = parser.add_argument_group("dataset")
    ds.add_argument("--root", type=str, required=True, help="Path to the benchmark dataset root directory")
    ds.add_argument("--benchmark", type=str, default="mvtec", help="Benchmark name")
    ds.add_argument("--categories", type=str, nargs="*", default=None, help="Categories to use (default: all)")
    ds.add_argument("--resize", type=int, default=None, help="Resize images to this square size")
    ds.add_argument("--max-train", type=int, default=None, help="Max training samples per category")
    ds.add_argument("--max-test", type=int, default=None, help="Max test samples per category")

    # ── model ────────────────────────────────────────────────────────
    md = parser.add_argument_group("model")
    md.add_argument("--backbone", type=str, default="resnet18", help="Backbone network")
    md.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for model feature extraction (default: auto: cuda -> mps -> cpu)",
    )
    md.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained backbone weights")
    md.add_argument("--coreset-ratio", type=float, default=0.1, help="Coreset sampling ratio")
    md.add_argument("--n-neighbors", type=int, default=1, help="Number of nearest neighbors for scoring")
    md.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # ── strategy ─────────────────────────────────────────────────────
    st = parser.add_argument_group("strategy")
    st.add_argument(
        "--strategy",
        type=str,
        default="naive",
        choices=["naive", "cumulative", "replay", "ste"],
        help="Continual-learning strategy",
    )
    st.add_argument(
        "--replay-buffer-fraction",
        type=float,
        default=0.2,
        help="Replay buffer size as a fraction of the average concept size (e.g. 0.20 = 20%%)",
    )

    # ── experiment ───────────────────────────────────────────────────
    ex = parser.add_argument_group("experiment")
    ex.add_argument("--n-runs", type=int, default=10, help="Number of experiment runs (seeds)")
    ex.add_argument("--master-seed", type=int, default=42, help="Master seed for generating per-run seeds")
    ex.add_argument(
        "--output-dir",
        type=str,
        default=str(default_results_root() / "patchcore"),
        help="Directory for result JSON files",
    )

    return parser.parse_args()


def build_strategy(strategy_name: str, model, model_creation_fn, dataset, replay_buffer_fraction: float):
    if strategy_name == "naive":
        return NaiveStrategy(model)
    elif strategy_name == "cumulative":
        return CumulativeStrategy(model)
    elif strategy_name == "replay":
        concept_sizes = [len(c.data) for c in dataset.train_concepts()]
        avg_concept_size = sum(concept_sizes) / len(concept_sizes) if concept_sizes else 0.0
        max_size = max(1, int(round(avg_concept_size * replay_buffer_fraction)))
        replay_buffer = AdaptiveBalancedReplayBuffer(
            selection_method=RandomSelection(), max_size=max_size
        )
        return ReplayEnhancedStrategy(model, replay_buffer)
    elif strategy_name == "ste":
        return SingleTaskExpertStrategy(model=model, model_creation_fn=model_creation_fn)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_single(args: argparse.Namespace, seed: int, run_index: int, total_runs: int, dataset) -> pathlib.Path:
    """Execute one full scenario for a given seed and return the output path."""
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"patchcore_seed_{seed}.log"

    with seed_log_handler(log_path):
        logger.info(f"═══ Run {run_index + 1}/{total_runs}  seed={seed} ═══")
        set_global_seeds(seed)

        shuffled_dataset = shuffle_dataset(dataset, seed)

        model_factory = lambda: build_model(args, seed)
        model = model_factory()
        logger.info(f"  Model device: {getattr(model, '_device', 'unknown')}")
        strategy = build_strategy(args.strategy, model, model_factory, shuffled_dataset, args.replay_buffer_fraction)
        callbacks = build_callbacks()

        scenario = ConceptIncrementalScenario(
            dataset=shuffled_dataset,
            strategy=strategy,
            callbacks=callbacks,
        )

        t0 = time.perf_counter()
        scenario.run()
        elapsed = time.perf_counter() - t0

        output_path = output_dir / f"patchcore_seed_{seed}.json"
        output_model = strategy.current_model() if hasattr(strategy, "current_model") else model
        JsonOutputWriter(output_path).write([output_model, shuffled_dataset, strategy, *callbacks])

        logger.info(f"    ✓ finished in {elapsed:.1f}s → {output_path}")

    return output_path


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if args.strategy == "ste" and args.n_runs > 1:
        logger.warning(
            "Strategy 'ste' resets the model for each concept, so extra seeds usually only reshuffle concept order. "
            "For curriculum ordering you will typically want --n-runs 1."
        )

    seeds = generate_seeds(args.master_seed, args.n_runs)
    logger.info(f"Master seed: {args.master_seed}  →  per-run seeds: {seeds}")

    resize_to = (args.resize, args.resize) if args.resize is not None else None

    # Load dataset once (deterministic, no randomness involved)
    dataset = read_visual_benchmark_dataset(
        root=args.root,
        benchmark=args.benchmark,
        categories=args.categories,
        data_mode="numpy",
        resize_to=resize_to,
        color_mode="rgb",
        max_train_samples_per_category=args.max_train,
        max_test_samples_per_category=args.max_test,
    )

    saved_paths: list[pathlib.Path] = []
    for run_index, seed in enumerate(seeds):
        path = run_single(args, seed, run_index, len(seeds), dataset)
        saved_paths.append(path)

    logger.info(f"All {len(saved_paths)} runs complete. Results in: {args.output_dir}/")
    for p in saved_paths:
        logger.info(f"  {p}")
