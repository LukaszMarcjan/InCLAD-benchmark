"""
Generic continual visual anomaly detection runner.

This script gives pyCLAD a single entrypoint for visual continual-learning
experiments, instead of maintaining one near-duplicate script per model.

Examples
--------
Replay with PaSTe:
    python examples/clvad/run_continual_visual_ad.py \
        --model paste \
        --strategy replay \
        --benchmark mvtec \
        --root /path/to/mvtec_ad \
        --backbone mobilenet_v2 \
        --pretrained \
        --epochs 50 \
        --batch-size 32 \
        --replay-buffer-size 100

Naive continual training with CFA:
    python examples/clvad/run_continual_visual_ad.py \
        --model cfa \
        --strategy naive \
        --benchmark visa \
        --root /path/to/visa \
        --backbone resnet18 \
        --pretrained

Replay with model-specific JSON overrides:
    python examples/clvad/run_continual_visual_ad.py \
        --model draem \
        --strategy replay \
        --benchmark mvtec \
        --root /path/to/mvtec_ad \
        --config-json '{"anomaly_source_root": "/path/to/dtd/images"}'

Replay with a fixed benchmark curriculum from ordering:
    python examples/clvad/run_continual_visual_ad.py \
        --model patchcore \
        --strategy naive \
        --benchmark mvtec \
        --root /path/to/mvtec_ad \
        --ordering-mode easy_to_hard
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import logging
import os
import pathlib
import random
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import torch

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.visual_dataset_registry import (
    read_registered_visual_benchmark_dataset,
    resolve_visual_benchmark_root,
)
from pyclad.data.readers.visual_dataset_ordering import resolve_benchmark_ordering_path
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.diagonal_average import DiagonalAverage
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.model import Model
from pyclad.output.output_writer import InfoProvider
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.baselines.ste import SingleTaskExpertStrategy
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection

logger = logging.getLogger(__name__)

DEFAULT_ORDERING_DIR = pathlib.Path(__file__).resolve().parents[1] / "visual_models" / "ordering"


@dataclass(frozen=True)
class ModelSpec:
    model_module: str
    model_name: str
    config_module: str
    config_name: str
    default_resize: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "cfa": ModelSpec("pyclad.models.cfa.cfa", "CFA", "pyclad.models.cfa.config", "CFAConfig", 224),
    "draem": ModelSpec("pyclad.models.draem.draem", "DRAEM", "pyclad.models.draem.config", "DRAEMConfig", 256),
    "efficientad": ModelSpec(
        "pyclad.models.efficientad.efficientad",
        "EfficientAD",
        "pyclad.models.efficientad.config",
        "EfficientADConfig",
        256,
    ),
    "fastflow": ModelSpec(
        "pyclad.models.fastflow.fastflow",
        "FastFlow",
        "pyclad.models.fastflow.config",
        "FastFlowConfig",
        256,
    ),
    "ganomaly": ModelSpec(
        "pyclad.models.ganomaly.ganomaly",
        "GANomaly",
        "pyclad.models.ganomaly.config",
        "GANomalyConfig",
        256,
    ),
    "padim": ModelSpec("pyclad.models.padim.padim", "PaDiM", "pyclad.models.padim.config", "PaDiMConfig", 224),
    "paste": ModelSpec("pyclad.models.paste.paste", "PaSTe", "pyclad.models.paste.config", "PaSTeConfig", 224),
    "patchcore": ModelSpec(
        "pyclad.models.patchcore.patchcore",
        "PatchCore",
        "pyclad.models.patchcore.config",
        "PatchCoreConfig",
        224,
    ),
    "rd4ad": ModelSpec("pyclad.models.rd4ad.rd4ad", "RD4AD", "pyclad.models.rd4ad.config", "RD4ADConfig", 224),
    "stfpm": ModelSpec("pyclad.models.stfpm.stfpm", "STFPM", "pyclad.models.stfpm.config", "STFPMConfig", 256),
    "unet": ModelSpec(
        "pyclad.models.unet.unet",
        "UNet",
        "pyclad.models.unet.config",
        "ReconstructionConfig",
        256,
    ),
}


class RunConfigProvider(InfoProvider):
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def info(self):
        return {"run_config": _make_json_safe(self._payload)}


def _timestamp_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class RunStatusProvider(InfoProvider):
    def __init__(self, output_path: pathlib.Path, log_path: pathlib.Path, total_concepts: int | None = None):
        self._payload: dict[str, Any] = {
            "state": "pending",
            "output_path": str(output_path),
            "log_path": str(log_path),
            "total_concepts": total_concepts,
            "completed_concepts": [],
            "completed_concepts_count": 0,
            "active_concept": None,
            "active_concept_index": None,
            "last_completed_concept": None,
            "last_completed_concept_index": None,
            "started_at": None,
            "finished_at": None,
            "updated_at": _timestamp_now(),
        }

    def set_total_concepts(self, total_concepts: int) -> None:
        self._payload["total_concepts"] = int(total_concepts)
        self._touch()

    def mark_run_started(self) -> None:
        if self._payload["started_at"] is None:
            self._payload["started_at"] = _timestamp_now()
        self._payload["state"] = "running"
        self._payload["finished_at"] = None
        self._touch()

    def mark_concept_started(self, concept_name: str, concept_index: int | None = None) -> None:
        self.mark_run_started()
        self._payload["active_concept"] = concept_name
        if concept_index is not None:
            self._payload["active_concept_index"] = int(concept_index)
        self._touch()

    def mark_concept_completed(self, concept_name: str, concept_index: int | None = None) -> None:
        self.mark_run_started()
        completed_concepts = self._payload["completed_concepts"]
        if concept_name not in completed_concepts:
            completed_concepts.append(concept_name)
        self._payload["completed_concepts_count"] = len(completed_concepts)
        self._payload["active_concept"] = None
        self._payload["active_concept_index"] = None
        self._payload["last_completed_concept"] = concept_name
        if concept_index is not None:
            self._payload["last_completed_concept_index"] = int(concept_index)
        self._touch()

    def mark_completed(self, elapsed_seconds: float | None = None) -> None:
        self._payload["state"] = "completed"
        self._payload["active_concept"] = None
        self._payload["active_concept_index"] = None
        self._payload["finished_at"] = _timestamp_now()
        self._payload["error_type"] = None
        self._payload["error_message"] = None
        self._payload["traceback"] = None
        if elapsed_seconds is not None:
            self._payload["elapsed_seconds"] = float(elapsed_seconds)
        self._touch()

    def mark_failed(self, exc: BaseException, traceback_text: str) -> None:
        self._payload["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        self._payload["finished_at"] = _timestamp_now()
        self._payload["error_type"] = type(exc).__name__
        self._payload["error_message"] = str(exc)
        self._payload["traceback"] = traceback_text
        self._touch()

    def info(self):
        return {"run_status": _make_json_safe(self._payload)}

    def completed_concepts(self) -> list[str]:
        return list(self._payload["completed_concepts"])

    def _touch(self) -> None:
        self._payload["updated_at"] = _timestamp_now()


class JsonCheckpointCallback(Callback):
    def __init__(
        self,
        output_path: pathlib.Path,
        run_status_provider: RunStatusProvider,
        write_snapshot: Callable[[], None],
    ):
        self._output_path = output_path
        self._run_status_provider = run_status_provider
        self._write_snapshot = write_snapshot
        self._concept_index = 0

    def before_scenario(self, *args, **kwargs):
        self._run_status_provider.mark_run_started()
        self._checkpoint("run start")

    def before_concept_processing(self, concept: Concept, *args, **kwargs):
        self._concept_index += 1
        self._run_status_provider.mark_concept_started(concept.name, self._concept_index)
        self._checkpoint(f"concept '{concept.name}' start")

    def after_concept_processing(self, concept: Concept, *args, **kwargs):
        self._run_status_provider.mark_concept_completed(concept.name, self._concept_index)
        self._checkpoint(f"concept '{concept.name}' completion")

    def _checkpoint(self, phase: str) -> None:
        try:
            self._write_snapshot()
        except Exception:
            logger.exception("Failed to write %s checkpoint to %s", phase, self._output_path)


def collect_result_providers(
    model: Model | None,
    run_dataset: ConceptsDataset | None,
    strategy: Any,
    callbacks: list[Callback],
    extra_providers: list[InfoProvider] | None = None,
) -> list[InfoProvider]:
    providers: list[InfoProvider] = []

    resolved_model = model
    current_model = getattr(strategy, "current_model", None) if strategy is not None else None
    if callable(current_model):
        maybe_model = current_model()
        if maybe_model is not None:
            resolved_model = maybe_model

    if resolved_model is not None:
        providers.append(resolved_model)
    if run_dataset is not None:
        providers.append(run_dataset)
    if strategy is not None:
        providers.append(strategy)

    providers.extend(callback for callback in callbacks if isinstance(callback, InfoProvider))

    if extra_providers is not None:
        providers.extend(extra_providers)

    return providers


def _ordered_metric_matrix(metric_matrix: dict[str, dict[str, float]], concepts_order: list[str]) -> list[list[float]]:
    if not concepts_order:
        return []

    values: list[list[float]] = []
    for learned_concept in concepts_order:
        learned_values = metric_matrix.get(learned_concept, {})
        values.append([learned_values.get(evaluated_concept, float("nan")) for evaluated_concept in concepts_order])
    return values


def _completed_concepts_for_callback(provider: Any, completed_concepts: list[str]) -> list[str]:
    learned_concepts = list(getattr(provider, "_learned_concepts", []))
    if not completed_concepts:
        return []

    completed_set = set(completed_concepts)
    return [concept_name for concept_name in learned_concepts if concept_name in completed_set]


def _serialize_concept_metric_callback(provider: ConceptMetricCallback, completed_concepts: list[str]) -> dict[str, Any]:
    concepts_order = _completed_concepts_for_callback(provider, completed_concepts)
    concept_level_matrix = _ordered_metric_matrix(provider._metric_matrix, concepts_order)
    lifelong_learning_metrics = {metric.name(): metric.compute(concept_level_matrix) for metric in provider._metrics}
    trimmed_metric_matrix = {
        learned_concept: {
            evaluated_concept: provider._metric_matrix.get(learned_concept, {}).get(evaluated_concept, float("nan"))
            for evaluated_concept in concepts_order
        }
        for learned_concept in concepts_order
    }

    return {
        f"concept_metric_callback_{provider._base_metric.name()}": {
            "base_metric_name": provider._base_metric.name(),
            "metrics": lifelong_learning_metrics,
            "concepts_order": concepts_order,
            "metric_matrix": trimmed_metric_matrix,
        }
    }


def _serialize_visual_pixel_metric_callback(provider: Any, completed_concepts: list[str]) -> dict[str, Any]:
    concepts_order = _completed_concepts_for_callback(provider, completed_concepts)
    payload: dict[str, Any] = {}

    for base_metric in provider._base_metrics:
        metric_matrix = provider._metric_matrices[base_metric.name()]
        concept_level_matrix = _ordered_metric_matrix(metric_matrix, concepts_order)
        lifelong_learning_metrics = {metric.name(): metric.compute(concept_level_matrix) for metric in provider._metrics}
        trimmed_metric_matrix = {
            learned_concept: {
                evaluated_concept: metric_matrix.get(learned_concept, {}).get(evaluated_concept, float("nan"))
                for evaluated_concept in concepts_order
            }
            for learned_concept in concepts_order
        }
        payload[f"pixel_concept_metric_callback_{base_metric.name()}"] = {
            "base_metric_name": base_metric.name(),
            "evaluation_level": "pixel",
            "threshold_selection": {
                "mode": provider._threshold_mode,
                "fixed_threshold": provider._fixed_threshold,
                "train_quantile": provider._threshold_quantile,
                "resolved_thresholds": {
                    concept_name: provider._resolved_thresholds[concept_name]
                    for concept_name in concepts_order
                    if concept_name in provider._resolved_thresholds
                },
            },
            "metrics": lifelong_learning_metrics,
            "concepts_order": concepts_order,
            "metric_matrix": trimmed_metric_matrix,
        }

    return payload


def _provider_snapshot_info(provider: InfoProvider, completed_concepts: list[str]) -> dict[str, Any] | None:
    try:
        if isinstance(provider, ConceptMetricCallback):
            return _serialize_concept_metric_callback(provider, completed_concepts)

        if (
            hasattr(provider, "_base_metrics")
            and hasattr(provider, "_metric_matrices")
            and hasattr(provider, "_resolved_thresholds")
        ):
            return _serialize_visual_pixel_metric_callback(provider, completed_concepts)

        return provider.info()
    except Exception:
        logger.exception("Skipping provider '%s' while writing snapshot", type(provider).__name__)
        return None


def build_snapshot_payload(providers: list[InfoProvider], completed_concepts: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    completed = list(completed_concepts or [])

    for provider in providers:
        provider_payload = _provider_snapshot_info(provider, completed)
        if provider_payload is None:
            continue
        payload.update(_make_json_safe(provider_payload))

    return payload


def _atomic_write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = pathlib.Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=4, default=lambda o: "<not serializable>")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def write_result_snapshot(
    output_path: pathlib.Path,
    providers: list[InfoProvider],
    completed_concepts: list[str] | None = None,
) -> None:
    _atomic_write_json(
        output_path,
        build_snapshot_payload(providers=providers, completed_concepts=completed_concepts),
    )


@contextlib.contextmanager
def seed_log_handler(log_path: pathlib.Path):
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continual visual anomaly detection experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True, choices=sorted(MODEL_SPECS.keys()), help="Model to train")
    parser.add_argument(
        "--strategy",
        type=str,
        default="replay",
        choices=["naive", "cumulative", "replay", "ste"],
        help="Continual-learning strategy",
    )

    ds = parser.add_argument_group("dataset")
    ds.add_argument("--benchmark", type=str, default="mvtec", help="Benchmark name")
    ds.add_argument("--root", type=str, default=None, help="Path to the benchmark root directory")
    ds.add_argument("--registry-path", type=str, default=None, help="Optional visual dataset registry JSON path")
    ds.add_argument("--categories", type=str, nargs="*", default=None, help="Categories to use (default: all)")
    ds.add_argument("--resize", type=int, default=None, help="Resize images to this square size")
    ds.add_argument("--max-train", type=int, default=None, help="Max training samples per category")
    ds.add_argument("--max-test", type=int, default=None, help="Max test samples per category")
    ds.add_argument(
        "--concept-order",
        type=str,
        nargs="*",
        default=None,
        help="Explicit concept order to use for continual learning",
    )
    ds.add_argument(
        "--ordering-file",
        type=str,
        default=None,
        help="CSV or JSON file containing a fixed concept order",
    )
    ds.add_argument(
        "--ordering-dir",
        type=str,
        default=str(DEFAULT_ORDERING_DIR),
        help="Directory containing benchmark ordering files",
    )
    ds.add_argument(
        "--ordering-mode",
        type=str,
        choices=["easy_to_hard", "hard_to_easy", "random"],
        default=None,
        help="Use a benchmark ordering from ordering-dir (or InCLAD-Bench split when --inclad-bench is set)",
    )
    ds.add_argument(
        "--inclad-bench",
        action="store_true",
        default=False,
        help="Load the dataset manifest from the InCLAD-Bench HuggingFace repository "
        "(anonmllab/inclad-bench) instead of reading from local folder structure. "
        "Images must still be available locally (--root or registry). "
        "When set, --ordering-mode selects the InCLAD ordering split "
        "(easy_to_hard, hard_to_easy); defaults to easy_to_hard.",
    )
    ds.add_argument(
        "--ordering-metric",
        type=str,
        default="roc-auc",
        help="Metric slug used in benchmark ordering filenames",
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
    md.add_argument("--backbone", type=str, default=None, help="Backbone override for models that support it")
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
        help="Enable/disable progress bars for models that support it",
    )

    md.add_argument("--variant", type=str, default=None, help="Variant override for models such as DRAEM")
    md.add_argument("--model-size", type=str, default=None, help="Model size override (for EfficientAD)")
    md.add_argument("--n-features", type=int, default=None, help="Feature count override for models that support it")
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
        help="JSON file with raw model-config overrides (field names must match the model config)",
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
    ex.add_argument("--n-runs", type=int, default=1, help="Number of experiment runs")
    ex.add_argument("--master-seed", type=int, default=42, help="Master seed for per-run seed generation")
    ex.add_argument("--output-dir", type=str, default=None, help="Directory for result JSON and logs")

    return parser.parse_args()


def default_results_root() -> pathlib.Path:
    return pathlib.Path(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def _count_samples(concept: Concept | None) -> int:
    if concept is None or concept.data is None:
        return 0
    return len(concept.data)


def _test_label_summary(concept: Concept | None) -> str:
    if concept is None or concept.labels is None:
        return "test_labels=unavailable"

    labels = np.asarray(concept.labels)
    normals = int(np.sum(labels == 0))
    anomalies = int(labels.size - normals)
    return f"test_normals={normals}, test_anomalies={anomalies}"


def build_dataset_order_lines(dataset: ConceptsDataset) -> list[str]:
    test_by_name = {concept.name: concept for concept in dataset.test_concepts()}
    total = len(dataset.train_concepts())
    lines: list[str] = []

    for index, train_concept in enumerate(dataset.train_concepts(), start=1):
        test_concept = test_by_name.get(train_concept.name)
        lines.append(
            f"{index:02d}/{total:02d} {train_concept.name}: "
            f"train={_count_samples(train_concept)}, "
            f"test={_count_samples(test_concept)} ({_test_label_summary(test_concept)})"
        )

    return lines


def log_dataset_overview(dataset: ConceptsDataset, header: str) -> None:
    logger.info(header)
    for line in build_dataset_order_lines(dataset):
        logger.info("  %s", line)


class ScenarioProgressCallback(Callback):
    def __init__(self, dataset: ConceptsDataset):
        self._train_concepts = list(dataset.train_concepts())
        self._test_by_name = {concept.name: concept for concept in dataset.test_concepts()}
        self._train_total = len(self._train_concepts)
        self._eval_total = len(dataset.test_concepts())
        self._current_train_index = 0
        self._current_eval_index = 0
        self._current_concept_name: str | None = None
        self._scenario_started_at: float | None = None
        self._concept_started_at: float | None = None
        self._training_started_at: float | None = None
        self._evaluation_started_at: float | None = None

    def before_scenario(self, *args, **kwargs):
        self._scenario_started_at = time.perf_counter()
        logger.info(
            "  Scenario plan: %d concept(s) to learn, %d evaluation(s) after each concept, %d total evaluation steps.",
            self._train_total,
            self._eval_total,
            self._train_total * self._eval_total,
        )

    def before_concept_processing(self, concept: Concept, *args, **kwargs):
        self._current_train_index += 1
        self._current_eval_index = 0
        self._current_concept_name = concept.name
        self._concept_started_at = time.perf_counter()
        logger.info(
            "  [%02d/%02d] Learning concept '%s' (train=%d, test=%d).",
            self._current_train_index,
            self._train_total,
            concept.name,
            _count_samples(concept),
            _count_samples(self._test_by_name.get(concept.name)),
        )

    def before_training(self, *args, **kwargs):
        self._training_started_at = time.perf_counter()
        if self._current_concept_name is not None:
            logger.info("      Training started for '%s'.", self._current_concept_name)

    def after_training(self, learned_concept: Concept, *args, **kwargs):
        elapsed = 0.0 if self._training_started_at is None else time.perf_counter() - self._training_started_at
        logger.info("      Training finished for '%s' in %.1fs.", learned_concept.name, elapsed)

    def before_evaluation(self, *args, **kwargs):
        self._evaluation_started_at = time.perf_counter()

    def after_evaluation(self, evaluated_concept: Concept, *args, **kwargs):
        self._current_eval_index += 1
        elapsed = 0.0 if self._evaluation_started_at is None else time.perf_counter() - self._evaluation_started_at
        logger.info(
            "      Eval %02d/%02d on '%s' finished in %.1fs (test=%d, %s).",
            self._current_eval_index,
            self._eval_total,
            evaluated_concept.name,
            elapsed,
            _count_samples(evaluated_concept),
            _test_label_summary(evaluated_concept),
        )

    def after_concept_processing(self, concept: Concept, *args, **kwargs):
        elapsed = 0.0 if self._concept_started_at is None else time.perf_counter() - self._concept_started_at
        logger.info(
            "  [%02d/%02d] Concept '%s' complete in %.1fs.",
            self._current_train_index,
            self._train_total,
            concept.name,
            elapsed,
        )

    def after_scenario(self, *args, **kwargs):
        elapsed = 0.0 if self._scenario_started_at is None else time.perf_counter() - self._scenario_started_at
        logger.info("  Scenario finished in %.1fs.", elapsed)


def resolve_output_dir(args: argparse.Namespace) -> pathlib.Path:
    if args.output_dir is not None:
        return pathlib.Path(args.output_dir)
    return default_results_root() / args.model / args.strategy


def _slugify_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _slugify_label_fragment(value: str) -> str:
    slug_chars: list[str] = []
    previous_was_separator = False
    for char in str(value).strip().lower():
        if char.isalnum():
            slug_chars.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            slug_chars.append("-")
            previous_was_separator = True
    return "".join(slug_chars).strip("-") or "value"


def ordering_output_suffix(args: argparse.Namespace) -> str:
    ordering_file = getattr(args, "ordering_file", None)
    if ordering_file:
        return f"_order-{_slugify_label_fragment(pathlib.Path(str(ordering_file)).stem)}"

    ordering_mode = getattr(args, "ordering_mode", None)
    if ordering_mode:
        return f"_order-{_slugify_label_fragment(str(ordering_mode))}"

    if getattr(args, "shuffle_concepts", False):
        return "_order-shuffle"

    if getattr(args, "concept_order", None):
        return "_order-custom"

    return ""


def replay_buffer_file_suffix(args: argparse.Namespace) -> str:
    if args.strategy != "replay":
        return ""
    return f"_buf-frac{_slugify_float(args.replay_buffer_fraction)}"


def resolve_replay_buffer_size(args: argparse.Namespace, dataset: ConceptsDataset) -> int:
    concept_sizes = [len(concept.data) for concept in dataset.train_concepts()]
    avg_concept_size = sum(concept_sizes) / len(concept_sizes) if concept_sizes else 0.0
    max_size = max(1, int(round(avg_concept_size * args.replay_buffer_fraction)))
    logger.info(
        "  Replay buffer: fraction=%.2f, avg_concept_size=%.1f, max_size=%d",
        args.replay_buffer_fraction,
        avg_concept_size,
        max_size,
    )
    return max_size


def generate_seeds(master_seed: int, n_runs: int) -> list[int]:
    rng = np.random.default_rng(master_seed)
    return [int(seed) for seed in rng.integers(0, 2**31 - 1, size=n_runs)]


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shuffle_dataset(dataset: ConceptsDataset, seed: int) -> ConceptsDataset:
    rng = np.random.default_rng(seed)
    pairs = list(zip(dataset.train_concepts(), dataset.test_concepts()))
    indices = rng.permutation(len(pairs))
    shuffled = [pairs[i] for i in indices]
    train_concepts, test_concepts = zip(*shuffled)
    return ConceptsDataset(
        name=dataset.name(),
        train_concepts=list(train_concepts),
        test_concepts=list(test_concepts),
    )


def reorder_dataset(dataset: ConceptsDataset, concept_order: list[str]) -> ConceptsDataset:
    train_by_name = {concept.name: concept for concept in dataset.train_concepts()}
    test_by_name = {concept.name: concept for concept in dataset.test_concepts()}
    return ConceptsDataset(
        name=dataset.name(),
        train_concepts=[train_by_name[name] for name in concept_order],
        test_concepts=[test_by_name[name] for name in concept_order],
    )


def resolve_fixed_concept_order(
    args: argparse.Namespace,
    dataset: ConceptsDataset,
) -> tuple[list[str] | None, dict[str, Any] | None]:
    # InCLAD-Bench ordering is already embedded in the HF manifest and applied
    # during dataset loading — no local ordering file needed.
    if getattr(args, "inclad_bench", False):
        return None, {
            "source": "inclad-bench",
            "ordering_mode": getattr(args, "ordering_mode", None),
        }

    requested_sources = sum(
        1
        for value in (args.concept_order, args.ordering_file, args.ordering_mode)
        if value not in (None, [])
    )
    if requested_sources > 1:
        raise ValueError(
            "Choose only one fixed ordering source: --concept-order, --ordering-file, or --ordering-mode."
        )

    if args.shuffle_concepts and requested_sources:
        raise ValueError("Fixed ordering cannot be combined with --shuffle-concepts.")

    if args.concept_order:
        raw_order: list[str] = []
        for item in args.concept_order:
            raw_order.extend(_parse_serialized_order(item))
        concept_order, projection_info = validate_and_project_concept_order(raw_order, dataset)
        return concept_order, {
            "source": "cli",
            "source_type": "concept_order",
            **projection_info,
        }

    if args.ordering_file:
        ordering_path = pathlib.Path(args.ordering_file)
        raw_order = load_concept_order_from_file(
            ordering_path=ordering_path,
            benchmark=args.benchmark,
            ordering_mode=args.ordering_mode,
            ordering_column=args.ordering_column,
        )
        concept_order, projection_info = validate_and_project_concept_order(raw_order, dataset)
        return concept_order, {
            "source": "file",
            "source_type": ordering_path.suffix.lower().lstrip(".") or "unknown",
            "ordering_file": str(ordering_path),
            "ordering_column": args.ordering_column,
            **projection_info,
        }

    if args.ordering_mode:
        ordering_path = resolve_ordering_path(args)
        raw_order = load_concept_order_from_file(
            ordering_path=ordering_path,
            benchmark=args.benchmark,
            ordering_mode=args.ordering_mode,
            ordering_column=args.ordering_column,
        )
        concept_order, projection_info = validate_and_project_concept_order(raw_order, dataset)
        return concept_order, {
            "source": "ordering",
            "source_type": "benchmark_curriculum",
            "ordering_file": str(ordering_path),
            "ordering_mode": args.ordering_mode,
            "ordering_metric": args.ordering_metric,
            "ordering_column": args.ordering_column,
            **projection_info,
        }

    return None, None


def resolve_ordering_path(args: argparse.Namespace) -> pathlib.Path:
    return resolve_benchmark_ordering_path(
        ordering_dir=pathlib.Path(args.ordering_dir),
        benchmark=args.benchmark,
        ordering_mode=args.ordering_mode,
        ordering_metric=args.ordering_metric,
    )


def load_concept_order_from_file(
    ordering_path: pathlib.Path,
    benchmark: str,
    ordering_mode: str | None,
    ordering_column: str,
) -> list[str]:
    if not ordering_path.exists():
        raise FileNotFoundError(f"Ordering file does not exist: {ordering_path}")

    suffix = ordering_path.suffix.lower()
    if suffix == ".csv":
        return load_concept_order_from_csv(ordering_path, benchmark, ordering_mode, ordering_column)
    if suffix == ".json":
        payload = json.loads(ordering_path.read_text(encoding="utf-8"))
        return _extract_order_from_payload(payload, benchmark, ordering_mode)
    raise ValueError(f"Unsupported ordering file format: {ordering_path}")


def load_concept_order_from_csv(
    ordering_path: pathlib.Path,
    benchmark: str,
    ordering_mode: str | None,
    ordering_column: str,
) -> list[str]:
    with ordering_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"Ordering CSV has no header: {ordering_path}")
        rows = list(reader)

    if ordering_column in reader.fieldnames:
        concepts = [row[ordering_column].strip() for row in rows if row.get(ordering_column, "").strip()]
        if not concepts:
            raise ValueError(
                f"Ordering CSV '{ordering_path}' contains column '{ordering_column}' but no concept names."
            )
        return concepts

    if ordering_mode and ordering_mode in reader.fieldnames:
        benchmark_field = "benchmark" if "benchmark" in reader.fieldnames else None
        for row in rows:
            if benchmark_field and row.get(benchmark_field, "").strip().lower() != benchmark.lower():
                continue
            return _parse_serialized_order(row.get(ordering_mode))
        raise ValueError(
            f"Ordering CSV '{ordering_path}' does not contain a row for benchmark '{benchmark}'."
        )

    raise ValueError(
        f"Ordering CSV '{ordering_path}' must contain either '{ordering_column}' or '{ordering_mode}' columns. "
        f"Available columns: {reader.fieldnames}"
    )


def _extract_order_from_payload(payload: Any, benchmark: str, ordering_mode: str | None) -> list[str]:
    if isinstance(payload, list):
        return _parse_serialized_order(payload)

    if not isinstance(payload, dict):
        raise ValueError("Ordering JSON must contain a list or object with a concept order.")

    for key in ("concept_order", "concepts", "order"):
        if key in payload:
            return _parse_serialized_order(payload[key])

    if ordering_mode and ordering_mode in payload:
        return _parse_serialized_order(payload[ordering_mode])

    benchmark_keys = (benchmark, benchmark.lower(), benchmark.upper())
    for key in benchmark_keys:
        if key in payload:
            return _extract_order_from_payload(payload[key], benchmark, ordering_mode)

    raise ValueError(
        "Could not find a concept order in the JSON payload. "
        "Expected one of: concept_order, concepts, order, ordering mode key, or benchmark key."
    )


def _parse_serialized_order(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        raise ValueError("Ordering value is empty.")
    if not isinstance(value, str):
        raise ValueError(f"Unsupported ordering payload type: {type(value)!r}")

    stripped = value.strip()
    if not stripped:
        raise ValueError("Ordering value is empty.")

    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("Serialized ordering JSON must decode to a list.")
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [item.strip() for item in stripped.split(",") if item.strip()]


def validate_and_project_concept_order(
    raw_order: list[str],
    dataset: ConceptsDataset,
) -> tuple[list[str], dict[str, Any]]:
    dataset_concepts = [concept.name for concept in dataset.train_concepts()]
    duplicate_order_items = _duplicates(raw_order)
    if duplicate_order_items:
        raise ValueError(f"Concept order contains duplicates: {duplicate_order_items}")

    selected = [concept for concept in raw_order if concept in dataset_concepts]
    missing = [concept for concept in dataset_concepts if concept not in selected]
    if missing:
        raise ValueError(
            "Concept order does not fully cover the dataset concepts. "
            f"Missing concepts: {missing}. Dataset concepts: {dataset_concepts}"
        )

    skipped = [concept for concept in raw_order if concept not in dataset_concepts]
    return selected, {
        "dataset_concepts": dataset_concepts,
        "raw_order_length": len(raw_order),
        "resolved_order_length": len(selected),
        "skipped_concepts": skipped,
    }


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def build_callbacks(dataset: ConceptsDataset):
    return [
        ScenarioProgressCallback(dataset),
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


def load_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    if args.config_file is not None:
        config_path = pathlib.Path(args.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        file_overrides = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(file_overrides, dict):
            raise ValueError("Config file must contain a JSON object")
        overrides.update(file_overrides)

    if args.config_json is not None:
        inline_overrides = json.loads(args.config_json)
        if not isinstance(inline_overrides, dict):
            raise ValueError("config-json must be a JSON object")
        overrides.update(inline_overrides)

    return overrides


def build_model_config(args: argparse.Namespace, seed: int):
    spec = MODEL_SPECS[args.model]
    config_cls = _load_attr(spec.config_module, spec.config_name)
    config_fields = set(config_cls.model_fields.keys())

    raw_overrides = load_config_overrides(args)
    unknown_raw = sorted(set(raw_overrides.keys()) - config_fields)
    if unknown_raw:
        raise ValueError(
            f"Unknown config override fields for model '{args.model}': {unknown_raw}. "
            f"Valid fields: {sorted(config_fields)}"
        )

    resize = int(args.resize) if args.resize is not None else spec.default_resize
    updates: dict[str, Any] = dict(raw_overrides)
    ignored_cli_options: list[str] = []

    if "input_size" in config_fields:
        updates["input_size"] = (resize, resize)
    if "random_seed" in config_fields:
        updates["random_seed"] = int(seed)

    def map_override(label: str, value: Any, *candidate_fields: str, transform: Callable[[Any], Any] | None = None) -> None:
        if value is None:
            return
        for field_name in candidate_fields:
            if field_name in config_fields:
                updates[field_name] = transform(value) if transform is not None else value
                return
        ignored_cli_options.append(label)

    map_override("backbone", args.backbone, "backbone_name")
    map_override("device", args.device, "device")
    map_override("batch_size", args.batch_size, "batch_size")
    map_override("epochs", args.epochs, "epochs")
    map_override("threshold_quantile", args.threshold_quantile, "threshold_quantile")
    map_override("show_training_progress", args.show_training_progress, "show_training_progress")
    map_override("variant", args.variant, "variant")
    map_override("model_size", args.model_size, "model_size")
    map_override("n_features", args.n_features, "n_features")
    map_override("coreset_ratio", args.coreset_ratio, "coreset_sampling_ratio")
    map_override("n_neighbors", args.n_neighbors, "n_neighbors", "k_neighbors")
    map_override(
        "ad_layers",
        args.ad_layers,
        "ad_layers",
        transform=lambda value: tuple(int(layer) for layer in value),
    )
    map_override(
        "student_bootstrap_layer",
        args.student_bootstrap_layer,
        "student_bootstrap_layer",
        transform=lambda value: None if int(value) < 0 else int(value),
    )
    map_override("teacher_weights_path", args.teacher_weights_path, "teacher_weights_path")
    map_override("penalty_source_root", args.penalty_source_root, "penalty_source_root")
    map_override("anomaly_source_root", args.anomaly_source_root, "anomaly_source_root")

    if args.pretrained is not None:
        map_override(
            "pretrained",
            args.pretrained,
            "pretrained_backbone",
            "pretrained_teacher",
            "pretrained_encoder",
        )

    if args.freeze_backbone is not None:
        map_override(
            "freeze_backbone",
            args.freeze_backbone,
            "freeze_backbone",
            "freeze_teacher",
            "freeze_encoder",
        )

    if ignored_cli_options:
        logger.warning(
            "Ignored CLI overrides for model '%s' because these config fields are not supported: %s",
            args.model,
            sorted(set(ignored_cli_options)),
        )

    base_config = config_cls()
    merged_config = {**base_config.model_dump(mode="python"), **updates}
    return config_cls(**merged_config)


def build_model(args: argparse.Namespace, seed: int) -> tuple[Model, Any]:
    spec = MODEL_SPECS[args.model]
    config = build_model_config(args, seed)
    model_cls = _load_attr(spec.model_module, spec.model_name)
    model = model_cls(config)
    return model, config


def build_strategy(
    strategy_name: str,
    model: Model,
    model_creation_fn: Callable[[], Model],
    replay_buffer_size: int = 0,
):
    if strategy_name == "naive":
        return NaiveStrategy(model)
    if strategy_name == "cumulative":
        logger.warning(
            "CumulativeStrategy in pyCLAD keeps accumulating data and calls fit() again on the same model instance. "
            "This is a useful upper bound, but it is not a strict reset-from-scratch joint-training baseline."
        )
        return CumulativeStrategy(model)
    if strategy_name == "replay":
        replay_buffer = AdaptiveBalancedReplayBuffer(
            selection_method=RandomSelection(),
            max_size=replay_buffer_size,
        )
        return ReplayEnhancedStrategy(model, replay_buffer)
    if strategy_name == "ste":
        return SingleTaskExpertStrategy(model=model, model_creation_fn=model_creation_fn)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def create_run_config_payload(
    args: argparse.Namespace,
    seed: int,
    output_path: pathlib.Path,
    log_path: pathlib.Path,
    config: Any | None = None,
    run_dataset: ConceptsDataset | None = None,
    ordering_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "seed": seed,
        "model": args.model,
        "strategy": args.strategy,
        "benchmark": args.benchmark,
        "root": args.root,
        "registry_path": args.registry_path,
        "categories": args.categories,
        "resize": args.resize,
        "shuffle_concepts": args.shuffle_concepts,
        "output_path": str(output_path),
        "log_path": str(log_path),
        "cli_args": vars(args),
    }
    if run_dataset is not None:
        payload["concept_order"] = [concept.name for concept in run_dataset.train_concepts()]
    if ordering_metadata is not None:
        payload["ordering_metadata"] = ordering_metadata
    if config is not None:
        payload["model_config"] = config.model_dump(mode="python")
    return payload


def run_single(
    args: argparse.Namespace,
    seed: int,
    run_index: int,
    total_runs: int,
    dataset: ConceptsDataset,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordering_suffix = ordering_output_suffix(args)
    replay_suffix = replay_buffer_file_suffix(args)
    output_path = output_dir / f"{args.model}_{args.strategy}{ordering_suffix}{replay_suffix}_seed_{seed}.json"
    log_path = output_dir / f"{args.model}_{args.strategy}{ordering_suffix}{replay_suffix}_seed_{seed}.log"

    with seed_log_handler(log_path):
        run_status = RunStatusProvider(output_path=output_path, log_path=log_path)
        run_payload = create_run_config_payload(
            args=args,
            seed=seed,
            output_path=output_path,
            log_path=log_path,
        )
        run_provider = RunConfigProvider(run_payload)

        model: Model | None = None
        config: Any | None = None
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
            logger.info("  Model device: %s", getattr(model, "_device", "unknown"))
            if hasattr(config, "show_training_progress"):
                logger.info("  Model-native training progress bars: %s", config.show_training_progress)
            else:
                logger.info("  Model-native training progress bars: not supported by this model")
            if hasattr(config, "batch_size"):
                logger.info("  Effective batch size: %s", getattr(config, "batch_size"))
            if hasattr(config, "epochs"):
                logger.info("  Effective epochs: %s", getattr(config, "epochs"))

            if args.strategy == "replay" and args.model in {"padim", "patchcore"}:
                logger.warning(
                    "Model '%s' is running with generic replay. This is valid in the current pyCLAD API, "
                    "but it is not the same as a dedicated %s-CL adaptation.",
                    args.model,
                    args.model.capitalize(),
                )

            logger.info("  [3/5] Building strategy and callbacks")
            replay_buffer_size = 0
            if args.strategy == "replay":
                replay_buffer_size = resolve_replay_buffer_size(args, run_dataset)
                run_payload["replay_buffer_fraction"] = args.replay_buffer_fraction
                run_payload["replay_buffer_size"] = replay_buffer_size
            strategy = build_strategy(
                args.strategy,
                model,
                model_factory,
                replay_buffer_size,
            )
            callbacks = build_callbacks(run_dataset)
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
            t0 = time.perf_counter()
            scenario.run()
            elapsed = time.perf_counter() - t0
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


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    return value


def _load_attr(module_name: str, attr_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


if __name__ == "__main__":
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
        "Dataset options: categories=%s, resize=%s, max_train=%s, max_test=%s",
        args.categories if args.categories is not None else "all",
        resize_to,
        args.max_train,
        args.max_test,
    )
    if args.strategy == "replay":
        logger.info("Replay buffer fraction: %.2f", args.replay_buffer_fraction)

    dataset_load_started_at = time.perf_counter()

    if args.inclad_bench:
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
        )
        saved_paths.append(path)

    logger.info("All %d runs complete.", len(saved_paths))
    for path in saved_paths:
        logger.info("  %s", path)
