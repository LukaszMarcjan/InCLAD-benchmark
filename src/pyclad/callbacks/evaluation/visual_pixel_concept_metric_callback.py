from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.evaluation.visual_pixel_utils import (
    index_registered_visual_test_samples,
    load_ground_truth_masks_for_samples,
)
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.base.pixel_threshold_utils import resolve_pixel_threshold
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)
from pyclad.output.output_writer import InfoProvider


class VisualPixelConceptMetricCallback(Callback, InfoProvider):
    def __init__(
        self,
        strategy: Any,
        benchmark: str,
        root: Optional[str | Path] = None,
        registry_path: Optional[str | Path] = None,
        categories: Optional[Sequence[str]] = None,
        max_test_samples_per_category: Optional[int] = None,
        base_metric: Optional[BaseMetric] = None,
        base_metrics: Optional[Sequence[BaseMetric]] = None,
        metrics: Iterable[ConceptLevelMetric] = (),
        skip_missing_anomaly_masks: bool = True,
        threshold_mode: str = "fixed",
        fixed_threshold: float = 0.5,
        threshold_quantile: float = 0.995,
    ):
        self._strategy = strategy
        if base_metrics is not None:
            self._base_metrics = list(base_metrics)
        else:
            self._base_metrics = [base_metric if base_metric is not None else _MissingMetric()]
        self._metric_matrices: Dict[str, Dict[str, Dict[str, float]]] = {
            metric.name(): defaultdict(dict) for metric in self._base_metrics
        }
        self._learned_concepts: List[str] = []
        self._training_concepts: Dict[str, Concept] = {}
        self._resolved_thresholds: Dict[str, float] = {}
        self._metrics = list(metrics)
        self._skip_missing_anomaly_masks = skip_missing_anomaly_masks
        self._threshold_mode = str(threshold_mode)
        self._fixed_threshold = float(fixed_threshold)
        self._threshold_quantile = float(threshold_quantile)
        self._test_samples_by_concept = index_registered_visual_test_samples(
            benchmark=benchmark,
            root=root,
            registry_path=registry_path,
            categories=categories,
            max_test_samples_per_category=max_test_samples_per_category,
        )

    def after_training(self, learned_concept: Concept):
        self._learned_concepts.append(learned_concept.name)
        self._training_concepts[learned_concept.name] = learned_concept

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        *args,
        **kwargs,
    ):
        for metric in self._base_metrics:
            assert (
                evaluated_concept.name not in self._metric_matrices[metric.name()][self._learned_concepts[-1]]
            ), "The same concept should not be evaluated twice after the same learned concept"

        model, threshold_key, threshold_train_concepts = self._resolve_model_context(evaluated_concept.name)
        score_maps_fn = getattr(model, "score_maps", None)
        if not callable(score_maps_fn):
            raise AttributeError(
                f"Model '{type(model).__name__}' does not expose score_maps(), so pixel-level evaluation is unavailable"
            )

        concept_samples = self._test_samples_by_concept.get(evaluated_concept.name, [])
        if len(concept_samples) != len(evaluated_concept.data):
            raise ValueError(
                "Pixel-level benchmark samples do not match the evaluated concept batch. "
                f"Concept '{evaluated_concept.name}' has {len(evaluated_concept.data)} examples in the scenario "
                f"but {len(concept_samples)} indexed benchmark test samples."
            )

        score_maps = np.asarray(score_maps_fn(evaluated_concept.data))
        if score_maps.ndim != 3:
            raise ValueError(
                f"Expected score_maps() to return a 3D array with shape (N, H, W), got {score_maps.shape}"
            )

        masks, kept_indices = load_ground_truth_masks_for_samples(
            concept_samples,
            resize_to=(int(score_maps.shape[1]), int(score_maps.shape[2])),
            skip_missing_anomaly_masks=self._skip_missing_anomaly_masks,
        )

        if len(kept_indices) == 0:
            metric_values = {metric.name(): float("nan") for metric in self._base_metrics}
        else:
            resolved_threshold = None
            if self._has_thresholded_metrics():
                resolved_threshold = self._resolve_threshold(
                    model=model,
                    cache_key=threshold_key,
                    train_concepts=threshold_train_concepts,
                )
                self._apply_runtime_threshold(resolved_threshold)

            metric_values = {}
            for metric in self._base_metrics:
                metric_values[metric.name()] = metric.compute(
                    anomaly_scores=score_maps[kept_indices],
                    y_pred=np.asarray([], dtype=np.uint8),
                    y_true=masks,
                )

        for metric in self._base_metrics:
            self._metric_matrices[metric.name()][self._learned_concepts[-1]][evaluated_concept.name] = metric_values[
                metric.name()
            ]

    def info(self) -> Dict[str, Any]:
        payload = {}
        for base_metric in self._base_metrics:
            metric_matrix = self._metric_matrices[base_metric.name()]
            concept_level_matrix = self._transform_to_ordered_matrix(metric_matrix, self._learned_concepts)
            lifelong_learning_metrics = {metric.name(): metric.compute(concept_level_matrix) for metric in self._metrics}
            payload[f"pixel_concept_metric_callback_{base_metric.name()}"] = {
                "base_metric_name": base_metric.name(),
                "evaluation_level": "pixel",
                "threshold_selection": {
                    "mode": self._threshold_mode,
                    "fixed_threshold": self._fixed_threshold,
                    "train_quantile": self._threshold_quantile,
                    "resolved_thresholds": {
                        concept_name: self._resolved_thresholds[concept_name]
                        for concept_name in self._learned_concepts
                        if concept_name in self._resolved_thresholds
                    },
                },
                "metrics": lifelong_learning_metrics,
                "concepts_order": self._learned_concepts,
                "metric_matrix": metric_matrix,
            }
        return payload

    def _resolve_model_context(self, concept_name: str):
        models = getattr(self._strategy, "_models", None)
        if isinstance(models, dict) and concept_name in models:
            train_concepts = []
            if concept_name in self._training_concepts:
                train_concepts.append(self._training_concepts[concept_name])
            return models[concept_name], concept_name, train_concepts

        current_model = getattr(self._strategy, "current_model", None)
        if callable(current_model):
            model = current_model()
            if model is not None:
                return model, self._learned_concepts[-1], self._seen_training_concepts()

        model = getattr(self._strategy, "_model", None)
        if model is not None:
            return model, self._learned_concepts[-1], self._seen_training_concepts()

        raise AttributeError(
            f"Could not resolve a model instance from strategy '{type(self._strategy).__name__}' for pixel evaluation"
        )

    def _seen_training_concepts(self) -> List[Concept]:
        return [
            self._training_concepts[concept_name]
            for concept_name in self._learned_concepts
            if concept_name in self._training_concepts
        ]

    def _has_thresholded_metrics(self) -> bool:
        return any(callable(getattr(metric, "set_runtime_threshold", None)) for metric in self._base_metrics)

    def _apply_runtime_threshold(self, threshold: float) -> None:
        for metric in self._base_metrics:
            setter = getattr(metric, "set_runtime_threshold", None)
            if callable(setter):
                setter(threshold)

    def _resolve_threshold(self, model: Any, cache_key: str, train_concepts: Sequence[Concept]) -> float:
        if cache_key in self._resolved_thresholds:
            return self._resolved_thresholds[cache_key]

        score_maps_fn = getattr(model, "score_maps", None)
        if not callable(score_maps_fn):
            raise AttributeError(
                f"Model '{type(model).__name__}' does not expose score_maps(), so adaptive pixel thresholding is unavailable"
            )

        flattened_train_scores: List[np.ndarray] = []
        for concept in train_concepts:
            if len(concept.data) == 0:
                continue
            concept_maps = np.asarray(score_maps_fn(concept.data))
            if concept_maps.ndim != 3:
                raise ValueError(
                    f"Expected score_maps() to return a 3D array with shape (N, H, W), got {concept_maps.shape}"
                )
            flattened_train_scores.append(concept_maps.reshape(-1).astype(np.float32, copy=False))

        train_scores = (
            np.concatenate(flattened_train_scores, axis=0)
            if flattened_train_scores
            else np.asarray([], dtype=np.float32)
        )
        threshold = resolve_pixel_threshold(
            train_scores,
            mode=self._threshold_mode,
            fixed_threshold=self._fixed_threshold,
            quantile=self._threshold_quantile,
        )
        self._resolved_thresholds[cache_key] = threshold
        return threshold

    @staticmethod
    def _transform_to_ordered_matrix(
        metric_matrix: Dict[str, Dict[str, float]], concepts_order: List[str]
    ) -> ConceptLevelMatrix:
        if len(concepts_order) == 0:
            return [[]]

        values = []
        for learned_concept in concepts_order:
            values.append([])
            for evaluated_concept in concepts_order:
                values[-1].append(metric_matrix[learned_concept][evaluated_concept])
        return values


class _MissingMetric(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        raise RuntimeError("A base metric must be provided for pixel-level evaluation")

    def name(self) -> str:
        return "MissingMetric"
