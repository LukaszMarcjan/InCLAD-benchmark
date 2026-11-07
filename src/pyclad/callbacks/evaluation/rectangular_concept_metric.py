"""Concept-metric callback for rectangular T x N matrices.

The default :class:`ConceptMetricCallback` assumes the train concepts and the
test concepts share names (square N x N matrix). When training concepts are
grouped via a step schedule but evaluation is still per category, the matrix
becomes T x N with T < N and the names do not align.

This callback tracks the train order and the test order independently and
emits a TxN matrix that the standard continual metrics consume row-major.
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)
from pyclad.output.output_writer import InfoProvider


class RectangularConceptMetricCallback(Callback, InfoProvider):
    def __init__(self, base_metric: BaseMetric, metrics: Iterable[ConceptLevelMetric]):
        self._base_metric: BaseMetric = base_metric
        self._metric_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._learned_train_concepts: List[str] = []
        self._seen_test_concepts: List[str] = []
        self._metrics = list(metrics)

    def after_training(self, learned_concept: Concept):
        self._learned_train_concepts.append(learned_concept.name)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        *args,
        **kwargs,
    ):
        train_key = self._learned_train_concepts[-1]
        assert (
            evaluated_concept.name not in self._metric_matrix[train_key]
        ), "The same concept should not be evaluated twice after the same learned concept"

        metric_value = self._base_metric.compute(
            anomaly_scores=anomaly_scores, y_true=y_true, y_pred=y_pred
        )
        self._metric_matrix[train_key][evaluated_concept.name] = metric_value

        if evaluated_concept.name not in self._seen_test_concepts:
            self._seen_test_concepts.append(evaluated_concept.name)

    def info(self) -> Dict[str, Any]:
        concept_level_matrix = self._transform_to_ordered_matrix(
            self._metric_matrix,
            train_order=self._learned_train_concepts,
            test_order=self._seen_test_concepts,
        )
        lifelong_learning_metrics = {
            metric.name(): metric.compute(concept_level_matrix) for metric in self._metrics
        }

        return {
            f"rectangular_concept_metric_callback_{self._base_metric.name()}": {
                "base_metric_name": self._base_metric.name(),
                "metrics": lifelong_learning_metrics,
                "train_order": self._learned_train_concepts,
                "test_order": self._seen_test_concepts,
                "metric_matrix": self._metric_matrix,
            }
        }

    @staticmethod
    def _transform_to_ordered_matrix(
        metric_matrix: Dict[str, Dict[str, float]],
        train_order: List[str],
        test_order: List[str],
    ) -> ConceptLevelMatrix:
        if not train_order or not test_order:
            return [[]]

        values: ConceptLevelMatrix = []
        for learned in train_order:
            row = []
            for evaluated in test_order:
                row.append(metric_matrix.get(learned, {}).get(evaluated, float("nan")))
            values.append(row)
        return values
