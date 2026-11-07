"""Rectangular concept-metric callback with schedule-aware metrics.

Extends :class:`RectangularConceptMetricCallback` (which only supports the
classic ``compute(matrix) -> float`` signature) with a parallel slot for
metrics that need a per-column ``first_seen_step`` mapping, such as
:class:`RectangularForgettingMeasure`,
:class:`RectangularForwardTransfer`, and
:class:`RectangularNewTaskAcquisition`.

The mapping is supplied at callback construction time as a
``Dict[str, int]`` keyed by *test concept name*. Internally the callback
realigns the mapping to its own ``_seen_test_concepts`` order before
calling each schedule-aware metric.

This keeps backward compatibility:

* legacy metrics (e.g. :class:`FinalStepAverage`) are passed via
  ``metrics`` and called as ``metric.compute(matrix)``;
* new metrics that need the mask are passed via
  ``schedule_aware_metrics`` and called as
  ``metric.compute(matrix, first_seen_steps)``.

Nothing in :class:`RectangularConceptMetricCallback` is modified.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from pyclad.callbacks.evaluation.rectangular_concept_metric import (
    RectangularConceptMetricCallback,
)
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import ConceptLevelMetric


class RectangularConceptMetricCallbackWithMask(RectangularConceptMetricCallback):
    """Variant that also computes schedule-aware metrics keyed by column."""

    def __init__(
        self,
        base_metric: BaseMetric,
        metrics: Iterable[ConceptLevelMetric] = (),
        schedule_aware_metrics: Iterable[Any] = (),
        first_seen_step: Mapping[str, int] | None = None,
    ):
        super().__init__(base_metric=base_metric, metrics=metrics)
        self._schedule_aware_metrics = list(schedule_aware_metrics)
        self._first_seen_step: Dict[str, int] = dict(first_seen_step or {})

    def info(self) -> Dict[str, Any]:
        payload = super().info()
        # Append schedule-aware results to the same callback payload so the
        # JSON snapshot stays compact and consumers find everything in one place.
        key = next(iter(payload))

        concept_level_matrix = RectangularConceptMetricCallback._transform_to_ordered_matrix(
            self._metric_matrix,
            train_order=self._learned_train_concepts,
            test_order=self._seen_test_concepts,
        )

        unresolved = self._unresolved_test_concepts()
        schedule_aware_results: Dict[str, float] = {}
        if not unresolved and self._schedule_aware_metrics:
            aligned_first_seen = self._aligned_first_seen()
            for metric in self._schedule_aware_metrics:
                schedule_aware_results[metric.name()] = float(
                    metric.compute(concept_level_matrix, aligned_first_seen)
                )

        payload[key]["schedule_aware_metrics"] = schedule_aware_results
        payload[key]["first_seen_step"] = {
            name: self._first_seen_step.get(name) for name in self._seen_test_concepts
        }
        if unresolved:
            payload[key]["schedule_aware_skipped_reason"] = (
                "first_seen_step missing for one or more test concepts; "
                "schedule-aware metrics not computed."
            )
            payload[key]["schedule_aware_unresolved_test_concepts"] = list(unresolved)
        return payload

    def _aligned_first_seen(self) -> List[int]:
        return [
            int(self._first_seen_step.get(name, -1))
            for name in self._seen_test_concepts
        ]

    def _unresolved_test_concepts(self) -> List[str]:
        return [
            name
            for name in self._seen_test_concepts
            if name not in self._first_seen_step
        ]
