"""Pixel-level rectangular callback with schedule-aware metrics.

Same goal as :class:`RectangularConceptMetricCallbackWithMask` for the
image-level pipeline: extend the pixel-level rectangular callback so that
the JSON snapshot also reports schedule-aware metrics
(:class:`RectangularForgettingMeasure`,
:class:`RectangularForwardTransfer`,
:class:`RectangularNewTaskAcquisition`).

Implementation note (important)
-------------------------------
The base example runner (``examples/clvad/run_continual_visual_ad.py``)
duck-types pixel callbacks during JSON serialization:

    if (hasattr(provider, "_base_metrics") and
        hasattr(provider, "_metric_matrices") and
        hasattr(provider, "_resolved_thresholds")):
        return _serialize_visual_pixel_metric_callback(provider, ...)

If we *inherit* from :class:`RectangularVisualPixelConceptMetricCallback`,
those three attributes get exposed on the subclass instance and the
duck-typing check fires first — bypassing our ``info()`` and silently
dropping the schedule-aware metrics from the snapshot.

To avoid touching the base runner we use **composition** instead: we hold
an inner :class:`RectangularVisualPixelConceptMetricCallback`, forward
the Callback hooks to it, but do *not* re-expose those private attributes
at the wrapper level. The wrapper is therefore serialized via the
``provider.info()`` fallback path, and our augmented payload makes it
into the JSON intact.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.evaluation.rectangular_visual_pixel_concept_metric_callback import (
    RectangularVisualPixelConceptMetricCallback,
)
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import ConceptLevelMetric
from pyclad.output.output_writer import InfoProvider


class RectangularVisualPixelConceptMetricCallbackWithMask(Callback, InfoProvider):
    """Wraps the rectangular pixel callback and emits schedule-aware metrics.

    Construct it exactly like :class:`RectangularVisualPixelConceptMetricCallback`
    plus two extras:

    schedule_aware_metrics:
        Iterable of metrics that take ``(matrix, first_seen_steps)``.
    first_seen_step:
        Mapping from test concept name to the index of the training step
        that first introduced that concept (use
        :func:`pyclad.data.readers.visual_step_schedule.compute_first_seen_step`).
    """

    def __init__(
        self,
        strategy: Any,
        benchmark: str,
        root: Optional[str] = None,
        registry_path: Optional[str] = None,
        categories: Optional[Sequence[str]] = None,
        max_test_samples_per_category: Optional[int] = None,
        base_metric: Optional[BaseMetric] = None,
        base_metrics: Optional[Sequence[BaseMetric]] = None,
        metrics: Iterable[ConceptLevelMetric] = (),
        schedule_aware_metrics: Iterable[Any] = (),
        first_seen_step: Mapping[str, int] | None = None,
        skip_missing_anomaly_masks: bool = True,
        threshold_mode: str = "fixed",
        fixed_threshold: float = 0.5,
        threshold_quantile: float = 0.995,
    ):
        self._inner = RectangularVisualPixelConceptMetricCallback(
            strategy=strategy,
            benchmark=benchmark,
            root=root,
            registry_path=registry_path,
            categories=categories,
            max_test_samples_per_category=max_test_samples_per_category,
            base_metric=base_metric,
            base_metrics=base_metrics,
            metrics=metrics,
            skip_missing_anomaly_masks=skip_missing_anomaly_masks,
            threshold_mode=threshold_mode,
            fixed_threshold=fixed_threshold,
            threshold_quantile=threshold_quantile,
        )
        self._schedule_aware_metrics = list(schedule_aware_metrics)
        self._first_seen_step: Dict[str, int] = dict(first_seen_step or {})

    # -- Callback hooks: delegate to the inner pixel callback. --
    def after_training(self, learned_concept: Concept) -> None:
        self._inner.after_training(learned_concept)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true,
        y_pred,
        anomaly_scores,
        *args,
        **kwargs,
    ) -> None:
        self._inner.after_evaluation(
            evaluated_concept=evaluated_concept,
            y_true=y_true,
            y_pred=y_pred,
            anomaly_scores=anomaly_scores,
            *args,
            **kwargs,
        )

    # -- Snapshot output: take inner.info() and augment with schedule-aware metrics. --
    def info(self) -> Dict[str, Any]:
        payload = self._inner.info()

        seen_test_concepts: Sequence[str] = self._inner._seen_test_concepts
        unresolved = [
            name for name in seen_test_concepts if name not in self._first_seen_step
        ]
        aligned_first_seen = [
            int(self._first_seen_step.get(name, -1)) for name in seen_test_concepts
        ]

        for key, entry in payload.items():
            metric_matrix_dict = entry.get("metric_matrix", {})
            train_order = entry.get("train_order", [])
            test_order = entry.get("test_order", [])
            concept_level_matrix = _build_rect_matrix(
                metric_matrix_dict, train_order, test_order
            )

            schedule_aware_results: Dict[str, float] = {}
            if not unresolved and self._schedule_aware_metrics:
                for metric in self._schedule_aware_metrics:
                    schedule_aware_results[metric.name()] = float(
                        metric.compute(concept_level_matrix, aligned_first_seen)
                    )

            entry["schedule_aware_metrics"] = schedule_aware_results
            entry["first_seen_step"] = {
                name: self._first_seen_step.get(name) for name in seen_test_concepts
            }
            if unresolved:
                entry["schedule_aware_skipped_reason"] = (
                    "first_seen_step missing for one or more test concepts; "
                    "schedule-aware metrics not computed."
                )
                entry["schedule_aware_unresolved_test_concepts"] = list(unresolved)

        return payload


def _build_rect_matrix(
    metric_matrix_dict: Mapping[str, Mapping[str, float]],
    train_order: Sequence[str],
    test_order: Sequence[str],
) -> List[List[float]]:
    if not train_order or not test_order:
        return [[]]
    values: List[List[float]] = []
    for learned in train_order:
        row = []
        for evaluated in test_order:
            row.append(
                metric_matrix_dict.get(learned, {}).get(evaluated, float("nan"))
            )
        values.append(row)
    return values
