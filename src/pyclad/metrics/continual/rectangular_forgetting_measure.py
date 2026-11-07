"""Schedule-aware Forgetting Measure for rectangular (T x N) metric matrices.

The standard :class:`ForgettingMeasure` assumes every column has been "seen"
by the very first training step, which is true for square (N x N) matrices
where each task is its own training step. In step-scheduled scenarios
(CDAD-style: ``14-1``, ``10-5``, ``3x5``, ``10-1x5``, ...) the training
schedule groups multiple categories into one step, while evaluation stays
per category. The matrix becomes T x N with T < N and columns introduced at
later training steps have "pre-training" rows whose values reflect
zero-shot/out-of-distribution behaviour, not forgetting.

Averaging the naive ForgettingMeasure over such columns biases the metric
toward zero (or even negative), because for not-yet-trained columns the
``peak - final`` quantity is negative (the model improved when it finally
saw the category). This metric fixes that by:

* restricting the per-column history to rows ``[first_seen_step, T-2]``;
* skipping columns whose ``first_seen_step >= T-1`` (never previously
  trained before the last step, so "forgetting" is undefined).

For each evaluated category ``k`` with ``first_seen_step = s_k <= T - 2``:

    f_k = max_{j in [s_k, T - 2]} M[j][k] - M[T - 1][k]

The metric averages ``f_k`` across the qualifying columns. Lower is better.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix


class RectangularForgettingMeasure:
    """Forgetting Measure restricted to columns trained before the final step.

    Use this in place of :class:`ForgettingMeasure` when the metric matrix is
    rectangular (T x N) because of a step schedule. For square matrices the
    two behave identically as long as ``first_seen_steps`` is the identity
    mapping ``[0, 1, ..., N-1]``.
    """

    def compute(
        self,
        metric_matrix: ConceptLevelMatrix,
        first_seen_steps: Sequence[int],
    ) -> float:
        rows = len(metric_matrix)
        if rows < 2:
            return 0.0

        cols = len(metric_matrix[0])
        if cols == 0:
            return 0.0
        if len(first_seen_steps) != cols:
            raise ValueError(
                f"first_seen_steps has length {len(first_seen_steps)} "
                f"but matrix has {cols} columns"
            )

        final_row = metric_matrix[-1]
        last_train_row = rows - 1
        values = []
        for col in range(cols):
            first_seen = int(first_seen_steps[col])
            # Need at least one row in [first_seen, T-2] to compute a peak.
            if first_seen >= last_train_row:
                continue

            history = [
                metric_matrix[row][col]
                for row in range(first_seen, last_train_row)
            ]
            history = [value for value in history if not _is_nan(value)]
            if not history:
                continue

            final = final_row[col]
            if _is_nan(final):
                continue

            values.append(max(history) - final)

        return float(np.nanmean(values)) if values else 0.0

    def name(self) -> str:
        return "RectangularForgettingMeasure"


def _is_nan(value: float) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
