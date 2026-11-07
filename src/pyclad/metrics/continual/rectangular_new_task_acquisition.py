"""Per-category acquisition score for rectangular (T x N) metric matrices.

For each evaluated category ``k`` with ``first_seen_step = s_k``,
``NewTaskAcquisition`` reads the value at row ``s_k`` and column ``k``:

    nta_k = M[s_k][k]

This is the model's performance on ``k`` *immediately after first training
on it*, before any subsequent step has had a chance to interfere. Averaging
nta_k across columns yields a single "how well does the model acquire new
tasks" score, independent of forgetting.

Together with :class:`RectangularForgettingMeasure`, this disentangles two
distinct failure modes:

* low NewTaskAcquisition -> the model never learned the task well in the
  first place (plasticity / capacity bottleneck);
* high forgetting with high acquisition -> the model learned but then
  forgot (stability bottleneck).

Higher is better.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix


class RectangularNewTaskAcquisition:
    def compute(
        self,
        metric_matrix: ConceptLevelMatrix,
        first_seen_steps: Sequence[int],
    ) -> float:
        rows = len(metric_matrix)
        if rows == 0:
            return 0.0
        cols = len(metric_matrix[0])
        if cols == 0:
            return 0.0
        if len(first_seen_steps) != cols:
            raise ValueError(
                f"first_seen_steps has length {len(first_seen_steps)} "
                f"but matrix has {cols} columns"
            )

        values = []
        for col in range(cols):
            first_seen = int(first_seen_steps[col])
            if first_seen < 0 or first_seen >= rows:
                continue
            value = metric_matrix[first_seen][col]
            if _is_nan(value):
                continue
            values.append(float(value))

        return float(np.nanmean(values)) if values else 0.0

    def name(self) -> str:
        return "RectangularNewTaskAcquisition"


def _is_nan(value: float) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
