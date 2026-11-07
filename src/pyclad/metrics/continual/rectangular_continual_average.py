"""Continual ROC-AUC (ROC) for rectangular (T x N) metric matrices.

In a grouped-component scenario a training step can introduce several
components, so the metric matrix has T training steps (rows) and N
evaluated components (columns), and ``first_seen_steps[c]`` is the step
f(c) that introduced component c. The paper (appendix on grouped-component
scenarios) defines the continual ROC-AUC as the mean of R[s][c] over all
steps s >= f(c):

    ROC = mean { R[s][c] : c = 0..N-1, f(c) <= s <= T-1 }

i.e. a single mean over every (s, c) entry with s >= f(c), not a mean of
per-column means. NaN or None entries are skipped. This is the ``ca`` value
of ``examples/clvad/camera_ready/build_grouped_tables.py``.

:class:`ContinualAverage` indexes the lower triangle by column position
(column j counts from row j on), which equals this definition only for a
square matrix with ``first_seen_steps == [0, 1, ..., N-1]``. Use this class
for step-scheduled (rectangular) matrices.

Higher is better. Returns NaN when no entry qualifies.
"""

from __future__ import annotations

import math
from typing import Sequence

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix


class RectangularContinualAverage:
    def compute(
        self,
        metric_matrix: ConceptLevelMatrix,
        first_seen_steps: Sequence[int],
    ) -> float:
        rows = len(metric_matrix)
        if rows == 0:
            return math.nan
        cols = len(metric_matrix[0])
        if len(first_seen_steps) != cols:
            raise ValueError(
                f"first_seen_steps has length {len(first_seen_steps)} "
                f"but matrix has {cols} columns"
            )

        values = []
        for col in range(cols):
            first_seen = _first_seen(first_seen_steps[col])
            for row in range(first_seen, rows):
                value = metric_matrix[row][col]
                if not _is_missing(value):
                    values.append(float(value))

        return sum(values) / len(values) if values else math.nan

    def name(self) -> str:
        return "RectangularContinualAverage"


def _first_seen(value) -> int:
    first_seen = int(value)
    if first_seen < 0:
        raise ValueError(f"first_seen_steps must be non-negative, got {first_seen}")
    return first_seen


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(value)
    except TypeError:
        return False
