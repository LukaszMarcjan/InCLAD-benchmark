"""Backward transfer (BWT) for rectangular (T x N) metric matrices.

In a grouped-component scenario the metric matrix has T training steps
(rows) and N evaluated components (columns), and ``first_seen_steps[c]`` is
the step f(c) that introduced component c. The paper (appendix on
grouped-component scenarios) defines BWT as the mean of
R[s][c] - R[f(c)][c] over all steps s > f(c), the rectangular counterpart
of Eq. (bwt) of the main text:

    BWT = mean { R[s][c] - R[f(c)][c] : c = 0..N-1, f(c) < s <= T-1 }

The mean is a single mean over all qualifying (s, c) pairs. Pairs with a
NaN or None entry are skipped, and BWT is 0.0 when no pair qualifies (for
example a single training step). This is the ``bwt`` value of
``examples/clvad/camera_ready/build_grouped_tables.py``.

:class:`BackwardTransfer` averages step-to-step differences
R[j][i] - R[j-1][i] by column position instead, so it gives different values
on rectangular matrices.

Higher is better; negative values indicate forgetting.
"""

from __future__ import annotations

import math
from typing import Sequence

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix
from pyclad.metrics.continual.rectangular_continual_average import (
    _first_seen,
    _is_missing,
)


class RectangularBackwardTransfer:
    def compute(
        self,
        metric_matrix: ConceptLevelMatrix,
        first_seen_steps: Sequence[int],
    ) -> float:
        rows = len(metric_matrix)
        if rows == 0:
            return 0.0
        cols = len(metric_matrix[0])
        if len(first_seen_steps) != cols:
            raise ValueError(
                f"first_seen_steps has length {len(first_seen_steps)} "
                f"but matrix has {cols} columns"
            )

        values = []
        for col in range(cols):
            first_seen = _first_seen(first_seen_steps[col])
            if first_seen >= rows - 1:
                continue
            reference = metric_matrix[first_seen][col]
            if _is_missing(reference):
                continue
            for row in range(first_seen + 1, rows):
                value = metric_matrix[row][col]
                if not _is_missing(value):
                    values.append(float(value) - float(reference))

        return sum(values) / len(values) if values else 0.0

    def name(self) -> str:
        return "RectangularBackwardTransfer"
