"""Schedule-aware Forward Transfer for rectangular (T x N) metric matrices.

In step-scheduled scenarios columns introduced at later training steps have
"pre-training" rows whose values reflect how well the model performs on a
category it has *not yet been trained on*. Averaging those values is a
direct measurement of forward transfer (sometimes called "zero-shot" or
"OOD" generalization to future tasks).

For each category ``k`` with ``first_seen_step = s_k > 0``:

    fwt_k = mean_{j in [0, s_k - 1]} M[j][k]

The metric averages ``fwt_k`` across qualifying columns. Columns with
``first_seen_step = 0`` (trained from the very first step) are skipped
because they have no "pre-training" history.

Higher is better. Compare against a model trained from scratch on ``k``;
positive values mean the model transfers prior knowledge productively.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix


class RectangularForwardTransfer:
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

        per_column_means = []
        for col in range(cols):
            first_seen = int(first_seen_steps[col])
            if first_seen <= 0:
                continue
            pre_train_values = [
                metric_matrix[row][col] for row in range(first_seen)
            ]
            pre_train_values = [v for v in pre_train_values if not _is_nan(v)]
            if not pre_train_values:
                continue
            per_column_means.append(float(np.mean(pre_train_values)))

        return float(np.nanmean(per_column_means)) if per_column_means else 0.0

    def name(self) -> str:
        return "RectangularForwardTransfer"


def _is_nan(value: float) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
