import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)


class ForgettingMeasure(ConceptLevelMetric):
    """Forgetting Measure (Chaudhry et al., 2018).

    For each evaluated task ``k`` (column) seen at least once before the final
    training step ``T-1``, forgetting is

        f_k = max_{j in [k, T-2]} M[j][k] - M[T-1][k]

    The metric averages ``f_k`` across columns. Works on both square (N x N)
    and rectangular (T x N) matrices: rows index training steps in order, and
    columns index the test concepts evaluated after every step.

    Lower is better.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        rows = len(metric_matrix)
        if rows < 2:
            return 0.0

        cols = len(metric_matrix[0])
        if cols == 0:
            return 0.0

        final_row = metric_matrix[-1]
        values = []
        for col in range(cols):
            history = [metric_matrix[row][col] for row in range(rows - 1)]
            history = [value for value in history if not _is_nan(value)]
            if not history:
                continue

            peak = max(history)
            final = final_row[col]
            if _is_nan(final):
                continue
            values.append(peak - final)

        return float(np.nanmean(values)) if values else 0.0

    def name(self) -> str:
        return "ForgettingMeasure"


def _is_nan(value: float) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
