import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)


class FinalStepAverage(ConceptLevelMetric):
    """Average of the metric over all test concepts at the final training step.

    Matches the ``A-AUROC`` figure used in CDAD-style continual anomaly
    detection benchmarks: train through the full sequence, then average the
    base metric across all test concepts after the last training step
    (i.e., the last row of the matrix).

    Works on both square (N x N) and rectangular (T x N) matrices.

    Higher is better.
    """

    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        if len(metric_matrix) == 0:
            return 0.0

        final_row = metric_matrix[-1]
        if len(final_row) == 0:
            return 0.0

        return float(np.nanmean(final_row))

    def name(self) -> str:
        return "FinalStepAverage"
