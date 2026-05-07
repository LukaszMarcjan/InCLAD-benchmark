import numpy as np

from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ConceptLevelMetric,
)


class DiagonalAverage(ConceptLevelMetric):
    def compute(self, metric_matrix: ConceptLevelMatrix) -> float:
        concepts_no = len(metric_matrix)
        if concepts_no == 0:
            return 0

        values = [metric_matrix[index][index] for index in range(concepts_no)]
        return float(np.nanmean(values)) if len(values) > 0 else 0

    def name(self) -> str:
        return "DiagonalAverage"
