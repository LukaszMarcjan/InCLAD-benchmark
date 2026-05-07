import numpy as np
from sklearn.metrics import f1_score

from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.base.pixel_threshold_utils import flatten_binary_masks, threshold_pixel_scores


class PixelF1Score(BaseMetric):
    def __init__(self, threshold: float = 0.5, zero_division: float = 0.0):
        self.threshold = float(threshold)
        self.zero_division = float(zero_division)
        self._runtime_threshold: float | None = None

    def set_runtime_threshold(self, threshold: float | None) -> None:
        self._runtime_threshold = None if threshold is None else float(threshold)

    def active_threshold(self) -> float:
        return self.threshold if self._runtime_threshold is None else self._runtime_threshold

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        y_pred_flat = flatten_binary_masks(threshold_pixel_scores(anomaly_scores, self.active_threshold()))
        y_true_flat = flatten_binary_masks(y_true)
        return float(f1_score(y_true=y_true_flat, y_pred=y_pred_flat, zero_division=self.zero_division))

    def name(self) -> str:
        return "Pixel-F1-Score"
