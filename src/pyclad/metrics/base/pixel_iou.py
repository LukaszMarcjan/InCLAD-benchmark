from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.base.pixel_threshold_utils import binary_confusion, threshold_pixel_scores


class PixelIoU(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self._runtime_threshold: float | None = None

    def set_runtime_threshold(self, threshold: float | None) -> None:
        self._runtime_threshold = None if threshold is None else float(threshold)

    def active_threshold(self) -> float:
        return self.threshold if self._runtime_threshold is None else self._runtime_threshold

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        y_pred_binary = threshold_pixel_scores(anomaly_scores, self.active_threshold())
        tp, fp, fn = binary_confusion(y_true, y_pred_binary)

        denominator = tp + fp + fn
        if denominator == 0:
            return 1.0
        return float(tp / denominator)

    def name(self) -> str:
        return "Pixel-IoU"
