from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label
from sklearn.metrics import auc

from pyclad.metrics.base.base_metric import BaseMetric


class PixelAUPRO(BaseMetric):
    """Per-Region Overlap area under the FPR-PRO curve.

    Score maps and ground-truth masks are flattened across the batch dimension,
    recall is computed per connected component of the ground truth, and the
    curve is integrated up to ``fpr_limit`` and normalized by it.
    """

    def __init__(
        self,
        fpr_limit: float = 0.3,
        num_thresholds: int = 200,
        target_size: Optional[int] = 256,
    ):
        if not 0.0 < fpr_limit <= 1.0:
            raise ValueError("fpr_limit must be in (0, 1]")
        if num_thresholds < 2:
            raise ValueError("num_thresholds must be at least 2")
        if target_size is not None and target_size <= 0:
            raise ValueError("target_size must be positive or None")
        self._fpr_limit = float(fpr_limit)
        self._num_thresholds = int(num_thresholds)
        self._target_size = target_size

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        scores = np.asarray(anomaly_scores, dtype=np.float32)
        masks = np.asarray(y_true).astype(bool)
        if scores.shape != masks.shape or scores.ndim != 3:
            raise ValueError(
                f"PixelAUPRO expects 3D (N,H,W) arrays with matching shapes, got {scores.shape} vs {masks.shape}"
            )
        if not masks.any() or masks.all():
            return float("nan")

        scores, masks = self._maybe_downscale(scores, masks)

        regions = []
        for image_idx in range(masks.shape[0]):
            labeled, n_regions = cc_label(masks[image_idx])
            for region_id in range(1, n_regions + 1):
                regions.append((image_idx, labeled == region_id))

        if not regions:
            return float("nan")

        normal_mask = ~masks
        normal_scores = scores[normal_mask]
        if normal_scores.size == 0:
            return float("nan")

        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, self._num_thresholds))

        pro_per_threshold = np.zeros_like(thresholds, dtype=np.float64)
        fpr_per_threshold = np.zeros_like(thresholds, dtype=np.float64)

        for t_idx, threshold in enumerate(thresholds):
            pred_pos = scores >= threshold
            recalls = []
            for image_idx, region_mask in regions:
                region_pred = pred_pos[image_idx][region_mask]
                recalls.append(region_pred.mean() if region_pred.size else 0.0)
            pro_per_threshold[t_idx] = float(np.mean(recalls))
            fpr_per_threshold[t_idx] = float((normal_scores >= threshold).mean())

        order = np.argsort(fpr_per_threshold)
        fpr_sorted = fpr_per_threshold[order]
        pro_sorted = pro_per_threshold[order]

        keep = fpr_sorted <= self._fpr_limit
        if keep.sum() < 2:
            return float("nan")

        fpr_clip = np.concatenate([fpr_sorted[keep], [self._fpr_limit]])
        pro_clip = np.concatenate([pro_sorted[keep], [pro_sorted[keep][-1]]])
        return float(auc(fpr_clip, pro_clip) / self._fpr_limit)

    def name(self) -> str:
        return "Pixel-AUPRO"

    def _maybe_downscale(self, scores: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._target_size is None:
            return scores, masks
        height, width = scores.shape[-2:]
        if max(height, width) <= self._target_size:
            return scores, masks

        scale = self._target_size / float(max(height, width))
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))

        scores_t = torch.from_numpy(scores).unsqueeze(1)
        scores_t = F.interpolate(scores_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        scores_resized = scores_t.squeeze(1).numpy().astype(np.float32, copy=False)

        masks_t = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
        masks_t = F.interpolate(masks_t, size=(new_h, new_w), mode="nearest")
        masks_resized = masks_t.squeeze(1).numpy().astype(bool, copy=False)

        return scores_resized, masks_resized
