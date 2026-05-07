from pathlib import Path

import numpy as np

from pyclad.callbacks.evaluation.visual_pixel_concept_metric_callback import (
    VisualPixelConceptMetricCallback,
)
from pyclad.data.concept import Concept
from pyclad.metrics.base.pixel_f1_score import PixelF1Score
from pyclad.metrics.base.pixel_roc_auc import PixelRocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage


def _write_rgb_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (4, 4)):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    array[..., 0] = color[0]
    array[..., 1] = color[1]
    array[..., 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, mask: np.ndarray):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


class _ModelWithMaps:
    def score_maps(self, data: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [[0.95, 0.05], [0.05, 0.05]],
                [[0.05, 0.05], [0.05, 0.05]],
            ],
            dtype=np.float32,
        )


class _AdaptiveThresholdModel:
    def score_maps(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        flattened = data.reshape(len(data), -1)
        maps = []
        for sample in flattened:
            first_value = float(sample[0])
            if first_value < 0.5:
                maps.append([[0.10, 0.20], [0.30, 0.40]])
            elif first_value < 1.5:
                maps.append([[0.95, 0.05], [0.05, 0.05]])
            else:
                maps.append([[0.05, 0.05], [0.05, 0.05]])
        return np.asarray(maps, dtype=np.float32)


class _StrategyStub:
    def __init__(self, model):
        self._model = model


def test_visual_pixel_concept_metric_callback_computes_per_concept_metric(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png", (10, 20, 30))
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png", (20, 30, 40))
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png", (30, 40, 50))

    anomaly_mask = np.zeros((4, 4), dtype=np.uint8)
    anomaly_mask[:2, :2] = 255
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png", anomaly_mask)

    callback = VisualPixelConceptMetricCallback(
        strategy=_StrategyStub(_ModelWithMaps()),
        benchmark="mvtec",
        root=root,
        base_metric=PixelRocAuc(),
        metrics=[ContinualAverage()],
    )

    concept = Concept(
        name="widget",
        data=np.random.default_rng(0).random((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
    )

    callback.after_training(Concept(name="widget", data=np.array([]), labels=np.array([])))
    callback.after_evaluation(
        evaluated_concept=concept,
        y_true=concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
    )

    info = callback.info()["pixel_concept_metric_callback_Pixel-ROC-AUC"]

    assert info["evaluation_level"] == "pixel"
    assert info["concepts_order"] == ["widget"]
    assert info["metric_matrix"]["widget"]["widget"] == 1.0


def test_visual_pixel_concept_metric_callback_supports_multiple_base_metrics(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png", (10, 20, 30))
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png", (20, 30, 40))
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png", (30, 40, 50))

    anomaly_mask = np.zeros((4, 4), dtype=np.uint8)
    anomaly_mask[:2, :2] = 255
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png", anomaly_mask)

    callback = VisualPixelConceptMetricCallback(
        strategy=_StrategyStub(_ModelWithMaps()),
        benchmark="mvtec",
        root=root,
        base_metrics=[PixelRocAuc(), PixelF1Score(threshold=0.5)],
        metrics=[ContinualAverage()],
    )

    concept = Concept(
        name="widget",
        data=np.random.default_rng(0).random((2, 2, 2, 3), dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
    )

    callback.after_training(Concept(name="widget", data=np.array([]), labels=np.array([])))
    callback.after_evaluation(
        evaluated_concept=concept,
        y_true=concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
    )

    info = callback.info()

    assert info["pixel_concept_metric_callback_Pixel-ROC-AUC"]["metric_matrix"]["widget"]["widget"] == 1.0
    assert info["pixel_concept_metric_callback_Pixel-F1-Score"]["metric_matrix"]["widget"]["widget"] == 1.0


def test_visual_pixel_callback_supports_train_quantile_thresholding(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png", (10, 20, 30))
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png", (20, 30, 40))
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png", (30, 40, 50))

    anomaly_mask = np.zeros((4, 4), dtype=np.uint8)
    anomaly_mask[:2, :2] = 255
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png", anomaly_mask)

    callback = VisualPixelConceptMetricCallback(
        strategy=_StrategyStub(_AdaptiveThresholdModel()),
        benchmark="mvtec",
        root=root,
        base_metrics=[PixelF1Score(threshold=0.99)],
        metrics=[ContinualAverage()],
        threshold_mode="train-quantile",
        fixed_threshold=0.99,
        threshold_quantile=0.75,
    )

    train_concept = Concept(
        name="widget",
        data=np.zeros((1, 2, 2, 3), dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
    )
    test_concept = Concept(
        name="widget",
        data=np.stack(
            [
                np.ones((2, 2, 3), dtype=np.float32),
                np.full((2, 2, 3), 2.0, dtype=np.float32),
            ],
            axis=0,
        ),
        labels=np.array([1, 0], dtype=np.int64),
    )

    callback.after_training(train_concept)
    callback.after_evaluation(
        evaluated_concept=test_concept,
        y_true=test_concept.labels,
        y_pred=np.array([1, 0], dtype=np.int64),
        anomaly_scores=np.array([0.95, 0.05], dtype=np.float32),
    )

    info = callback.info()["pixel_concept_metric_callback_Pixel-F1-Score"]
    resolved_threshold = info["threshold_selection"]["resolved_thresholds"]["widget"]

    assert np.isclose(resolved_threshold, 0.325)
    assert info["metric_matrix"]["widget"]["widget"] == 1.0
