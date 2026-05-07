import importlib.util
import sys
from pathlib import Path

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.diagonal_average import DiagonalAverage
from pyclad.metrics.continual.forward_transfer import ForwardTransfer


def _load_visual_runner_module():
    module_path = Path(__file__).resolve().parents[3] / "examples" / "clvad" / "run_continual_visual_ad.py"
    spec = importlib.util.spec_from_file_location("test_visual_runner_module_for_metrics", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeBaseMetric:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name


class _FakePixelMetricCallback:
    def __init__(self):
        base_metric = _FakeBaseMetric("PixelROC-AUC")
        self._base_metrics = [base_metric]
        self._metric_matrices = {base_metric.name(): {"a": {"a": 0.7}}}
        self._metrics = [ContinualAverage(), BackwardTransfer(), ForwardTransfer(), DiagonalAverage()]
        self._learned_concepts = ["a", "b"]
        self._threshold_mode = "train-quantile"
        self._fixed_threshold = 0.5
        self._threshold_quantile = 0.995
        self._resolved_thresholds = {"a": 0.42, "b": 0.77}


def test_runner_snapshot_trims_incomplete_image_metric_row():
    runner = _load_visual_runner_module()
    callback = ConceptMetricCallback(
        base_metric=_FakeBaseMetric("ROC-AUC"),
        metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer(), DiagonalAverage()],
    )
    callback._learned_concepts = ["a", "b"]
    callback._metric_matrix = {"a": {"a": 0.9}}

    payload = runner.build_snapshot_payload([callback], completed_concepts=["a"])

    assert payload["concept_metric_callback_ROC-AUC"]["concepts_order"] == ["a"]
    assert payload["concept_metric_callback_ROC-AUC"]["metric_matrix"] == {"a": {"a": 0.9}}
    assert payload["concept_metric_callback_ROC-AUC"]["metrics"]["DiagonalAverage"] == 0.9


def test_runner_snapshot_trims_incomplete_pixel_metric_row():
    runner = _load_visual_runner_module()
    callback = _FakePixelMetricCallback()

    payload = runner.build_snapshot_payload([callback], completed_concepts=["a"])

    assert payload["pixel_concept_metric_callback_PixelROC-AUC"]["concepts_order"] == ["a"]
    assert payload["pixel_concept_metric_callback_PixelROC-AUC"]["metric_matrix"] == {"a": {"a": 0.7}}
    assert payload["pixel_concept_metric_callback_PixelROC-AUC"]["threshold_selection"]["resolved_thresholds"] == {
        "a": 0.42
    }
