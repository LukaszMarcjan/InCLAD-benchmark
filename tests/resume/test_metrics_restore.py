"""Rehydrating evaluation callbacks from the partial result JSON.

"Restored intact" is checked three ways: the values match, the container still
behaves like the live one (``defaultdict``, so the next row can be written), and
the duplicate-cell assertion still trips — that assertion is the tripwire which
catches a resume that restored a row and then re-evaluated it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.resume.metrics_restore import (
    load_result_payload,
    prime_json_checkpoint_callback,
    restore_callbacks,
    restore_run_status,
    restore_training_concepts,
    sha256_of_file,
)
from pyclad.resume.protocol import CheckpointResultMismatch

from .conftest import CONCEPTS, CountingMetric, MatrixMean, make_dataset


def _result_payload(completed, *, base: float = 1.0):
    matrix = {
        learned: {evaluated: base + index * 10 + offset for offset, evaluated in enumerate(CONCEPTS)}
        for index, learned in enumerate(completed)
    }
    return {
        "concept_metric_callback_CountingMetric": {
            "base_metric_name": "CountingMetric",
            "metrics": {"MatrixMean": 0.0},
            "concepts_order": list(completed),
            "metric_matrix": matrix,
        },
        "pixel_concept_metric_callback_PixelFake": {
            "base_metric_name": "PixelFake",
            "evaluation_level": "pixel",
            "threshold_selection": {
                "mode": "train_quantile",
                "fixed_threshold": 0.5,
                "train_quantile": 0.995,
                "resolved_thresholds": {name: 7.7235 + index for index, name in enumerate(completed)},
            },
            "metrics": {},
            "concepts_order": list(completed),
            "metric_matrix": matrix,
        },
        "time_evaluation_callback": {
            "time_by_concept": {name: {"train_time": 100.0 + i, "eval_time": 5.0} for i, name in enumerate(CONCEPTS)},
            "train_time_total": sum(100.0 + i for i in range(len(completed))),
            "eval_time_total": 42.0,
        },
    }


class _PixelFake:
    def name(self) -> str:
        return "PixelFake"

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return 0.0


class _PixelCallbackStub:
    """Same attribute surface the runner's snapshot serializer duck-types on."""

    def __init__(self):
        self._base_metrics = [_PixelFake()]
        self._metric_matrices = {"PixelFake": defaultdict(dict)}
        self._learned_concepts = []
        self._training_concepts = {}
        self._resolved_thresholds = {}
        self._metrics = []


def test_concept_metric_matrix_is_restored_intact():
    completed = list(CONCEPTS[:3])
    callback = ConceptMetricCallback(base_metric=CountingMetric(), metrics=[MatrixMean()])

    report = restore_callbacks([callback], _result_payload(completed), completed)

    assert callback._learned_concepts == completed
    assert callback._metric_matrix["bottle"]["bottle"] == 1.0
    assert callback._metric_matrix["capsule"]["grid"] == pytest.approx(1.0 + 20 + 4)
    assert [entry for entry in report.restored if "ConceptMetricCallback" in entry]

    # The container must still be a live defaultdict: the next concept's row is
    # written with self._metric_matrix[name][...] = value and would KeyError.
    callback.after_training(Concept(name="carpet", data=np.zeros((1, 2))))
    callback.after_evaluation(
        evaluated_concept=Concept(name="bottle", data=np.zeros((1, 2))),
        y_true=np.array([0]),
        y_pred=np.array([0]),
        anomaly_scores=np.array([0.5]),
    )
    assert callback._metric_matrix["carpet"]["bottle"] == pytest.approx(0.5)


def test_restored_rows_still_trip_the_double_evaluation_assertion():
    completed = list(CONCEPTS[:2])
    callback = ConceptMetricCallback(base_metric=CountingMetric(), metrics=[])
    restore_callbacks([callback], _result_payload(completed), completed)

    # Pretend a buggy resume re-evaluates a row it just restored.
    with pytest.raises(AssertionError, match="should not be evaluated twice"):
        callback.after_evaluation(
            evaluated_concept=Concept(name="bottle", data=np.zeros((1, 2))),
            y_true=np.array([0]),
            y_pred=np.array([0]),
            anomaly_scores=np.array([0.5]),
        )


def test_pixel_thresholds_and_training_concepts_are_restored():
    completed = list(CONCEPTS[:3])
    callback = _PixelCallbackStub()
    dataset = make_dataset()

    restore_callbacks([callback], _result_payload(completed), completed)
    restore_training_concepts([callback], dataset.train_concepts(), completed)

    assert callback._learned_concepts == completed
    # Restored, never recomputed: recomputing against a model that is now at a
    # later concept can give a different threshold and shift every pixel metric.
    assert callback._resolved_thresholds["bottle"] == pytest.approx(7.7235)
    assert set(callback._training_concepts) == set(completed)
    assert isinstance(callback._metric_matrices["PixelFake"], defaultdict)


def test_pixel_callback_with_a_changed_metric_set_aborts():
    completed = list(CONCEPTS[:2])
    callback = _PixelCallbackStub()
    payload = _result_payload(completed)
    del payload["pixel_concept_metric_callback_PixelFake"]

    with pytest.raises(CheckpointResultMismatch, match="no block for this pixel metric"):
        restore_callbacks([callback], payload, completed)


def test_time_callback_keeps_the_earlier_segments_wall_clock():
    completed = list(CONCEPTS[:3])
    callback = TimeEvaluationCallback()

    restore_callbacks([callback], _result_payload(completed), completed)

    assert callback._train_time_total == pytest.approx(303.0)
    assert callback._eval_time_total == pytest.approx(42.0)
    assert callback._time_by_concept["bottle"]["train_time"] == pytest.approx(100.0)
    # Still a defaultdict, so an unseen concept does not explode.
    assert callback._time_by_concept["unseen"]["train_time"] == 0


def test_missing_row_for_a_committed_concept_aborts():
    completed = list(CONCEPTS[:3])
    payload = _result_payload(completed)
    del payload["concept_metric_callback_CountingMetric"]["metric_matrix"]["capsule"]
    callback = ConceptMetricCallback(base_metric=CountingMetric(), metrics=[])

    with pytest.raises(CheckpointResultMismatch, match="no metric row"):
        restore_callbacks([callback], payload, completed)


def test_result_json_recorded_under_a_different_order_aborts():
    completed = list(CONCEPTS[:3])
    payload = _result_payload(completed)
    payload["concept_metric_callback_CountingMetric"]["concepts_order"] = ["cable", "bottle", "capsule"]
    callback = ConceptMetricCallback(base_metric=CountingMetric(), metrics=[])

    with pytest.raises(CheckpointResultMismatch, match="does not start with"):
        restore_callbacks([callback], payload, completed)


def test_json_longer_than_the_checkpoint_is_truncated_not_rejected(caplog):
    """Falling back to an older checkpoint leaves a result JSON that is ahead."""
    completed = list(CONCEPTS[:2])
    payload = _result_payload(list(CONCEPTS[:4]))
    callback = ConceptMetricCallback(base_metric=CountingMetric(), metrics=[])

    restore_callbacks([callback], payload, completed)

    assert callback._learned_concepts == completed
    assert set(callback._metric_matrix) == set(completed)


def test_result_json_sha_mismatch_aborts(tmp_path: Path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    digest = sha256_of_file(path)
    assert load_result_payload(path, expected_sha256=digest) == {"a": 1}

    path.write_text(json.dumps({"a": 2}), encoding="utf-8")
    with pytest.raises(CheckpointResultMismatch, match="changed since the checkpoint"):
        load_result_payload(path, expected_sha256=digest)


def test_missing_result_json_names_the_fix(tmp_path: Path):
    with pytest.raises(CheckpointResultMismatch, match="--output-dir"):
        load_result_payload(tmp_path / "nope.json")


def test_committed_concept_absent_from_the_dataset_aborts():
    dataset = make_dataset(CONCEPTS[:2])
    callback = _PixelCallbackStub()
    with pytest.raises(CheckpointResultMismatch, match="absent from the dataset"):
        restore_training_concepts([callback], dataset.train_concepts(), list(CONCEPTS[:3]))


def test_nan_cells_survive_the_round_trip():
    completed = list(CONCEPTS[:1])
    payload = _result_payload(completed)
    payload["concept_metric_callback_CountingMetric"]["metric_matrix"]["bottle"]["cable"] = None
    callback = ConceptMetricCallback(base_metric=CountingMetric(), metrics=[])

    restore_callbacks([callback], payload, completed)
    assert math.isnan(callback._metric_matrix["bottle"]["cable"])


def test_run_status_and_snapshot_numbering_continue():
    class _Status:
        def __init__(self):
            self.calls = []

        def mark_concept_completed(self, name, index=None):
            self.calls.append((name, index))

    class _JsonCheckpoint:
        _concept_index = 0

    status = _Status()
    restore_run_status(status, list(CONCEPTS[:3]))
    assert status.calls == [("bottle", 1), ("cable", 2), ("capsule", 3)]

    snapshot = _JsonCheckpoint()
    prime_json_checkpoint_callback(snapshot, 3)
    assert snapshot._concept_index == 3
