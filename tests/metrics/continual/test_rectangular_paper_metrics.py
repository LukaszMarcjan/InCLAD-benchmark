"""Rectangular ROC and BWT match the grouped-table builder of the paper.

The three matrices below are copied from completed grouped-grid runs
(manifest rows ``mvtec rn18 padim naive 14-1 eth`` image ROC-AUC,
``mpdd rn18 patchcore cl 4-1x2 hte`` pixel ROC-AUC and
``btech wr50 padim cl 1x3 eth`` image ROC-AUC; rows in train_order, columns
in test_order).
"""

import importlib.util
import math
import pathlib
import sys

import pytest

from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.rectangular_backward_transfer import (
    RectangularBackwardTransfer,
)
from pyclad.metrics.continual.rectangular_continual_average import (
    RectangularContinualAverage,
)
from pyclad.metrics.continual.rectangular_matrix import matrix_from_result_block

CAMERA_READY_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples" / "clvad" / "camera_ready"

MVTEC_14_1 = (
    [
        [0.9753968253968255, 0.9517663043478262, 0.9570175438596491, 0.8001443001443002, 0.7448680351906158,
         0.9625, 0.7202023988005997, 0.8388888888888889, 0.8711878009630819, 0.7925000000000001,
         0.7620798319327732, 0.7786131996658312, 0.7024331870761867, 0.5798319327731092, 0.45881069285324605],
        [0.5801587301587301, 0.4052309782608695, 0.7789473684210526, 0.4393939393939394, 0.4535679374389051,
         0.7739285714285714, 0.5438530734632684, 0.4888888888888888, 0.5212680577849117, 0.43583333333333335,
         0.6394432773109243, 0.6975772765246449, 0.3797367371360192, 0.5234679237548678, 0.7787779596290235],
    ],
    [0] * 14 + [1],
)
MPDD_4_1X2_PIXEL = (
    [
        [0.9707462162149849, 0.9812045959558083, 0.8027306804584277, 0.9702319805803832, 0.9589108402417976,
         0.9438973192138349],
        [0.8882043755002687, 0.881242796253042, 0.9314447279563652, 0.8642206509823509, 0.9927434803974059,
         0.940392170414448],
        [0.9010331530174915, 0.8691524081632986, 0.9100654162583515, 0.8477167247293678, 0.9406410132994222,
         0.995355621388538],
    ],
    [0, 0, 0, 0, 1, 2],
)
BTECH_1X3 = (
    [
        [0.9632317073170732, 0.5694849368318756, 0.5598333333333334],
        [0.9632317073170732, 0.9990281827016521, 0.5656666666666667],
        [0.9632317073170732, 0.9990281827016521, 0.8431666666666667],
    ],
    [0, 1, 2],
)

# (matrix, first_seen_steps, ROC, BWT) as reported by build_grouped_tables.py
REAL_CASES = [
    pytest.param(*MVTEC_14_1, 0.6854311828264702, -0.26972386826719724, id="mvtec-14-1"),
    pytest.param(*MPDD_4_1X2_PIXEL, 0.9164489227437005, -0.04542768451740618, id="mpdd-4-1x2-pixel"),
    pytest.param(*BTECH_1X3, 0.9551530256701984, 0.0, id="btech-1x3"),
]


def _as_block(matrix, first_seen_steps):
    """Store a matrix the way the step-schedule runner writes it to the result JSON."""
    steps = [f"step_{s}" for s in range(len(matrix))]
    components = [f"c{c}" for c in range(len(first_seen_steps))]
    return {
        "metric_matrix": {step: dict(zip(components, row)) for step, row in zip(steps, matrix)},
        "train_order": steps,
        "test_order": components,
        "first_seen_step": dict(zip(components, first_seen_steps)),
    }


@pytest.fixture(scope="module")
def builder():
    path = CAMERA_READY_DIR / "build_grouped_tables.py"
    sys.path.insert(0, str(CAMERA_READY_DIR))
    try:
        spec = importlib.util.spec_from_file_location("build_grouped_tables_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CAMERA_READY_DIR))
    return module


@pytest.mark.parametrize("matrix,first_seen,roc,bwt", REAL_CASES)
def test_matches_builder_on_real_matrices(builder, matrix, first_seen, roc, bwt):
    expected = builder.metrics(_as_block(matrix, first_seen))
    assert RectangularContinualAverage().compute(matrix, first_seen) == expected["ca"]
    assert RectangularBackwardTransfer().compute(matrix, first_seen) == expected["bwt"]


@pytest.mark.parametrize("matrix,first_seen,roc,bwt", REAL_CASES)
def test_reported_values(matrix, first_seen, roc, bwt):
    assert RectangularContinualAverage().compute(matrix, first_seen) == pytest.approx(roc, abs=1e-15)
    assert RectangularBackwardTransfer().compute(matrix, first_seen) == pytest.approx(bwt, abs=1e-15)


@pytest.mark.parametrize("matrix,first_seen,roc,bwt", REAL_CASES)
def test_matrix_from_result_block(matrix, first_seen, roc, bwt):
    assert matrix_from_result_block(_as_block(matrix, first_seen)) == (matrix, first_seen)


def test_matrix_from_result_block_uses_train_and_test_order():
    block = {
        "metric_matrix": {"step_1": {"a": 3.0, "b": 4.0}, "step_0": {"a": 1.0, "b": 2.0}},
        "train_order": ["step_0", "step_1"],
        "test_order": ["b", "a"],
        "first_seen_step": {"a": 0, "b": 1},
    }
    assert matrix_from_result_block(block) == ([[2.0, 1.0], [4.0, 3.0]], [1, 0])


def test_position_indexed_library_metrics_differ_on_rectangular_matrix():
    matrix, first_seen = MVTEC_14_1
    assert ContinualAverage().compute(matrix) != pytest.approx(RectangularContinualAverage().compute(matrix, first_seen))
    assert BackwardTransfer().compute(matrix) != pytest.approx(RectangularBackwardTransfer().compute(matrix, first_seen))


def test_identity_first_seen_on_square_matrix_matches_continual_average():
    matrix = [[0.8, 0.1, 0.2], [0.6, 0.9, 0.3], [0.5, 0.7, 1.0]]
    assert RectangularContinualAverage().compute(matrix, [0, 1, 2]) == pytest.approx(ContinualAverage().compute(matrix))
    # Eq. (bwt): mean of R[s][c] - R[c][c] over s > c
    expected_bwt = ((0.6 - 0.8) + (0.5 - 0.8) + (0.7 - 0.9)) / 3
    assert RectangularBackwardTransfer().compute(matrix, [0, 1, 2]) == pytest.approx(expected_bwt)


def test_missing_entries_are_skipped(builder):
    matrix = [[0.9, 0.2, float("nan")], [0.7, 0.8, 0.1], [float("nan"), 0.6, 0.9]]
    first_seen = [0, 1, 2]
    expected = builder.metrics(_as_block(matrix, first_seen))
    assert RectangularContinualAverage().compute(matrix, first_seen) == pytest.approx(expected["ca"])
    assert RectangularBackwardTransfer().compute(matrix, first_seen) == pytest.approx(expected["bwt"])
    # None is treated like NaN
    with_none = [[0.9, 0.2, None], [0.7, 0.8, 0.1], [None, 0.6, 0.9]]
    assert RectangularContinualAverage().compute(with_none, first_seen) == pytest.approx(expected["ca"])
    assert RectangularBackwardTransfer().compute(with_none, first_seen) == pytest.approx(expected["bwt"])


def test_single_step_bwt_is_zero_and_roc_is_row_mean():
    matrix = [[0.5, 0.7, 0.9]]
    assert RectangularBackwardTransfer().compute(matrix, [0, 0, 0]) == 0.0
    assert RectangularContinualAverage().compute(matrix, [0, 0, 0]) == pytest.approx(0.7)


def test_empty_matrix():
    assert math.isnan(RectangularContinualAverage().compute([], []))
    assert RectangularBackwardTransfer().compute([], []) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        RectangularContinualAverage().compute([[0.1, 0.2]], [0])
    with pytest.raises(ValueError):
        RectangularBackwardTransfer().compute([[0.1, 0.2]], [0])


def test_negative_first_seen_raises():
    with pytest.raises(ValueError):
        RectangularContinualAverage().compute([[0.1, 0.2]], [0, -1])
    with pytest.raises(ValueError):
        RectangularBackwardTransfer().compute([[0.1, 0.2], [0.3, 0.4]], [-1, 0])


def test_names():
    assert RectangularContinualAverage().name() == "RectangularContinualAverage"
    assert RectangularBackwardTransfer().name() == "RectangularBackwardTransfer"
