"""Read a rectangular (T x N) matrix from a step-schedule result JSON block.

The v2 step-schedule runner stores each rectangular metric as a block such as
``rectangular_concept_metric_callback_ROC-AUC`` with the keys
``metric_matrix`` (``{step: {component: value}}``), ``train_order`` (steps),
``test_order`` (components) and ``first_seen_step`` (``{component: step}``).
:func:`matrix_from_result_block` turns such a block into the
``(metric_matrix, first_seen_steps)`` pair taken by the ``Rectangular*``
metrics, with rows in ``train_order`` and columns in ``test_order``, as
``examples/clvad/camera_ready/build_grouped_tables.py`` does.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Tuple

from pyclad.metrics.continual.concepts_metric import ConceptLevelMatrix


def matrix_from_result_block(block: Mapping[str, Any]) -> Tuple[ConceptLevelMatrix, List[int]]:
    matrix = block["metric_matrix"]
    rows = block.get("train_order") or list(matrix)
    cols = block.get("test_order")
    if not cols:
        raise KeyError("result block has no test_order")
    first_seen = block.get("first_seen_step") or {}
    metric_matrix = [[matrix[row][col] for col in cols] for row in rows]
    first_seen_steps = [int(first_seen[col]) for col in cols]
    return metric_matrix, first_seen_steps
