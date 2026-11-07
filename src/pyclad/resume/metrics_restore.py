"""Rehydrate evaluation callbacks from the partial result JSON.

The state checkpoint deliberately does **not** carry the metric matrices. The
runner already persists them atomically at every concept boundary, and the
result JSON is the published artifact; storing them twice creates two sources of
truth that can disagree. The manifest instead records the JSON's path and
sha256, and this module reads them back from it.

Three details here are the difference between a correct resume and a plausible,
wrong one:

* the restored matrices must be ``defaultdict(dict)``. The evaluation callbacks
  index ``self._metric_matrix[learned_concept]`` before assigning into it, so a
  plain ``dict`` raises ``KeyError`` on the first row written after resume.
* ``_resolved_thresholds`` must be restored, not recomputed. It is a cache keyed
  on concept name, and recomputing it after a restore — against a model that is
  now at a later concept — can give a different threshold and shift every
  Pixel-F1/Dice/IoU relative to the uninterrupted run.
* ``_training_concepts`` must be repopulated for the skipped concepts even
  though ``after_training`` never fires for them. The pixel threshold is a
  quantile over score maps of *all* seen training concepts; leave that dict
  empty and the threshold at concept *k* is computed from concept *k* alone.
  Because the lookup filters by the learned-concept list, this has to happen
  after the learned lists are restored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pyclad.data.concept import Concept
from pyclad.resume.protocol import CheckpointResultMismatch

logger = logging.getLogger(__name__)

__all__ = [
    "RestoreReport",
    "sha256_of_file",
    "load_result_payload",
    "restore_callbacks",
    "restore_training_concepts",
    "restore_run_status",
    "prime_json_checkpoint_callback",
]

_HASH_CHUNK = 8 * 1024 * 1024


@dataclass
class RestoreReport:
    """What was found and restored, for logging and for the result JSON."""

    restored: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        return {
            "restored_callbacks": list(self.restored),
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
        }


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result_payload(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Read the partial result JSON that pairs with a checkpoint.

    A JSON that has been rewritten since the checkpoint committed cannot be
    paired with it: restoring from it is exactly how a metric matrix ends up
    with holes that nothing downstream notices.
    """
    result_path = Path(path).expanduser()
    if not result_path.is_file():
        raise CheckpointResultMismatch(
            f"The partial result JSON {result_path} recorded in the checkpoint manifest does not exist. "
            "Point --output-dir at the original run, or pass --resume off to start over."
        )
    if expected_sha256:
        actual = sha256_of_file(result_path)
        if actual != expected_sha256:
            raise CheckpointResultMismatch(
                f"The partial result JSON {result_path} changed since the checkpoint was committed "
                f"(manifest={expected_sha256[:12]}..., file={actual[:12]}...). Another process wrote it, so "
                "the metrics and the model state no longer describe the same run."
            )
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointResultMismatch(f"Partial result JSON {result_path} is unreadable: {error}") from error


@dataclass(frozen=True)
class _Layout:
    learned_attr: str
    order_key: str
    test_attr: str | None
    test_order_key: str | None
    prefixes: tuple[str, ...]


def _layout_for(callback: Any, *, pixel: bool) -> _Layout:
    rectangular = hasattr(callback, "_learned_train_concepts")
    if rectangular:
        prefix = "rectangular_pixel_concept_metric_callback_" if pixel else "rectangular_concept_metric_callback_"
        return _Layout(
            learned_attr="_learned_train_concepts",
            order_key="train_order",
            test_attr="_seen_test_concepts",
            test_order_key="test_order",
            prefixes=(prefix,),
        )
    prefix = "pixel_concept_metric_callback_" if pixel else "concept_metric_callback_"
    return _Layout(
        learned_attr="_learned_concepts",
        order_key="concepts_order",
        test_attr=None,
        test_order_key=None,
        prefixes=(prefix,),
    )


def _find_block(payload: Mapping[str, Any], layout: _Layout, metric_name: str) -> Mapping[str, Any] | None:
    for prefix in layout.prefixes:
        block = payload.get(f"{prefix}{metric_name}")
        if isinstance(block, Mapping):
            return block
    return None


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, str):
        return math.nan
    return float(value)


def _assert_prefix(recorded_order: Sequence[str], completed: Sequence[str], *, where: str) -> None:
    recorded = [str(name) for name in recorded_order]
    expected = [str(name) for name in completed]
    if recorded[: len(expected)] != expected:
        raise CheckpointResultMismatch(
            f"{where}: the result JSON records concept order {recorded} which does not start with the "
            f"{len(expected)} committed concepts {expected}. The JSON and the checkpoint describe different runs."
        )
    if len(recorded) > len(expected):
        logger.warning(
            "%s: the result JSON holds %d concepts but only %d are committed in the checkpoint manifest; "
            "the extra rows are dropped so the matrices match the restored model state.",
            where,
            len(recorded),
            len(expected),
        )


def _restore_matrix(
    matrix: Mapping[str, Any],
    completed: Sequence[str],
    *,
    where: str,
) -> defaultdict:
    restored: defaultdict = defaultdict(dict)
    for learned_concept in completed:
        row = matrix.get(learned_concept)
        if not isinstance(row, Mapping):
            raise CheckpointResultMismatch(
                f"{where}: the result JSON has no metric row for committed concept '{learned_concept}'"
            )
        restored[learned_concept] = {str(name): _as_float(value) for name, value in row.items()}
    return restored


def _restore_concept_metric_callback(
    callback: Any,
    payload: Mapping[str, Any],
    completed: Sequence[str],
    report: RestoreReport,
) -> bool:
    layout = _layout_for(callback, pixel=False)
    metric_name = callback._base_metric.name()
    block = _find_block(payload, layout, metric_name)
    if block is None:
        report.skipped.append(f"{type(callback).__name__}[{metric_name}] (no matching block in result JSON)")
        return False

    where = f"{type(callback).__name__}[{metric_name}]"
    _assert_prefix(block.get(layout.order_key, []), completed, where=where)
    callback._metric_matrix = _restore_matrix(block.get("metric_matrix", {}), completed, where=where)
    setattr(callback, layout.learned_attr, [str(name) for name in completed])
    if layout.test_attr and layout.test_order_key:
        setattr(callback, layout.test_attr, [str(name) for name in block.get(layout.test_order_key, [])])
    report.restored.append(where)
    return True


def _restore_pixel_metric_callback(
    callback: Any,
    payload: Mapping[str, Any],
    completed: Sequence[str],
    report: RestoreReport,
) -> bool:
    layout = _layout_for(callback, pixel=True)
    matrices: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    restored_any = False

    for base_metric in callback._base_metrics:
        metric_name = base_metric.name()
        block = _find_block(payload, layout, metric_name)
        where = f"{type(callback).__name__}[{metric_name}]"
        if block is None:
            if completed:
                raise CheckpointResultMismatch(
                    f"{where}: the result JSON has no block for this pixel metric, but "
                    f"{len(completed)} concepts are committed. The metric set changed between runs, so the "
                    "restored matrix would have holes."
                )
            matrices[metric_name] = defaultdict(dict)
            continue
        _assert_prefix(block.get(layout.order_key, []), completed, where=where)
        matrices[metric_name] = _restore_matrix(block.get("metric_matrix", {}), completed, where=where)
        selection = block.get("threshold_selection", {})
        for concept_name, value in (selection.get("resolved_thresholds") or {}).items():
            thresholds[str(concept_name)] = _as_float(value)
        if layout.test_attr and layout.test_order_key:
            setattr(callback, layout.test_attr, [str(name) for name in block.get(layout.test_order_key, [])])
        report.restored.append(where)
        restored_any = True

    if not restored_any and completed:
        return False

    callback._metric_matrices = matrices
    callback._resolved_thresholds = thresholds
    setattr(callback, layout.learned_attr, [str(name) for name in completed])
    return True


def _restore_time_callback(
    callback: Any,
    payload: Mapping[str, Any],
    report: RestoreReport,
) -> bool:
    block = payload.get("time_evaluation_callback")
    if not isinstance(block, Mapping):
        report.skipped.append(f"{type(callback).__name__} (no time_evaluation_callback block)")
        return False

    restored = defaultdict(lambda: {"train_time": 0, "eval_time": 0})
    for concept_name, values in (block.get("time_by_concept") or {}).items():
        restored[str(concept_name)] = {
            "train_time": float(values.get("train_time", 0.0) or 0.0),
            "eval_time": float(values.get("eval_time", 0.0) or 0.0),
        }
    callback._time_by_concept = restored
    callback._train_time_total = float(block.get("train_time_total", 0.0) or 0.0)
    callback._eval_time_total = float(block.get("eval_time_total", 0.0) or 0.0)
    report.restored.append(f"{type(callback).__name__} (wall clock)")
    return True


def _is_pixel_metric_callback(callback: Any) -> bool:
    return (
        hasattr(callback, "_base_metrics")
        and hasattr(callback, "_metric_matrices")
        and hasattr(callback, "_resolved_thresholds")
    )


def _is_concept_metric_callback(callback: Any) -> bool:
    return hasattr(callback, "_base_metric") and hasattr(callback, "_metric_matrix")


def _is_time_callback(callback: Any) -> bool:
    return hasattr(callback, "_time_by_concept") and hasattr(callback, "_train_time_total")


def restore_callbacks(
    callbacks: Iterable[Any],
    result_payload: Mapping[str, Any],
    completed_concepts: Sequence[str],
    *,
    report: RestoreReport | None = None,
) -> RestoreReport:
    """Restore every evaluation callback that has a matching JSON block.

    Duck-typed on the same attributes the runner's snapshot serializer uses, so
    the rectangular callback variants are handled without naming them here.
    """
    report = report or RestoreReport()
    completed = [str(name) for name in completed_concepts]

    for callback in callbacks:
        if _is_pixel_metric_callback(callback):
            _restore_pixel_metric_callback(callback, result_payload, completed, report)
        elif _is_concept_metric_callback(callback):
            _restore_concept_metric_callback(callback, result_payload, completed, report)
        elif _is_time_callback(callback):
            _restore_time_callback(callback, result_payload, report)
    return report


def restore_training_concepts(
    callbacks: Iterable[Any],
    train_concepts: Sequence[Concept],
    completed_concepts: Sequence[str],
    *,
    report: RestoreReport | None = None,
) -> RestoreReport:
    """Repopulate ``_training_concepts`` for concepts that will be skipped.

    Must run *after* :func:`restore_callbacks`: the pixel callback looks up
    training concepts through its learned-concept list, so an entry added before
    that list is restored would be invisible.
    """
    report = report or RestoreReport()
    completed = [str(name) for name in completed_concepts]
    by_name = {concept.name: concept for concept in train_concepts}

    missing = [name for name in completed if name not in by_name]
    if missing:
        raise CheckpointResultMismatch(
            f"Committed concepts {missing} are absent from the dataset handed to this run; "
            "the benchmark, its categories or its ordering changed since the checkpoint was written."
        )

    for callback in callbacks:
        training_concepts = getattr(callback, "_training_concepts", None)
        if training_concepts is None:
            continue
        for name in completed:
            training_concepts[name] = by_name[name]
        report.restored.append(f"{type(callback).__name__} (training concepts for pixel thresholds)")
    return report


def restore_run_status(provider: Any, completed_concepts: Sequence[str]) -> None:
    """Continue the run-status block rather than restarting its counters."""
    for index, name in enumerate(completed_concepts):
        provider.mark_concept_completed(str(name), index + 1)


def prime_json_checkpoint_callback(callback: Any, completed_count: int) -> None:
    """Number the resumed concept correctly in the result snapshot."""
    callback._concept_index = int(completed_count)
