"""Step-schedule grouping for visual continual anomaly detection benchmarks.

Wraps the existing visual benchmark readers without modifying them. Given an
ordered list of per-category concepts, groups consecutive categories into
multi-class training tasks according to a CDAD-style schedule such as
``"14-1"``, ``"10-5"``, ``"3x5"`` or ``"10-1x5"``.

Train concepts are merged into one Concept per scheduled step; test concepts
are kept per-category by default so evaluation matrices still report
per-class AUROC (matching how CDAD-style papers report results).

Supports MVTec (15 categories) and VisA (12 categories).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.visual_benchmark_reader import (
    VisualBenchmarkSpec,
    read_visual_benchmark_dataset,
)
from pyclad.data.readers.visual_dataset_registry import (
    read_registered_visual_benchmark_dataset,
)

StepSchedule = List[int]
StepScheduleLike = Union[str, Sequence[int]]

_SCHEDULE_TOKEN_RE = re.compile(r"^\s*(\d+)\s*(?:[x×*]\s*(\d+))?\s*$")


def parse_step_schedule(spec: StepScheduleLike, total_categories: Optional[int] = None) -> StepSchedule:
    """Parse a step-schedule spec into an explicit list of class counts per step.

    Accepted string forms (separated by ``-``):

    - ``"14-1"`` -> ``[14, 1]``
    - ``"10-5"`` -> ``[10, 5]``
    - ``"3x5"`` -> ``[3, 3, 3, 3, 3]``  (also ``3×5`` or ``3*5``)
    - ``"10-1x5"`` -> ``[10, 1, 1, 1, 1, 1]``
    - ``"8-1x4"`` -> ``[8, 1, 1, 1, 1]``

    A sequence of ints is returned as-is after validation.

    If ``total_categories`` is given, the parsed schedule must sum to that value.
    """
    if isinstance(spec, str):
        schedule = _parse_schedule_string(spec)
    else:
        schedule = [int(value) for value in spec]

    if not schedule:
        raise ValueError("Step schedule must contain at least one step.")
    if any(step <= 0 for step in schedule):
        raise ValueError(f"Step schedule entries must be positive: {schedule}")

    if total_categories is not None and sum(schedule) != total_categories:
        raise ValueError(
            f"Step schedule {schedule} sums to {sum(schedule)}, "
            f"expected {total_categories}."
        )

    return schedule


def group_concepts_by_schedule(
    concepts: Sequence[Concept],
    schedule: StepSchedule,
    name_prefix: str = "step",
    name_mode: str = "indexed",
    name_separator: str = "+",
) -> List[Concept]:
    """Concatenate consecutive concepts into grouped concepts per scheduled step.

    Parameters
    ----------
    concepts:
        Ordered concepts (one per original category).
    schedule:
        Class counts per step. ``sum(schedule)`` must equal ``len(concepts)``.
    name_mode:
        ``"indexed"`` -> ``"{name_prefix}_{i}"``;
        ``"joined"``  -> categories joined with ``name_separator``.
    """
    if sum(schedule) != len(concepts):
        raise ValueError(
            f"Schedule sum ({sum(schedule)}) does not match number of concepts ({len(concepts)})."
        )
    if name_mode not in {"indexed", "joined"}:
        raise ValueError(f"name_mode must be 'indexed' or 'joined', got {name_mode!r}")

    grouped: List[Concept] = []
    cursor = 0
    for step_index, step_size in enumerate(schedule):
        chunk = list(concepts[cursor : cursor + step_size])
        cursor += step_size

        if name_mode == "joined":
            group_name = name_separator.join(concept.name for concept in chunk)
        else:
            group_name = f"{name_prefix}_{step_index}"

        grouped.append(
            Concept(
                name=group_name,
                data=_concatenate_concept_arrays([concept.data for concept in chunk]),
                labels=_concatenate_optional_labels([concept.labels for concept in chunk]),
            )
        )
    return grouped


def apply_step_schedule(
    dataset: ConceptsDataset,
    schedule: StepScheduleLike,
    group_test: bool = False,
    dataset_name: Optional[str] = None,
    name_mode: str = "indexed",
    name_separator: str = "+",
) -> ConceptsDataset:
    """Return a new ConceptsDataset with concepts grouped according to ``schedule``.

    Train concepts are always grouped. Test concepts are grouped only if
    ``group_test=True``; otherwise the original per-category test concepts are
    preserved so evaluation reports per-class metrics (the CDAD convention).
    """
    train_concepts = dataset.train_concepts()
    parsed = parse_step_schedule(schedule, total_categories=len(train_concepts))

    grouped_train = group_concepts_by_schedule(
        concepts=train_concepts,
        schedule=parsed,
        name_mode=name_mode,
        name_separator=name_separator,
    )

    if group_test:
        test_concepts = dataset.test_concepts()
        if len(test_concepts) != len(train_concepts):
            raise ValueError(
                "group_test=True requires test_concepts to align 1:1 with train_concepts "
                f"(got {len(test_concepts)} test vs {len(train_concepts)} train)."
            )
        grouped_test = group_concepts_by_schedule(
            concepts=test_concepts,
            schedule=parsed,
            name_mode=name_mode,
            name_separator=name_separator,
        )
    else:
        grouped_test = list(dataset.test_concepts())

    resolved_name = dataset_name or f"{dataset.name()}__{_format_schedule(parsed)}"
    return ConceptsDataset(
        name=resolved_name,
        train_concepts=grouped_train,
        test_concepts=grouped_test,
    )


def read_step_scheduled_visual_benchmark_dataset(
    benchmark: Union[str, VisualBenchmarkSpec],
    schedule: StepScheduleLike,
    root: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    dataset_name: Optional[str] = None,
    data_mode: str = "numpy",
    resize_to: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
    group_test: bool = False,
    name_mode: str = "indexed",
    name_separator: str = "+",
) -> ConceptsDataset:
    """Read a visual benchmark and group concepts according to a step schedule.

    Equivalent to ``read_visual_benchmark_dataset(...)`` followed by
    :func:`apply_step_schedule`. Pass ``root`` for an explicit dataset directory.
    """
    if root is None:
        base = read_registered_visual_benchmark_dataset(
            benchmark=benchmark,
            categories=categories,
            dataset_name=dataset_name,
            data_mode=data_mode,
            resize_to=resize_to,
            color_mode=color_mode,
            max_train_samples_per_category=max_train_samples_per_category,
            max_test_samples_per_category=max_test_samples_per_category,
        )
    else:
        base = read_visual_benchmark_dataset(
            root=root,
            benchmark=benchmark,
            categories=categories,
            dataset_name=dataset_name,
            data_mode=data_mode,
            resize_to=resize_to,
            color_mode=color_mode,
            max_train_samples_per_category=max_train_samples_per_category,
            max_test_samples_per_category=max_test_samples_per_category,
        )

    return apply_step_schedule(
        dataset=base,
        schedule=schedule,
        group_test=group_test,
        dataset_name=dataset_name,
        name_mode=name_mode,
        name_separator=name_separator,
    )


def read_step_scheduled_mvtec(
    schedule: StepScheduleLike,
    root: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    dataset_name: Optional[str] = None,
    data_mode: str = "numpy",
    resize_to: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
    group_test: bool = False,
    name_mode: str = "indexed",
) -> ConceptsDataset:
    """Convenience entry point for MVTec with a CDAD-style step schedule."""
    return read_step_scheduled_visual_benchmark_dataset(
        benchmark="mvtec",
        schedule=schedule,
        root=root,
        categories=categories,
        dataset_name=dataset_name,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
        group_test=group_test,
        name_mode=name_mode,
    )


def read_step_scheduled_visa(
    schedule: StepScheduleLike,
    root: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    dataset_name: Optional[str] = None,
    data_mode: str = "numpy",
    resize_to: Optional[Tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
    group_test: bool = False,
    name_mode: str = "indexed",
) -> ConceptsDataset:
    """Convenience entry point for VisA with a CDAD-style step schedule."""
    return read_step_scheduled_visual_benchmark_dataset(
        benchmark="visa",
        schedule=schedule,
        root=root,
        categories=categories,
        dataset_name=dataset_name,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
        group_test=group_test,
        name_mode=name_mode,
    )


def compute_first_seen_step(
    ordered_concept_names: Sequence[str],
    schedule: StepScheduleLike,
) -> List[int]:
    """Return per-category first-training-step indices for a step schedule.

    Given the per-category training order (length N) and a step schedule
    that partitions those N categories into T consecutive groups, returns a
    list of length N where entry ``i`` is the index of the training step
    that first includes ``ordered_concept_names[i]``.

    Example
    -------
    For MVTec ``14-1`` with alphabetical ordering::

        ordered_concept_names = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
            'transistor', 'wood', 'zipper',
        ]
        compute_first_seen_step(ordered_concept_names, '14-1')
        # -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    Notes
    -----
    The return order matches ``ordered_concept_names``. Callbacks that
    record test concepts in a different order (e.g. order of first
    appearance during evaluation) must rebuild the list by lookup; use
    :func:`first_seen_step_for_test_order` for that.
    """
    parsed = parse_step_schedule(schedule, total_categories=len(ordered_concept_names))
    first_seen: List[int] = []
    cursor_step = 0
    remaining_in_step = parsed[0]
    for _ in range(len(ordered_concept_names)):
        while remaining_in_step == 0:
            cursor_step += 1
            remaining_in_step = parsed[cursor_step]
        first_seen.append(cursor_step)
        remaining_in_step -= 1
    return first_seen


def first_seen_step_for_test_order(
    ordered_concept_names: Sequence[str],
    schedule: StepScheduleLike,
    test_order: Sequence[str],
) -> List[int]:
    """Re-align first-seen indices to an arbitrary evaluation order.

    ``ordered_concept_names`` is the training order. ``test_order`` is the
    column order used by the rectangular callback (typically the order in
    which test concepts first appeared during evaluation). Returns a list
    of length ``len(test_order)`` whose entries are first-training-step
    indices keyed by name.

    Raises ``KeyError`` if any name in ``test_order`` is missing from
    ``ordered_concept_names``.
    """
    train_index = compute_first_seen_step(ordered_concept_names, schedule)
    name_to_step = dict(zip(ordered_concept_names, train_index))
    aligned: List[int] = []
    for name in test_order:
        if name not in name_to_step:
            raise KeyError(
                f"Test concept {name!r} not present in training order; "
                "cannot determine its first-seen step."
            )
        aligned.append(name_to_step[name])
    return aligned


def _parse_schedule_string(spec: str) -> StepSchedule:
    if not spec or not spec.strip():
        raise ValueError("Step schedule string is empty.")

    schedule: StepSchedule = []
    for raw_token in spec.split("-"):
        match = _SCHEDULE_TOKEN_RE.match(raw_token)
        if match is None:
            raise ValueError(
                f"Invalid step-schedule token {raw_token!r} in {spec!r}. "
                "Expected forms: 'N' or 'NxM' (e.g. '14', '3x5')."
            )
        size = int(match.group(1))
        repeat = int(match.group(2)) if match.group(2) is not None else 1
        schedule.extend([size] * repeat)
    return schedule


def _format_schedule(schedule: StepSchedule) -> str:
    parts: List[str] = []
    i = 0
    while i < len(schedule):
        j = i
        while j < len(schedule) and schedule[j] == schedule[i]:
            j += 1
        run = j - i
        parts.append(f"{schedule[i]}x{run}" if run > 1 else str(schedule[i]))
        i = j
    return "-".join(parts)


def _concatenate_concept_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(list(arrays), axis=0)


def _concatenate_optional_labels(
    label_arrays: Sequence[Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    if all(labels is None for labels in label_arrays):
        return None
    if any(labels is None for labels in label_arrays):
        raise ValueError("Cannot group concepts where some have labels and others do not.")
    return np.concatenate([np.asarray(labels) for labels in label_arrays], axis=0)
