import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def _create_upper_diagonal_mask(tasks_no: int) -> np.array:
    mask = np.zeros((tasks_no, tasks_no))
    for i in range(tasks_no):
        for j in range(tasks_no):
            if j > i:
                mask[i, j] = True
    return mask


def plot_metric_heatmap(
    matrix: Dict,
    concepts_order: List[str],
    output_path: pathlib.Path = None,
    names_mapping: Dict[str, str] = None,
    xlabel: str = "Evaluating on concept",
    ylabel: str = "After learning concept",
    title: str = "Performance Heatmap",
    annotate: bool = False,
    color_palette: str = "plasma",
    figsize: tuple = (6, 5),
    ignore_upper_diagonal: bool = False,
):
    sns.set_theme(style="darkgrid")
    sns.set(rc={"figure.figsize": figsize})

    data = []  # learned_concept, evaluated_concept, metric_value

    for learned_concept in concepts_order:
        for evaluated_concept in concepts_order:
            metric_value = matrix[learned_concept][evaluated_concept]
            data.append(
                [
                    learned_concept if names_mapping is None else names_mapping[learned_concept],
                    evaluated_concept if names_mapping is None else names_mapping[evaluated_concept],
                    metric_value,
                ]
            )

    df = pd.DataFrame(data, columns=["learned_concept", "evaluated_concept", "metric_value"])
    df = df.pivot(index="learned_concept", columns="evaluated_concept", values="metric_value")
    df = df.reindex(index=concepts_order, columns=concepts_order)
    p: Axes = sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        center=0.5,
        cmap=sns.color_palette(color_palette, as_cmap=True),
        annot=annotate,
        mask=None if ignore_upper_diagonal else _create_upper_diagonal_mask(len(concepts_order)),
    )
    p.set_xlabel(xlabel)
    p.set_ylabel(ylabel)
    p.set_title(title)

    if output_path is not None:
        plt.tight_layout()
        plt.savefig(output_path)

    return p


def plot_rectangular_metric_heatmap(
    matrix: Dict[str, Dict[str, float]],
    train_order: List[str],
    test_order: List[str],
    output_path: pathlib.Path = None,
    row_labels: Dict[str, str] = None,
    col_labels: Dict[str, str] = None,
    xlabel: str = "Evaluated on category",
    ylabel: str = "After training step",
    title: str = "Performance Heatmap",
    annotate: bool = False,
    color_palette: str = "plasma",
    figsize: tuple = (10, 10),
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = None,
    mask_upper_diagonal: bool = False,
    show_tick_labels: bool = True,
    show_cbar: bool = True,
    annot_size: float = None,
):
    """Plot a rectangular ``T x N`` continual-learning metric matrix.

    Unlike :func:`plot_metric_heatmap`, the rows (training steps,
    ``train_order``) and columns (evaluated categories, ``test_order``) are
    indexed independently, so the matrix does not have to be square. This is the
    layout produced by
    :class:`pyclad.callbacks.evaluation.rectangular_concept_metric.RectangularConceptMetricCallback`
    where ``matrix[train_step][category]`` holds the metric value. Missing
    entries are rendered as NaN (blank cells).
    """
    sns.set_theme(style="darkgrid")

    rows = [row_labels.get(step, step) for step in train_order] if row_labels else list(train_order)
    cols = [col_labels.get(cat, cat) for cat in test_order] if col_labels else list(test_order)

    values = np.full((len(train_order), len(test_order)), np.nan)
    for i, step in enumerate(train_order):
        step_row = matrix.get(step, {})
        for j, category in enumerate(test_order):
            value = step_row.get(category)
            if value is not None:
                values[i, j] = value

    df = pd.DataFrame(values, index=rows, columns=cols)

    mask = None
    if mask_upper_diagonal and len(train_order) == len(test_order):
        mask = np.triu(np.ones_like(values, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        vmin=vmin,
        vmax=vmax,
        center=(vmin + vmax) / 2,
        cmap=sns.color_palette(color_palette, as_cmap=True),
        annot=annotate,
        fmt=".2g",  # same number format as plot_metric_heatmap / generate_heatmaps.py
        annot_kws={"size": annot_size} if annot_size else None,
        mask=mask,
        cbar=show_cbar,
        cbar_kws={"label": cbar_label} if (show_cbar and cbar_label) else None,
        xticklabels=show_tick_labels,
        yticklabels=show_tick_labels,
        ax=ax,
    )
    # Pass an empty string / None to drop any of these captions.
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    if title:
        ax.set_title(title)

    if output_path is not None:
        fig.tight_layout()
        fig.savefig(output_path)

    return ax
