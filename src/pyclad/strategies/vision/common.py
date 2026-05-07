from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RunningGaussianStats:
    sample_count: int
    feature_sum: np.ndarray
    feature_outer_sum: np.ndarray
    feature_map_size: tuple[int, int]
    selected_feature_indices: np.ndarray


@dataclass
class TaskMemoryBank:
    task_banks: list[np.ndarray]
    feature_map_size: tuple[int, int] | None = None
    memory_budget: int | None = None  # informational only; not used for logic


@dataclass
class RunningCFAMemory:
    descriptor_mean: torch.Tensor | None = None
    batch_count: int = 0
    descriptor_map_size: tuple[int, int] | None = None
