from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


REPLAY_BUFFER_MODES = ("fixed", "avg-concept-fraction", "per-concept-budget")


@dataclass(frozen=True)
class ReplayBufferBudgetPlan:
    mode: str
    total_concepts: int
    total_train_samples: int
    average_train_concept_size: float
    requested_fixed_size: int | None
    requested_avg_concept_fraction: float | None
    requested_per_concept_budget: int | None
    resolved_max_size: int

    @property
    def resolved_budget_per_concept(self) -> float:
        if self.total_concepts <= 0:
            return 0.0
        return float(self.resolved_max_size) / float(self.total_concepts)

    @property
    def resolved_fraction_of_total_train_samples(self) -> float:
        if self.total_train_samples <= 0:
            return 0.0
        return float(self.resolved_max_size) / float(self.total_train_samples)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_concepts": self.total_concepts,
            "total_train_samples": self.total_train_samples,
            "average_train_concept_size": self.average_train_concept_size,
            "requested_fixed_size": self.requested_fixed_size,
            "requested_avg_concept_fraction": self.requested_avg_concept_fraction,
            "requested_per_concept_budget": self.requested_per_concept_budget,
            "resolved_max_size": self.resolved_max_size,
            "resolved_budget_per_concept": self.resolved_budget_per_concept,
            "resolved_fraction_of_total_train_samples": self.resolved_fraction_of_total_train_samples,
        }


def resolve_replay_buffer_budget(
    train_concept_sizes: Sequence[int],
    *,
    mode: str,
    fixed_size: int,
    avg_concept_fraction: float | None,
    per_concept_budget: int | None,
) -> ReplayBufferBudgetPlan:
    if mode not in REPLAY_BUFFER_MODES:
        raise ValueError(f"Unsupported replay buffer mode: {mode}")

    normalized_sizes = [max(0, int(size)) for size in train_concept_sizes]
    total_concepts = len(normalized_sizes)
    total_train_samples = int(sum(normalized_sizes))
    average_train_concept_size = (
        float(total_train_samples) / float(total_concepts)
        if total_concepts > 0
        else 0.0
    )

    requested_fixed_size = None
    requested_avg_concept_fraction = None
    requested_per_concept_budget = None

    if mode == "fixed":
        if int(fixed_size) <= 0:
            raise ValueError(f"Replay buffer size must be positive, got {fixed_size}")
        requested_fixed_size = int(fixed_size)
        resolved_max_size = int(fixed_size)
    elif mode == "avg-concept-fraction":
        if avg_concept_fraction is None or float(avg_concept_fraction) <= 0.0:
            raise ValueError(
                "Replay buffer fraction must be positive when mode='avg-concept-fraction'"
            )
        requested_avg_concept_fraction = float(avg_concept_fraction)
        resolved_max_size = max(1, int(round(total_train_samples * requested_avg_concept_fraction)))
    else:
        if per_concept_budget is None or int(per_concept_budget) <= 0:
            raise ValueError(
                "Replay buffer per-concept budget must be positive when mode='per-concept-budget'"
            )
        requested_per_concept_budget = int(per_concept_budget)
        resolved_max_size = max(1, requested_per_concept_budget * max(total_concepts, 1))

    return ReplayBufferBudgetPlan(
        mode=mode,
        total_concepts=total_concepts,
        total_train_samples=total_train_samples,
        average_train_concept_size=average_train_concept_size,
        requested_fixed_size=requested_fixed_size,
        requested_avg_concept_fraction=requested_avg_concept_fraction,
        requested_per_concept_budget=requested_per_concept_budget,
        resolved_max_size=resolved_max_size,
    )
