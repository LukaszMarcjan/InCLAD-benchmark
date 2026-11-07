"""Fakes shared by the resume tests.

Everything here is intentionally tiny and CPU-only: the point of putting the
resume layer in ``src/pyclad`` rather than in an example script is that it can
be tested without spinning up a fifteen-concept visual benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import torch

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import ConceptLevelMetric
from pyclad.resume.fingerprint import current_provenance
from pyclad.resume.protocol import SIDECAR_KEY, build_envelope, require_payload_keys
from pyclad.resume.store import CheckpointStore
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy

CONCEPTS = ("bottle", "cable", "capsule", "carpet", "grid")


class CountingMetric(BaseMetric):
    """Deterministic per-(learned, evaluated) value, so matrices are checkable."""

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return float(np.mean(anomaly_scores))

    def name(self) -> str:
        return "CountingMetric"


class MatrixMean(ConceptLevelMetric):
    def compute(self, metric_matrix) -> float:
        values = [value for row in metric_matrix for value in row]
        return float(np.mean(values)) if values else float("nan")

    def name(self) -> str:
        return "MatrixMean"


class FakeState:
    """A minimal :class:`ResumableState` with one tensor and one counter."""

    STATE_KIND = "fake_state"
    STATE_VERSION = 1

    def __init__(self, *, key_size: int = 4, seed: int = 11):
        self.key_size = key_size
        self.seed = seed
        self.learned: list[str] = []
        self.weights = torch.zeros(key_size)
        self.cache_invalidations = 0
        self.save_calls = 0

    def fit(self, concept_id: str) -> None:
        self.learned.append(concept_id)
        self.weights = self.weights + len(self.learned)

    def score(self) -> float:
        return float(len(self.learned))

    def save_state(self) -> dict[str, Any]:
        self.save_calls += 1
        return {"learned": list(self.learned), "weights": self.weights.clone()}

    def load_state(self, payload: Mapping[str, Any]) -> None:
        require_payload_keys(payload, ("learned", "weights"), state_kind=self.STATE_KIND)
        self.learned = [str(name) for name in payload["learned"]]
        self.weights = torch.as_tensor(payload["weights"]).clone()

    def state_fingerprint(self) -> dict[str, Any]:
        return {"key_size": self.key_size, "seed": self.seed}

    def _invalidate_prediction_cache(self) -> None:
        self.cache_invalidations += 1


class FakeStateWithSidecar(FakeState):
    """State whose bulk lives out of band, like ReplayCAD's memory bank."""

    STATE_KIND = "fake_state_sidecar"

    def __init__(self, *, bank_rows: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.bank = np.zeros((bank_rows, 3), dtype=np.float32)

    def fit(self, concept_id: str) -> None:
        super().fit(concept_id)
        self.bank = np.full_like(self.bank, float(len(self.learned)))

    def save_state(self) -> dict[str, Any]:
        payload = super().save_state()
        payload[SIDECAR_KEY] = {"memory_bank": self.bank}
        return payload

    def load_state(self, payload: Mapping[str, Any]) -> None:
        super().load_state(payload)
        sidecars = payload.get(SIDECAR_KEY, {})
        require_payload_keys(sidecars, ("memory_bank",), state_kind=self.STATE_KIND)
        self.bank = np.asarray(sidecars["memory_bank"])


class RecordingStrategy(ConceptAwareStrategy):
    """Concept-aware strategy that records every call the scenario makes."""

    def __init__(self, state: FakeState):
        self._model = state
        self.learn_calls: list[str] = []
        self.predict_calls: list[str] = []
        self.resume_restore_calls: list[int] = []

    def name(self) -> str:
        return "Recording"

    def learn(self, data: np.ndarray, concept_id: str) -> None:
        self.learn_calls.append(concept_id)
        self._model.fit(concept_id)

    def predict(self, data: np.ndarray, concept_id: str):
        self.predict_calls.append(concept_id)
        scores = np.full(len(data), self._model.score(), dtype=np.float64)
        return (scores > 0).astype(int), scores

    def resume_restore(self, payload: Mapping[str, Any], plan) -> None:
        self.resume_restore_calls.append(plan.resume_index)


class RecordingIncrementalStrategy(ConceptIncrementalStrategy):
    def __init__(self, state: FakeState):
        self._model = state
        self.learn_calls: list[int] = []
        self.predict_calls: list[int] = []

    def name(self) -> str:
        return "RecordingIncremental"

    def learn(self, data: np.ndarray) -> None:
        self.learn_calls.append(len(data))
        self._model.fit(f"concept_{len(self.learn_calls)}")

    def predict(self, data: np.ndarray):
        self.predict_calls.append(len(data))
        scores = np.full(len(data), self._model.score(), dtype=np.float64)
        return (scores > 0).astype(int), scores


class EventRecorder:
    """Callback that records every event, so skipped concepts can be asserted."""

    def __init__(self):
        self.events: list[tuple[str, str | None]] = []

    def before_scenario(self, *args, **kwargs):
        self.events.append(("before_scenario", None))

    def after_scenario(self, *args, **kwargs):
        self.events.append(("after_scenario", None))

    def before_concept_processing(self, concept, *args, **kwargs):
        self.events.append(("before_concept_processing", concept.name))

    def before_training(self, *args, **kwargs):
        self.events.append(("before_training", None))

    def after_training(self, learned_concept, *args, **kwargs):
        self.events.append(("after_training", learned_concept.name))

    def before_evaluation(self, *args, **kwargs):
        self.events.append(("before_evaluation", None))

    def after_evaluation(self, evaluated_concept, *args, **kwargs):
        self.events.append(("after_evaluation", evaluated_concept.name))

    def after_concept_processing(self, concept, *args, **kwargs):
        self.events.append(("after_concept_processing", concept.name))

    def names_for(self, event: str) -> list[str]:
        return [name for kind, name in self.events if kind == event and name is not None]


def make_dataset(concepts: Sequence[str] = CONCEPTS, *, samples: int = 3) -> ConceptsDataset:
    train = [
        Concept(name=name, data=np.full((samples, 2), index, dtype=np.float64)) for index, name in enumerate(concepts)
    ]
    test = [
        Concept(
            name=name,
            data=np.full((samples, 2), index, dtype=np.float64),
            labels=np.array([0] * (samples - 1) + [1]),
        )
        for index, name in enumerate(concepts)
    ]
    return ConceptsDataset(name="fake", train_concepts=train, test_concepts=test)


def make_envelope(
    *,
    index: int,
    concept_id: str,
    concept_order: Sequence[str],
    state: Mapping[str, Any] | None = None,
    state_kind: str = "fake_state",
    state_version: int = 1,
    fingerprint: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    phase: str = "after_training",
    train_seconds: float | None = None,
) -> dict[str, Any]:
    return build_envelope(
        state_kind=state_kind,
        state_version=state_version,
        provenance=dict(provenance) if provenance is not None else current_provenance(),
        run={
            "run_id": "testrun",
            "seed": 1,
            "concept_index": index,
            "concept_id": concept_id,
            "concept_order": list(concept_order),
            "phase": phase,
            "train_seconds": train_seconds,
        },
        determinism={"guarantee": "bit_identical", "caveats": []},
        fingerprint=dict(fingerprint or {"key_size": 4, "seed": 11}),
        state=dict(state or {"learned": [concept_id], "weights": torch.ones(2)}),
    )


def write_result_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "ckpt", "runid1234ab", keep=2)
