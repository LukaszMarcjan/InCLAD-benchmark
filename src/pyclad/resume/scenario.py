"""Scenarios that can start from the middle of a concept sequence.

``ConceptAwareScenario.run()`` is a bare ``for`` loop with no skip hook and no
callback veto, so resuming needs a scenario rather than a wrapper. These classes
are structurally identical to the core ones and diverge in exactly one place:
the loop head.

Skipping has to happen at the scenario level, not inside a strategy, for two
reasons. A no-op ``learn()`` would still run the full evaluation sweep for every
skipped concept — fifteen test concepts at roughly three minutes each, per
skipped concept, which is most of what resuming was supposed to save. And
``predict()`` would then be called against a detector that, for ReplayCAD, was
never fitted at all.

A skipped concept fires **no callbacks whatsoever**. In particular it must not
fire ``after_training``: that would append to every metric callback's
``_learned_concepts`` and corrupt the row ordering of the matrices that were
just restored. Everything a skipped concept would have recorded is restored
once, before the loop, by :mod:`pyclad.resume.plan`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.composite_callback import CallbackComposite
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.resume.protocol import CheckpointOrderMismatch, describe_sequence
from pyclad.strategies.strategy import ConceptAwareStrategy, ConceptIncrementalStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "ResumePlan",
    "ResumableConceptAwareScenario",
    "ResumableConceptIncrementalScenario",
]


@dataclass(frozen=True)
class ResumePlan:
    """What the scenario must skip, and what it must only re-evaluate."""

    #: Concepts already completed, in the resolved order. A prefix, always.
    skip_concepts: tuple[str, ...]
    #: Index of the first concept this process actually processes.
    resume_index: int
    #: True when the concept at ``resume_index`` was already trained (its state
    #: is on disk as ``pending``) and only its evaluation sweep is missing.
    evaluate_pending_concept: bool
    #: The full resolved order, used as a per-iteration guard inside the loop.
    concept_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.resume_index < 0:
            raise ValueError(f"resume_index must be non-negative, got {self.resume_index}")
        if len(self.skip_concepts) != self.resume_index:
            raise ValueError(
                f"resume_index ({self.resume_index}) must equal len(skip_concepts) ({len(self.skip_concepts)})"
            )
        if self.concept_order and tuple(self.concept_order[: self.resume_index]) != tuple(self.skip_concepts):
            raise ValueError(
                "skip_concepts must be the leading prefix of concept_order; "
                f"got {list(self.skip_concepts)} vs {list(self.concept_order[: self.resume_index])}"
            )
        if self.evaluate_pending_concept and self.resume_index >= len(self.concept_order):
            raise ValueError("evaluate_pending_concept was requested but there is no concept left to evaluate")

    @classmethod
    def fresh(cls, concept_order: Sequence[str] = ()) -> "ResumePlan":
        return cls(
            skip_concepts=(),
            resume_index=0,
            evaluate_pending_concept=False,
            concept_order=tuple(str(name) for name in concept_order),
        )

    @classmethod
    def after_completed(
        cls,
        concept_order: Sequence[str],
        completed: int,
        *,
        evaluate_pending_concept: bool = False,
    ) -> "ResumePlan":
        """Build a plan that skips the first *completed* concepts."""
        order = tuple(str(name) for name in concept_order)
        return cls(
            skip_concepts=order[:completed],
            resume_index=completed,
            evaluate_pending_concept=evaluate_pending_concept,
            concept_order=order,
        )

    @property
    def resumed(self) -> bool:
        return self.resume_index > 0

    def info(self) -> dict:
        return {
            "resume_index": self.resume_index,
            "skipped_concepts": list(self.skip_concepts),
            "evaluate_pending_concept": self.evaluate_pending_concept,
        }


class _ResumableScenarioBase:
    """Shared loop for the two resumable scenarios."""

    def __init__(
        self, dataset: ConceptsDataset, strategy, callbacks: List[Callback], *, plan: ResumePlan | None = None
    ):
        self._dataset = dataset
        self._strategy = strategy
        self._callbacks = callbacks
        self._plan = plan if plan is not None else ResumePlan.fresh([c.name for c in dataset.train_concepts()])

    @property
    def plan(self) -> ResumePlan:
        return self._plan

    def _learn(self, train_concept: Concept) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _predict(self, test_concept: Concept):  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self) -> None:
        plan = self._plan
        train_concepts = self._dataset.train_concepts()
        self._assert_dataset_order(train_concepts)

        callback_composite = CallbackComposite(self._callbacks)
        callback_composite.before_scenario()

        for index, train_concept in enumerate(train_concepts):
            if index < plan.resume_index:
                # An explicit raise rather than `assert`: python -O strips
                # asserts, and this guard is the last line of defence against
                # writing concept k's metrics into concept j's matrix row.
                if train_concept.name != plan.skip_concepts[index]:
                    raise CheckpointOrderMismatch(
                        f"Concept at index {index} is '{train_concept.name}' but the checkpoint recorded "
                        f"'{plan.skip_concepts[index]}' as already completed; "
                        f"resolved order {describe_sequence([c.name for c in train_concepts])}"
                    )
                logger.info("Resume: skipping already-completed concept %s", train_concept.name)
                continue

            replay_only = plan.evaluate_pending_concept and index == plan.resume_index

            logger.info("Starting training on concept %s", train_concept.name)
            callback_composite.before_concept_processing(concept=train_concept)
            # before_training still fires in replay-only mode: callbacks that
            # time a concept start their clock here, and leaving that clock at
            # zero would record an epoch-sized training time.
            callback_composite.before_training()

            if replay_only:
                logger.info(
                    "Resume: concept %s was already trained (pending checkpoint); re-running its evaluation only",
                    train_concept.name,
                )
            else:
                self._learn(train_concept)

            callback_composite.after_training(learned_concept=train_concept)

            for test_concept in self._dataset.test_concepts():
                logger.info("Starting evaluation of concept %s", test_concept.name)
                callback_composite.before_evaluation()
                y_predicted, anomaly_scores = self._predict(test_concept)
                callback_composite.after_evaluation(
                    evaluated_concept=test_concept,
                    y_true=test_concept.labels,
                    y_pred=y_predicted,
                    anomaly_scores=anomaly_scores,
                )

            callback_composite.after_concept_processing(concept=train_concept)

        callback_composite.after_scenario()

    def _assert_dataset_order(self, train_concepts: Sequence[Concept]) -> None:
        expected = tuple(self._plan.concept_order)
        if not expected:
            return
        actual = tuple(concept.name for concept in train_concepts)
        if actual != expected:
            raise CheckpointOrderMismatch(
                "The dataset handed to the scenario is not in the order the resume plan was built on.\n"
                f"  plan:    {describe_sequence(list(expected))}\n"
                f"  dataset: {describe_sequence(list(actual))}"
            )


class ResumableConceptAwareScenario(_ResumableScenarioBase):
    """``ConceptAwareScenario`` that can skip an already-completed prefix."""

    def __init__(
        self,
        dataset: ConceptsDataset,
        strategy: ConceptAwareStrategy,
        callbacks: List[Callback],
        *,
        plan: ResumePlan | None = None,
    ):
        super().__init__(dataset, strategy, callbacks, plan=plan)

    def _learn(self, train_concept: Concept) -> None:
        self._strategy.learn(data=train_concept.data, concept_id=train_concept.name)

    def _predict(self, test_concept: Concept):
        return self._strategy.predict(data=test_concept.data, concept_id=test_concept.name)


class ResumableConceptIncrementalScenario(_ResumableScenarioBase):
    """``ConceptIncrementalScenario`` counterpart.

    Needed because One-for-More has no strategy of its own: it runs as a bare
    model under ``NaiveStrategy``, which the runner routes through the
    *incremental* scenario. Note that this scenario passes no ``concept_id`` —
    the strategy signature genuinely differs.
    """

    def __init__(
        self,
        dataset: ConceptsDataset,
        strategy: ConceptIncrementalStrategy,
        callbacks: List[Callback],
        *,
        plan: ResumePlan | None = None,
    ):
        super().__init__(dataset, strategy, callbacks, plan=plan)

    def _learn(self, train_concept: Concept) -> None:
        self._strategy.learn(data=train_concept.data)

    def _predict(self, test_concept: Concept):
        return self._strategy.predict(data=test_concept.data)
