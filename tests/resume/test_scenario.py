"""The resumable scenarios must skip *exactly* the completed concepts.

"Exactly" has two halves: a skipped concept must fire no callback at all (an
``after_training`` would append to every metric callback's learned-concept list
and shift every matrix row), and a non-skipped concept must behave precisely as
it does in the core scenario.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyclad.resume.protocol import CheckpointOrderMismatch
from pyclad.resume.scenario import (
    ResumableConceptAwareScenario,
    ResumableConceptIncrementalScenario,
    ResumePlan,
)
from pyclad.scenarios.concept_aware import ConceptAwareScenario

from .conftest import (
    CONCEPTS,
    EventRecorder,
    FakeState,
    RecordingIncrementalStrategy,
    RecordingStrategy,
    make_dataset,
)


def test_fresh_plan_matches_the_core_scenario_event_for_event():
    dataset = make_dataset()

    core_recorder = EventRecorder()
    ConceptAwareScenario(dataset, RecordingStrategy(FakeState()), [core_recorder]).run()

    resumable_recorder = EventRecorder()
    ResumableConceptAwareScenario(
        dataset, RecordingStrategy(FakeState()), [resumable_recorder], plan=ResumePlan.fresh(CONCEPTS)
    ).run()

    assert resumable_recorder.events == core_recorder.events


def test_resume_skips_exactly_the_completed_concepts():
    dataset = make_dataset()
    strategy = RecordingStrategy(FakeState())
    recorder = EventRecorder()
    plan = ResumePlan.after_completed(CONCEPTS, 3)

    ResumableConceptAwareScenario(dataset, strategy, [recorder], plan=plan).run()

    assert strategy.learn_calls == ["carpet", "grid"]
    assert recorder.names_for("before_concept_processing") == ["carpet", "grid"]
    assert recorder.names_for("after_training") == ["carpet", "grid"]
    assert recorder.names_for("after_concept_processing") == ["carpet", "grid"]
    # Skipping the whole iteration, not just training: the evaluation sweep of a
    # skipped concept is most of what resuming saves.
    assert strategy.predict_calls == list(CONCEPTS) * 2
    assert recorder.events[0][0] == "before_scenario"
    assert recorder.events[-1][0] == "after_scenario"


def test_skipped_concepts_fire_no_callbacks_whatsoever():
    dataset = make_dataset()
    recorder = EventRecorder()
    plan = ResumePlan.after_completed(CONCEPTS, 2)

    ResumableConceptAwareScenario(dataset, RecordingStrategy(FakeState()), [recorder], plan=plan).run()

    touched = {name for _, name in recorder.events if name is not None}
    assert "bottle" in touched and "cable" in touched, "they are still evaluated as test concepts"
    for event in ("before_concept_processing", "after_training", "after_concept_processing"):
        assert "bottle" not in recorder.names_for(event)
        assert "cable" not in recorder.names_for(event)


def test_pending_replay_evaluates_without_re_training():
    dataset = make_dataset()
    strategy = RecordingStrategy(FakeState())
    recorder = EventRecorder()
    plan = ResumePlan.after_completed(CONCEPTS, 2, evaluate_pending_concept=True)

    ResumableConceptAwareScenario(dataset, strategy, [recorder], plan=plan).run()

    # 'capsule' is at resume_index: already trained before the crash, so it is
    # evaluated but not re-fitted.
    assert strategy.learn_calls == ["carpet", "grid"]
    assert recorder.names_for("after_training") == ["capsule", "carpet", "grid"]
    assert recorder.names_for("before_training") == []  # before_training carries no name
    assert len([kind for kind, _ in recorder.events if kind == "before_training"]) == 3
    assert strategy.predict_calls == list(CONCEPTS) * 3


def test_plan_rejects_an_inconsistent_prefix():
    with pytest.raises(ValueError, match="leading prefix"):
        ResumePlan(
            skip_concepts=("cable", "bottle"),
            resume_index=2,
            evaluate_pending_concept=False,
            concept_order=tuple(CONCEPTS),
        )
    with pytest.raises(ValueError, match="resume_index"):
        ResumePlan(
            skip_concepts=("bottle",),
            resume_index=2,
            evaluate_pending_concept=False,
            concept_order=tuple(CONCEPTS),
        )


def test_reordered_dataset_aborts_before_any_training():
    reordered = list(CONCEPTS)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    dataset = make_dataset(reordered)
    strategy = RecordingStrategy(FakeState())
    plan = ResumePlan.after_completed(CONCEPTS, 2)

    with pytest.raises(CheckpointOrderMismatch, match="not in the order"):
        ResumableConceptAwareScenario(dataset, strategy, [], plan=plan).run()

    assert strategy.learn_calls == [], "nothing may be trained once the order is suspect"


def test_per_iteration_guard_survives_python_dash_O():
    """The in-loop order check is a raise, not an assert: -O strips asserts."""
    dataset = make_dataset()
    plan = ResumePlan(
        skip_concepts=("bottle", "capsule"),
        resume_index=2,
        evaluate_pending_concept=False,
        concept_order=("bottle", "capsule", "cable", "carpet", "grid"),
    )
    scenario = ResumableConceptAwareScenario(dataset, RecordingStrategy(FakeState()), [], plan=plan)
    # The dataset-level guard fires first; force the per-iteration path by
    # letting the plan agree with the dataset order but not with itself.
    scenario._plan = ResumePlan.after_completed(CONCEPTS, 2)
    object.__setattr__(scenario._plan, "skip_concepts", ("bottle", "grid"))
    with pytest.raises(CheckpointOrderMismatch, match="already completed"):
        scenario.run()


def test_incremental_scenario_skips_the_same_way():
    dataset = make_dataset()
    strategy = RecordingIncrementalStrategy(FakeState())
    recorder = EventRecorder()
    plan = ResumePlan.after_completed(CONCEPTS, 4)

    ResumableConceptIncrementalScenario(dataset, strategy, [recorder], plan=plan).run()

    assert strategy.learn_calls == [3]
    assert recorder.names_for("after_concept_processing") == ["grid"]
    assert np.array(strategy.predict_calls).tolist() == [3] * len(CONCEPTS)


def test_plan_without_order_defaults_to_the_dataset():
    dataset = make_dataset()
    scenario = ResumableConceptAwareScenario(dataset, RecordingStrategy(FakeState()), [])
    assert scenario.plan.concept_order == tuple(CONCEPTS)
    assert scenario.plan.resume_index == 0
