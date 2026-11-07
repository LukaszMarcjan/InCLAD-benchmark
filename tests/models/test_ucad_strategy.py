import numpy as np

from pyclad.strategies.vision.ucad import UCADStrategy


class _FakeUCAD:
    def __init__(self):
        self.calls = []

    def begin_continual_run(self):
        self.calls.append("begin")

    def fit_task(self, data, concept_id=None):
        self.calls.append(("fit", len(data), concept_id))

    def predict(self, data):
        self.calls.append(("predict", len(data)))
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=np.float32)


def test_ucad_strategy_keeps_training_identity_but_ignores_test_identity():
    model = _FakeUCAD()
    strategy = UCADStrategy(model)
    data = np.zeros((2, 4), dtype=np.float32)

    strategy.learn(data, concept_id="task-a")
    strategy.learn(data[:1], concept_id="task-b")
    strategy.predict(data, concept_id="must-not-route-oracle")

    assert model.calls == [
        "begin",
        ("fit", 2, "task-a"),
        ("fit", 1, "task-b"),
        ("predict", 2),
    ]
    assert strategy.additional_info()["task_agnostic_inference"] is True
