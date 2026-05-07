import numpy as np

from pyclad.strategies.vision.padim_cl import PaDiMCLStrategy


class _FakePaDiM:
    """Minimal stub that records which PaDiM CL methods are called."""

    def __init__(self):
        self.calls: list[str] = []

    def begin_continual_run(self):
        self.calls.append("begin")

    def fit_task_gaussian(self, data):
        self.calls.append(f"fit:{len(data)}")

    def update_threshold_from_reference_data(self, data):
        self.calls.append(f"threshold:{len(data)}")

    def predict(self, data):
        self.calls.append(f"predict:{len(data)}")
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=np.float32)


def test_padim_cl_strategy_initializes_once_and_delegates_predict():
    model = _FakePaDiM()
    strategy = PaDiMCLStrategy(model)
    batch_a = np.zeros((2, 4), dtype=np.float32)
    batch_b = np.zeros((1, 4), dtype=np.float32)

    strategy.learn(batch_a)
    strategy.learn(batch_b)
    y_pred, scores = strategy.predict(batch_a)

    # begin_continual_run is called exactly once (on the first task).
    # Each task: fit_task_gaussian + update_threshold.
    assert model.calls == [
        "begin",
        "fit:2",
        "threshold:2",
        "fit:1",
        "threshold:1",
        "predict:2",
    ]
    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
    assert strategy._tasks_seen == 2
