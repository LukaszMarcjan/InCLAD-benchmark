import numpy as np

from pyclad.strategies.vision.cfa_cl import CFACLStrategy


class _FakeBuffer:
    def __init__(self):
        self._data = np.zeros((0, 3), dtype=np.float32)
        self.updated = 0

    def data(self):
        return self._data

    def update(self, data):
        self._data = data.copy()
        self.updated += 1

    def info(self):
        return {"updated": self.updated}


class _FakeCFA:
    """Minimal stub that records which CFA CL methods are called."""

    def __init__(self):
        self.calls: list[str] = []

    def begin_continual_run(self):
        self.calls.append("begin")

    def initialize_training_bank(self, data):
        self.calls.append(f"init_bank:{len(data)}")

    def fit_descriptor_only(self, data):
        self.calls.append(f"fit:{len(data)}")

    def rebuild_task_inference_banks(self, task_samples):
        self.calls.append(f"rebuild:{len(task_samples)}")

    def update_threshold_from_reference_data(self, data):
        self.calls.append(f"threshold:{len(data)}")

    def predict(self, data):
        self.calls.append(f"predict:{len(data)}")
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=np.float32)


def test_cfa_cl_strategy_uses_per_task_banks_and_replay():
    buffer = _FakeBuffer()
    model = _FakeCFA()
    strategy = CFACLStrategy(model=model, buffer=buffer)
    batch_a = np.zeros((2, 3), dtype=np.float32)
    batch_b = np.zeros((1, 3), dtype=np.float32)

    strategy.learn(batch_a)
    strategy.learn(batch_b)
    strategy.predict(batch_a)

    # Task 0 (no replay yet):
    #   begin → init_bank:2 → fit:2 → rebuild:1 → threshold:2
    # Task 1 (batch_a in replay buffer, len=2; train_data = batch_a+batch_b = 3):
    #   init_bank:3 → fit:3 → rebuild:2 → threshold:1
    assert model.calls == [
        "begin",
        "init_bank:2",
        "fit:2",
        "rebuild:1",
        "threshold:2",
        "init_bank:3",
        "fit:3",
        "rebuild:2",
        "threshold:1",
        "predict:2",
    ]
    assert buffer.updated == 2
    # Strategy tracks one array per task.
    assert len(strategy._task_samples) == 2
