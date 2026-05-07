import types

import numpy as np

from pyclad.strategies.vision.patchcore_cl import PatchCoreCLStrategy

_BANK_SIZE = 100  # synthetic bank size returned by the fake model


class _FakePatchCore:
    """Minimal stub that records which PatchCore CL methods are called."""

    def __init__(self):
        self.calls: list[str] = []
        self._task_memory_state = types.SimpleNamespace(task_banks=[])

    def begin_continual_run(self):
        self.calls.append("begin")

    def build_current_task_bank(self, data):
        self.calls.append(f"build:{len(data)}")
        return np.zeros((_BANK_SIZE, 4), dtype=np.float32)

    def shrink_bank_weighted(self, bank, target_size, reference_bank):
        self.calls.append(f"shrink:{len(bank)}→{target_size}")
        return np.zeros((target_size, 4), dtype=np.float32)

    def set_task_banks(self, task_banks):
        self.calls.append(f"set:{len(task_banks)}")
        self._task_memory_state.task_banks = list(task_banks)

    def update_threshold_from_reference_data(self, data):
        self.calls.append(f"threshold:{len(data)}")

    def predict(self, data):
        self.calls.append(f"predict:{len(data)}")
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=np.float32)


def test_patchcore_cl_strategy_rebalances_after_first_task():
    model = _FakePatchCore()
    # previous_ratio=0.25 → previous budget = 25% of current bank size = 25 patches.
    strategy = PatchCoreCLStrategy(model=model, previous_ratio=0.25)
    batch_a = np.zeros((2, 4), dtype=np.float32)
    batch_b = np.zeros((3, 4), dtype=np.float32)

    strategy.learn(batch_a)
    strategy.learn(batch_b)
    y_pred, scores = strategy.predict(batch_a)

    # Task 0: begin → build:2 → set:1 (no previous banks) → threshold:2
    # Task 1: build:3 → shrink previous bank (100 → 25) → set:2 → threshold:3
    previous_budget = max(1, int(0.25 * _BANK_SIZE))  # 25
    per_task_budget = max(1, previous_budget // 1)     # 25 (one previous task)
    assert model.calls == [
        "begin",
        f"build:2",
        "set:1",
        "threshold:2",
        "build:3",
        f"shrink:{_BANK_SIZE}→{per_task_budget}",
        "set:2",
        "threshold:3",
        "predict:2",
    ]
    assert len(model._task_memory_state.task_banks) == 2
    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
