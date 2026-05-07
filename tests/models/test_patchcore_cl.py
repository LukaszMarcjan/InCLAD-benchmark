import numpy as np

from pyclad.models.patchcore.config import PatchCoreConfig
from pyclad.models.patchcore.patchcore_cl import ContinualPatchCore
from pyclad.strategies.vision.patchcore_cl import PatchCoreCLStrategy


def test_continual_patchcore_respects_budget_and_predicts():
    rng = np.random.default_rng(1)
    data_a = rng.random((1, 64, 64, 3), dtype=np.float32)
    data_b = rng.random((1, 64, 64, 3), dtype=np.float32)
    model = ContinualPatchCore(
        PatchCoreConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            pretrained_backbone=False,
            batch_size=1,
            pretrain_embed_dimension=32,
            target_embed_dimension=32,
            n_neighbors=1,
            threshold=0.5,
            smoothing_sigma=0.0,
        )
    )
    # previous_ratio=0.25: previous tasks share 25% of the current task's coreset.
    strategy = PatchCoreCLStrategy(model=model, previous_ratio=0.25)

    strategy.learn(data_a)
    strategy.learn(data_b)
    y_pred, scores = strategy.predict(data_a)

    assert y_pred.shape == (1,)
    assert scores.shape == (1,)
    assert model._task_memory_state is not None
    assert len(model._task_memory_state.task_banks) == 2

    # After 2 tasks: task_banks[0] is shrunk to ≤ 25% of task_banks[1] size.
    bank_0_size = len(model._task_memory_state.task_banks[0])
    bank_1_size = len(model._task_memory_state.task_banks[1])
    assert bank_0_size <= max(1, int(0.25 * bank_1_size)), (
        f"Previous bank size {bank_0_size} exceeds 25% budget of current bank ({bank_1_size})"
    )
