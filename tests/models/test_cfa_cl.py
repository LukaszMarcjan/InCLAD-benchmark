import numpy as np

from pyclad.models.cfa.cfa_cl import ContinualCFA
from pyclad.models.cfa.config import CFAConfig
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.selection.random import RandomSelection
from pyclad.strategies.vision.cfa_cl import CFACLStrategy


def test_continual_cfa_updates_memory_and_predicts():
    rng = np.random.default_rng(2)
    data_a = rng.random((1, 64, 64, 3), dtype=np.float32)
    data_b = rng.random((1, 64, 64, 3), dtype=np.float32)
    model = ContinualCFA(
        CFAConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            pretrained_backbone=False,
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
            score_smoothing_kernel=1,
            score_smoothing_sigma=0.0,
        )
    )
    buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=2)
    strategy = CFACLStrategy(model=model, buffer=buffer)

    strategy.learn(data_a)
    strategy.learn(data_b)
    y_pred, scores = strategy.predict(data_a)

    assert y_pred.shape == (1,)
    assert scores.shape == (1,)
    assert model.continual_state_info()["continual_num_tasks"] >= 2
