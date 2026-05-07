import numpy as np

from pyclad.models.padim.config import PaDiMConfig
from pyclad.models.padim.padim_cl import ContinualPaDiM
from pyclad.strategies.vision.padim_cl import PaDiMCLStrategy


def test_continual_padim_learns_multiple_tasks_and_predicts():
    rng = np.random.default_rng(0)
    data_a = rng.random((1, 64, 64, 3), dtype=np.float32)
    data_b = rng.random((1, 64, 64, 3), dtype=np.float32)
    model = ContinualPaDiM(
        PaDiMConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            pretrained_backbone=False,
            batch_size=1,
            n_features=8,
            threshold=0.5,
        )
    )
    strategy = PaDiMCLStrategy(model)

    strategy.learn(data_a)
    strategy.learn(data_b)
    y_pred, scores = strategy.predict(data_a)

    assert y_pred.shape == (1,)
    assert scores.shape == (1,)
    assert model.continual_state_info()["continual_num_tasks"] == 2
