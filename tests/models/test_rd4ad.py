import numpy as np
import pytest

from pyclad.models.rd4ad.config import RD4ADConfig
from pyclad.models.rd4ad.rd4ad import RD4AD


@pytest.mark.parametrize("backbone_name", ["resnet18", "resnet34", "resnet50", "wide_resnet50_2"])
def test_rd4ad_predict_shapes(backbone_name: str):
    data = np.random.default_rng(0).random((1, 32, 32, 3), dtype=np.float32)
    model = RD4AD(
        RD4ADConfig(
            input_size=(32, 32),
            backbone_name=backbone_name,
            pretrained_encoder=False,
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
            score_smoothing_kernel=1,
            score_smoothing_sigma=0.0,
        )
    )

    y_pred, scores = model.predict(data)
    score_maps = model.score_maps(data)

    assert y_pred.shape == (1,)
    assert scores.shape == (1,)
    assert score_maps.shape == (1, 32, 32)


def test_rd4ad_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 32, 32, 3), dtype=np.float32)
    model = RD4AD(
        RD4ADConfig(
            input_size=(32, 32),
            pretrained_encoder=False,
            batch_size=2,
            epochs=1,
            show_training_progress=False,
            score_smoothing_kernel=1,
            score_smoothing_sigma=0.0,
        )
    )

    model.fit(data)
    y_pred, scores = model.predict(data)

    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
