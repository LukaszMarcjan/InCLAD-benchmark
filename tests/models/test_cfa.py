import numpy as np
import pytest

from pyclad.models.cfa.cfa import CFA
from pyclad.models.cfa.config import CFAConfig


def test_cfa_requires_fit_before_predict():
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = CFA(
        CFAConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            pretrained_backbone=False,
            batch_size=1,
            epochs=0,
            show_training_progress=False,
        )
    )

    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict(data)


@pytest.mark.parametrize("backbone_name", ["resnet18", "mobilenet_v2", "efficientnet_b0"])
def test_cfa_fit_predict_shapes(backbone_name: str):
    data = np.random.default_rng(1).random((2, 64, 64, 3), dtype=np.float32)
    model = CFA(
        CFAConfig(
            input_size=(64, 64),
            backbone_name=backbone_name,
            pretrained_backbone=False,
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
            score_smoothing_kernel=1,
            score_smoothing_sigma=0.0,
        )
    )

    model.fit(data)
    y_pred, scores = model.predict(data)
    score_maps = model.score_maps(data)

    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
    assert score_maps.shape == (2, 64, 64)


def test_cfa_fit_smoke_run():
    data = np.random.default_rng(2).random((2, 64, 64, 3), dtype=np.float32)
    model = CFA(
        CFAConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            pretrained_backbone=False,
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
