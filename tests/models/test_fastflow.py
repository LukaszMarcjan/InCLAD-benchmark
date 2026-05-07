import numpy as np
import pytest

from pyclad.models.fastflow.config import FastFlowConfig
from pyclad.models.fastflow.fastflow import FastFlow


@pytest.mark.parametrize(
    "backbone_name",
    ["resnet18", "resnet50", "wide_resnet50_2", "mobilenet_v2", "efficientnet_b0", "efficientnet_v2_s"],
)
def test_fastflow_predict_shapes(backbone_name: str):
    data = np.random.default_rng(0).random((1, 32, 32, 3), dtype=np.float32)
    model = FastFlow(
        FastFlowConfig(
            input_size=(32, 32),
            backbone_name=backbone_name,
            pretrained_backbone=False,
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
        )
    )

    y_pred, scores = model.predict(data)
    score_maps = model.score_maps(data)

    assert y_pred.shape == (1,)
    assert scores.shape == (1,)
    assert score_maps.shape == (1, 32, 32)


def test_fastflow_custom_backbone_return_nodes():
    data = np.random.default_rng(2).random((1, 32, 32, 3), dtype=np.float32)
    model = FastFlow(
        FastFlowConfig(
            input_size=(32, 32),
            backbone_name="mobilenet_v2",
            backbone_return_nodes=("features.1", "features.6"),
            pretrained_backbone=False,
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
        )
    )

    _, scores = model.predict(data)
    assert scores.shape == (1,)


def test_fastflow_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 64, 64, 3), dtype=np.float32)
    model = FastFlow(
        FastFlowConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            pretrained_backbone=False,
            batch_size=2,
            epochs=1,
            show_training_progress=False,
        )
    )

    model.fit(data)
    y_pred, scores = model.predict(data)

    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
