import numpy as np

from pyclad.models.efficientad.config import EfficientADConfig
from pyclad.models.efficientad.efficientad import EfficientAD


def test_efficientad_predict_shapes():
    data = np.random.default_rng(0).random((2, 256, 256, 3), dtype=np.float32)
    model = EfficientAD(
        EfficientADConfig(
            input_size=(256, 256),
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
            teacher_out_channels=32,
        )
    )

    y_pred, scores = model.predict(data)
    score_maps = model.score_maps(data)

    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
    assert score_maps.shape == (2, 256, 256)


def test_efficientad_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 256, 256, 3), dtype=np.float32)
    model = EfficientAD(
        EfficientADConfig(
            input_size=(256, 256),
            batch_size=1,
            epochs=1,
            show_training_progress=False,
            teacher_out_channels=32,
        )
    )

    model.fit(data)
    y_pred, scores = model.predict(data)

    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
