import numpy as np

from pyclad.models.ganomaly.config import GANomalyConfig
from pyclad.models.ganomaly.ganomaly import GANomaly


def test_ganomaly_predict_shapes_with_non_power_of_two_input():
    data = np.random.default_rng(0).random((1, 48, 32, 3), dtype=np.float32)
    model = GANomaly(
        GANomalyConfig(
            input_size=(48, 32),
            n_features=8,
            latent_vec_size=16,
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
    assert score_maps.shape == (1, 48, 32)


def test_ganomaly_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 64, 64, 3), dtype=np.float32)
    model = GANomaly(
        GANomalyConfig(
            input_size=(64, 64),
            n_features=8,
            latent_vec_size=16,
            batch_size=2,
            epochs=1,
            show_training_progress=False,
        )
    )

    model.fit(data)
    y_pred, scores = model.predict(data)

    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
