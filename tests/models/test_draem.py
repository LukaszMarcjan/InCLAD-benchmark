import numpy as np

from pyclad.models.draem.config import DRAEMConfig
from pyclad.models.draem.draem import DRAEM


def test_draem_predict_shapes():
    data = np.random.default_rng(0).random((4, 32, 32, 3), dtype=np.float32)
    model = DRAEM(
        DRAEMConfig(
            input_size=(32, 32),
            batch_size=2,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
            score_smoothing_kernel=5,
            reconstructive_base_width=16,
            discriminative_base_width=8,
        )
    )

    y_pred, scores = model.predict(data)
    score_maps = model.score_maps(data)
    reconstructions = model.reconstruct(data)

    assert y_pred.shape == (4,)
    assert scores.shape == (4,)
    assert score_maps.shape == (4, 32, 32)
    assert reconstructions.shape == (4, 3, 32, 32)


def test_draem_fit_smoke_run():
    data = np.random.default_rng(1).random((4, 32, 32, 3), dtype=np.float32)
    model = DRAEM(
        DRAEMConfig(
            input_size=(32, 32),
            batch_size=2,
            epochs=1,
            show_training_progress=False,
            score_smoothing_kernel=5,
            reconstructive_base_width=16,
            discriminative_base_width=8,
        )
    )

    model.fit(data)
    y_pred, scores = model.predict(data)

    assert y_pred.shape == (4,)
    assert scores.shape == (4,)


def test_draem_backbone_variant_predict_shapes():
    data = np.random.default_rng(2).random((2, 64, 64, 3), dtype=np.float32)
    model = DRAEM(
        DRAEMConfig(
            variant="backbone",
            input_size=(64, 64),
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
            score_smoothing_kernel=5,
            backbone_name="resnet18",
            pretrained_backbone=False,
            freeze_backbone=False,
            discriminative_base_width=8,
        )
    )

    y_pred, scores = model.predict(data)
    score_maps = model.score_maps(data)
    reconstructions = model.reconstruct(data)

    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
    assert score_maps.shape == (2, 64, 64)
    assert reconstructions.shape == (2, 3, 64, 64)
