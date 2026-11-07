from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from pyclad.models.patchcore.config import PatchCoreConfig
from pyclad.models.patchcore.builder import build
from pyclad.models.patchcore.patchcore import PatchCore


def test_patchcore_efficientnet_b0_default_nodes_and_forward_shapes():
    module = build(
        PatchCoreConfig(
            input_size=(64, 64),
            backbone_name="efficientnet_b0",
            pretrained_backbone=False,
        )
    )

    assert module.return_nodes == ("features.3", "features.5")

    outputs = module(torch.zeros(1, 3, 64, 64))

    assert len(outputs) == 2
    assert outputs[0].shape[0] == outputs[1].shape[0] == 1


def test_patchcore_reuses_maps_from_the_immediately_preceding_prediction():
    model = PatchCore.__new__(PatchCore)
    model.config = SimpleNamespace(threshold=None)
    model._threshold = 0.5
    model._last_prediction_data = None
    model._last_prediction_maps = None
    maps = np.ones((2, 4, 4), dtype=np.float32)
    model._predict_scores_and_masks = Mock(
        return_value=(np.asarray([0.1, 0.9], dtype=np.float32), maps)
    )
    data = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    y_pred, scores = model.predict(data)
    returned_maps = model.score_maps(data)

    assert y_pred.tolist() == [0, 1]
    assert np.allclose(scores, [0.1, 0.9])
    assert returned_maps is maps
    model._predict_scores_and_masks.assert_called_once_with(data)
