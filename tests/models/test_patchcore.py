import torch

from pyclad.models.patchcore.config import PatchCoreConfig
from pyclad.models.patchcore.builder import build


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
