import torch

from pyclad.models.padim.config import PaDiMConfig
from pyclad.models.padim.builder import build


def test_padim_efficientnet_b0_default_nodes_and_forward_shapes():
    module = build(
        PaDiMConfig(
            input_size=(64, 64),
            backbone_name="efficientnet_b0",
            pretrained_backbone=False,
        )
    )

    assert module.return_nodes == ("features.2", "features.3", "features.5")

    outputs = module(torch.zeros(1, 3, 64, 64))

    assert len(outputs) == 3
    assert all(output.shape[0] == 1 for output in outputs)
