import torch

from pyclad.models.stfpm.config import STFPMConfig
from pyclad.models.stfpm.builder import build


def test_stfpm_efficientnet_b0_default_nodes_and_forward_shapes():
    module = build(
        STFPMConfig(
            input_size=(64, 64),
            backbone_name="efficientnet_b0",
            pretrained_teacher=False,
            pretrained_student=False,
        )
    )

    assert module.return_nodes == ("features.2", "features.3", "features.5")

    teacher_features, student_features = module(torch.zeros(1, 3, 64, 64))

    assert len(teacher_features) == len(student_features) == 3
    assert all(feature.shape[0] == 1 for feature in teacher_features)
    assert all(feature.shape[0] == 1 for feature in student_features)
