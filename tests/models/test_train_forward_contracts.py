import torch

from pyclad.models.cfa.architecture import CFAArchitecture
from pyclad.models.fastflow.architecture import FastFlowArchitecture
from pyclad.models.ganomaly.architecture import GANomalyArchitecture
from pyclad.models.paste.architecture import PaSTeArchitecture
from pyclad.models.rd4ad.architecture import RD4ADArchitecture


def test_fastflow_forward_train_restores_training_contract():
    model = FastFlowArchitecture(
        in_channels=3,
        input_size=(64, 64),
        backbone_name="resnet18",
        backbone_return_nodes=None,
        pretrained_backbone=False,
        freeze_backbone=True,
        normalize_features=True,
        flow_steps=2,
        conv3x3_only=False,
        hidden_ratio=1.0,
        affine_clamping=2.0,
        score_mode="max",
    )

    model.eval()
    hidden_variables, log_jacobians = model.forward_train(torch.randn(2, 3, 64, 64))

    assert model.training
    assert not model.feature_extractor.training
    assert model.fast_flow_blocks.training
    assert len(hidden_variables) == len(log_jacobians) == len(model.return_nodes)
    assert all(hidden_variable.ndim == 4 and hidden_variable.shape[0] == 2 for hidden_variable in hidden_variables)
    assert all(log_jacobian.ndim == 1 and log_jacobian.shape[0] == 2 for log_jacobian in log_jacobians)


def test_ganomaly_forward_train_restores_training_contract():
    model = GANomalyArchitecture(
        input_size=(64, 64),
        in_channels=3,
        n_features=8,
        latent_vec_size=16,
        extra_layers=0,
        add_final_conv_layer=True,
    )

    model.eval()
    padded, fake, latent_i, latent_o = model.forward_train(torch.randn(2, 3, 64, 64))

    assert model.training
    assert model.generator.training
    assert model.discriminator.training
    assert padded.shape == fake.shape == (2, 3, 64, 64)
    assert latent_i.shape[0] == latent_o.shape[0] == 2


def test_paste_forward_train_restores_training_contract():
    model = PaSTeArchitecture(
        backbone_name="resnet18",
        ad_layers=(1, 2, 3),
        student_bootstrap_layer=0,
        pretrained_teacher=False,
        pretrained_student=False,
        freeze_teacher=True,
        input_size=(64, 64),
    )

    model.eval()
    teacher_features, student_features = model.forward_train(torch.randn(2, 3, 64, 64))

    assert model.training
    assert not model.teacher.training
    assert model.student.training
    assert len(teacher_features) == len(student_features) == 3
    assert all(feature.ndim == 4 and feature.shape[0] == 2 for feature in teacher_features)
    assert all(feature.ndim == 4 and feature.shape[0] == 2 for feature in student_features)


def test_rd4ad_forward_train_restores_training_contract():
    model = RD4ADArchitecture(
        backbone_name="resnet18",
        input_size=(64, 64),
        pretrained_encoder=False,
        freeze_encoder=True,
        score_smoothing_kernel=1,
        score_smoothing_sigma=0.0,
    )

    model.eval()
    teacher_features, bottleneck_features, student_features = model.forward_train(torch.randn(2, 3, 64, 64))

    assert model.training
    assert not model.encoder.training
    assert model.bn.training
    assert model.decoder.training
    assert len(teacher_features) == len(student_features) == 3
    assert bottleneck_features.ndim == 4 and bottleneck_features.shape[0] == 2
    assert all(feature.ndim == 4 and feature.shape[0] == 2 for feature in teacher_features)
    assert all(feature.ndim == 4 and feature.shape[0] == 2 for feature in student_features)


def test_cfa_forward_train_restores_training_contract():
    model = CFAArchitecture(
        in_channels=3,
        input_size=(64, 64),
        backbone_name="resnet18",
        backbone_return_nodes=None,
        pretrained_backbone=False,
        freeze_backbone=True,
        gamma_c=1,
        gamma_d=1,
        k_neighbors=3,
        repulsion_neighbors=3,
        nu=1e-3,
        alpha=1e-1,
        radius_init=1e-5,
        score_smoothing_kernel=1,
        score_smoothing_sigma=0.0,
        score_mode="max",
        random_seed=0,
    )

    with torch.no_grad():
        descriptor_map = model.describe(torch.randn(1, 3, 64, 64))
        flat_descriptors = model._flatten_descriptors(descriptor_map)
        model.memory_bank = flat_descriptors[0].transpose(0, 1).contiguous()

    model.eval()
    loss = model.forward_train(torch.randn(2, 3, 64, 64))

    assert model.training
    assert not model.feature_extractor.training
    assert model.descriptor.training
    assert loss.ndim == 0
