import numpy as np
import pytest
import torch

from pyclad.models.one_for_more import OneForMore, OneForMoreConfig
from pyclad.models.one_for_more.architecture import AnomalyMaskedConditioner
from pyclad.models.one_for_more.projection import IterativeSubspaceProjector


def _tiny_config(**updates) -> OneForMoreConfig:
    values = {
        "input_size": (32, 32),
        "latent_downsample": 8,
        "base_channels": 8,
        "conditioner_depth": 0,
        "conditioner_heads": 1,
        "conditioner_feedforward": 16,
        "diffusion_steps": 10,
        "inference_start_step": 9,
        "inference_steps": 1,
        "epochs": 0,
        "incremental_epochs": 0,
        "batch_size": 2,
        "projection_batches": 1,
        "projection_max_columns": 32,
        "projection_energy_threshold": 0.9,
        "backbone_name": None,
        "score_smoothing_sigma": 0.0,
        "threshold": 0.5,
        "show_training_progress": False,
        "device": "cpu",
    }
    values.update(updates)
    return OneForMoreConfig(**values)


def test_isvd_basis_reaches_energy_threshold_and_projects_gradient():
    matrix = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    projector = IterativeSubspaceProjector(energy_threshold=0.8, max_columns=None)

    projector.update("layer", matrix)

    basis = projector.bases["layer"]
    represented = basis @ (basis.T @ matrix)
    retained_ratio = represented.square().sum() / matrix.square().sum()
    assert retained_ratio >= 0.8

    gradient = torch.tensor([[1.0, 2.0, 3.0]])
    projected = projector.project_gradient("layer", gradient)
    assert torch.allclose(projected @ basis, torch.zeros((1, basis.shape[1])), atol=1e-6)


def test_isvd_updates_without_retaining_activation_columns():
    projector = IterativeSubspaceProjector(energy_threshold=0.95, max_columns=4)
    projector.update("layer", torch.randn(3, 20))
    projector.update("layer", torch.randn(3, 20))

    summary = projector.summary()
    assert summary.layers == 1
    assert summary.dimensions["layer"] == 3
    assert summary.ranks["layer"] <= 3
    assert not hasattr(projector, "activations")


def test_one_for_more_config_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="divisible"):
        OneForMoreConfig(input_size=(30, 32), latent_downsample=8)


def test_anomaly_masked_conditioner_preserves_latent_shape():
    config = _tiny_config(
        input_size=(64, 64),
        conditioner_depth=1,
        conditioner_heads=1,
        neighbor_size=(3, 3),
    )
    conditioner = AnomalyMaskedConditioner(config).eval()

    output = conditioner(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 8, 8, 8)
    assert torch.isfinite(output).all()


def test_one_for_more_predict_and_pixel_map_shapes(monkeypatch):
    data = np.random.default_rng(0).random((3, 32, 32, 3), dtype=np.float32)
    model = OneForMore(_tiny_config())
    original = model._batch_reconstruct
    reconstruction_calls = 0

    def counted(images, batch_index):
        nonlocal reconstruction_calls
        reconstruction_calls += 1
        return original(images, batch_index)

    monkeypatch.setattr(model, "_batch_reconstruct", counted)

    labels, scores = model.predict(data)
    score_maps = model.score_maps(data)
    assert reconstruction_calls == 2
    reconstructions = model.reconstruct(data)

    assert labels.shape == (3,)
    assert scores.shape == (3,)
    assert score_maps.shape == (3, 32, 32)
    assert reconstructions.shape == (3, 3, 32, 32)
    assert np.isfinite(scores).all()
    assert np.isfinite(score_maps).all()


def test_one_for_more_two_task_smoke_run_builds_projection_bases():
    rng = np.random.default_rng(1)
    model = OneForMore(_tiny_config(epochs=1, incremental_epochs=1))

    model.fit(rng.random((2, 32, 32, 3), dtype=np.float32))
    first_summary = model.additional_info()
    model.fit(rng.random((2, 32, 32, 3), dtype=np.float32))
    second_summary = model.additional_info()

    assert first_summary["tasks_seen"] == 1
    assert first_summary["projection_layers"] > 0
    assert second_summary["tasks_seen"] == 2
    assert second_summary["projection_total_rank"] >= first_summary["projection_total_rank"]
    assert second_summary["last_loss"] is not None
