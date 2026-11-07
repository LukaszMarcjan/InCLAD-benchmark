from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from pyclad.models.ucad.config import UCADConfig
from pyclad.models.ucad.structure import PrecomputedStructureMaskGenerator
from pyclad.models.ucad.ucad import UCAD, structure_contrastive_loss
from pyclad.strategies.vision.ucad import UCADStrategy


class _TinyPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.proj(images)
        return patches.flatten(2).transpose(1, 2)


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.proj(self.norm(tokens))


class _TinyViT(nn.Module):
    embed_dim = 8
    num_prefix_tokens = 1

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.patch_embed = _TinyPatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])

    def _pos_embed(self, patches: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.expand(len(patches), -1, -1)
        return torch.cat([cls, patches], dim=1)


def _config(**updates) -> UCADConfig:
    values = {
        "backbone_name": "tiny-test-vit",
        "pretrained_backbone": False,
        "input_size": (8, 8),
        "batch_size": 2,
        "feature_layer": 2,
        "prompt_depth": 2,
        "prompt_length": 1,
        "prompt_init": "zero",
        "target_embed_dimension": 8,
        "key_size": 4,
        "knowledge_size": 4,
        "coreset_projection_dimension": 4,
        "coreset_starting_points": 2,
        "epochs": 0,
        "structure_mode": "none",
        "smoothing_sigma": 0.0,
        "device": "cpu",
    }
    values.update(updates)
    return UCADConfig(**values)


@pytest.fixture
def tiny_backbone(monkeypatch):
    monkeypatch.setattr("pyclad.models.ucad.ucad._build_vit_backbone", lambda config: _TinyViT())


def test_structure_contrastive_loss_is_finite_and_differentiable():
    features = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])

    loss = structure_contrastive_loss(features, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_ucad_learns_multiple_tasks_and_exposes_inclad_contracts(tiny_backbone):
    rng = np.random.default_rng(11)
    task_a = rng.integers(0, 80, size=(3, 8, 8, 3), dtype=np.uint8)
    task_b = rng.integers(175, 255, size=(3, 8, 8, 3), dtype=np.uint8)
    model = UCAD(_config())
    strategy = UCADStrategy(model)

    strategy.learn(task_a, concept_id="dark")
    strategy.learn(task_b, concept_id="bright")
    labels, scores = strategy.predict(np.concatenate([task_a[:1], task_b[:1]]))
    maps = model.score_maps(task_a[:2])
    selected = model.selected_task_indices(np.concatenate([task_a[:1], task_b[:1]]))

    assert labels.shape == (2,)
    assert scores.shape == (2,)
    assert maps.shape == (2, 8, 8)
    assert selected.shape == (2,)
    assert len(model._task_memories) == 2
    assert [memory.concept_id for memory in model._task_memories] == ["dark", "bright"]
    assert all(memory.key.shape == (4, 8) for memory in model._task_memories)
    assert all(memory.knowledge.shape == (4, 8) for memory in model._task_memories)
    assert all(memory.prompt.shape == (2, 2, 1, 1, 8) for memory in model._task_memories)
    assert model.additional_info()["variant"] == "CPM-only"


def test_ucad_reuses_inference_for_image_and_pixel_metrics(tiny_backbone, monkeypatch):
    data = np.random.default_rng(12).integers(0, 255, size=(3, 8, 8, 3), dtype=np.uint8)
    model = UCAD(_config())
    model.fit_task(data, concept_id="task")
    original = model._predict_scores_and_masks
    calls = 0

    def counted(values):
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(model, "_predict_scores_and_masks", counted)
    evaluation_data = data.copy()
    model.predict(evaluation_data)
    model.score_maps(evaluation_data)

    assert calls == 1


def test_ucad_optimizes_prompt_when_structure_labels_are_supplied(tiny_backbone):
    rng = np.random.default_rng(21)
    data = rng.integers(0, 255, size=(2, 8, 8, 3), dtype=np.uint8)
    masks = np.zeros((2, 8, 8), dtype=np.int64)
    masks[:, :, 4:] = 1
    model = UCAD(_config(epochs=1, learning_rate=1e-3))

    model.fit_task(data, concept_id="structured", structure_masks=masks)

    memory = model._task_memories[0]
    assert memory.final_scl_loss is not None
    assert np.isfinite(memory.final_scl_loss)
    assert torch.count_nonzero(memory.prompt).item() > 0


def test_precomputed_structure_masks_support_official_mvtec_layout(tmp_path):
    mask_dir = tmp_path / "bottle" / "train" / "good"
    mask_dir.mkdir(parents=True)
    Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(mask_dir / "000.png")
    Image.fromarray(np.array([[4, 5], [6, 7]], dtype=np.uint8)).save(mask_dir / "001.png")
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    masks = PrecomputedStructureMaskGenerator(tmp_path).generate(images, "bottle", task_index=0)

    assert masks.shape == (2, 4, 4)
    assert set(np.unique(masks[0])) == {0, 1, 2, 3}
    assert set(np.unique(masks[1])) == {4, 5, 6, 7}


def test_precomputed_structure_masks_support_inclad_md_layout(tmp_path):
    mask_dir = tmp_path / "mvtec" / "bottle" / "train" / "good"
    mask_dir.mkdir(parents=True)
    Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(mask_dir / "000.png")
    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)

    masks = PrecomputedStructureMaskGenerator(tmp_path).generate(images, "mvtec__bottle", task_index=0)

    assert masks.shape == (1, 4, 4)
    assert set(np.unique(masks[0])) == {0, 1, 2, 3}


def test_ucad_config_rejects_prompt_layers_after_feature_layer():
    with pytest.raises(ValueError, match="prompt_depth"):
        _config(feature_layer=1, prompt_depth=2)


def test_stock_timm_vit_uses_prefix_attention_and_backpropagates():
    pytest.importorskip("timm")
    model = UCAD(
        _config(
            backbone_name="vit_tiny_patch16_224",
            input_size=(32, 32),
            feature_layer=1,
            prompt_depth=1,
            target_embed_dimension=32,
        )
    )
    prompt = nn.Parameter(model._initial_prompt())
    images = torch.rand(2, 3, 32, 32)

    tokens = model._forward_patch_tokens(images, prompt)
    tokens.square().mean().backward()

    assert model._supports_prefix_prompt(model.module.blocks[0])
    assert prompt.shape == (1, 2, 1, 3, 64)
    assert prompt.grad is not None
    assert torch.count_nonzero(prompt.grad).item() > 0
