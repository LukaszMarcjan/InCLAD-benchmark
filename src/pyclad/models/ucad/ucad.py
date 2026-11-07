from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.models.patchcore.patchcore import ApproximateGreedyCoresetSampler, RescaleSegmentor
from pyclad.models.ucad.config import UCADConfig
from pyclad.models.ucad.structure import build_structure_mask_generator
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class UCADTaskMemory:
    """One entry in UCAD's key-prompt-knowledge memory."""

    key: np.ndarray
    prompt: torch.Tensor
    knowledge: np.ndarray
    concept_id: Optional[str] = None
    final_scl_loss: Optional[float] = None


def structure_contrastive_loss(
    patch_features: torch.Tensor,
    region_labels: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """UCAD SCL objective over all pairs of patch features.

    Features assigned to the same SAM region are pulled together, while
    features from different regions are pushed apart.  This follows the loss
    implemented by the authors and keeps the self-pairs used there.
    """

    if patch_features.ndim != 3:
        raise ValueError(f"patch_features must have shape (B, P, D), got {patch_features.shape}")
    if region_labels.shape != patch_features.shape[:2]:
        raise ValueError(f"region_labels must have shape {patch_features.shape[:2]}, got {region_labels.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    normalized = F.normalize(patch_features, dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2)) / temperature
    same_region = (region_labels.unsqueeze(1) == region_labels.unsqueeze(2)).to(similarity.dtype)
    return (-similarity * same_region + torch.exp(similarity) * (1.0 - same_region)).mean()


def _build_vit_backbone(config: UCADConfig) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("UCAD requires timm. Install pyclad with the 'ucad' extra.") from exc

    return timm.create_model(
        config.backbone_name,
        pretrained=config.pretrained_backbone,
        num_classes=0,
        img_size=config.input_size,
    )


class UCAD(Model):
    """Unsupervised Continual Anomaly Detection with learned prompts.

    This is a clean-room implementation of the AAAI 2024 method.  It exposes
    pyCLAD's standard image-level ``predict`` contract and the ``score_maps``
    contract used by InCLAD's pixel metrics.
    """

    def __init__(self, config: Optional[UCADConfig] = None):
        self.config = config or UCADConfig()
        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=3,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )
        self.module = _build_vit_backbone(self.config).to(self._device).eval()
        self._embed_dim = int(self.module.embed_dim)
        first_attention = getattr(self.module.blocks[0], "attn", None)
        self._num_heads = int(getattr(first_attention, "num_heads", 1))
        if self._embed_dim % self._num_heads != 0:
            raise ValueError(
                f"UCAD embedding dimension {self._embed_dim} is not divisible by "
                f"the attention head count {self._num_heads}"
            )
        self._head_dim = self._embed_dim // self._num_heads
        self._validate_backbone()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

        self._segmentor = RescaleSegmentor(
            device=self._device,
            target_size=self.config.input_size,
            smoothing=self.config.smoothing_sigma,
        )
        self._coreset_sampler_seed = int(self.config.random_seed)
        self._structure_masks = build_structure_mask_generator(self.config, str(self._device))
        self._task_memories: list[UCADTaskMemory] = []
        self._feature_map_size: Optional[tuple[int, int]] = None
        self._threshold = self.config.threshold
        self._last_prediction_data: Optional[np.ndarray] = None
        self._last_prediction_scores: Optional[np.ndarray] = None
        self._last_prediction_masks: Optional[np.ndarray] = None

    def _validate_backbone(self) -> None:
        required = ["patch_embed", "_pos_embed", "patch_drop", "norm_pre", "blocks", "embed_dim"]
        missing = [name for name in required if not hasattr(self.module, name)]
        if missing:
            raise TypeError(
                f"UCAD backbone '{self.config.backbone_name}' is not a compatible timm ViT; missing {missing}"
            )
        if self.config.feature_layer > len(self.module.blocks):
            raise ValueError(
                f"feature_layer={self.config.feature_layer} exceeds backbone depth={len(self.module.blocks)}"
            )

    def begin_continual_run(self) -> None:
        self._task_memories = []
        self._feature_map_size = None
        self._threshold = self.config.threshold
        self._invalidate_prediction_cache()

    def _invalidate_prediction_cache(self) -> None:
        self._last_prediction_data = None
        self._last_prediction_scores = None
        self._last_prediction_masks = None

    def _initial_prompt(self) -> torch.Tensor:
        shape = (
            self.config.prompt_depth,
            2,
            self.config.prompt_length,
            self._num_heads,
            self._head_dim,
        )
        if self.config.warm_start_prompts and self._task_memories:
            return self._task_memories[-1].prompt.detach().clone().to(self._device)
        if self.config.prompt_init == "zero":
            return torch.zeros(shape, dtype=torch.float32, device=self._device)
        generator = torch.Generator()
        generator.manual_seed(self.config.random_seed + len(self._task_memories))
        return (
            torch.empty(shape, dtype=torch.float32)
            .uniform_(
                -1.0,
                1.0,
                generator=generator,
            )
            .to(self._device)
        )

    @staticmethod
    def _supports_prefix_prompt(block: nn.Module) -> bool:
        attention = getattr(block, "attn", None)
        return all(
            hasattr(component, attribute)
            for component, attribute in (
                (block, "norm1"),
                (block, "norm2"),
                (block, "mlp"),
                (block, "ls1"),
                (block, "ls2"),
                (block, "drop_path1"),
                (block, "drop_path2"),
                (attention, "qkv"),
                (attention, "num_heads"),
                (attention, "scale"),
                (attention, "attn_drop"),
                (attention, "proj"),
                (attention, "proj_drop"),
            )
        )

    def _forward_block_with_prefix_prompt(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        layer_prompt: torch.Tensor,
    ) -> torch.Tensor:
        """Run a stock timm ViT block with learned key/value prefixes."""

        if not self._supports_prefix_prompt(block):
            # A small additive fallback keeps custom ViT-like backbones usable;
            # the default timm backbone always takes the prefix-tuning path.
            prompt_vector = layer_prompt.reshape(-1, self._embed_dim).mean(dim=0).view(1, 1, -1)
            return block(tokens + prompt_vector)

        attention = block.attn
        normalized = block.norm1(tokens)
        batch_size, token_count, embed_dim = normalized.shape
        qkv = (
            attention.qkv(normalized)
            .reshape(batch_size, token_count, 3, attention.num_heads, embed_dim // attention.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        query = getattr(attention, "q_norm", nn.Identity())(query)
        key = getattr(attention, "k_norm", nn.Identity())(key)

        key_prefix = layer_prompt[0].permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
        value_prefix = layer_prompt[1].permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1)
        key = torch.cat([key_prefix.to(key.dtype), key], dim=2)
        value = torch.cat([value_prefix.to(value.dtype), value], dim=2)

        weights = torch.matmul(query * attention.scale, key.transpose(-2, -1))
        weights = attention.attn_drop(weights.softmax(dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, token_count, embed_dim)
        attended = attention.proj_drop(attention.proj(attended))

        tokens = tokens + block.drop_path1(block.ls1(attended))
        tokens = tokens + block.drop_path2(block.ls2(block.mlp(block.norm2(tokens))))
        return tokens

    def _forward_patch_tokens(
        self,
        images: torch.Tensor,
        prompt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.module.patch_embed(images)
        x = self.module._pos_embed(x)
        x = self.module.patch_drop(x)
        x = self.module.norm_pre(x)
        prefix_tokens = int(getattr(self.module, "num_prefix_tokens", 1))

        for block_index, block in enumerate(self.module.blocks[: self.config.feature_layer]):
            if prompt is not None and block_index < self.config.prompt_depth:
                x = self._forward_block_with_prefix_prompt(block, x, prompt[block_index])
            else:
                x = block(x)
        return x[:, prefix_tokens:]

    def _map_dimension(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] == self.config.target_embed_dimension:
            return features
        flattened = features.reshape(-1, 1, features.shape[-1])
        mapped = F.adaptive_avg_pool1d(flattened, self.config.target_embed_dimension)
        return mapped.reshape(*features.shape[:-1], self.config.target_embed_dimension)

    @staticmethod
    def _infer_grid_size(patch_count: int) -> tuple[int, int]:
        side = int(round(math.sqrt(patch_count)))
        if side * side != patch_count:
            raise ValueError(f"UCAD currently requires a square ViT patch grid, got {patch_count} patch tokens")
        return side, side

    def _extract_patch_embeddings(
        self,
        data: np.ndarray,
        prompt: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        if len(data) == 0:
            return np.empty((0, self.config.target_embed_dimension), dtype=np.float32), (0, 0)
        transformed = self._preprocessor.transform(data)
        loader = DataLoader(TensorDataset(transformed), batch_size=self.config.batch_size, shuffle=False)
        batches: list[np.ndarray] = []
        grid_size: Optional[tuple[int, int]] = None
        self.module.eval()
        with torch.no_grad():
            for (images,) in loader:
                tokens = self._forward_patch_tokens(
                    images.to(self._device, dtype=torch.float32),
                    None if prompt is None else prompt.to(self._device),
                )
                grid_size = self._infer_grid_size(tokens.shape[1])
                mapped = self._map_dimension(tokens)
                batches.append(mapped.reshape(-1, mapped.shape[-1]).cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(batches, axis=0), grid_size

    def _select_fixed_coreset(self, features: np.ndarray, target_size: int) -> np.ndarray:
        if len(features) <= target_size:
            return features.astype(np.float32, copy=False)
        sampler = ApproximateGreedyCoresetSampler(
            percentage=min(1.0, (target_size + 0.25) / len(features)),
            device=self._device,
            number_of_starting_points=self.config.coreset_starting_points,
            dimension_to_project_features_to=self.config.coreset_projection_dimension,
            random_seed=self._coreset_sampler_seed + len(self._task_memories),
        )
        selected = sampler.run(features)
        return selected[:target_size].astype(np.float32, copy=False)

    def _train_prompt(
        self,
        data: np.ndarray,
        structure_masks: Optional[np.ndarray],
        initial_prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[float]]:
        if structure_masks is None or self.config.epochs == 0:
            return initial_prompt.detach(), None
        if len(structure_masks) != len(data):
            raise ValueError(f"UCAD received {len(structure_masks)} structure masks for {len(data)} images")

        images = self._preprocessor.transform(data)
        masks = torch.as_tensor(np.asarray(structure_masks), dtype=torch.float32).unsqueeze(1)
        prompt = nn.Parameter(initial_prompt.detach().clone())
        optimizer = torch.optim.Adam(
            [prompt],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = torch.Generator()
        generator.manual_seed(self.config.random_seed + len(self._task_memories))
        loader = DataLoader(
            TensorDataset(images, masks),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.module.eval()
        final_loss: Optional[float] = None
        for epoch in range(self.config.epochs):
            epoch_losses: list[float] = []
            for image_batch, mask_batch in loader:
                image_batch = image_batch.to(self._device, dtype=torch.float32)
                mask_batch = mask_batch.to(self._device)
                tokens = self._forward_patch_tokens(image_batch, prompt)
                grid_size = self._infer_grid_size(tokens.shape[1])
                labels = F.interpolate(mask_batch, size=grid_size, mode="nearest").to(torch.long)
                labels = labels[:, 0].reshape(len(image_batch), -1)
                loss = structure_contrastive_loss(
                    tokens,
                    labels,
                    temperature=self.config.contrastive_temperature,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([prompt], self.config.gradient_clip_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            if epoch_losses:
                final_loss = float(np.mean(epoch_losses))
            logger.info(
                "UCAD task %d SCL epoch %d/%d loss=%s",
                len(self._task_memories),
                epoch + 1,
                self.config.epochs,
                "nan" if final_loss is None else f"{final_loss:.6f}",
            )
        return prompt.detach(), final_loss

    def fit_task(
        self,
        data: np.ndarray,
        concept_id: Optional[str] = None,
        structure_masks: Optional[np.ndarray] = None,
    ) -> None:
        if len(data) == 0:
            raise ValueError("UCAD cannot learn an empty task")
        self._invalidate_prediction_cache()

        task_index = len(self._task_memories)
        key_features, grid_size = self._extract_patch_embeddings(data, prompt=None)
        if self._feature_map_size is None:
            self._feature_map_size = grid_size
        elif self._feature_map_size != grid_size:
            raise ValueError(
                f"All UCAD tasks must share the same feature grid; expected {self._feature_map_size}, got {grid_size}"
            )
        key = self._select_fixed_coreset(key_features, self.config.key_size)

        if structure_masks is None:
            structure_masks = self._structure_masks.generate(data, concept_id, task_index)
        if structure_masks is None and self.config.epochs > 0:
            logger.info(
                "UCAD task %d is running the CPM-only ablation because structure_mode='none'; "
                "no SCL prompt optimization was performed.",
                task_index,
            )
        prompt, final_scl_loss = self._train_prompt(data, structure_masks, self._initial_prompt())
        knowledge_features, _ = self._extract_patch_embeddings(data, prompt=prompt)
        knowledge = self._select_fixed_coreset(knowledge_features, self.config.knowledge_size)

        self._task_memories.append(
            UCADTaskMemory(
                key=key,
                prompt=prompt.cpu(),
                knowledge=knowledge,
                concept_id=concept_id,
                final_scl_loss=final_scl_loss,
            )
        )
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
        else:
            reference_scores = self.score_data(data)
            self._threshold = (
                0.0
                if len(reference_scores) == 0
                else float(np.quantile(reference_scores, self.config.threshold_quantile))
            )

    def fit(self, data: np.ndarray):
        self.fit_task(data)

    def _ensure_fitted(self) -> None:
        if not self._task_memories or self._feature_map_size is None:
            raise RuntimeError("UCAD must learn at least one task before prediction")

    def _task_selection_indices(self, data: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        if len(self._task_memories) == 1:
            return np.zeros(len(data), dtype=np.int64)
        query_features, grid_size = self._extract_patch_embeddings(data, prompt=None)
        patch_count = grid_size[0] * grid_size[1]
        queries = query_features.reshape(len(data), patch_count, -1)

        task_distances = np.empty((len(self._task_memories), len(data)), dtype=np.float32)
        for task_index, memory in enumerate(self._task_memories):
            index = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(memory.key)
            for image_index, query in enumerate(queries):
                distances, _ = index.kneighbors(query)
                task_distances[task_index, image_index] = float(distances[:, 0].mean())
        return np.argmin(task_distances, axis=0).astype(np.int64)

    def selected_task_indices(self, data: np.ndarray) -> np.ndarray:
        """Return task identities selected from UCAD keys (diagnostic API)."""
        return self._task_selection_indices(data)

    def _predict_scores_and_masks(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_fitted()
        if len(data) == 0:
            return (
                np.asarray([], dtype=np.float32),
                np.empty((0, *self.config.input_size), dtype=np.float32),
            )

        selected_tasks = self._task_selection_indices(data)
        image_scores = np.empty(len(data), dtype=np.float32)
        patch_maps = np.empty((len(data), *self._feature_map_size), dtype=np.float32)

        for task_index in np.unique(selected_tasks):
            sample_indices = np.flatnonzero(selected_tasks == task_index)
            memory = self._task_memories[int(task_index)]
            embeddings, grid_size = self._extract_patch_embeddings(
                np.asarray(data)[sample_indices],
                prompt=memory.prompt,
            )
            patch_count = grid_size[0] * grid_size[1]
            index = NearestNeighbors(
                n_neighbors=min(self.config.n_neighbors, len(memory.knowledge)),
                n_jobs=1,
            ).fit(memory.knowledge)
            distances, _ = index.kneighbors(embeddings)
            scores = distances.mean(axis=1).astype(np.float32, copy=False).reshape(len(sample_indices), patch_count)
            image_scores[sample_indices] = scores.max(axis=1)
            patch_maps[sample_indices] = scores.reshape(len(sample_indices), *grid_size)

        masks = self._segmentor.convert_to_segmentation(patch_maps)
        return image_scores, masks

    def _prediction_for(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reuse one inference pass for adjacent image- and pixel-level metrics."""
        if (
            data is self._last_prediction_data
            and self._last_prediction_scores is not None
            and self._last_prediction_masks is not None
        ):
            return self._last_prediction_scores, self._last_prediction_masks
        scores, masks = self._predict_scores_and_masks(data)
        self._last_prediction_data = data
        self._last_prediction_scores = scores
        self._last_prediction_masks = masks
        return scores, masks

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)
        return self._prediction_for(data)[0]

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            target = self.config.input_size if resize_to_input else (0, 0)
            return np.empty((0, *target), dtype=np.float32)
        _, masks = self._prediction_for(data)
        if resize_to_input:
            return masks
        resized = F.interpolate(
            torch.from_numpy(masks[:, None]),
            size=self._feature_map_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized[:, 0].numpy().astype(np.float32, copy=False)

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score_data(data)
        threshold = 0.0 if self._threshold is None else float(self._threshold)
        return (scores > threshold).astype(int), scores

    def name(self) -> str:
        return "UCAD"

    def additional_info(self) -> Dict:
        key_sizes = [int(len(memory.key)) for memory in self._task_memories]
        knowledge_sizes = [int(len(memory.knowledge)) for memory in self._task_memories]
        prompt_shapes = [list(memory.prompt.shape) for memory in self._task_memories]
        memory_bytes = sum(
            memory.key.nbytes + memory.knowledge.nbytes + memory.prompt.numel() * memory.prompt.element_size()
            for memory in self._task_memories
        )
        return {
            "method": "UCAD",
            "variant": "CPM+SCL" if self.config.structure_mode != "none" else "CPM-only",
            "paper": "Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt",
            "backbone": self.config.backbone_name,
            "pretrained_backbone": self.config.pretrained_backbone,
            "feature_layer": self.config.feature_layer,
            "input_size": self.config.input_size,
            "target_embed_dimension": self.config.target_embed_dimension,
            "structure_mode": self.config.structure_mode,
            "tasks_seen": len(self._task_memories),
            "task_concepts": [memory.concept_id for memory in self._task_memories],
            "key_sizes": key_sizes,
            "knowledge_sizes": knowledge_sizes,
            "prompt_shapes": prompt_shapes,
            "final_scl_losses": [memory.final_scl_loss for memory in self._task_memories],
            "key_prompt_knowledge_bytes": int(memory_bytes),
            "threshold": self._threshold,
            "device": str(self._device),
        }
