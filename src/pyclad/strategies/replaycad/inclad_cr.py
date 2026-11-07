"""InCLAD camera-ready profile of ReplayCAD (``inclad-cr``).

The paper presets in :mod:`pyclad.strategies.replaycad.config` stay the
default. This module adds one uniform protocol for the five single-dataset
InCLAD benchmarks; ``examples/clvad/run_replaycad_cr.py`` activates it with
:func:`install`. Differences from the paper presets:

* LDM-256 (``CompVis/ldm-text2im-large-256``) for every dataset, VisA
  included. InCLAD feeds 256 px images, and the authors' VisA scripts already
  use LDM-256 for macaroni1, macaroni2 and candle.
* Trainable conditions stay in float32 while the frozen diffusion model runs
  under bfloat16 autocast. Training the conditions directly in bfloat16 rounds
  most AdamW updates of the mask MLP away: 81-83 % of its weights never left
  their initial values in the earlier bfloat16 runs.
* The authors' learning-rate rule ``lr = n_gpu * batch * 5e-5``
  (``add_mask.yaml:2``) for one GPU and a batch of 16, embeddings at
  ``lr * 100`` (``ddpm2.py:1574``), AdamW weight decay 0.01, one
  textual-inversion template per training sample and the initializer word
  "screw" that the authors' training scripts pass for every MVTec class
  (``tran_mvtec.sh``).
* SAM masks from four corner prompts (paper Fig. 11a) with a full-frame
  fallback for degenerate masks; full frame for texture concepts and DAGM.
* No mask augmentation at generation, the authors' setting for 9 of 14 MVTec
  classes (``txt2img_with_mask.py:198-230``).
* Lower-case token names: the slow BERT tokenizer of the LDM pipeline maps
  mixed-case added tokens such as ``<rc-Class8-00>`` to ``[UNK]``.
* Replay images are cached as soon as a concept is compressed, so one artifact
  bank per dataset serves every ordering of that dataset.
* Replay of a concept whose training images are grey stored as RGB is
  converted to grey, because the LDM prior generates colour.
* InCLAD-MD (``inclad-md``, 27 concepts named ``<dataset>__<category>``) uses
  the same settings; each concept keeps the mask rule of its source dataset.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import Field, model_validator

from pyclad.strategies.replaycad.artifacts import ReplayCADArtifact, concept_slug
from pyclad.strategies.replaycad.backend import DiffusersReplayCADBackend, ReplayCADBackend, _torch_load
from pyclad.strategies.replaycad.config import ReplayCADConfig
from pyclad.strategies.replaycad.masks import ReplayCADMaskGenerator, SAMMaskGenerator
from pyclad.strategies.replaycad.memory import ReplayCADMemory

logger = logging.getLogger(__name__)

#: Bump the version whenever a setting below changes. The base guard compares
#: ``preset``; :data:`PROFILE_GUARD_KEYS` covers the fields the base guard does not know.
PROFILE_PRESET = "inclad-cr-v1-ldm-256"
GREY_METADATA_KEY = "training_images_grey"
INCLAD_CR_BENCHMARKS = ("btech", "dagm", "mpdd", "mvtec", "visa", "inclad-md")
INCLAD_MD = "inclad-md"
INCLAD_MD_ALIASES = ("inclad-md", "incladmd", "multidataset", "multidataset-seq-v1", "md")
DEFAULT_COMPRESSION_STEPS = 6_000
DEFAULT_REPLAY_SAMPLES = 200
# lr = n_gpu * batch_size * base_lr = 1 * 16 * 5e-5; token embeddings use lr * 100.
SPATIAL_LEARNING_RATE = 8e-4
SEMANTIC_LEARNING_RATE = 8e-2
TEXTURE_CONCEPTS: Dict[str, Tuple[str, ...]] = {
    "btech": ("2",),
    "mvtec": ("carpet", "grid", "leather", "tile", "wood"),
}
#: InCLAD-MD concepts with a full-frame mask: the texture concepts above under their MD names
#: (the MD manifest zero-pads the BTAD categories) and every DAGM class.
MD_FULL_FRAME_CONCEPTS: Tuple[str, ...] = (
    *(f"mvtec__{concept}" for concept in TEXTURE_CONCEPTS["mvtec"]),
    *(f"btech__{int(concept):02d}" for concept in TEXTURE_CONCEPTS["btech"]),
    *(f"dagm__Class{index}" for index in range(1, 11)),
)
PROFILE_GUARD_KEYS = (
    "weight_decay",
    "prompt_templates",
    "default_mask_mode",
    "mask_modes",
    "mask_min_foreground",
    "mask_max_foreground",
    "autocast_dtype",
    "augment_masks",
    "inference_steps",
    "guidance_scale",
    "generation_batch_size",
    "replay_samples_per_concept",
)

# Prompt templates from rinongal/textual_inversion (MIT, Rinon Gal et al.). ReplayCAD samples one
# uniformly per training step (personalized.py:218); generation instead uses a fixed prompt.
IMAGENET_TEMPLATES_SMALL: Tuple[str, ...] = (
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
    "an illustration of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "an illustration of a clean {}",
    "an illustration of a dirty {}",
    "a dark photo of the {}",
    "an illustration of my {}",
    "an illustration of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "an illustration of the {}",
    "a good photo of the {}",
    "an illustration of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "an illustration of the clean {}",
    "a rendition of a {}",
    "an illustration of a nice {}",
    "a good photo of a {}",
    "an illustration of the nice {}",
    "an illustration of the small {}",
    "an illustration of the weird {}",
    "an illustration of the large {}",
    "an illustration of a cool {}",
    "an illustration of a small {}",
    "a depiction of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a depiction of a clean {}",
    "a depiction of a dirty {}",
    "a dark photo of the {}",
    "a depiction of my {}",
    "a depiction of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a depiction of the {}",
    "a good photo of the {}",
    "a depiction of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a depiction of the clean {}",
    "a rendition of a {}",
    "a depiction of a nice {}",
    "a good photo of a {}",
    "a depiction of the nice {}",
    "a depiction of the small {}",
    "a depiction of the weird {}",
    "a depiction of the large {}",
    "a depiction of a cool {}",
    "a depiction of a small {}",
)


@dataclass(frozen=True)
class InCLADCRSettings:
    """Process-wide settings that the base runner has no flags for."""

    #: One artifact bank per dataset, ``<bank_dir>/<benchmark>``, shared by all orderings.
    bank_dir: Optional[Path] = None


_SETTINGS = InCLADCRSettings()


class InCLADCRReplayCADConfig(ReplayCADConfig):
    """ReplayCAD configuration of the ``inclad-cr`` profile."""

    benchmark: Literal["btech", "dagm", "mpdd", "mvtec", "visa", "inclad-md"]
    preset: Literal["inclad-cr-v1-ldm-256"]

    weight_decay: float = Field(default=1e-2, ge=0.0)
    prompt_templates: Tuple[str, ...] = IMAGENET_TEMPLATES_SMALL
    default_mask_mode: Literal["corner-background", "area", "full-frame"] = "corner-background"
    mask_min_foreground: float = Field(default=0.02, ge=0.0, le=1.0)
    mask_max_foreground: float = Field(default=0.98, ge=0.0, le=1.0)
    autocast_dtype: Literal["bfloat16", "none"] = "bfloat16"

    @model_validator(mode="after")
    def validate_profile(self):
        if self.torch_dtype != "float32":
            raise ValueError(
                "The inclad-cr profile keeps the trainable conditions in float32 and runs the frozen "
                f"diffusion model under autocast; got torch_dtype={self.torch_dtype!r}"
            )
        if not self.cache_generated:
            raise ValueError("The inclad-cr profile shares replay images across orderings; enable cache_generated")
        if self.mask_min_foreground >= self.mask_max_foreground:
            raise ValueError("mask_min_foreground must be smaller than mask_max_foreground")
        if not self.prompt_templates or any("{}" not in template for template in self.prompt_templates):
            raise ValueError("Every prompt template needs one '{}' placeholder")
        return self

    @classmethod
    def inclad_cr_preset(
        cls,
        benchmark: str,
        artifact_dir: Path,
        *,
        sam_checkpoint: Optional[Path] = None,
        precomputed_mask_root: Optional[Path] = None,
        mask_backend: str = "sam",
        device: str = "cuda",
        seed: int = 42,
        compression_steps: Optional[int] = None,
        replay_samples_per_concept: int = DEFAULT_REPLAY_SAMPLES,
        masks_per_concept: int = 5,
        local_files_only: bool = False,
    ) -> "InCLADCRReplayCADConfig":
        benchmark_key = normalize_benchmark(benchmark)
        if benchmark_key not in INCLAD_CR_BENCHMARKS:
            raise ValueError(f"The inclad-cr profile supports {', '.join(INCLAD_CR_BENCHMARKS)}; got {benchmark!r}")
        if _SETTINGS.bank_dir is not None:
            artifact_dir = _SETTINGS.bank_dir / benchmark_key
        return cls(
            benchmark=benchmark_key,
            preset=PROFILE_PRESET,
            artifact_dir=artifact_dir,
            diffusion_family="ldm",
            model_id="CompVis/ldm-text2im-large-256",
            resolution=256,
            condition_dim=1280,
            mask_group_width=128,
            mask_projection_width=200,
            compression_steps=DEFAULT_COMPRESSION_STEPS if compression_steps is None else compression_steps,
            compression_batch_size=16,
            semantic_learning_rate=SEMANTIC_LEARNING_RATE,
            spatial_learning_rate=SPATIAL_LEARNING_RATE,
            timestep_sampling="uniform",
            inference_steps=50,
            guidance_scale=10.0,
            generation_batch_size=8,
            prompt_prefix="a photo of",
            initializer_word="screw",
            replay_samples_per_concept=replay_samples_per_concept,
            masks_per_concept=masks_per_concept,
            mask_backend=mask_backend,
            sam_checkpoint=sam_checkpoint,
            precomputed_mask_root=precomputed_mask_root,
            mask_modes={concept: "full-frame" for concept in _full_frame_concepts(benchmark_key)},
            default_mask_mode="full-frame" if benchmark_key == "dagm" else "corner-background",
            augment_masks=False,
            device=device,
            seed=seed,
            local_files_only=local_files_only,
            cache_generated=True,
        )

    @classmethod
    def paper_preset(cls, benchmark: str, artifact_dir: Path, **kwargs) -> "InCLADCRReplayCADConfig":
        """Keep the base preset name so the unchanged base runner builds this profile."""
        return cls.inclad_cr_preset(benchmark, artifact_dir, **kwargs)


def normalize_benchmark(name: str) -> str:
    folded = str(name).strip().lower().replace("_", "-")
    return INCLAD_MD if folded in INCLAD_MD_ALIASES else folded


def _full_frame_concepts(benchmark: str) -> Tuple[str, ...]:
    if benchmark == INCLAD_MD:
        return MD_FULL_FRAME_CONCEPTS
    return TEXTURE_CONCEPTS.get(benchmark, ())


def _relative_drift(initial, final, decay_factor: float) -> float:
    """Distance from where the weights would be without learning, relative to their size.

    Without a usable gradient AdamW either only decays a parameter (zero gradient, ending at
    ``initial * decay_factor``) or skips it entirely (no gradient, ending at ``initial``); a value
    near zero means the condition did not learn.
    """
    initial = initial.double()
    final = final.double()
    scale = float(initial.norm())
    if not scale:
        return 0.0
    return min(float((final - initial * decay_factor).norm()), float((final - initial).norm())) / scale


class InCLADCRDiffusersBackend(DiffusersReplayCADBackend):
    """Diffusers backend with float32 conditions, bf16 autocast and template prompts."""

    def _autocast(self):
        if self.config.autocast_dtype == "none" or not str(self.config.device).startswith("cuda"):
            return contextlib.nullcontext()
        return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)

    def _without_autocast(self):
        if not str(self.config.device).startswith("cuda"):
            return contextlib.nullcontext()
        return self._torch.autocast(device_type="cuda", enabled=False)

    def _semantic_token_names(self, concept_id: str) -> List[str]:
        slug = concept_slug(concept_id).replace(".", "-").lower()
        return [f"<rc-{slug}-{index:02d}>" for index in range(self.config.semantic_tokens)]

    def _install_tokens(self, token_names: Sequence[str], initialize: bool) -> List[int]:
        token_ids = super()._install_tokens(token_names, initialize)
        unknown_id = getattr(self._tokenizer, "unk_token_id", None)
        if unknown_id is not None and int(unknown_id) in token_ids:
            raise ValueError(f"ReplayCAD token names {list(token_names)[:2]}... map to the tokenizer's unknown token")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError(f"ReplayCAD token names {list(token_names)[:2]}... do not map to distinct token ids")
        return token_ids

    def _encode_mask_collection(self, masks: np.ndarray):
        # compress() encodes masks outside autocast; generation must give the MLP the same inputs.
        with self._without_autocast():
            return super()._encode_mask_collection(masks)

    def _template_prompt(self, token_names: Sequence[str], rng: np.random.Generator) -> str:
        template = self.config.prompt_templates[int(rng.integers(len(self.config.prompt_templates)))]
        return template.format(" ".join(token_names))

    def compress(
        self,
        concept_id: str,
        images: np.ndarray,
        masks: np.ndarray,
        artifact: ReplayCADArtifact,
    ) -> Dict[str, Any]:
        self._load()
        torch = self._torch
        if len(images) == 0:
            raise ValueError(f"Cannot compress empty concept '{concept_id}'")
        if len(images) != len(masks):
            raise ValueError(f"ReplayCAD received {len(images)} images and {len(masks)} masks")

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        token_names = self._semantic_token_names(concept_id)
        token_ids = self._install_tokens(token_names, initialize=True)
        embedding_layer = self._text_encoder.get_input_embeddings()
        embedding_layer.weight.requires_grad_(True)
        initial_embeddings = embedding_layer.weight.detach().clone()

        trainable_rows = torch.zeros(
            embedding_layer.weight.shape[0],
            dtype=torch.bool,
            device=embedding_layer.weight.device,
        )
        trainable_rows[token_ids] = True

        def filter_embedding_gradient(gradient):
            row_mask = trainable_rows[:, None].to(dtype=gradient.dtype)
            return gradient * row_mask

        hook = embedding_layer.weight.register_hook(filter_embedding_gradient)
        projection = torch.nn.Sequential(
            torch.nn.Linear(self.config.mask_group_width, self.config.mask_projection_width),
            torch.nn.ReLU(),
        ).to(device=self.config.device, dtype=self._dtype)
        initial_projection = {key: value.detach().clone() for key, value in projection.state_dict().items()}

        # The embedding matrix is optimized as a whole with a gradient mask, so AdamW's own
        # weight decay would shrink every frozen vocabulary row. Decay is applied by hand to
        # the trained token rows only, before each step, exactly as AdamW orders it.
        optimizer = torch.optim.AdamW(
            [
                {"params": [embedding_layer.weight], "lr": self.config.semantic_learning_rate, "weight_decay": 0.0},
                {
                    "params": projection.parameters(),
                    "lr": self.config.spatial_learning_rate,
                    "weight_decay": self.config.weight_decay,
                },
            ],
        )
        token_decay = 1.0 - self.config.semantic_learning_rate * self.config.weight_decay

        image_latents = self._encode_image_collection(images)
        mask_latents = self._encode_mask_collection(masks)
        if len(image_latents) != len(mask_latents):
            raise RuntimeError("Image and mask latent collections have different lengths")

        rng = np.random.default_rng(self.config.seed)
        started = time.perf_counter()
        final_loss = math.nan
        try:
            for step in range(self.config.compression_steps):
                indices = rng.integers(0, len(image_latents), size=self.config.compression_batch_size)
                latents = image_latents[indices].to(device=self.config.device, dtype=self._dtype)
                mask_batch = mask_latents[indices].to(device=self.config.device, dtype=self._dtype)
                prompts = [self._template_prompt(token_names, rng) for _ in indices]

                with self._autocast():
                    noise = torch.randn_like(latents)
                    timesteps = self._sample_timesteps(len(indices))
                    noisy_latents = self._scheduler.add_noise(latents, noise, timesteps)
                    text = self._encode_text(prompts, requires_grad=True)
                    spatial = self._spatial_tokens(mask_batch, projection)
                    context = torch.cat([text, spatial.to(dtype=text.dtype)], dim=1)
                    prediction = self._unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=context,
                    ).sample

                prediction_type = getattr(self._scheduler.config, "prediction_type", "epsilon")
                if prediction_type == "v_prediction":
                    target = self._scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="mean")

                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        f"ReplayCAD compression for '{concept_id}' produced a non-finite loss "
                        f"at step {step + 1}/{self.config.compression_steps} "
                        f"(inclad-cr, autocast={self.config.autocast_dtype})"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if token_decay != 1.0:
                    with torch.no_grad():
                        embedding_layer.weight[token_ids] *= token_decay
                optimizer.step()
                final_loss = float(loss.detach().cpu())

                if step == 0 or (step + 1) % 500 == 0 or step + 1 == self.config.compression_steps:
                    logger.info(
                        "ReplayCAD compression %s: step %d/%d loss=%.6f",
                        concept_id,
                        step + 1,
                        self.config.compression_steps,
                        final_loss,
                    )
        finally:
            hook.remove()
            embedding_layer.weight.requires_grad_(False)

        steps = self.config.compression_steps
        spatial_decay = (1.0 - self.config.spatial_learning_rate * self.config.weight_decay) ** steps
        with torch.no_grad():
            changed_rows = (embedding_layer.weight != initial_embeddings).any(dim=1)
            frozen_rows_changed = int((changed_rows & ~trainable_rows).sum())
            final_projection = projection.state_dict()
            spatial_drift = _relative_drift(
                torch.cat([initial_projection[key].flatten() for key in initial_projection]),
                torch.cat([final_projection[key].flatten() for key in initial_projection]),
                spatial_decay,
            )
            semantic_drift = _relative_drift(
                initial_embeddings[token_ids], embedding_layer.weight[token_ids], token_decay**steps
            )
        if frozen_rows_changed:
            raise RuntimeError(
                f"ReplayCAD compression for '{concept_id}' changed {frozen_rows_changed} frozen vocabulary rows"
            )

        semantic_embeddings = embedding_layer.weight[token_ids].detach().cpu().float()
        projection_state = {key: value.detach().cpu().float() for key, value in final_projection.items()}
        torch.save(
            {
                "semantic_embeddings": semantic_embeddings,
                "projection_state": projection_state,
                "token_names": token_names,
                "condition_dim": self.config.condition_dim,
                "mask_group_width": self.config.mask_group_width,
                "mask_projection_width": self.config.mask_projection_width,
            },
            artifact.condition_path,
        )
        return {
            "compression_seconds": time.perf_counter() - started,
            "compression_steps": steps,
            "final_compression_loss": final_loss,
            "semantic_tokens": self.config.semantic_tokens,
            "spatial_tokens": self.config.spatial_tokens,
            "token_names": token_names,
            "profile": self.config.preset,
            "autocast_dtype": self.config.autocast_dtype,
            "weight_decay": self.config.weight_decay,
            "spatial_gradient_drift": spatial_drift,
            "semantic_gradient_drift": semantic_drift,
        }

    def generate(self, artifact: ReplayCADArtifact, count: int, seed: int) -> np.ndarray:
        self._load()
        # Installing the tokens resizes the embedding matrix in a fresh process (bank reuse,
        # restarted job); keep that outside autocast. The base call then finds them installed.
        condition = _torch_load(artifact.condition_path)
        self._install_tokens(list(condition["token_names"]), initialize=False)
        with self._autocast():
            return super().generate(artifact, count, seed)


class InCLADCRSAMMaskGenerator(SAMMaskGenerator):
    """SAM masks with a per-benchmark default mode and a full-frame fallback."""

    def __init__(self, config: InCLADCRReplayCADConfig):
        super().__init__(config)
        self.statistics: Dict[str, Dict[str, Any]] = {}

    def generate(self, images: np.ndarray, concept_id: str) -> np.ndarray:
        mode = self.config.mask_modes.get(concept_id, self.config.default_mask_mode)
        fallbacks = 0
        masks: List[np.ndarray] = []
        if mode == "full-frame":
            masks = [np.ones(np.asarray(image).shape[:2], dtype=np.uint8) for image in images]
        else:
            self._load()
            for image in images:
                array = np.asarray(image)
                if mode == "area":
                    try:
                        mask = self._generate_one(array, concept_id, mode)
                    except RuntimeError as error:
                        # Only "SAM proposed nothing" falls back; CUDA errors and OOM must surface.
                        if "did not produce a mask" not in str(error):
                            raise
                        mask = None
                else:
                    mask = self._generate_one(array, concept_id, mode)
                foreground = float(np.mean(mask > 0)) if mask is not None else 0.0
                if mask is None or not self.config.mask_min_foreground <= foreground <= self.config.mask_max_foreground:
                    mask = np.ones(array.shape[:2], dtype=np.uint8)
                    fallbacks += 1
                masks.append(np.asarray(mask, dtype=np.uint8))
        self.statistics[concept_id] = {
            "mask_mode": mode,
            "mask_full_frame_fallbacks": fallbacks,
            "mask_mean_foreground_fraction": float(np.mean([np.mean(mask > 0) for mask in masks])) if masks else 0.0,
        }
        if fallbacks:
            logger.info("ReplayCAD masks for %s: %d/%d full-frame fallbacks", concept_id, fallbacks, len(masks))
        return np.stack(masks).astype(np.uint8, copy=False)


def _is_grey_rgb(images: np.ndarray) -> bool:
    array = np.asarray(images)
    return (
        array.ndim == 4
        and array.shape[-1] == 3
        and np.array_equal(array[..., 0], array[..., 1])
        and np.array_equal(array[..., 0], array[..., 2])
    )


def _to_grey_rgb(images: np.ndarray) -> np.ndarray:
    # ITU-R 601-2 luma, the weights of PIL's "L" conversion.
    grey = np.asarray(images)[..., :3].astype(np.float32) @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    if np.issubdtype(images.dtype, np.integer):
        grey = np.clip(np.rint(grey), np.iinfo(images.dtype).min, np.iinfo(images.dtype).max)
    return np.repeat(grey[..., None], 3, axis=-1).astype(images.dtype, copy=False)


class InCLADCRReplayCADMemory(ReplayCADMemory):
    """ReplayCAD memory whose artifact directory is a bank shared by all orderings."""

    def __init__(
        self,
        config: InCLADCRReplayCADConfig,
        *,
        backend: Optional[ReplayCADBackend] = None,
        mask_generator: Optional[ReplayCADMaskGenerator] = None,
    ):
        if mask_generator is None and config.mask_backend == "sam":
            mask_generator = InCLADCRSAMMaskGenerator(config)
        self._grey_concepts: Dict[str, bool] = {}
        super().__init__(
            config,
            backend=backend or InCLADCRDiffusersBackend(config),
            mask_generator=mask_generator,
        )

    def compress(self, concept_id: str, images: np.ndarray) -> ReplayCADArtifact:
        existing = self._artifact_by_concept(concept_id)
        reused = existing is not None and existing.complete
        # A run sees the training images of each concept it compresses or reuses. A resumed
        # run does not, so the writer also stores the flag in the artifact metadata.
        self._grey_concepts[concept_id] = _is_grey_rgb(images)
        artifact = super().compress(concept_id=concept_id, images=images)
        if not reused:
            self._record_mask_statistics(artifact)
            self._record_grey_flag(artifact)
        # Every ordering of the dataset replays this concept with the same seed, so generate its
        # images now; the bank is then complete even for the last concept of this ordering.
        self._ensure_generated_cache(artifact)
        return artifact

    def generate_previous(self, seed: Optional[int] = None) -> np.ndarray:
        replay = super().generate_previous(seed)
        samples = self.config.replay_samples_per_concept
        for index, artifact in enumerate(self._artifacts):
            if self._is_grey_concept(artifact):
                window = slice(index * samples, (index + 1) * samples)
                replay[window] = _to_grey_rgb(replay[window])
        return replay

    def _ensure_generated_cache(self, artifact: ReplayCADArtifact) -> None:
        seed = self._concept_seed(self.config.seed, artifact.concept_id)
        if self._cache_path(artifact, seed).is_file():
            return
        started = time.perf_counter()
        generated = self._backend.generate(
            artifact=artifact,
            count=self.config.replay_samples_per_concept,
            seed=seed,
        )
        if len(generated) != self.config.replay_samples_per_concept:
            raise RuntimeError(
                f"ReplayCAD backend generated {len(generated)} samples for '{artifact.concept_id}', "
                f"expected {self.config.replay_samples_per_concept}"
            )
        self._save_generated_cache(artifact, seed, generated)
        self._generation_seconds += time.perf_counter() - started

    def _save_generated_cache(self, artifact: ReplayCADArtifact, seed: int, images: np.ndarray) -> None:
        if not self.config.cache_generated:
            return
        path = self._cache_path(artifact, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a reader of the shared bank never sees a half-written cache file.
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, images=np.asarray(images, dtype=np.uint8))
        os.replace(temporary, path)

    def _write_run_config(self) -> None:
        path = self.root / "replaycad_config.json"
        existed = path.is_file()
        super()._write_run_config()
        if not existed:
            return
        existing = json.loads(path.read_text(encoding="utf-8"))
        payload = self.config.model_dump(mode="json")
        mismatches = [key for key in PROFILE_GUARD_KEYS if existing.get(key) != payload.get(key)]
        if mismatches:
            raise ValueError(
                f"ReplayCAD artifact bank {self.root} was built with different inclad-cr settings; "
                f"different fields: {mismatches}"
            )

    def _record_grey_flag(self, artifact: ReplayCADArtifact) -> None:
        metadata = artifact.read_metadata()
        metadata[GREY_METADATA_KEY] = bool(self._grey_concepts[artifact.concept_id])
        artifact.write_metadata(metadata)

    def _is_grey_concept(self, artifact: ReplayCADArtifact) -> bool:
        if artifact.concept_id not in self._grey_concepts:
            metadata = artifact.read_metadata()
            if GREY_METADATA_KEY not in metadata:
                raise ValueError(
                    f"ReplayCAD artifact of '{artifact.concept_id}' was activated without its training images "
                    f"and its metadata has no '{GREY_METADATA_KEY}' flag, so its replay colour is unknown"
                )
            self._grey_concepts[artifact.concept_id] = bool(metadata[GREY_METADATA_KEY])
        return self._grey_concepts[artifact.concept_id]

    def _record_mask_statistics(self, artifact: ReplayCADArtifact) -> None:
        statistics = getattr(self._mask_generator, "statistics", {}).get(artifact.concept_id)
        if statistics is None:
            return
        metadata = artifact.read_metadata()
        metadata.update(statistics)
        artifact.write_metadata(metadata)
        self._write_manifest()


def _resolve_fixed_threshold_without_training_scan(original: Callable) -> Callable:
    """Return the fixed pixel threshold before scoring every seen training image.

    The base resolver computes score maps of all seen training images and then, in ``fixed``
    mode, ignores them. The result is identical; only the scan is skipped.
    """

    def resolve(self, model, cache_key, train_concepts):
        if self._threshold_mode == "fixed":
            self._resolved_thresholds[cache_key] = float(self._fixed_threshold)
            return self._resolved_thresholds[cache_key]
        return original(self, model, cache_key, train_concepts)

    return resolve


def current_settings() -> InCLADCRSettings:
    return _SETTINGS


def install(*, bank_dir: Optional[Path] = None, gpu_knn: bool = False) -> Callable[[], None]:
    """Make the base runner build this profile; returns a function that undoes it.

    ``run_continual_visual_ad.py`` imports the ReplayCAD config and memory classes
    inside its functions, so replacing the module attributes before it runs is
    enough; the runner itself stays unchanged. ``gpu_knn`` also moves PatchCore's
    nearest-neighbour search to the GPU (:mod:`pyclad.models.patchcore.gpu_knn`).
    """
    global _SETTINGS
    import pyclad.strategies.replaycad as package
    from pyclad.callbacks.evaluation import visual_pixel_concept_metric_callback as pixel_callback_module
    from pyclad.strategies.replaycad import config as config_module

    pixel_callback = pixel_callback_module.VisualPixelConceptMetricCallback
    previous_settings = _SETTINGS
    replaced = [
        (config_module, "ReplayCADConfig", config_module.ReplayCADConfig),
        (package, "ReplayCADConfig", package.ReplayCADConfig),
        (package, "ReplayCADMemory", package.ReplayCADMemory),
        (pixel_callback, "_resolve_threshold", pixel_callback.__dict__["_resolve_threshold"]),
    ]
    _SETTINGS = InCLADCRSettings(bank_dir=None if bank_dir is None else Path(bank_dir))
    config_module.ReplayCADConfig = InCLADCRReplayCADConfig
    package.ReplayCADConfig = InCLADCRReplayCADConfig
    package.ReplayCADMemory = InCLADCRReplayCADMemory
    pixel_callback._resolve_threshold = _resolve_fixed_threshold_without_training_scan(
        pixel_callback.__dict__["_resolve_threshold"]
    )
    uninstall_gpu_knn = None
    if gpu_knn:
        from pyclad.models.patchcore import gpu_knn as gpu_knn_module

        uninstall_gpu_knn = gpu_knn_module.install()

    def uninstall() -> None:
        global _SETTINGS
        for owner, name, value in replaced:
            setattr(owner, name, value)
        if uninstall_gpu_knn is not None:
            uninstall_gpu_knn()
        _SETTINGS = previous_settings

    return uninstall
