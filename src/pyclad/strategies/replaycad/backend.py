from __future__ import annotations

import abc
import inspect
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image

from pyclad.strategies.replaycad.artifacts import ReplayCADArtifact, concept_slug
from pyclad.strategies.replaycad.config import ReplayCADConfig

logger = logging.getLogger(__name__)


class ReplayCADBackend(abc.ABC):
    @abc.abstractmethod
    def compress(
        self,
        concept_id: str,
        images: np.ndarray,
        masks: np.ndarray,
        artifact: ReplayCADArtifact,
    ) -> Dict[str, Any]:
        """Learn and persist a semantic embedding and spatial projection."""

    @abc.abstractmethod
    def generate(self, artifact: ReplayCADArtifact, count: int, seed: int) -> np.ndarray:
        """Generate normal historical images represented by *artifact*."""

    def release_device_memory(self) -> None:
        """Release optional accelerator state between ReplayCAD phases.

        Backends that do not keep accelerator-resident state need not
        override this hook.
        """


def _torch_load(path: Path):
    import torch

    parameters = inspect.signature(torch.load).parameters
    kwargs = {"map_location": "cpu"}
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


class DiffusersReplayCADBackend(ReplayCADBackend):
    """ReplayCAD backend built from licensed Diffusers model components.

    It implements the paper objective directly: the diffusion backbone, image
    encoder and text encoder remain frozen; only K semantic token embeddings
    and the per-concept mask projection are optimized. Spatial tokens are
    concatenated with text conditioning before U-Net cross-attention.
    """

    def __init__(self, config: ReplayCADConfig):
        self.config = config
        self._pipeline = None
        self._text_encoder = None
        self._tokenizer = None
        self._unet = None
        self._vae = None
        self._scheduler = None
        self._torch = None
        self._dtype = None

    def _load(self) -> None:
        if self._pipeline is not None:
            # ``release_device_memory`` moves the already-loaded pipeline to
            # CPU while PatchCore builds its coreset.  Reactivate the same
            # weights instead of downloading/reloading them.
            self._pipeline.to(self.config.device)
            return
        try:
            import torch
            from diffusers import DDIMScheduler, DiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "ReplayCAD's diffusion backend requires diffusers and transformers. "
                "Install pyclad with the 'replaycad' extra."
            ) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"ReplayCAD is configured for '{self.config.device}', but CUDA is not available. "
                "Use --replaycad-device cpu only for tiny smoke tests."
            )

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.config.torch_dtype]
        pipeline = DiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            local_files_only=self.config.local_files_only,
        )
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.to(self.config.device)

        if self.config.diffusion_family == "stable-diffusion":
            text_encoder = pipeline.text_encoder
            vae = pipeline.vae
        else:
            text_encoder = pipeline.bert
            vae = pipeline.vqvae

        self._pipeline = pipeline
        self._text_encoder = text_encoder
        self._tokenizer = pipeline.tokenizer
        self._unet = pipeline.unet
        self._vae = vae
        self._scheduler = pipeline.scheduler
        self._torch = torch
        self._dtype = dtype

        for module in (self._text_encoder, self._unet, self._vae):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        actual_dim = int(getattr(self._unet.config, "cross_attention_dim", self.config.condition_dim))
        if actual_dim != self.config.condition_dim:
            raise ValueError(
                f"ReplayCAD preset expects condition_dim={self.config.condition_dim}, "
                f"but '{self.config.model_id}' exposes {actual_dim}"
            )

    def release_device_memory(self) -> None:
        """Move diffusion weights off CUDA before the detector is fitted.

        ReplayCAD generation/compression and PatchCore never run at the same
        time.  Keeping both models on the GPU caused avoidable H100 OOMs in
        PatchCore's coreset projection.  Moving the frozen pipeline to CPU
        changes neither weights nor generated samples.
        """
        if self._pipeline is None or self._torch is None:
            return
        if not self.config.device.startswith("cuda"):
            return
        self._pipeline.to("cpu")
        self._torch.cuda.empty_cache()
        logger.info("ReplayCAD diffusion pipeline moved to CPU before detector fitting")

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

        optimizer = torch.optim.AdamW(
            [
                {"params": [embedding_layer.weight], "lr": self.config.semantic_learning_rate},
                {"params": projection.parameters(), "lr": self.config.spatial_learning_rate},
            ],
            weight_decay=0.0,
        )

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

                noise = torch.randn_like(latents)
                timesteps = self._sample_timesteps(len(indices))
                noisy_latents = self._scheduler.add_noise(latents, noise, timesteps)
                context = self._conditioning(token_names, mask_batch, projection, requires_grad=True)
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
                        f"at step {step + 1}/{self.config.compression_steps} with "
                        f"torch_dtype={self.config.torch_dtype}. Retry with bfloat16 on a "
                        "supported GPU, or float32 if bfloat16 is still unstable."
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
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

        semantic_embeddings = embedding_layer.weight[token_ids].detach().cpu().float()
        projection_state = {key: value.detach().cpu().float() for key, value in projection.state_dict().items()}
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
            "compression_steps": self.config.compression_steps,
            "final_compression_loss": final_loss,
            "semantic_tokens": self.config.semantic_tokens,
            "spatial_tokens": self.config.spatial_tokens,
            "token_names": token_names,
        }

    def generate(self, artifact: ReplayCADArtifact, count: int, seed: int) -> np.ndarray:
        self._load()
        torch = self._torch
        if count <= 0:
            return np.empty((0, self.config.resolution, self.config.resolution, 3), dtype=np.uint8)

        condition = _torch_load(artifact.condition_path)
        token_names = list(condition["token_names"])
        token_ids = self._install_tokens(token_names, initialize=False)
        embedding_layer = self._text_encoder.get_input_embeddings()
        with torch.no_grad():
            embedding_values = condition["semantic_embeddings"].to(
                device=embedding_layer.weight.device,
                dtype=embedding_layer.weight.dtype,
            )
            embedding_layer.weight[token_ids] = embedding_values

        projection = torch.nn.Sequential(
            torch.nn.Linear(self.config.mask_group_width, self.config.mask_projection_width),
            torch.nn.ReLU(),
        ).to(device=self.config.device, dtype=self._dtype)
        projection.load_state_dict(condition["projection_state"])
        projection.eval()

        stored_masks = artifact.load_masks()
        if len(stored_masks) == 0:
            raise RuntimeError(f"ReplayCAD artifact for '{artifact.concept_id}' contains no masks")
        rng = np.random.default_rng(seed)
        generated: List[np.ndarray] = []
        started = time.perf_counter()

        with torch.no_grad():
            for offset in range(0, count, self.config.generation_batch_size):
                batch_size = min(self.config.generation_batch_size, count - offset)
                selected = stored_masks[rng.integers(0, len(stored_masks), size=batch_size)]
                if self.config.augment_masks:
                    selected = np.stack([self._augment_mask(mask, rng) for mask in selected])
                mask_latents = self._encode_mask_collection(selected).to(
                    device=self.config.device,
                    dtype=self._dtype,
                )
                condition_context = self._conditioning(
                    token_names,
                    mask_latents,
                    projection,
                    requires_grad=False,
                )
                unconditional_context = self._unconditional_context(batch_size, condition_context.shape[1])

                generator = torch.Generator(device=self.config.device)
                generator.manual_seed(seed + offset)
                latents = self._sample_images(
                    condition_context=condition_context,
                    unconditional_context=unconditional_context,
                    generator=generator,
                )
                generated.extend(self._decode_latents(latents))

        logger.info(
            "ReplayCAD generated %d samples for %s in %.1fs",
            count,
            artifact.concept_id,
            time.perf_counter() - started,
        )
        return np.stack(generated[:count]).astype(np.uint8, copy=False)

    def _semantic_token_names(self, concept_id: str) -> List[str]:
        slug = concept_slug(concept_id).replace(".", "-")
        return [f"<rc-{slug}-{index:02d}>" for index in range(self.config.semantic_tokens)]

    def _install_tokens(self, token_names: Sequence[str], initialize: bool) -> List[int]:
        torch = self._torch
        tokenizer = self._tokenizer
        text_encoder = self._text_encoder
        old_size = len(tokenizer)
        tokenizer.add_tokens(list(token_names))
        if len(tokenizer) != old_size:
            text_encoder.resize_token_embeddings(len(tokenizer))
            text_encoder.to(device=self.config.device, dtype=self._dtype)
            for parameter in text_encoder.parameters():
                parameter.requires_grad_(False)

        token_ids = [int(tokenizer.convert_tokens_to_ids(name)) for name in token_names]
        if initialize:
            initializer_ids = tokenizer.encode(self.config.initializer_word, add_special_tokens=False)
            if not initializer_ids:
                raise ValueError(f"Initializer word '{self.config.initializer_word}' does not tokenize")
            embedding_layer = text_encoder.get_input_embeddings()
            with torch.no_grad():
                initial = embedding_layer.weight[int(initializer_ids[0])].detach().clone()
                embedding_layer.weight[token_ids] = initial[None, :].repeat(len(token_ids), 1)
        return token_ids

    def _prompt(self, token_names: Sequence[str]) -> str:
        return f"{self.config.prompt_prefix} {' '.join(token_names)}"

    def _encode_text(self, prompts: Sequence[str], requires_grad: bool):
        torch = self._torch
        encoder_limit = getattr(self._text_encoder.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(self._tokenizer, "model_max_length", None)
        limits = [int(limit) for limit in (encoder_limit, tokenizer_limit) if limit is not None and int(limit) > 0]
        max_length = min(limits) if limits else 77
        inputs = self._tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.config.device)
        if requires_grad:
            result = self._text_encoder(input_ids)
        else:
            with torch.no_grad():
                result = self._text_encoder(input_ids)
        hidden = result.last_hidden_state if hasattr(result, "last_hidden_state") else result[0]
        return hidden.to(dtype=self._dtype)

    def _conditioning(self, token_names, mask_latents, projection, requires_grad: bool):
        batch_size = len(mask_latents)
        text = self._encode_text([self._prompt(token_names)] * batch_size, requires_grad=requires_grad)
        spatial = self._spatial_tokens(mask_latents, projection)
        return self._torch.cat([text, spatial], dim=1)

    def _unconditional_context(self, batch_size: int, target_length: int):
        torch = self._torch
        text = self._encode_text([self.config.negative_prompt] * batch_size, requires_grad=False)
        if text.shape[1] > target_length:
            return text[:, :target_length]
        if text.shape[1] < target_length:
            padding = torch.zeros(
                (batch_size, target_length - text.shape[1], text.shape[2]),
                device=text.device,
                dtype=text.dtype,
            )
            text = torch.cat([text, padding], dim=1)
        return text

    def _spatial_tokens(self, mask_latents, projection):
        torch = self._torch
        flattened = mask_latents.flatten(start_dim=1)
        if flattened.shape[1] % self.config.mask_group_width != 0:
            raise ValueError(
                f"Encoded masks have {flattened.shape[1]} values, not divisible by "
                f"mask_group_width={self.config.mask_group_width}"
            )
        grouped = flattened.reshape(len(flattened), -1, self.config.mask_group_width)
        projected = projection(grouped)
        total = projected.shape[1] * projected.shape[2]
        if total % self.config.condition_dim != 0:
            raise ValueError(f"Projected spatial feature size {total} is incompatible with condition dimension")
        return projected.reshape(len(projected), -1, self.config.condition_dim)

    def _sample_timesteps(self, batch_size: int):
        torch = self._torch
        total = int(self._scheduler.config.num_train_timesteps)
        if self.config.timestep_sampling == "uniform":
            return torch.randint(0, total, (batch_size,), device=self.config.device, dtype=torch.long)
        probabilities = torch.arange(total, device=self.config.device, dtype=torch.float32).pow(3)
        probabilities[0] = torch.finfo(probabilities.dtype).eps
        probabilities /= probabilities.sum()
        return torch.multinomial(probabilities, batch_size, replacement=True).long()

    def _encode_image_collection(self, images: np.ndarray):
        batches = []
        for offset in range(0, len(images), self.config.compression_batch_size):
            tensor = self._image_tensor(images[offset : offset + self.config.compression_batch_size])
            batches.append(self._encode_tensor(tensor).detach().cpu())
        return self._torch.cat(batches, dim=0)

    def _encode_mask_collection(self, masks: np.ndarray):
        arrays = np.asarray(masks)
        if arrays.ndim != 3:
            raise ValueError(f"Expected masks with shape (N,H,W), got {arrays.shape}")
        images = np.repeat((arrays > 0)[..., None], 3, axis=-1).astype(np.uint8) * 255
        batches = []
        for offset in range(0, len(images), self.config.compression_batch_size):
            tensor = self._image_tensor(images[offset : offset + self.config.compression_batch_size])
            batches.append(self._encode_tensor(tensor).detach().cpu())
        return self._torch.cat(batches, dim=0)

    def _image_tensor(self, images: Iterable[np.ndarray]):
        torch = self._torch
        arrays = []
        for image in images:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            if array.dtype != np.uint8:
                if array.size and float(array.max()) <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            pil = Image.fromarray(array[..., :3]).resize(
                (self.config.resolution, self.config.resolution),
                resample=Image.Resampling.BILINEAR,
            )
            arrays.append(np.asarray(pil, dtype=np.float32) / 127.5 - 1.0)
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
        return tensor.to(device=self.config.device, dtype=self._dtype)

    def _encode_tensor(self, tensor):
        torch = self._torch
        with torch.no_grad():
            encoded = self._vae.encode(tensor)
            if hasattr(encoded, "latent_dist"):
                latents = encoded.latent_dist.sample()
            elif hasattr(encoded, "latents"):
                latents = encoded.latents
            elif isinstance(encoded, tuple):
                latents = encoded[0]
            else:
                raise TypeError(f"Unsupported VAE encoder output: {type(encoded)!r}")
            return latents * self._latent_scaling_factor()

    def _latent_scaling_factor(self) -> float:
        return self.config.latent_scaling_factor

    def _vae_scale_factor(self) -> int:
        block_channels = getattr(self._vae.config, "block_out_channels", None)
        return 2 ** (len(block_channels) - 1) if block_channels else 8

    def _sample_images(self, condition_context, unconditional_context, generator):
        torch = self._torch
        batch_size = len(condition_context)
        scale = self._vae_scale_factor()
        shape = (
            batch_size,
            int(self._unet.config.in_channels),
            self.config.resolution // scale,
            self.config.resolution // scale,
        )
        latents = torch.randn(shape, generator=generator, device=self.config.device, dtype=self._dtype)
        latents *= float(getattr(self._scheduler, "init_noise_sigma", 1.0))
        self._scheduler.set_timesteps(self.config.inference_steps, device=self.config.device)

        for timestep in self._scheduler.timesteps:
            model_input = torch.cat([latents, latents], dim=0)
            if hasattr(self._scheduler, "scale_model_input"):
                model_input = self._scheduler.scale_model_input(model_input, timestep)
            context = torch.cat([unconditional_context, condition_context], dim=0)
            predicted = self._unet(model_input, timestep, encoder_hidden_states=context).sample
            predicted_unconditional, predicted_condition = predicted.chunk(2)
            guided = predicted_unconditional + self.config.guidance_scale * (
                predicted_condition - predicted_unconditional
            )
            step_kwargs = {}
            parameters = inspect.signature(self._scheduler.step).parameters
            if "eta" in parameters:
                step_kwargs["eta"] = 0.0
            if "generator" in parameters:
                step_kwargs["generator"] = generator
            latents = self._scheduler.step(guided, timestep, latents, **step_kwargs).prev_sample
        return latents

    def _decode_latents(self, latents) -> List[np.ndarray]:
        torch = self._torch
        with torch.no_grad():
            decoded = self._vae.decode(latents / self._latent_scaling_factor())
            tensor = decoded.sample if hasattr(decoded, "sample") else decoded[0]
            tensor = (tensor / 2 + 0.5).clamp(0, 1)
            arrays = tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return [(array * 255.0).round().astype(np.uint8) for array in arrays]

    def _augment_mask(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
        angle = rng.uniform(-self.config.max_mask_rotation_degrees, self.config.max_mask_rotation_degrees)
        image = image.rotate(float(angle), resample=Image.Resampling.NEAREST, fillcolor=0)
        array = np.asarray(image)
        max_x = int(round(array.shape[1] * self.config.max_mask_shift_fraction))
        max_y = int(round(array.shape[0] * self.config.max_mask_shift_fraction))
        shift_x = int(rng.integers(-max_x, max_x + 1)) if max_x else 0
        shift_y = int(rng.integers(-max_y, max_y + 1)) if max_y else 0
        shifted = np.roll(array, shift=(shift_y, shift_x), axis=(0, 1))
        if shift_y > 0:
            shifted[:shift_y, :] = 0
        elif shift_y < 0:
            shifted[shift_y:, :] = 0
        if shift_x > 0:
            shifted[:, :shift_x] = 0
        elif shift_x < 0:
            shifted[:, shift_x:] = 0
        return (shifted > 0).astype(np.uint8)
