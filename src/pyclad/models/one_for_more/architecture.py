from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pyclad.models.one_for_more.config import OneForMoreConfig
from pyclad.models.vision.backbones import TorchvisionFeatureExtractor


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = math.log(10_000.0) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device, dtype=torch.float32))
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _position_embedding_2d(
    height: int,
    width: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quarter = max(channels // 4, 1)
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10_000.0 ** (omega / max(quarter - 1, 1)))
    y = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1) * omega.unsqueeze(0)
    x = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(1) * omega.unsqueeze(0)
    y_embedding = torch.cat([y.sin(), y.cos()], dim=1)[:, None, :].expand(-1, width, -1)
    x_embedding = torch.cat([x.sin(), x.cos()], dim=1)[None, :, :].expand(height, -1, -1)
    embedding = torch.cat([y_embedding, x_embedding], dim=2).reshape(height * width, -1)
    if embedding.shape[1] < channels:
        embedding = F.pad(embedding, (0, channels - embedding.shape[1]))
    return embedding[:, :channels].to(dtype=dtype)


class FixedImageCodec(nn.Module):
    """Deterministic offline fallback for the frozen factor-eight VAE."""

    def __init__(self, downsample: int):
        super().__init__()
        self.downsample = int(downsample)
        self.latent_channels = 3

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(images, kernel_size=self.downsample, stride=self.downsample)

    def decode(self, latents: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(latents, size=output_size, mode="bilinear", align_corners=False).clamp(-1.0, 1.0)


class DiffusersVAECodec(nn.Module):
    """Frozen diffusers AutoencoderKL adapter loaded only when requested."""

    def __init__(self, config: OneForMoreConfig):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "vae_model_id requires diffusers; install pyclad[one-for-more] or pyclad[replaycad]"
            ) from exc

        kwargs = {"local_files_only": config.vae_local_files_only}
        if config.vae_subfolder is not None:
            kwargs["subfolder"] = config.vae_subfolder
        self.vae = AutoencoderKL.from_pretrained(config.vae_model_id, **kwargs)
        self.vae.requires_grad_(False).eval()
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.latent_channels = int(getattr(self.vae.config, "latent_channels", config.vae_latent_channels))

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images).latent_dist.mode() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        decoded = self.vae.decode(latents / self.scaling_factor).sample
        if tuple(decoded.shape[-2:]) != tuple(output_size):
            decoded = F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)
        return decoded.clamp(-1.0, 1.0)

    def train(self, mode: bool = True):
        super().train(False)
        self.vae.eval()
        return self


class AnomalyMaskedConditioner(nn.Module):
    """AMN: local CNN encoder followed by neighbor-masked self-attention."""

    def __init__(self, config: OneForMoreConfig):
        super().__init__()
        channels = config.base_channels
        first = max(channels // 4, 8)
        second = max(channels // 2, 8)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(first, first, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(first, second, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(second, second, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(second, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )
        self.neighbor_size = config.neighbor_size

        if config.conditioner_depth > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=config.conditioner_heads,
                dim_feedforward=config.conditioner_feedforward,
                dropout=config.conditioner_dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.non_local_encoder: Optional[nn.Module] = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.conditioner_depth,
            )
        else:
            self.non_local_encoder = None
        self._mask_cache: dict[tuple[int, int, str], Optional[torch.Tensor]] = {}

    def local_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.local_encoder(images)

    def _neighbor_mask(self, height: int, width: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.neighbor_size is None:
            return None
        key = (height, width, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]

        neighbor_h, neighbor_w = self.neighbor_size
        mask = torch.zeros((height * width, height * width), dtype=torch.bool, device=device)
        for row in range(height):
            for column in range(width):
                row_start = max(row - neighbor_h // 2, 0)
                row_end = min(row + neighbor_h // 2 + 1, height)
                col_start = max(column - neighbor_w // 2, 0)
                col_end = min(column + neighbor_w // 2 + 1, width)
                query_index = row * width + column
                for blocked_row in range(row_start, row_end):
                    start = blocked_row * width + col_start
                    end = blocked_row * width + col_end
                    mask[query_index, start:end] = True

        # Tiny smoke-test feature maps can have no non-local token at all.
        # Falling back to global attention avoids all-masked softmax rows.
        result = None if bool(mask.all(dim=1).any()) else mask
        self._mask_cache[key] = result
        return result

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        local = self.local_features(images)
        if self.non_local_encoder is None:
            return local

        batch, channels, height, width = local.shape
        tokens = local.flatten(2).transpose(1, 2)
        position = _position_embedding_2d(height, width, channels, local.device, local.dtype)
        tokens = tokens + position.unsqueeze(0)
        tokens = self.non_local_encoder(tokens, mask=self._neighbor_mask(height, width, local.device))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class ResidualTimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, value: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(value)


class ConditionalDenoiser(nn.Module):
    def __init__(self, latent_channels: int, base_channels: int):
        super().__init__()
        time_channels = base_channels * 4
        self.time_channels = time_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.base_channels = base_channels
        self.input = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.input_block = ResidualTimeBlock(base_channels, base_channels, time_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.down_block1 = ResidualTimeBlock(base_channels * 2, base_channels * 2, time_channels)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.middle = ResidualTimeBlock(base_channels * 4, base_channels * 4, time_channels)
        self.up1 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.up_block1 = ResidualTimeBlock(base_channels * 4, base_channels * 2, time_channels)
        self.up2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.up_block2 = ResidualTimeBlock(base_channels * 2, base_channels, time_channels)
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output = nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(_sinusoidal_embedding(timesteps, self.base_channels))
        hidden0 = self.input(noisy_latents)
        if tuple(control.shape[-2:]) != tuple(hidden0.shape[-2:]):
            control = F.interpolate(control, size=hidden0.shape[-2:], mode="bilinear", align_corners=False)
        hidden0 = hidden0 + control
        hidden1 = self.input_block(hidden0, time_embedding)
        hidden2 = self.down_block1(self.down1(hidden1), time_embedding)
        hidden3 = self.middle(self.down2(hidden2), time_embedding)

        up1 = F.interpolate(hidden3, size=hidden2.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(up1)
        up1 = self.up_block1(torch.cat([up1, hidden2], dim=1), time_embedding)
        up2 = F.interpolate(up1, size=hidden1.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up2(up2)
        up2 = self.up_block2(torch.cat([up2, hidden1], dim=1), time_embedding)
        return self.output(F.silu(self.output_norm(up2)))


class OneForMoreArchitecture(nn.Module):
    def __init__(self, config: OneForMoreConfig):
        super().__init__()
        self.config = config
        self.codec = (
            DiffusersVAECodec(config)
            if config.vae_model_id is not None
            else FixedImageCodec(config.latent_downsample)
        )
        self.conditioner = AnomalyMaskedConditioner(config)
        self.denoiser = ConditionalDenoiser(self.codec.latent_channels, config.base_channels)

        betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps, dtype=torch.float32)
        alphas_cumulative = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumulative", alphas_cumulative, persistent=True)

        if config.backbone_name is None:
            self.feature_extractor: Optional[TorchvisionFeatureExtractor] = None
        else:
            self.feature_extractor = TorchvisionFeatureExtractor(
                backbone_name=config.backbone_name,
                return_nodes=config.backbone_return_nodes,
                pretrained=config.pretrained_backbone,
                freeze=True,
            )
            self.feature_extractor.eval()

    @property
    def continual_module(self) -> nn.Module:
        # Codec and feature extractor are frozen in the method; only these two
        # components participate in gradient projection.
        return nn.ModuleDict({"conditioner": self.conditioner, "denoiser": self.denoiser})

    def train(self, mode: bool = True):
        super().train(mode)
        self.codec.eval()
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
        return self

    def _alpha(self, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
        alpha = self.alphas_cumulative[timesteps]
        return alpha.reshape(alpha.shape[0], *((1,) * (ndim - 1)))

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha(timesteps, clean.ndim)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        control = self.conditioner(condition_images)
        return self.denoiser(noisy_latents, timesteps, control), control

    def training_loss(
        self,
        clean_images: torch.Tensor,
        corrupted_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            clean_latents = self.codec.encode(clean_images)
        timesteps = torch.randint(
            0,
            self.config.diffusion_steps,
            (clean_latents.shape[0],),
            device=clean_latents.device,
        )
        noise = torch.randn_like(clean_latents)
        noisy = self.q_sample(clean_latents, timesteps, noise)
        predicted_noise, control = self.predict_noise(noisy, timesteps, corrupted_images)
        with torch.no_grad():
            clean_local_features = self.conditioner.local_features(clean_images)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        condition_loss = F.mse_loss(control, clean_local_features)
        total = diffusion_loss + self.config.condition_loss_weight * condition_loss
        return total, {"diffusion": diffusion_loss.detach(), "condition": condition_loss.detach()}

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor, seed: int) -> torch.Tensor:
        self.eval()
        clean_latents = self.codec.encode(images)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            clean_latents.shape,
            generator=generator,
            dtype=clean_latents.dtype,
        ).to(clean_latents.device)
        start = torch.full(
            (clean_latents.shape[0],),
            self.config.inference_start_step,
            dtype=torch.long,
            device=clean_latents.device,
        )
        current = self.q_sample(clean_latents, start, noise)
        schedule = torch.linspace(
            self.config.inference_start_step,
            0,
            self.config.inference_steps,
            device=clean_latents.device,
        ).round().long()

        predicted_clean = clean_latents
        for index, step in enumerate(schedule):
            timesteps = torch.full_like(start, int(step.item()))
            predicted_noise, _ = self.predict_noise(current, timesteps, images)
            alpha = self._alpha(timesteps, current.ndim)
            predicted_clean = (current - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-8)
            if index + 1 < len(schedule):
                next_steps = torch.full_like(start, int(schedule[index + 1].item()))
                next_alpha = self._alpha(next_steps, current.ndim)
                current = next_alpha.sqrt() * predicted_clean + (1.0 - next_alpha).sqrt() * predicted_noise

        return self.codec.decode(predicted_clean, output_size=self.config.input_size)

    @staticmethod
    def _imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
        images = (images.clamp(-1.0, 1.0) + 1.0) * 0.5
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (images - mean) / std

    @torch.no_grad()
    def anomaly_maps(self, images: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is None:
            return torch.linalg.vector_norm(images - reconstructions, dim=1)

        self.feature_extractor.eval()
        input_features = self.feature_extractor(self._imagenet_normalize(images))
        reconstruction_features = self.feature_extractor(self._imagenet_normalize(reconstructions))
        anomaly_map = images.new_zeros((images.shape[0], *self.config.input_size))
        for input_feature, reconstruction_feature in zip(input_features, reconstruction_features):
            component = torch.linalg.vector_norm(input_feature - reconstruction_feature, dim=1, keepdim=True)
            component = F.interpolate(
                component,
                size=self.config.input_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map = anomaly_map + component[:, 0]
        return anomaly_map
