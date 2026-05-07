from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pyclad.models.ganomaly.standard import Discriminator, Generator


def _next_power_of_two(value: int) -> int:
    return 2 ** math.ceil(math.log2(max(1, value)))


class GANomalyArchitecture(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        in_channels: int,
        n_features: int,
        latent_vec_size: int,
        extra_layers: int,
        add_final_conv_layer: bool,
    ):
        super().__init__()

        self.input_size = tuple(int(dimension) for dimension in input_size)
        padded_side = _next_power_of_two(max(self.input_size))
        self.model_input_size = (padded_side, padded_side)
        self.pad = self._compute_padding(self.input_size, self.model_input_size)

        self.generator = Generator(
            input_size=self.model_input_size,
            latent_vec_size=latent_vec_size,
            num_input_channels=in_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        self.discriminator = Discriminator(
            input_size=self.model_input_size,
            num_input_channels=in_channels,
            n_features=n_features,
            extra_layers=extra_layers,
        )

        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

    @staticmethod
    def _compute_padding(input_size: tuple[int, int], model_input_size: tuple[int, int]) -> tuple[int, int, int, int]:
        input_height, input_width = input_size
        target_height, target_width = model_input_size

        padding_h = target_height - input_height
        padding_w = target_width - input_width
        top = padding_h // 2
        bottom = padding_h - top
        left = padding_w // 2
        right = padding_w - left
        return left, right, top, bottom

    @staticmethod
    def weights_init(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if classname.find("Conv") != -1 and hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def pad_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if self.pad == (0, 0, 0, 0):
            return batch
        left, right, top, bottom = self.pad
        return F.pad(batch, pad=[left, right, top, bottom])

    def crop_to_input_size(self, batch: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.pad
        if self.pad == (0, 0, 0, 0):
            return batch

        height_end = batch.shape[-2] - bottom if bottom > 0 else batch.shape[-2]
        width_end = batch.shape[-1] - right if right > 0 else batch.shape[-1]
        return batch[..., top:height_end, left:width_end]

    def _generator_outputs(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_batch = self.pad_batch(batch)
        fake, latent_i, latent_o = self.generator(padded_batch)
        return padded_batch, fake, latent_i, latent_o

    def forward_train(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            self.train()
        return self._generator_outputs(batch)

    def forward(self, batch: torch.Tensor):
        padded_batch, fake, latent_i, latent_o = self._generator_outputs(batch)
        if self.training:
            return padded_batch, fake, latent_i, latent_o

        scores = torch.mean(torch.pow(latent_i - latent_o, 2), dim=1).view(-1)
        anomaly_maps = torch.mean(torch.pow(padded_batch - fake, 2), dim=1, keepdim=True)
        anomaly_maps = self.crop_to_input_size(anomaly_maps)
        return anomaly_maps[:, 0], scores
