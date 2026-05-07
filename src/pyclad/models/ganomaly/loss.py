from __future__ import annotations

import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self, wadv: float = 1.0, wcon: float = 50.0, wenc: float = 1.0):
        super().__init__()
        self.loss_enc = nn.SmoothL1Loss()
        self.loss_adv = nn.MSELoss()
        self.loss_con = nn.L1Loss()

        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def forward(
        self,
        latent_i: torch.Tensor,
        latent_o: torch.Tensor,
        images: torch.Tensor,
        fake: torch.Tensor,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> torch.Tensor:
        error_enc = self.loss_enc(latent_i, latent_o)
        error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(pred_real, pred_fake)
        return error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        error_real = self.loss_bce(
            pred_real,
            torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device),
        )
        error_fake = self.loss_bce(
            pred_fake,
            torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device),
        )
        return (error_real + error_fake) * 0.5
