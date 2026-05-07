from __future__ import annotations

import inspect
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.ganomaly.architecture import GANomalyArchitecture
from pyclad.models.ganomaly.builder import build
from pyclad.models.ganomaly.config import GANomalyConfig
from pyclad.models.ganomaly.loss import DiscriminatorLoss, GeneratorLoss
from pyclad.models.model import Model
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import (
    BestWeightsCallback,
    resolve_device,
    to_float,
    trainer_device_config,
)

_MONITOR_METRIC = "train_generator_loss"


class GANomaly(Model):
    def __init__(self, config: Optional[GANomalyConfig] = None):
        self.config = config or GANomalyConfig()
        self._validate_config(self.config)

        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=self.config.in_channels,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )

        network = build(self.config)
        self.module = GANomalyModule(
            network=network,
            learning_rate=self.config.learning_rate,
            adam_betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
            adversarial_weight=self.config.adversarial_weight,
            contextual_weight=self.config.contextual_weight,
            encoding_weight=self.config.encoding_weight,
        )

        self._threshold = self.config.threshold
        self._last_generator_loss: Optional[float] = None
        self._last_discriminator_loss: Optional[float] = None

    @staticmethod
    def _validate_config(config: GANomalyConfig) -> None:
        if len(config.normalize_mean) != config.in_channels or len(config.normalize_std) != config.in_channels:
            raise ValueError("normalize_mean/std length must match in_channels")
        if config.threshold_quantile <= 0.0 or config.threshold_quantile >= 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")

    def _prepare_batches(self, data: np.ndarray, shuffle: bool) -> DataLoader:
        x_t = self._preprocessor.transform(data)
        dataset = TensorDataset(x_t)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    @staticmethod
    def _resize_maps(score_maps: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if tuple(score_maps.shape[-2:]) == tuple(output_size):
            return score_maps
        resized = F.interpolate(score_maps[:, None, :, :], size=output_size, mode="bilinear", align_corners=False)
        return resized[:, 0]

    def fit(self, data: np.ndarray):
        if len(data) == 0 or self.config.epochs == 0:
            return

        callbacks: list[pl.Callback] = []
        best_weights_callback: Optional[BestWeightsCallback] = None

        if self.config.early_stopping_patience is not None:
            early_stopping_kwargs = {
                "monitor": _MONITOR_METRIC,
                "mode": "min",
                "patience": self.config.early_stopping_patience,
                "min_delta": float(self.config.early_stopping_min_delta),
            }
            if "check_on_train_epoch_end" in inspect.signature(EarlyStopping.__init__).parameters:
                early_stopping_kwargs["check_on_train_epoch_end"] = True
            callbacks.append(EarlyStopping(**early_stopping_kwargs))

            if self.config.early_stopping_restore_best:
                best_weights_callback = BestWeightsCallback(
                    monitor=_MONITOR_METRIC,
                    min_delta=float(self.config.early_stopping_min_delta),
                )
                callbacks.append(best_weights_callback)

        accelerator, devices = trainer_device_config(self._device)
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=self.config.show_training_progress,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )

        trainer.fit(self.module, train_dataloaders=self._prepare_batches(data, shuffle=True))
        self.module = self.module.to(self._device)
        self._last_generator_loss = to_float(trainer.callback_metrics.get("train_generator_loss"))
        self._last_discriminator_loss = to_float(trainer.callback_metrics.get("train_discriminator_loss"))

        if (
            self.config.early_stopping_patience is not None
            and self.config.early_stopping_restore_best
            and best_weights_callback is not None
            and best_weights_callback.best_state_dict is not None
        ):
            self.module.network.load_state_dict(best_weights_callback.best_state_dict)

        scores = None if self.config.threshold is not None else self.score_data(data)
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
        elif scores is None or len(scores) == 0:
            self._threshold = 0.0
        else:
            self._threshold = float(np.quantile(scores, self.config.threshold_quantile))

    def _forward_inference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.module = self.module.to(self._device)
        self.module.eval()
        with torch.no_grad():
            score_maps, anomaly_scores = self.module(x)
        return score_maps, anomaly_scores

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        target_size = self._preprocessor.spatial_size(data)
        all_maps: list[np.ndarray] = []

        for (batch_x,) in self._prepare_batches(data, shuffle=False):
            batch_maps, _ = self._forward_inference(batch_x.to(self._device, dtype=torch.float32))
            if resize_to_input:
                batch_maps = self._resize_maps(batch_maps, output_size=target_size)
            all_maps.append(batch_maps.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(all_maps, axis=0) if all_maps else np.asarray([], dtype=np.float32)

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        all_scores: list[np.ndarray] = []
        for (batch_x,) in self._prepare_batches(data, shuffle=False):
            _, batch_scores = self._forward_inference(batch_x.to(self._device, dtype=torch.float32))
            all_scores.append(batch_scores.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(all_scores, axis=0) if all_scores else np.asarray([], dtype=np.float32)

    def _resolve_threshold(self, scores: np.ndarray) -> float:
        if self.config.threshold is not None:
            return float(self.config.threshold)
        if self._threshold is not None:
            return float(self._threshold)
        if len(scores) == 0:
            return 0.0
        return float(np.quantile(scores, self.config.threshold_quantile))

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        anomaly_scores = self.score_data(data)
        threshold = self._resolve_threshold(anomaly_scores)
        y_pred = (anomaly_scores > threshold).astype(int)
        return y_pred, anomaly_scores

    def name(self) -> str:
        return "GANomaly"

    def additional_info(self) -> Dict:
        return {
            "threshold": self._threshold,
            "input_size": self.config.input_size,
            "model_input_size": self.module.network.model_input_size,
            "n_features": self.config.n_features,
            "latent_vec_size": self.config.latent_vec_size,
            "extra_layers": self.config.extra_layers,
            "add_final_conv_layer": self.config.add_final_conv_layer,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "adversarial_weight": self.config.adversarial_weight,
            "contextual_weight": self.config.contextual_weight,
            "encoding_weight": self.config.encoding_weight,
            "threshold_quantile": self.config.threshold_quantile,
            "last_generator_loss": self._last_generator_loss,
            "last_discriminator_loss": self._last_discriminator_loss,
        }


class GANomalyModule(pl.LightningModule):
    def __init__(
        self,
        network: GANomalyArchitecture,
        learning_rate: float,
        adam_betas: tuple[float, float],
        weight_decay: float,
        adversarial_weight: float,
        contextual_weight: float,
        encoding_weight: float,
    ):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay
        self.generator_loss_fn = GeneratorLoss(
            wadv=adversarial_weight,
            wcon=contextual_weight,
            wenc=encoding_weight,
        )
        self.discriminator_loss_fn = DiscriminatorLoss()
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=["network", "generator_loss_fn", "discriminator_loss_fn"])

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        g_opt, d_opt = self.optimizers()

        padded, fake, latent_i, latent_o = self.network.forward_train(x)

        self.toggle_optimizer(g_opt)
        pred_real_for_g, _ = self.network.discriminator(padded)
        pred_fake_for_g, _ = self.network.discriminator(fake)
        g_loss = self.generator_loss_fn(latent_i, latent_o, padded, fake, pred_real_for_g, pred_fake_for_g)
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
        self.untoggle_optimizer(g_opt)

        self.toggle_optimizer(d_opt)
        pred_real_for_d, _ = self.network.discriminator(padded.detach())
        pred_fake_for_d, _ = self.network.discriminator(fake.detach())
        d_loss = self.discriminator_loss_fn(pred_real_for_d, pred_fake_for_d)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        self.untoggle_optimizer(d_opt)

        total_loss = g_loss.detach() + d_loss.detach()
        self.log("train_generator_loss", g_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_discriminator_loss", d_loss.detach(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False)
        return total_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        discriminator_optimizer = torch.optim.Adam(
            self.network.discriminator.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
        generator_optimizer = torch.optim.Adam(
            self.network.generator.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
        return [generator_optimizer, discriminator_optimizer]
