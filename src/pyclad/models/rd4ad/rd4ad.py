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

from pyclad.models.model import Model
from pyclad.models.rd4ad.architecture import RD4ADArchitecture
from pyclad.models.rd4ad.builder import build
from pyclad.models.rd4ad.config import RD4ADConfig
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import (
    BestWeightsCallback,
    resolve_device,
    to_float,
    trainer_device_config,
)


class RD4AD(Model):
    def __init__(self, config: Optional[RD4ADConfig] = None):
        self.config = config or RD4ADConfig()
        self._validate_config(self.config)

        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=self.config.in_channels,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )

        network = build(self.config)
        self.module = RD4ADModule(
            network=network,
            learning_rate=self.config.learning_rate,
            adam_betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

        self._threshold = self.config.threshold
        self._last_loss: Optional[float] = None

    @staticmethod
    def _validate_config(config: RD4ADConfig) -> None:
        if config.in_channels != 3:
            raise ValueError("RD4AD currently supports RGB inputs only and requires in_channels=3")
        if len(config.normalize_mean) != 3 or len(config.normalize_std) != 3:
            raise ValueError("normalize_mean and normalize_std must contain three values")
        if config.score_smoothing_kernel % 2 == 0:
            raise ValueError("score_smoothing_kernel must be odd")
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

    def fit(self, data: np.ndarray):
        if len(data) == 0 or self.config.epochs == 0:
            return

        callbacks: list[pl.Callback] = []
        best_weights_callback: Optional[BestWeightsCallback] = None

        if self.config.early_stopping_patience is not None:
            early_stopping_kwargs = {
                "monitor": "train_loss",
                "mode": "min",
                "patience": self.config.early_stopping_patience,
                "min_delta": float(self.config.early_stopping_min_delta),
            }
            if "check_on_train_epoch_end" in inspect.signature(EarlyStopping.__init__).parameters:
                early_stopping_kwargs["check_on_train_epoch_end"] = True
            callbacks.append(EarlyStopping(**early_stopping_kwargs))

            if self.config.early_stopping_restore_best:
                best_weights_callback = BestWeightsCallback(
                    monitor="train_loss",
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
        self._last_loss = to_float(trainer.callback_metrics.get("train_loss"))

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

    @staticmethod
    def _resize_maps(score_maps: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if tuple(score_maps.shape[-2:]) == tuple(output_size):
            return score_maps
        resized = F.interpolate(score_maps[:, None, :, :], size=output_size, mode="bilinear", align_corners=False)
        return resized[:, 0]

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

    def _image_scores_from_maps(self, score_maps: np.ndarray) -> np.ndarray:
        if self.config.score_mode == "mean":
            return score_maps.mean(axis=(1, 2)).astype(np.float32, copy=False)
        return score_maps.max(axis=(1, 2)).astype(np.float32, copy=False)

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        score_maps = self.score_maps(data, resize_to_input=False)
        return self._image_scores_from_maps(score_maps)

    def _resolve_threshold(self, scores: np.ndarray) -> float:
        if self.config.threshold is not None:
            return float(self.config.threshold)
        if self._threshold is not None:
            return float(self._threshold)
        if len(scores) == 0:
            return 0.0
        return float(np.quantile(scores, self.config.threshold_quantile))

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        anomaly_scores = self.score_data(data)
        threshold = self._resolve_threshold(anomaly_scores)
        y_pred = (anomaly_scores > threshold).astype(int)
        return y_pred, anomaly_scores

    def name(self) -> str:
        return "RD4AD"

    def additional_info(self) -> Dict:
        return {
            "threshold": self._threshold,
            "backbone": self.config.backbone_name,
            "input_size": self.config.input_size,
            "pretrained_encoder": self.config.pretrained_encoder,
            "freeze_encoder": self.config.freeze_encoder,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "score_smoothing_kernel": self.config.score_smoothing_kernel,
            "score_smoothing_sigma": self.config.score_smoothing_sigma,
            "score_mode": self.config.score_mode,
            "threshold_quantile": self.config.threshold_quantile,
            "last_loss": self._last_loss,
        }


class RD4ADModule(pl.LightningModule):
    def __init__(
        self,
        network: RD4ADArchitecture,
        learning_rate: float,
        adam_betas: tuple[float, float],
        weight_decay: float,
    ):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["network"])

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        teacher_features, _, student_features = self.network.forward_train(x)
        loss = RD4ADArchitecture.cosine_loss(teacher_features, student_features)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        parameters = list(self.network.decoder.parameters()) + list(self.network.bn.parameters())
        if not self.network.freeze_encoder:
            parameters = list(self.network.encoder.parameters()) + parameters

        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
