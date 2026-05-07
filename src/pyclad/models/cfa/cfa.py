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

from pyclad.models.cfa.architecture import CFAArchitecture
from pyclad.models.cfa.builder import build
from pyclad.models.cfa.config import CFAConfig
from pyclad.models.model import Model
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import (
    BestWeightsCallback,
    resolve_device,
    to_float,
    trainer_device_config,
)


class CFA(Model):
    def __init__(self, config: Optional[CFAConfig] = None):
        self.config = config or CFAConfig()

        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=self.config.in_channels,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )
        network = build(self.config)
        self.module = CFAModule(
            network=network,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            use_amsgrad=self.config.use_amsgrad,
        )

        self._threshold = self.config.threshold
        self._last_loss: Optional[float] = None

    def _prepare_batches(self, data: np.ndarray, shuffle: bool) -> DataLoader:
        x_t = self._preprocessor.transform(data)
        dataset = TensorDataset(x_t)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def _ensure_fitted(self) -> None:
        if not self.module.network.has_memory_bank:
            raise RuntimeError("CFA must be fitted before scoring or predicting")

    @staticmethod
    def _resize_maps(score_maps: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if tuple(score_maps.shape[-2:]) == tuple(output_size):
            return score_maps
        resized = F.interpolate(score_maps[:, None, :, :], size=output_size, mode="bilinear", align_corners=False)
        return resized[:, 0]

    def fit(self, data: np.ndarray):
        if len(data) == 0:
            return

        self.module = self.module.to(self._device)
        memory_loader = self._prepare_batches(data, shuffle=False)
        self.module.network.initialize_memory_bank(memory_loader, device=self._device)

        callbacks: list[pl.Callback] = []
        best_weights_callback: Optional[BestWeightsCallback] = None

        if self.config.epochs > 0 and self.config.early_stopping_patience is not None:
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

        if self.config.epochs > 0:
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
            self._last_loss = to_float(trainer.callback_metrics.get("train_loss"))

            if (
                self.config.early_stopping_patience is not None
                and self.config.early_stopping_restore_best
                and best_weights_callback is not None
                and best_weights_callback.best_state_dict is not None
            ):
                self.module.network.load_state_dict(best_weights_callback.best_state_dict)

        self.module = self.module.to(self._device)

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

        self._ensure_fitted()

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

        self._ensure_fitted()

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
        return "CFA"

    def additional_info(self) -> Dict:
        return {
            "threshold": self._threshold,
            "backbone": self.config.backbone_name,
            "feature_layers": self.module.network.return_nodes,
            "input_size": self.config.input_size,
            "pretrained_backbone": self.config.pretrained_backbone,
            "freeze_backbone": self.config.freeze_backbone,
            "gamma_c": self.config.gamma_c,
            "gamma_d": self.config.gamma_d,
            "k_neighbors": self.config.k_neighbors,
            "repulsion_neighbors": self.config.repulsion_neighbors,
            "descriptor_out_channels": self.module.network.descriptor_out_channels,
            "memory_bank_shape": tuple(int(dimension) for dimension in self.module.network.memory_bank.shape),
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "score_smoothing_kernel": self.config.score_smoothing_kernel,
            "score_smoothing_sigma": self.config.score_smoothing_sigma,
            "score_mode": self.config.score_mode,
            "threshold_quantile": self.config.threshold_quantile,
            "last_loss": self._last_loss,
        }


class CFAModule(pl.LightningModule):
    def __init__(
        self,
        network: CFAArchitecture,
        learning_rate: float,
        weight_decay: float,
        use_amsgrad: bool,
    ):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_amsgrad = use_amsgrad

        self.save_hyperparameters(ignore=["network"])

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = self.network.forward_train(x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        parameters = [parameter for parameter in self.network.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.use_amsgrad,
        )
