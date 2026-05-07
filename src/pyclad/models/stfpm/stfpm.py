import inspect
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.models.stfpm.builder import build
from pyclad.models.stfpm.config import STFPMConfig
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import (
    BestWeightsCallback,
    resolve_device,
    to_float,
    trainer_device_config,
)


class STFPM(Model):
    def __init__(self, config: Optional[STFPMConfig] = None):
        self.config = config or STFPMConfig()

        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=3,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )

        network = build(self.config)
        self.module = STFPMModule(
            network=network,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        self._threshold = self.config.threshold
        self._last_loss: Optional[float] = None
        self._feature_map_size: Optional[tuple[int, int]] = None

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
    def _pair_distance_map(teacher_feature: torch.Tensor, student_feature: torch.Tensor) -> torch.Tensor:
        teacher_feature = F.normalize(teacher_feature, dim=1)
        student_feature = F.normalize(student_feature, dim=1)
        return torch.sum((teacher_feature - student_feature) ** 2, dim=1, keepdim=True)

    def _forward_feature_pyramid(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.module = self.module.to(self._device)
        self.module.eval()
        with torch.no_grad():
            return self.module(x)

    def _batch_score_maps(self, x: torch.Tensor) -> torch.Tensor:
        teacher_features, student_features = self._forward_feature_pyramid(x)
        reference_size = teacher_features[0].shape[-2:]
        self._feature_map_size = (int(reference_size[0]), int(reference_size[1]))

        aggregated_map: Optional[torch.Tensor] = None
        for teacher_feature, student_feature in zip(teacher_features, student_features):
            distance_map = self._pair_distance_map(teacher_feature, student_feature)
            if distance_map.shape[-2:] != reference_size:
                distance_map = F.interpolate(distance_map, size=reference_size, mode="bilinear", align_corners=False)

            if aggregated_map is None:
                aggregated_map = distance_map
            else:
                aggregated_map = aggregated_map * distance_map

        return aggregated_map[:, 0]

    @staticmethod
    def _resize_maps(score_maps: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if tuple(score_maps.shape[-2:]) == tuple(output_size):
            return score_maps
        resized = F.interpolate(score_maps[:, None, :, :], size=output_size, mode="bilinear", align_corners=False)
        return resized[:, 0]

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        target_size = self._preprocessor.spatial_size(data)
        all_maps: list[np.ndarray] = []

        for (batch_x,) in self._prepare_batches(data, shuffle=False):
            batch_maps = self._batch_score_maps(batch_x.to(self._device, dtype=torch.float32))
            if resize_to_input:
                batch_maps = self._resize_maps(batch_maps, output_size=target_size)
            all_maps.append(batch_maps.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(all_maps, axis=0) if all_maps else np.asarray([], dtype=np.float32)

    def _image_scores_from_maps(self, score_maps: np.ndarray) -> np.ndarray:
        if self.config.score_mode == "max":
            return score_maps.max(axis=(1, 2)).astype(np.float32, copy=False)
        return score_maps.mean(axis=(1, 2)).astype(np.float32, copy=False)

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
        return "STFPM"

    def additional_info(self) -> Dict:
        return {
            "threshold": self._threshold,
            "backbone": self.config.backbone_name,
            "feature_layers": self.module.network.return_nodes,
            "input_size": self.config.input_size,
            "pretrained_teacher": self.config.pretrained_teacher,
            "pretrained_student": self.config.pretrained_student,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "momentum": self.config.momentum,
            "weight_decay": self.config.weight_decay,
            "score_mode": self.config.score_mode,
            "threshold_quantile": self.config.threshold_quantile,
            "last_loss": self._last_loss,
        }


class STFPMModule(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["network"])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.network.teacher.eval()
        with torch.no_grad():
            teacher_features = self.network.teacher(x)
        student_features = self.network.student(x)
        return teacher_features, student_features

    @staticmethod
    def feature_loss(teacher_features: list[torch.Tensor], student_features: list[torch.Tensor]) -> torch.Tensor:
        loss = teacher_features[0].new_tensor(0.0)
        for teacher_feature, student_feature in zip(teacher_features, student_features):
            teacher_feature = F.normalize(teacher_feature, dim=1)
            student_feature = F.normalize(student_feature, dim=1)
            loss = loss + torch.sum((teacher_feature - student_feature) ** 2, dim=1).mean()
        return loss

    def training_step(self, batch, batch_idx):
        x = batch[0]
        teacher_features, student_features = self(x)
        loss = self.feature_loss(teacher_features, student_features)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.SGD(
            self.network.student.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
