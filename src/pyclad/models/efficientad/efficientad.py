from __future__ import annotations

import copy
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

from pyclad.models.efficientad.architecture import EfficientADArchitecture, reduce_tensor_elems
from pyclad.models.efficientad.builder import build
from pyclad.models.efficientad.config import EfficientADConfig
from pyclad.models.efficientad.simulator import EfficientADTrainDataset, load_penalty_image_paths
from pyclad.models.model import Model
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import resolve_device

_MONITOR_METRIC = "train_loss"


def _trainer_device_config(device: torch.device) -> tuple[str, int | list[int]]:
    if device.type == "cpu":
        return "cpu", 1
    if device.type == "cuda":
        if device.index is None:
            return "gpu", 1
        return "gpu", [device.index]
    if device.type == "mps":
        return "mps", 1
    return "auto", 1


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class _BestWeightsCallback(pl.Callback):
    def __init__(self, monitor: str, min_delta: float):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.best_state_dict: Optional[dict[str, torch.Tensor]] = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        current_loss = _to_float(trainer.callback_metrics.get(self.monitor))
        if current_loss is None:
            return
        if self.best_loss is None or (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.best_state_dict = copy.deepcopy(pl_module.network.state_dict())


@torch.no_grad()
def _compute_teacher_channel_stats(
    network: EfficientADArchitecture,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    network = network.to(device).eval()

    channel_sum: Optional[torch.Tensor] = None
    channel_sum_sqr: Optional[torch.Tensor] = None
    n_elements = 0

    for (batch_x,) in dataloader:
        x = batch_x.to(device, dtype=torch.float32)
        teacher_output = network.teacher(x)

        if channel_sum is None or channel_sum_sqr is None:
            channel_sum = torch.zeros((teacher_output.shape[1],), dtype=torch.float32, device=device)
            channel_sum_sqr = torch.zeros((teacher_output.shape[1],), dtype=torch.float32, device=device)

        channel_sum += torch.sum(teacher_output, dim=(0, 2, 3))
        channel_sum_sqr += torch.sum(teacher_output**2, dim=(0, 2, 3))
        n_elements += int(teacher_output[:, 0].numel())

    if channel_sum is None or channel_sum_sqr is None or n_elements == 0:
        raise ValueError("Cannot compute teacher channel statistics from an empty dataloader")

    channel_mean = channel_sum / n_elements
    channel_var = torch.clamp(channel_sum_sqr / n_elements - channel_mean**2, min=1e-12)
    channel_std = torch.sqrt(channel_var)

    return channel_mean[None, :, None, None], channel_std[None, :, None, None]


@torch.no_grad()
def _compute_map_quantiles(
    network: EfficientADArchitecture,
    dataloader: DataLoader,
    device: torch.device,
    quantile_low: float,
    quantile_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    network = network.to(device).eval()

    maps_st: list[torch.Tensor] = []
    maps_ae: list[torch.Tensor] = []

    for (batch_x,) in dataloader:
        x = batch_x.to(device, dtype=torch.float32)
        map_st, map_ae = network.component_maps(x, normalize=False)
        maps_st.append(map_st)
        maps_ae.append(map_ae)

    if len(maps_st) == 0 or len(maps_ae) == 0:
        raise ValueError("Cannot compute map quantiles from an empty dataloader")

    maps_st_flat = reduce_tensor_elems(torch.cat(maps_st, dim=0))
    maps_ae_flat = reduce_tensor_elems(torch.cat(maps_ae, dim=0))

    qa_st = torch.quantile(maps_st_flat, quantile_low)
    qb_st = torch.quantile(maps_st_flat, quantile_high)
    qa_ae = torch.quantile(maps_ae_flat, quantile_low)
    qb_ae = torch.quantile(maps_ae_flat, quantile_high)

    return qa_st, qb_st, qa_ae, qb_ae


class EfficientAD(Model):
    def __init__(self, config: Optional[EfficientADConfig] = None):
        self.config = config or EfficientADConfig()
        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=self.config.in_channels,
        )

        network = build(self.config)
        self.module = EfficientADModule(
            network=network,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            use_lr_scheduler=self.config.use_lr_scheduler,
            lr_step_size=max(1, int(self.config.epochs * self.config.lr_decay_fraction)) if self.config.epochs > 0 else 1,
            lr_gamma=self.config.lr_decay_gamma,
        )

        self._threshold = self.config.threshold
        self._last_loss: Optional[float] = None
        self._penalty_images_count = 0

    def _prepare_inference_batches(self, data: np.ndarray) -> DataLoader:
        x_t = self._preprocessor.transform(data)
        dataset = TensorDataset(x_t)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _normalization_loader(self, clean_images: torch.Tensor) -> DataLoader:
        return DataLoader(
            TensorDataset(clean_images),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _build_training_dataloader(self, clean_images: torch.Tensor) -> DataLoader:
        penalty_image_paths = load_penalty_image_paths(
            source_root=self.config.penalty_source_root,
            source_paths=self.config.penalty_source_paths,
        )
        self._penalty_images_count = len(penalty_image_paths)

        train_dataset = EfficientADTrainDataset(
            clean_images=clean_images,
            input_size=self.config.input_size,
            penalty_image_paths=penalty_image_paths,
            penalty_resize_scale=self.config.penalty_resize_scale,
            penalty_grayscale_probability=self.config.penalty_grayscale_probability,
            random_seed=self.config.random_seed,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

    @staticmethod
    def _resize_maps(score_maps: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if tuple(score_maps.shape[-2:]) == tuple(output_size):
            return score_maps
        resized = F.interpolate(score_maps, size=output_size, mode="bilinear", align_corners=False)
        return resized

    def fit(self, data: np.ndarray):
        if len(data) == 0 or self.config.epochs == 0:
            return

        clean_images = self._preprocessor.transform(data)
        normalization_loader = self._normalization_loader(clean_images)
        training_loader = self._build_training_dataloader(clean_images)

        self.module = self.module.to(self._device)
        teacher_mean, teacher_std = _compute_teacher_channel_stats(
            network=self.module.network,
            dataloader=normalization_loader,
            device=self._device,
        )
        self.module.network.set_teacher_normalization(teacher_mean, teacher_std)

        callbacks: list[pl.Callback] = []
        best_weights_callback: Optional[_BestWeightsCallback] = None

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
                best_weights_callback = _BestWeightsCallback(
                    monitor=_MONITOR_METRIC,
                    min_delta=float(self.config.early_stopping_min_delta),
                )
                callbacks.append(best_weights_callback)

        accelerator, devices = _trainer_device_config(self._device)
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

        trainer.fit(self.module, train_dataloaders=training_loader)
        self.module = self.module.to(self._device)
        self._last_loss = _to_float(trainer.callback_metrics.get(_MONITOR_METRIC))

        if (
            self.config.early_stopping_patience is not None
            and self.config.early_stopping_restore_best
            and best_weights_callback is not None
            and best_weights_callback.best_state_dict is not None
        ):
            self.module.network.load_state_dict(best_weights_callback.best_state_dict)

        qa_st, qb_st, qa_ae, qb_ae = _compute_map_quantiles(
            network=self.module.network,
            dataloader=normalization_loader,
            device=self._device,
            quantile_low=self.config.map_quantile_low,
            quantile_high=self.config.map_quantile_high,
        )
        self.module.network.set_map_quantiles(qa_st, qb_st, qa_ae, qb_ae)

        scores = None if self.config.threshold is not None else self.score_data(data)
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
        elif scores is None or len(scores) == 0:
            self._threshold = 0.0
        else:
            self._threshold = float(np.quantile(scores, self.config.threshold_quantile))

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        target_size = self._preprocessor.spatial_size(data)
        self.module = self.module.to(self._device).eval()
        all_maps: list[np.ndarray] = []

        with torch.no_grad():
            for (batch_x,) in self._prepare_inference_batches(data):
                x = batch_x.to(self._device, dtype=torch.float32)
                batch_maps = self.module.network.anomaly_map(x, normalize=True, apply_padding=True)
                if resize_to_input:
                    batch_maps = self._resize_maps(batch_maps, output_size=target_size)
                all_maps.append(batch_maps[:, 0].detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(all_maps, axis=0) if all_maps else np.asarray([], dtype=np.float32)

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        self.module = self.module.to(self._device).eval()
        all_scores: list[np.ndarray] = []

        with torch.no_grad():
            for (batch_x,) in self._prepare_inference_batches(data):
                x = batch_x.to(self._device, dtype=torch.float32)
                batch_maps = self.module.network.anomaly_map(x, normalize=True, apply_padding=False)[:, 0]
                if self.config.score_mode == "mean":
                    batch_scores = batch_maps.mean(dim=(1, 2))
                else:
                    batch_scores = torch.amax(batch_maps, dim=(1, 2))
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
        return "EfficientAD"

    def additional_info(self) -> Dict:
        return {
            "threshold": self._threshold,
            "input_size": self.config.input_size,
            "model_size": self.config.model_size,
            "teacher_out_channels": self.config.teacher_out_channels,
            "padding": self.config.padding,
            "pad_maps": self.config.pad_maps,
            "map_padding": self.config.map_padding,
            "teacher_weights_path": self.config.teacher_weights_path,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "use_lr_scheduler": self.config.use_lr_scheduler,
            "lr_decay_fraction": self.config.lr_decay_fraction,
            "lr_decay_gamma": self.config.lr_decay_gamma,
            "student_hard_quantile": self.config.student_hard_quantile,
            "map_quantile_low": self.config.map_quantile_low,
            "map_quantile_high": self.config.map_quantile_high,
            "score_mode": self.config.score_mode,
            "threshold_quantile": self.config.threshold_quantile,
            "penalty_source_root": self.config.penalty_source_root,
            "penalty_source_paths": self.config.penalty_source_paths,
            "penalty_source_images_count": self._penalty_images_count,
            "penalty_resize_scale": self.config.penalty_resize_scale,
            "penalty_grayscale_probability": self.config.penalty_grayscale_probability,
            "ae_augmentation_min": self.config.ae_augmentation_min,
            "ae_augmentation_max": self.config.ae_augmentation_max,
            "teacher_stats_ready": bool(self.module.network.teacher_stats_ready.item()),
            "map_quantiles_ready": bool(self.module.network.map_quantiles_ready.item()),
            "device": str(self._device),
            "last_loss": self._last_loss,
            "uses_lightning": True,
        }


class EfficientADModule(pl.LightningModule):
    def __init__(
        self,
        network: EfficientADArchitecture,
        learning_rate: float,
        weight_decay: float,
        use_lr_scheduler: bool,
        lr_step_size: int,
        lr_gamma: float,
    ):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        self.save_hyperparameters(ignore=["network"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.anomaly_map(x, normalize=True, apply_padding=True)

    def training_step(self, batch, batch_idx):
        del batch_idx
        clean_image, penalty_image, has_penalty = batch
        losses = self.network.compute_losses(
            batch=clean_image,
            batch_penalty=penalty_image,
            has_penalty=has_penalty,
        )

        self.log("train_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_student_teacher_loss", losses["student_teacher_loss"], on_step=False, on_epoch=True)
        self.log("train_hard_loss", losses["hard_loss"], on_step=False, on_epoch=True)
        self.log("train_penalty_loss", losses["penalty_loss"], on_step=False, on_epoch=True)
        self.log("train_autoencoder_loss", losses["autoencoder_loss"], on_step=False, on_epoch=True)
        self.log(
            "train_student_autoencoder_loss",
            losses["student_autoencoder_loss"],
            on_step=False,
            on_epoch=True,
        )

        return losses["loss"]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            list(self.network.student.parameters()) + list(self.network.autoencoder.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if not self.use_lr_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, self.lr_step_size),
            gamma=self.lr_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
