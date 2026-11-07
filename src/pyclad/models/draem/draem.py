from __future__ import annotations

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

from pyclad.models.draem.config import DRAEMConfig
from pyclad.models.draem.loss import DRAEMLoss
from pyclad.models.draem.simulator import SyntheticAnomalyDataset, load_source_images
from pyclad.models.draem.standard import DiscriminativeSubNetwork, ReconstructiveSubNetwork, weights_init
from pyclad.models.draem.variant import BackboneReconstructiveSubNetwork
from pyclad.models.model import Model
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import (
    BestWeightsCallback,
    resolve_device,
    to_float,
    trainer_device_config,
)


class DRAEMArchitecture(nn.Module):
    def __init__(self, reconstructive_subnetwork: nn.Module, discriminative_subnetwork: nn.Module):
        super().__init__()
        self.reconstructive_subnetwork = reconstructive_subnetwork
        self.discriminative_subnetwork = discriminative_subnetwork

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction = self.reconstructive_subnetwork(x)
        segmentation_logits = self.discriminative_subnetwork(torch.cat([reconstruction, x], dim=1))
        return reconstruction, segmentation_logits


def build_architecture(config: DRAEMConfig) -> nn.Module:
    if config.variant == "standard":
        reconstructive_subnetwork = ReconstructiveSubNetwork(
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            base_width=config.reconstructive_base_width,
        )
    elif config.variant == "backbone":
        reconstructive_subnetwork = BackboneReconstructiveSubNetwork(
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            input_size=config.input_size,
            backbone_name=config.backbone_name,
            return_nodes=config.backbone_return_nodes,
            pretrained=config.pretrained_backbone,
            freeze=config.freeze_backbone,
            decoder_channels=config.backbone_decoder_channels,
            normalize_mean=config.backbone_normalize_mean,
            normalize_std=config.backbone_normalize_std,
        )
    else:
        raise ValueError(f"Unsupported DRAEM variant '{config.variant}'")

    discriminative_subnetwork = DiscriminativeSubNetwork(
        in_channels=config.in_channels * 2,
        out_channels=2,
        base_width=config.discriminative_base_width,
    )

    if config.initialize_weights:
        if config.variant == "standard":
            reconstructive_subnetwork.apply(weights_init)
        elif hasattr(reconstructive_subnetwork, "initialize_decoder_weights"):
            reconstructive_subnetwork.initialize_decoder_weights()
        discriminative_subnetwork.apply(weights_init)

    return DRAEMArchitecture(
        reconstructive_subnetwork=reconstructive_subnetwork,
        discriminative_subnetwork=discriminative_subnetwork,
    )


class DRAEM(Model):
    def __init__(self, config: Optional[DRAEMConfig] = None):
        self.config = config or DRAEMConfig()

        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=self.config.in_channels,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )

        network = build_architecture(self.config)
        self.module = DRAEMModule(
            network=network,
            reconstruction_l2_weight=self.config.reconstruction_l2_weight,
            reconstruction_ssim_weight=self.config.reconstruction_ssim_weight,
            reconstruction_ssim_window=self.config.reconstruction_ssim_window,
            segmentation_loss_weight=self.config.segmentation_loss_weight,
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            use_lr_scheduler=self.config.use_lr_scheduler,
            lr_milestones=self._resolve_lr_milestones(self.config),
            lr_gamma=self.config.lr_gamma,
        )

        self._threshold = self.config.threshold
        self._last_loss: Optional[float] = None
        self._source_images_count = 0

    @staticmethod
    def _resolve_lr_milestones(config: DRAEMConfig) -> tuple[int, ...]:
        if config.lr_milestones is not None:
            return tuple(sorted(set(int(milestone) for milestone in config.lr_milestones if milestone > 0)))

        if config.epochs <= 1:
            return ()

        default_milestones = {max(1, int(config.epochs * 0.8)), max(1, int(config.epochs * 0.9))}
        return tuple(sorted(default_milestones))

    def _prepare_inference_batches(self, data: np.ndarray) -> DataLoader:
        x_t = self._preprocessor.transform(data)
        dataset = TensorDataset(x_t)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _build_training_dataloader(self, data: np.ndarray) -> DataLoader:
        clean_images = self._preprocessor.transform(data)
        source_images = load_source_images(
            input_size=self.config.input_size,
            in_channels=self.config.in_channels,
            source_root=self.config.anomaly_source_root,
            source_paths=self.config.anomaly_source_paths,
        )
        self._source_images_count = len(source_images)

        train_dataset = SyntheticAnomalyDataset(
            clean_images=clean_images,
            source_images=source_images,
            allow_random_color_fallback=self.config.allow_random_color_fallback,
            texture_augmentation=self.config.texture_augmentation,
            texture_augmentation_count=self.config.texture_augmentation_count,
            perlin_scale_min=self.config.perlin_scale_min,
            perlin_scale_max=self.config.perlin_scale_max,
            perlin_threshold=self.config.perlin_threshold,
            anomaly_blend_range=self.config.anomaly_blend_range,
            anomaly_probability=self.config.anomaly_probability,
            rotation_degrees=self.config.rotation_degrees,
            rotation_probability=self.config.rotation_probability,
            random_seed=self.config.random_seed,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
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

        trainer.fit(self.module, train_dataloaders=self._build_training_dataloader(data))
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

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        self.module = self.module.to(self._device)
        self.module.eval()

        reconstructions: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_x,) in self._prepare_inference_batches(data):
                x = batch_x.to(self._device, dtype=torch.float32)
                reconstruction, _ = self.module(x)
                reconstructions.append(reconstruction.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(reconstructions, axis=0) if reconstructions else np.asarray([], dtype=np.float32)

    def _smooth_maps(self, score_maps: torch.Tensor) -> torch.Tensor:
        if self.config.score_smoothing_kernel <= 1:
            return score_maps
        padding = self.config.score_smoothing_kernel // 2
        return F.avg_pool2d(
            score_maps,
            kernel_size=self.config.score_smoothing_kernel,
            stride=1,
            padding=padding,
        )

    @staticmethod
    def _resize_maps(score_maps: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if tuple(score_maps.shape[-2:]) == tuple(output_size):
            return score_maps
        resized = F.interpolate(score_maps, size=output_size, mode="bilinear", align_corners=False)
        return resized

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True, smooth: bool = False) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        target_size = self._preprocessor.spatial_size(data)
        self.module = self.module.to(self._device)
        self.module.eval()
        all_maps: list[np.ndarray] = []

        with torch.no_grad():
            for (batch_x,) in self._prepare_inference_batches(data):
                x = batch_x.to(self._device, dtype=torch.float32)
                _, segmentation_logits = self.module(x)
                batch_maps = torch.softmax(segmentation_logits, dim=1)[:, 1:2, :, :]
                if smooth:
                    batch_maps = self._smooth_maps(batch_maps)
                if resize_to_input:
                    batch_maps = self._resize_maps(batch_maps, output_size=target_size)
                all_maps.append(batch_maps[:, 0].detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(all_maps, axis=0) if all_maps else np.asarray([], dtype=np.float32)

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)
        score_maps = self.score_maps(data, resize_to_input=False, smooth=True)
        return score_maps.max(axis=(1, 2)).astype(np.float32, copy=False)

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
        return "DRAEM"

    def additional_info(self) -> Dict:
        return {
            **self.config.model_dump(),
            "threshold": self._threshold,
            "anomaly_source_images_count": self._source_images_count,
            "device": str(self._device),
            "last_loss": self._last_loss,
            "uses_lightning": True,
        }


class DRAEMModule(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        reconstruction_l2_weight: float,
        reconstruction_ssim_weight: float,
        reconstruction_ssim_window: int,
        segmentation_loss_weight: float,
        focal_alpha: float | None,
        focal_gamma: float,
        learning_rate: float,
        weight_decay: float,
        use_lr_scheduler: bool,
        lr_milestones: tuple[int, ...],
        lr_gamma: float,
    ):
        super().__init__()
        self.network = network
        self.loss_fn = DRAEMLoss(
            reconstruction_l2_weight=reconstruction_l2_weight,
            reconstruction_ssim_weight=reconstruction_ssim_weight,
            reconstruction_ssim_window=reconstruction_ssim_window,
            segmentation_loss_weight=segmentation_loss_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma

        self.save_hyperparameters(ignore=["network"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        target, anomalous_input, anomaly_mask = batch
        reconstruction, segmentation_logits = self(anomalous_input)
        segmentation_probabilities = torch.softmax(segmentation_logits, dim=1)
        losses = self.loss_fn(
            target=target,
            reconstruction=reconstruction,
            segmentation_probabilities=segmentation_probabilities,
            anomaly_mask=anomaly_mask,
        )

        self.log("train_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_reconstruction_loss", losses["reconstruction_loss"], on_step=False, on_epoch=True)
        self.log("train_segmentation_loss", losses["segmentation_loss"], on_step=False, on_epoch=True)
        self.log("train_l2_loss", losses["l2_loss"], on_step=False, on_epoch=True)
        self.log("train_ssim_loss", losses["ssim_loss"], on_step=False, on_epoch=True)

        return losses["loss"]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            [
                {"params": self.network.reconstructive_subnetwork.parameters(), "lr": self.learning_rate},
                {"params": self.network.discriminative_subnetwork.parameters(), "lr": self.learning_rate},
            ],
            weight_decay=self.weight_decay,
        )

        if not self.use_lr_scheduler or len(self.lr_milestones) == 0:
            return optimizer

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.lr_milestones),
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
