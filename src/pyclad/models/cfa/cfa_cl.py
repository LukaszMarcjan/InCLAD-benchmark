from __future__ import annotations

import inspect
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from pyclad.models.cfa.cfa import CFA
from pyclad.models.vision.utils import BestWeightsCallback, to_float, trainer_device_config

class ContinualCFA(CFA):
    """CFA with continual-learning extensions.

    Design
    ------
    Each task gets its own *inference bank*: a CFA memory bank built from
    that task's training images using the **current** (post-training)
    descriptor.  At inference time every image is assigned to the task whose
    bank gives the lowest anomaly score, and that bank is used to produce
    the final score and segmentation map.

    During training the model uses a single *training bank* (initialised from
    the concatenation of the current task and any replay data) as the fixed
    hypersphere target for the soft-boundary CFA loss — exactly as in the
    naive single-task baseline.  After the descriptor is updated, all
    per-task inference banks are rebuilt with the new descriptor weights so
    they stay in sync.

    Bank management is delegated to the companion
    :class:`~pyclad.strategies.vision.cfa_cl.CFACLStrategy`.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._task_inference_banks: List[torch.Tensor] = []

    def begin_continual_run(self) -> None:
        """Reset internal CL state; call once before the first task."""
        self.reset_continual_state()

    def reset_continual_state(self) -> None:
        self._task_inference_banks = []
        self.module = self.module.to(self._device)
        self.module.network.memory_bank = torch.empty((0, 0), dtype=torch.float32, device=self._device)
        self._threshold = self.config.threshold
        self._last_loss = None

    def initialize_training_bank(self, data: np.ndarray) -> None:
        """Initialise ``module.network.memory_bank`` from *data* using the current descriptor.

        This bank is used as the fixed hypersphere target during
        :meth:`fit_descriptor_only`.  It is later superseded by the per-task
        inference banks built in :meth:`rebuild_task_inference_banks`.
        """
        if len(data) == 0:
            return
        loader = self._prepare_batches(data, shuffle=False)
        self.module.network.initialize_memory_bank(loader, device=self._device)

    def rebuild_task_inference_banks(self, task_samples: List[np.ndarray]) -> None:
        """Rebuild all per-task inference banks using the current (post-training) descriptor.

        For each task, a fresh bank is built from that task's stored training
        images.  Because the descriptor is updated after every ``learn()``
        call, this ensures that all banks live in the same embedding space.

        After rebuilding, ``module.network.memory_bank`` is set to the last
        task's bank so that :meth:`_ensure_fitted` and threshold calibration
        continue to work.
        """
        self._task_inference_banks = []
        for samples in task_samples:
            loader = self._prepare_batches(samples, shuffle=False)
            self.module.network.initialize_memory_bank(loader, device=self._device)
            # Clone so each stored bank is independent.
            self._task_inference_banks.append(self.module.network.memory_bank.clone())

        # Leave memory_bank pointing at the last task's bank so that
        # _ensure_fitted() passes and update_threshold_from_reference_data()
        # (which calls the overridden score_data) can run.
        if self._task_inference_banks:
            self.module.network.memory_bank = self._task_inference_banks[-1]

    def _score_and_maps_with_bank(
        self,
        data: np.ndarray,
        bank: torch.Tensor,
        resize_to_input: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(anomaly_scores, score_maps)`` for *data* scored against *bank*.

        Temporarily swaps ``module.network.memory_bank`` to *bank*, runs the
        forward inference pass, then restores the original bank.
        """
        target_size = self._preprocessor.spatial_size(data)
        all_scores: list[np.ndarray] = []
        all_maps: list[np.ndarray] = []

        old_bank = self.module.network.memory_bank
        try:
            self.module.network.memory_bank = bank.to(self._device)
            for (batch_x,) in self._prepare_batches(data, shuffle=False):
                batch_maps, batch_scores = self._forward_inference(
                    batch_x.to(self._device, dtype=torch.float32)
                )
                if resize_to_input:
                    batch_maps = self._resize_maps(batch_maps, output_size=target_size)
                all_scores.append(batch_scores.detach().cpu().numpy().astype(np.float32, copy=False))
                all_maps.append(batch_maps.detach().cpu().numpy().astype(np.float32, copy=False))
        finally:
            self.module.network.memory_bank = old_bank

        scores = np.concatenate(all_scores, axis=0) if all_scores else np.asarray([], dtype=np.float32)
        maps = np.concatenate(all_maps, axis=0) if all_maps else np.asarray([], dtype=np.float32)
        return scores, maps

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)
        if not self._task_inference_banks:
            return super().score_data(data)

        # Score against each task bank; return the minimum per image.
        all_scores = np.stack(
            [self._score_and_maps_with_bank(data, bank)[0] for bank in self._task_inference_banks],
            axis=0,
        )  # (n_tasks, n_samples)
        return all_scores.min(axis=0)  # (n_samples,)

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)
        if not self._task_inference_banks:
            return super().score_maps(data, resize_to_input=resize_to_input)

        # Run one forward pass per task bank (scores + maps in one shot).
        results = [
            self._score_and_maps_with_bank(data, bank, resize_to_input=resize_to_input)
            for bank in self._task_inference_banks
        ]
        all_scores = np.stack([r[0] for r in results], axis=0)  # (n_tasks, n_samples)
        all_maps = np.stack([r[1] for r in results], axis=0)    # (n_tasks, n_samples, H, W)
        selected_task_indices = np.argmin(all_scores, axis=0)   # (n_samples,)
        return all_maps[selected_task_indices, np.arange(len(data))]

    def fit_descriptor_only(self, data: np.ndarray) -> None:
        """Train the descriptor network on *data* using the fixed training bank.

        The memory bank must already be initialised (via
        :meth:`initialize_training_bank`) before calling this method.
        """
        if len(data) == 0:
            return
        self._ensure_fitted()

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

    def update_threshold_from_reference_data(self, data: np.ndarray) -> None:
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
            return
        if len(data) == 0:
            self._threshold = 0.0
            return
        scores = self.score_data(data)
        self._threshold = 0.0 if len(scores) == 0 else float(np.quantile(scores, self.config.threshold_quantile))

    def continual_state_info(self) -> Dict:
        return {
            "continual_state": "running" if self._task_inference_banks else "empty",
            "continual_num_tasks": len(self._task_inference_banks),
            "continual_task_bank_sizes": [int(b.shape[1]) for b in self._task_inference_banks],
        }

    def additional_info(self) -> Dict:
        return {**super().additional_info(), **self.continual_state_info()}
