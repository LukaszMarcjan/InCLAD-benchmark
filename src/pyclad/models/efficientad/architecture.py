from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import functional as TF

from pyclad.models.efficientad.standard import AutoEncoder


def reduce_tensor_elems(tensor: torch.Tensor, max_elems: int = 2**24) -> torch.Tensor:
    tensor = torch.flatten(tensor)
    if len(tensor) > max_elems:
        indices = torch.randperm(len(tensor), device=tensor.device)[:max_elems]
        tensor = tensor[indices]
    return tensor


class EfficientADArchitecture(nn.Module):
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        autoencoder: AutoEncoder,
        teacher_out_channels: int,
        padding: bool,
        pad_maps: bool,
        map_padding: int,
        student_hard_quantile: float,
        ae_augmentation_min: float,
        ae_augmentation_max: float,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.teacher_out_channels = teacher_out_channels
        self.padding = padding
        self.pad_maps = pad_maps
        self.map_padding = map_padding
        self.student_hard_quantile = student_hard_quantile
        self.ae_augmentation_min = ae_augmentation_min
        self.ae_augmentation_max = ae_augmentation_max

        self.register_buffer("teacher_mean", torch.zeros((1, teacher_out_channels, 1, 1), dtype=torch.float32))
        self.register_buffer("teacher_std", torch.ones((1, teacher_out_channels, 1, 1), dtype=torch.float32))
        self.register_buffer("teacher_stats_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("qa_st", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("qb_st", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("qa_ae", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("qb_ae", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("map_quantiles_ready", torch.tensor(False, dtype=torch.bool))

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def set_teacher_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.teacher_mean.copy_(mean.detach().to(device=self.teacher_mean.device, dtype=self.teacher_mean.dtype))
        std = std.detach().to(device=self.teacher_std.device, dtype=self.teacher_std.dtype)
        self.teacher_std.copy_(torch.clamp(std, min=1e-6))
        self.teacher_stats_ready.fill_(True)

    def set_map_quantiles(self, qa_st: torch.Tensor, qb_st: torch.Tensor, qa_ae: torch.Tensor, qb_ae: torch.Tensor) -> None:
        self.qa_st.copy_(qa_st.detach().to(device=self.qa_st.device, dtype=self.qa_st.dtype))
        self.qb_st.copy_(qb_st.detach().to(device=self.qb_st.device, dtype=self.qb_st.dtype))
        self.qa_ae.copy_(qa_ae.detach().to(device=self.qa_ae.device, dtype=self.qa_ae.dtype))
        self.qb_ae.copy_(qb_ae.detach().to(device=self.qb_ae.device, dtype=self.qb_ae.dtype))
        self.map_quantiles_ready.fill_(True)

    def _normalize_teacher_output(self, teacher_output: torch.Tensor) -> torch.Tensor:
        if bool(self.teacher_stats_ready.item()):
            return (teacher_output - self.teacher_mean) / self.teacher_std
        return teacher_output

    def _normalize_map(self, anomaly_map: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        return 0.1 * (anomaly_map - start) / torch.clamp(end - start, min=1e-6)

    def _choose_random_aug_image(self, batch: torch.Tensor) -> torch.Tensor:
        augmented_images: list[torch.Tensor] = []
        for image in batch:
            coefficient = float(
                torch.empty(1, device=image.device).uniform_(self.ae_augmentation_min, self.ae_augmentation_max).item()
            )
            choice = int(torch.randint(0, 3, (1,), device=image.device).item())
            if choice == 0:
                augmented = TF.adjust_brightness(image, coefficient)
            elif choice == 1:
                augmented = TF.adjust_contrast(image, coefficient)
            else:
                augmented = TF.adjust_saturation(image, coefficient)
            augmented_images.append(augmented.clamp(0.0, 1.0))
        return torch.stack(augmented_images, dim=0)

    def compute_student_teacher_distance(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            teacher_output = self._normalize_teacher_output(teacher_output)

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels], 2)
        return student_output, distance_st

    def compute_losses(
        self,
        batch: torch.Tensor,
        batch_penalty: Optional[torch.Tensor] = None,
        has_penalty: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        student_output, distance_st = self.compute_student_teacher_distance(batch)

        distance_st = reduce_tensor_elems(distance_st)
        hard_distance = torch.quantile(distance_st, self.student_hard_quantile)
        loss_hard = torch.mean(distance_st[distance_st >= hard_distance])

        if batch_penalty is not None and has_penalty is not None and bool(has_penalty.any().item()):
            selected_penalty = batch_penalty[has_penalty]
            student_output_penalty = self.student(selected_penalty)[:, : self.teacher_out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
        else:
            loss_penalty = batch.new_tensor(0.0)

        augmented_batch = self._choose_random_aug_image(batch)
        ae_output = self.autoencoder(augmented_batch, image_size=batch.shape[-2:])

        with torch.no_grad():
            teacher_output_aug = self.teacher(augmented_batch)
            teacher_output_aug = self._normalize_teacher_output(teacher_output_aug)

        student_output_ae = self.student(augmented_batch)[:, self.teacher_out_channels :]
        distance_ae = torch.pow(teacher_output_aug - ae_output, 2)
        distance_stae = torch.pow(ae_output - student_output_ae, 2)

        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_st = loss_hard + loss_penalty
        loss = loss_st + loss_ae + loss_stae

        return {
            "loss": loss,
            "student_teacher_loss": loss_st,
            "hard_loss": loss_hard,
            "penalty_loss": loss_penalty,
            "autoencoder_loss": loss_ae,
            "student_autoencoder_loss": loss_stae,
        }

    def component_maps(self, batch: torch.Tensor, normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        ae_output = self.autoencoder(batch, image_size=batch.shape[-2:])

        map_st = torch.mean(distance_st, dim=1, keepdim=True)
        map_ae = torch.mean(torch.pow(ae_output - student_output[:, self.teacher_out_channels :], 2), dim=1, keepdim=True)

        if normalize and bool(self.map_quantiles_ready.item()):
            map_st = self._normalize_map(map_st, self.qa_st, self.qb_st)
            map_ae = self._normalize_map(map_ae, self.qa_ae, self.qb_ae)

        return map_st, map_ae

    def anomaly_map(self, batch: torch.Tensor, normalize: bool = True, apply_padding: bool = True) -> torch.Tensor:
        map_st, map_ae = self.component_maps(batch, normalize=normalize)
        combined = 0.5 * map_st + 0.5 * map_ae
        if apply_padding and self.pad_maps and not self.padding and self.map_padding > 0:
            combined = F.pad(combined, (self.map_padding, self.map_padding, self.map_padding, self.map_padding))
        return combined
