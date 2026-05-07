from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from pyclad.models.rd4ad.standard import build_decoder, build_encoder_and_bn, supported_backbone_names


class RD4ADArchitecture(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        input_size: tuple[int, int],
        pretrained_encoder: bool,
        freeze_encoder: bool,
        score_smoothing_kernel: int,
        score_smoothing_sigma: float,
    ):
        super().__init__()

        if backbone_name not in supported_backbone_names():
            raise ValueError(
                f"Unsupported RD4AD backbone '{backbone_name}'. Supported backbones: {', '.join(supported_backbone_names())}"
            )

        self.input_size = input_size
        self.freeze_encoder = freeze_encoder
        self.score_smoothing_kernel = score_smoothing_kernel
        self.score_smoothing_sigma = score_smoothing_sigma

        self.encoder, self.bn = build_encoder_and_bn(backbone_name=backbone_name, pretrained=pretrained_encoder)
        self.decoder = build_decoder(backbone_name=backbone_name)

        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        else:
            self.encoder.train(mode)
        self.bn.train(mode)
        self.decoder.train(mode)
        return self

    def _encode(self, batch: torch.Tensor) -> list[torch.Tensor]:
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(batch)
        return self.encoder(batch)

    def _forward_features(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        teacher_features = self._encode(batch)
        bottleneck_features = self.bn(teacher_features)
        student_features = self.decoder(bottleneck_features)
        return teacher_features, bottleneck_features, student_features

    def forward_train(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        if not self.training:
            self.train()
        return self._forward_features(batch)

    def forward(self, batch: torch.Tensor):
        teacher_features, bottleneck_features, student_features = self._forward_features(batch)
        if self.training:
            return teacher_features, bottleneck_features, student_features
        return self.post_process(teacher_features, student_features)

    def post_process(
        self,
        teacher_features: list[torch.Tensor],
        student_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anomaly_map: torch.Tensor | None = None

        for teacher_feature, student_feature in zip(teacher_features, student_features):
            component_map = 1.0 - F.cosine_similarity(student_feature, teacher_feature, dim=1)
            component_map = component_map.unsqueeze(1)
            component_map = F.interpolate(
                component_map,
                size=self.input_size,
                mode="bilinear",
                align_corners=True,
            )

            if anomaly_map is None:
                anomaly_map = component_map
            else:
                anomaly_map = anomaly_map + component_map

        if anomaly_map is None:
            raise ValueError("RD4AD received empty feature lists during post-processing")

        if self.score_smoothing_kernel > 1 and self.score_smoothing_sigma > 0.0:
            sigma = [self.score_smoothing_sigma, self.score_smoothing_sigma]
            kernel = [self.score_smoothing_kernel, self.score_smoothing_kernel]
            anomaly_map = gaussian_blur(anomaly_map, kernel_size=kernel, sigma=sigma)

        anomaly_scores = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1).values
        return anomaly_map[:, 0], anomaly_scores

    @staticmethod
    def cosine_loss(teacher_features: list[torch.Tensor], student_features: list[torch.Tensor]) -> torch.Tensor:
        loss = teacher_features[0].new_tensor(0.0)
        cosine_similarity = torch.nn.CosineSimilarity(dim=1)

        for teacher_feature, student_feature in zip(teacher_features, student_features):
            loss = loss + torch.mean(
                1.0
                - cosine_similarity(
                    teacher_feature.view(teacher_feature.shape[0], -1),
                    student_feature.view(student_feature.shape[0], -1),
                )
            )

        return loss
