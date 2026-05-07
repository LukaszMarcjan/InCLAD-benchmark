from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from pyclad.models.cfa.standard import Descriptor
from pyclad.models.vision.backbones import TorchvisionFeatureExtractor

_SUPPORTED_BACKBONES: dict[str, tuple[str, ...]] = {
    "resnet18": ("layer2", "layer3"),
    "resnet34": ("layer2", "layer3"),
    "resnet50": ("layer2", "layer3"),
    "wide_resnet50_2": ("layer2", "layer3"),
    "mobilenet_v2": ("features.6", "features.13"),
    "efficientnet_b0": ("features.3", "features.5"),
    "efficientnet_b1": ("features.3", "features.5"),
    "efficientnet_b2": ("features.3", "features.5"),
    "efficientnet_b3": ("features.3", "features.5"),
    "efficientnet_b4": ("features.3", "features.5"),
    "efficientnet_v2_s": ("features.3", "features.5"),
    "efficientnet_v2_m": ("features.3", "features.5"),
    "efficientnet_v2_l": ("features.3", "features.5"),
}


def supported_backbone_names() -> tuple[str, ...]:
    return tuple(_SUPPORTED_BACKBONES.keys())


def default_cfa_return_nodes(backbone_name: str) -> tuple[str, ...]:
    if backbone_name not in _SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported CFA backbone '{backbone_name}'. Supported backbones: {', '.join(supported_backbone_names())}"
        )
    return _SUPPORTED_BACKBONES[backbone_name]


class CFAArchitecture(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: tuple[int, int],
        backbone_name: str,
        backbone_return_nodes: tuple[str, ...] | None,
        pretrained_backbone: bool,
        freeze_backbone: bool,
        gamma_c: int,
        gamma_d: int,
        k_neighbors: int,
        repulsion_neighbors: int,
        nu: float,
        alpha: float,
        radius_init: float,
        score_smoothing_kernel: int,
        score_smoothing_sigma: float,
        score_mode: str,
        random_seed: int,
    ):
        super().__init__()

        try:
            return_nodes = backbone_return_nodes or default_cfa_return_nodes(backbone_name)
        except ValueError:
            if backbone_return_nodes is None:
                raise
            return_nodes = backbone_return_nodes

        self.input_size = tuple(input_size)
        self.freeze_backbone = freeze_backbone
        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.k_neighbors = k_neighbors
        self.repulsion_neighbors = repulsion_neighbors
        self.nu = nu
        self.alpha = alpha
        self.score_smoothing_kernel = score_smoothing_kernel
        self.score_smoothing_sigma = score_smoothing_sigma
        self.score_mode = score_mode
        self.random_seed = random_seed
        self.return_nodes = tuple(return_nodes)

        self.feature_extractor = TorchvisionFeatureExtractor(
            backbone_name=backbone_name,
            return_nodes=self.return_nodes,
            pretrained=pretrained_backbone,
            freeze=freeze_backbone,
        )
        self.feature_shapes = tuple(self._infer_feature_shapes(input_size=self.input_size, in_channels=in_channels))
        self.total_feature_channels = int(sum(shape[0] for shape in self.feature_shapes))
        self.descriptor = Descriptor(
            gamma_d=self.gamma_d,
            feature_map_channels=self.total_feature_channels,
            backbone_name=backbone_name,
        )
        self.radius = nn.Parameter(torch.full((1,), float(radius_init), dtype=torch.float32))

        self.register_buffer("memory_bank", torch.empty((0, 0), dtype=torch.float32), persistent=True)
        self._descriptor_map_size = (
            int(self.feature_shapes[0][1]),
            int(self.feature_shapes[0][2]),
        )

    @property
    def descriptor_out_channels(self) -> int:
        return int(self.descriptor.output_channels)

    @property
    def has_memory_bank(self) -> bool:
        return self.memory_bank.numel() > 0

    def _infer_feature_shapes(self, input_size: tuple[int, int], in_channels: int) -> list[tuple[int, int, int]]:
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        with torch.no_grad():
            try:
                device = next(self.feature_extractor.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            dummy = torch.zeros((1, in_channels, input_size[0], input_size[1]), dtype=torch.float32, device=device)
            features = self.feature_extractor(dummy)
        self.feature_extractor.train(was_training)
        return [tuple(int(dimension) for dimension in feature.shape[1:]) for feature in features]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train(mode)
        self.descriptor.train(mode)
        return self

    def _extract_features(self, batch: torch.Tensor) -> list[torch.Tensor]:
        if self.freeze_backbone:
            with torch.no_grad():
                return self.feature_extractor(batch)
        return self.feature_extractor(batch)

    def describe(self, batch: torch.Tensor) -> torch.Tensor:
        return self.descriptor(self._extract_features(batch))

    @staticmethod
    def _flatten_descriptors(descriptor_map: torch.Tensor) -> torch.Tensor:
        return descriptor_map.permute(0, 2, 3, 1).reshape(descriptor_map.shape[0], -1, descriptor_map.shape[1])

    def _squared_distances(self, flat_descriptors: torch.Tensor) -> torch.Tensor:
        if not self.has_memory_bank:
            raise RuntimeError("CFA memory bank is not initialized. Call fit() before scoring or predicting.")

        descriptor_norm = torch.sum(flat_descriptors**2, dim=2, keepdim=True)
        center_norm = torch.sum(self.memory_bank**2, dim=0, keepdim=True)
        cross_term = 2.0 * torch.matmul(flat_descriptors, self.memory_bank)
        return torch.clamp(descriptor_norm + center_norm - cross_term, min=1e-12)

    def soft_boundary_loss(self, flat_descriptors: torch.Tensor) -> torch.Tensor:
        distances = self._squared_distances(flat_descriptors)
        total_neighbors = min(distances.shape[-1], self.k_neighbors + self.repulsion_neighbors)
        nearest = distances.topk(total_neighbors, largest=False).values

        k_neighbors = min(self.k_neighbors, nearest.shape[-1])
        attraction = nearest[:, :, :k_neighbors] - self.radius.square()
        attraction_loss = (1.0 / self.nu) * torch.mean(torch.maximum(torch.zeros_like(attraction), attraction))

        if nearest.shape[-1] > self.repulsion_neighbors:
            repulsion = self.radius.square() - nearest[:, :, self.repulsion_neighbors :]
            repulsion_loss = (1.0 / self.nu) * torch.mean(
                torch.maximum(torch.zeros_like(repulsion), repulsion - self.alpha)
            )
        else:
            repulsion_loss = attraction_loss.new_tensor(0.0)

        return attraction_loss + repulsion_loss

    @torch.no_grad()
    def initialize_memory_bank(self, dataloader, device: torch.device) -> None:
        was_training = self.training
        self.to(device)
        self.eval()

        centroid: torch.Tensor | None = None
        n_batches = 0

        for batch in dataloader:
            batch_x = batch[0] if isinstance(batch, (tuple, list)) else batch
            descriptor_map = self.describe(batch_x.to(device, dtype=torch.float32))
            batch_centroid = descriptor_map.mean(dim=0, keepdim=True)

            if centroid is None:
                centroid = batch_centroid
            else:
                centroid = ((centroid * n_batches) + batch_centroid) / (n_batches + 1)
            n_batches += 1

        if centroid is None:
            raise ValueError("Cannot initialize CFA memory bank from an empty dataloader")

        self._descriptor_map_size = (int(centroid.shape[-2]), int(centroid.shape[-1]))
        patch_centers = centroid.permute(0, 2, 3, 1).reshape(-1, centroid.shape[1]).detach()

        if self.gamma_c > 1 and len(patch_centers) > 1:
            n_clusters = max(1, len(patch_centers) // self.gamma_c)
            n_clusters = min(n_clusters, len(patch_centers))
            kmeans = KMeans(n_clusters=n_clusters, max_iter=3000, random_state=self.random_seed, n_init="auto")
            cluster_centers = kmeans.fit(patch_centers.cpu().numpy()).cluster_centers_.astype(np.float32, copy=False)
            memory_bank = torch.from_numpy(cluster_centers).to(device=device, dtype=patch_centers.dtype)
        else:
            memory_bank = patch_centers.to(device=device)

        self.memory_bank = memory_bank.transpose(0, 1).contiguous()
        self.train(was_training)

    def forward(self, batch: torch.Tensor):
        descriptor_map = self.describe(batch)
        flat_descriptors = self._flatten_descriptors(descriptor_map)

        if self.training:
            return self.soft_boundary_loss(flat_descriptors)

        distances = torch.sqrt(self._squared_distances(flat_descriptors))
        n_neighbors = min(self.k_neighbors, distances.shape[-1])
        nearest = distances.topk(n_neighbors, largest=False).values
        weighted_distance = F.softmin(nearest, dim=-1)[:, :, 0] * nearest[:, :, 0]

        anomaly_map = weighted_distance.reshape(
            descriptor_map.shape[0],
            1,
            descriptor_map.shape[-2],
            descriptor_map.shape[-1],
        )
        anomaly_map = F.interpolate(
            anomaly_map,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        if self.score_smoothing_kernel > 1 and self.score_smoothing_sigma > 0.0:
            sigma = [self.score_smoothing_sigma, self.score_smoothing_sigma]
            kernel = [self.score_smoothing_kernel, self.score_smoothing_kernel]
            anomaly_map = gaussian_blur(anomaly_map, kernel_size=kernel, sigma=sigma)

        anomaly_map = anomaly_map[:, 0]
        if self.score_mode == "mean":
            anomaly_scores = anomaly_map.mean(dim=(1, 2))
        else:
            anomaly_scores = anomaly_map.view(anomaly_map.shape[0], -1).max(dim=1).values

        return anomaly_map, anomaly_scores

    def forward_train(self, batch: torch.Tensor) -> torch.Tensor:
        if not self.training:
            self.train()
        descriptor_map = self.describe(batch)
        flat_descriptors = self._flatten_descriptors(descriptor_map)
        return self.soft_boundary_loss(flat_descriptors)
