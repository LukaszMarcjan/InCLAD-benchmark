from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.models.patchcore.builder import build
from pyclad.models.patchcore.config import PatchCoreConfig
from pyclad.models.vision.preprocessing import ImagePreprocessor
from pyclad.models.vision.utils import resolve_device


class MeanMapper(nn.Module):
    def __init__(self, preprocessing_dim: int):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class FeaturePreprocessor(nn.Module):
    def __init__(self, input_dims: Sequence[int], output_dim: int):
        super().__init__()
        self.input_dims = tuple(input_dims)
        self.output_dim = output_dim
        self.preprocessing_modules = nn.ModuleList([MeanMapper(output_dim) for _ in input_dims])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        reduced = [module(feature) for module, feature in zip(self.preprocessing_modules, features)]
        return torch.stack(reduced, dim=1)


class FeatureAggregator(nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class PatchMaker:
    def __init__(self, patchsize: int, stride: int):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)

        number_of_total_patches = []
        for spatial_size in features.shape[-2:]:
            n_patches = (spatial_size + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        unfolded_features = unfolded_features.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    @staticmethod
    def unpatch_scores(scores: np.ndarray, batch_size: int) -> np.ndarray:
        return scores.reshape(batch_size, -1, *scores.shape[1:])

    @staticmethod
    def score(x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        while x_t.ndim > 1:
            x_t = torch.max(x_t, dim=-1).values
        return x_t.numpy() if isinstance(x, np.ndarray) else x_t


class RescaleSegmentor:
    def __init__(self, device: torch.device, target_size: tuple[int, int], smoothing: float):
        self.device = device
        self.target_size = target_size
        self.smoothing = smoothing

    def convert_to_segmentation(self, patch_scores: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)

            scores = patch_scores.to(self.device).unsqueeze(1)
            scores = F.interpolate(scores, size=self.target_size, mode="bilinear", align_corners=False)
            scores = scores.squeeze(1).cpu().numpy()

        if self.smoothing <= 0:
            return scores.astype(np.float32, copy=False)
        return np.asarray(
            [ndimage.gaussian_filter(score, sigma=self.smoothing) for score in scores],
            dtype=np.float32,
        )


class ApproximateGreedyCoresetSampler:
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        random_seed: int = 0,
    ):
        if percentage <= 0.0 or percentage > 1.0:
            raise ValueError("percentage must be in (0, 1]")

        self.percentage = percentage
        self.device = device
        self.number_of_starting_points = number_of_starting_points
        self.dimension_to_project_features_to = dimension_to_project_features_to
        self.random_seed = random_seed

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] == self.dimension_to_project_features_to:
            return features.to(self.device)

        with torch.random.fork_rng():
            torch.manual_seed(self.random_seed)
            mapper = nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False).to(self.device)
        return mapper(features.to(self.device))

    @staticmethod
    def _compute_batchwise_differences(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(
        self,
        features: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Greedy coreset selection.

        If *weights* (shape ``(N,)``, on the same device as *features*) are
        provided, the selection criterion becomes ``weights[p] * min_dist(p, S)``
        instead of plain ``min_dist(p, S)``.  This biases sampling towards
        points that are both diverse *within* the bank and far from an external
        reference (e.g. the current-task bank), whose NN distances were used to
        build the weight vector.
        """
        number_of_starting_points = min(self.number_of_starting_points, len(features))
        rng = np.random.default_rng(self.random_seed)
        start_points = rng.choice(len(features), number_of_starting_points, replace=False).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(features, features[start_points])
        approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix, axis=-1).reshape(-1, 1)

        coreset_indices = []
        num_coreset_samples = max(1, int(len(features) * self.percentage))

        with torch.no_grad():
            for _ in range(num_coreset_samples):
                scores = approximate_coreset_anchor_distances.squeeze(-1)
                if weights is not None:
                    scores = scores * weights
                select_idx = torch.argmax(scores).item()
                coreset_indices.append(select_idx)

                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)

    def run(self, features: np.ndarray) -> np.ndarray:
        if self.percentage == 1.0:
            return features

        feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
        reduced_features = self._reduce_features(feature_tensor)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        return feature_tensor[sample_indices].cpu().numpy().astype(np.float32, copy=False)

    def run_weighted(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Like :meth:`run`, but with per-point importance weights.

        ``weights[i]`` should be a non-negative scalar — larger values mean
        "keep this point preferentially".  A natural choice is the NN distance
        from point *i* to the current-task memory bank: patches that are far
        from the current task are unique to the previous task and therefore more
        important to retain.

        The weight vector is normalised by its mean before use so that, when all
        weights are equal, the result matches plain :meth:`run`.
        """
        if self.percentage == 1.0:
            return features

        feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
        weight_tensor = torch.from_numpy(weights.astype(np.float32, copy=False)).to(self.device)

        # Normalise so mean weight == 1; avoids scale effects while keeping
        # relative ordering.  Guard against degenerate all-zero case.
        w_mean = weight_tensor.mean()
        if w_mean > 0:
            weight_tensor = weight_tensor / w_mean

        reduced_features = self._reduce_features(feature_tensor)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features, weights=weight_tensor)
        return feature_tensor[sample_indices].cpu().numpy().astype(np.float32, copy=False)


class PatchCore(Model):
    def __init__(self, config: Optional[PatchCoreConfig] = None):
        self.config = config or PatchCoreConfig()

        self._device = resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=3,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )
        self.module = build(self.config).to(self._device).eval()
        self._patch_maker = PatchMaker(patchsize=self.config.patchsize, stride=self.config.patchstride)

        feature_dimensions = self.module.infer_out_channels(self.config.input_size)
        self._feature_preprocessor = FeaturePreprocessor(
            input_dims=feature_dimensions,
            output_dim=self.config.pretrain_embed_dimension,
        ).to(self._device)
        self._feature_aggregator = FeatureAggregator(target_dim=self.config.target_embed_dimension).to(self._device)
        self._segmentor = RescaleSegmentor(
            device=self._device,
            target_size=self.config.input_size,
            smoothing=self.config.smoothing_sigma,
        )
        self._coreset_sampler = ApproximateGreedyCoresetSampler(
            percentage=self.config.coreset_sampling_ratio,
            device=self._device,
            number_of_starting_points=self.config.coreset_starting_points,
            dimension_to_project_features_to=self.config.coreset_projection_dimension,
            random_seed=self.config.random_seed,
        )

        self._memory_bank: Optional[np.ndarray] = None
        self._feature_map_size: Optional[tuple[int, int]] = None
        self._threshold = self.config.threshold
        self._nn_index: Optional[NearestNeighbors] = None

    def _prepare_batches(self, data: np.ndarray) -> DataLoader:
        x_t = self._preprocessor.transform(data)
        dataset = TensorDataset(x_t)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    @staticmethod
    def _align_feature_maps(features: list[torch.Tensor], patch_shapes: list[list[int]]) -> list[torch.Tensor]:
        reference_shape = patch_shapes[0]
        for index in range(1, len(features)):
            feat = features[index]
            dims = patch_shapes[index]

            feat = feat.reshape(feat.shape[0], dims[0], dims[1], *feat.shape[2:])
            feat = feat.permute(0, -3, -2, -1, 1, 2)
            permuted_shape = feat.shape
            feat = feat.reshape(-1, *feat.shape[-2:])
            feat = F.interpolate(
                feat.unsqueeze(1),
                size=(reference_shape[0], reference_shape[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            feat = feat.reshape(*permuted_shape[:-2], reference_shape[0], reference_shape[1])
            feat = feat.permute(0, -2, -1, 1, 2, 3)
            features[index] = feat.reshape(len(feat), -1, *feat.shape[-3:])

        return features

    def _embed(self, images: torch.Tensor, provide_patch_shapes: bool = False):
        with torch.no_grad():
            features = self.module(images)
            patches_with_shapes = [self._patch_maker.patchify(f, return_spatial_info=True) for f in features]
            patch_shapes = [s for _, s in patches_with_shapes]
            features = [f for f, _ in patches_with_shapes]

            features = self._align_feature_maps(features, patch_shapes)
            features = [f.reshape(-1, *f.shape[-3:]) for f in features]
            features = self._feature_aggregator(self._feature_preprocessor(features))

        embeddings = features.detach().cpu().numpy().astype(np.float32, copy=False)
        if provide_patch_shapes:
            return embeddings, patch_shapes
        return embeddings

    def _extract_embeddings(self, data: np.ndarray, provide_patch_shapes: bool = False):
        if len(data) == 0:
            empty = np.asarray([], dtype=np.float32)
            return (empty, []) if provide_patch_shapes else empty

        self.module = self.module.to(self._device).eval()
        self._feature_preprocessor = self._feature_preprocessor.to(self._device).eval()
        self._feature_aggregator = self._feature_aggregator.to(self._device).eval()

        embeddings = []
        patch_shapes = None
        for (batch_x,) in self._prepare_batches(data):
            result = self._embed(batch_x.to(self._device, dtype=torch.float32), provide_patch_shapes)
            if provide_patch_shapes:
                embeddings.append(result[0])
                patch_shapes = result[1]
            else:
                embeddings.append(result)

        all_embeddings = np.concatenate(embeddings, axis=0) if embeddings else np.asarray([], dtype=np.float32)
        return (all_embeddings, patch_shapes) if provide_patch_shapes else all_embeddings

    def _fit_nn_index(self) -> None:
        self._ensure_fitted()
        self._nn_index = NearestNeighbors(
            n_neighbors=min(self.config.n_neighbors, len(self._memory_bank)),
            n_jobs=1,
        )
        self._nn_index.fit(self._memory_bank)

    def _ensure_fitted(self) -> None:
        if self._memory_bank is None or self._feature_map_size is None:
            raise RuntimeError("PatchCore must be fitted before scoring or predicting")

    def fit(self, data: np.ndarray):
        if len(data) == 0:
            return

        embeddings, patch_shapes = self._extract_embeddings(data, provide_patch_shapes=True)
        self._feature_map_size = tuple(patch_shapes[0])
        self._memory_bank = self._coreset_sampler.run(embeddings)
        self._fit_nn_index()

        anomaly_scores = None if self.config.threshold is not None else self.score_data(data)
        if self.config.threshold is not None:
            self._threshold = float(self.config.threshold)
        elif anomaly_scores is None or len(anomaly_scores) == 0:
            self._threshold = 0.0
        else:
            self._threshold = float(np.quantile(anomaly_scores, self.config.threshold_quantile))

    def _patch_scores(self, embeddings: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        if self._nn_index is None:
            self._fit_nn_index()

        distances, _ = self._nn_index.kneighbors(embeddings)
        return np.mean(distances, axis=-1).astype(np.float32, copy=False)

    def _predict_scores_and_masks(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        embeddings, patch_shapes = self._extract_embeddings(data, provide_patch_shapes=True)
        patch_scores = self._patch_scores(embeddings)

        batch_size = len(data)
        unpatched = self._patch_maker.unpatch_scores(patch_scores, batch_size=batch_size)

        image_scores = self._patch_maker.score(unpatched.reshape(*unpatched.shape[:2], -1)).astype(
            np.float32, copy=False
        )

        scales = patch_shapes[0]
        masks = self._segmentor.convert_to_segmentation(
            unpatched.reshape(batch_size, scales[0], scales[1]).astype(np.float32, copy=False)
        )
        return image_scores, masks

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        _, masks = self._predict_scores_and_masks(data)
        if resize_to_input:
            return masks

        target_size = self._feature_map_size
        map_tensor = torch.from_numpy(masks[:, None, :, :])
        resized = F.interpolate(map_tensor, size=target_size, mode="bilinear", align_corners=False)
        return resized[:, 0].numpy().astype(np.float32, copy=False)

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        image_scores, _ = self._predict_scores_and_masks(data)
        return image_scores

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
        return "PatchCore"

    def additional_info(self) -> Dict:
        return {
            "threshold": self._threshold,
            "device": str(self._device),
            "backbone": self.config.backbone_name,
            "feature_layers": getattr(self.module, "return_nodes", self.config.backbone_return_nodes),
            "input_size": self.config.input_size,
            "pretrain_embed_dimension": self.config.pretrain_embed_dimension,
            "target_embed_dimension": self.config.target_embed_dimension,
            "patchsize": self.config.patchsize,
            "patchstride": self.config.patchstride,
            "pretrained_backbone": self.config.pretrained_backbone,
            "coreset_sampling_ratio": self.config.coreset_sampling_ratio,
            "memory_bank_size": len(self._memory_bank) if self._memory_bank is not None else None,
            "n_neighbors": self.config.n_neighbors,
            "threshold_quantile": self.config.threshold_quantile,
        }
