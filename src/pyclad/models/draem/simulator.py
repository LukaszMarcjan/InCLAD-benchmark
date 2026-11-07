from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from pyclad.models.vision.preprocessing import ImagePreprocessor

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _fade(values: np.ndarray) -> np.ndarray:
    return 6.0 * values**5 - 15.0 * values**4 + 10.0 * values**3


def rand_perlin_2d(shape: tuple[int, int], resolution: tuple[int, int], rng: np.random.Generator) -> torch.Tensor:
    resolution_y = max(1, int(resolution[0]))
    resolution_x = max(1, int(resolution[1]))

    yy = np.linspace(0, resolution_y, shape[0], endpoint=False, dtype=np.float32)
    xx = np.linspace(0, resolution_x, shape[1], endpoint=False, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")

    y0 = np.floor(grid_y).astype(np.int32)
    x0 = np.floor(grid_x).astype(np.int32)
    fy = grid_y - y0
    fx = grid_x - x0
    y1 = np.clip(y0 + 1, 0, resolution_y)
    x1 = np.clip(x0 + 1, 0, resolution_x)

    angles = 2.0 * np.pi * rng.random((resolution_y + 1, resolution_x + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32, copy=False)

    g00 = gradients[y0, x0]
    g10 = gradients[y1, x0]
    g01 = gradients[y0, x1]
    g11 = gradients[y1, x1]

    n00 = g00[..., 0] * fy + g00[..., 1] * fx
    n10 = g10[..., 0] * (fy - 1.0) + g10[..., 1] * fx
    n01 = g01[..., 0] * fy + g01[..., 1] * (fx - 1.0)
    n11 = g11[..., 0] * (fy - 1.0) + g11[..., 1] * (fx - 1.0)

    sy = _fade(fy)
    sx = _fade(fx)
    nx0 = n00 * (1.0 - sy) + n10 * sy
    nx1 = n01 * (1.0 - sy) + n11 * sy
    noise = nx0 * (1.0 - sx) + nx1 * sx
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    return torch.from_numpy(noise.astype(np.float32, copy=False))


def _resolve_image_paths(source_root: Optional[str], source_paths: Optional[Sequence[str]]) -> list[Path]:
    resolved_paths: list[Path] = []

    def _collect_from_path(path_str: str) -> None:
        path = Path(path_str)
        if not path.exists():
            return
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            resolved_paths.append(path)
            return
        if path.is_dir():
            for extension in SUPPORTED_IMAGE_EXTENSIONS:
                resolved_paths.extend(sorted(path.rglob(f"*{extension}")))

    if source_root is not None:
        _collect_from_path(source_root)
    if source_paths is not None:
        for path in source_paths:
            _collect_from_path(path)

    return list(dict.fromkeys(resolved_paths))


def load_source_images(
    input_size: tuple[int, int],
    in_channels: int,
    source_root: Optional[str],
    source_paths: Optional[Sequence[str]],
) -> list[torch.Tensor]:
    image_paths = _resolve_image_paths(source_root=source_root, source_paths=source_paths)
    if len(image_paths) == 0:
        return []

    preprocessor = ImagePreprocessor(input_size=input_size, in_channels=in_channels)
    images: list[torch.Tensor] = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("L" if in_channels == 1 else "RGB")
            image_array = np.asarray(image, dtype=np.uint8)

        if image_array.ndim == 2:
            image_array = image_array[..., None]

        images.append(preprocessor.transform(image_array[None, ...])[0])

    return images


class SyntheticAnomalyDataset(Dataset):
    def __init__(
        self,
        clean_images: torch.Tensor,
        source_images: Sequence[torch.Tensor],
        allow_random_color_fallback: bool,
        texture_augmentation: bool,
        texture_augmentation_count: int,
        perlin_scale_min: int,
        perlin_scale_max: int,
        perlin_threshold: float,
        anomaly_blend_range: tuple[float, float],
        anomaly_probability: float,
        rotation_degrees: float,
        rotation_probability: float,
        random_seed: int,
    ):
        if clean_images.ndim != 4:
            raise ValueError(f"Expected clean_images to be a 4D tensor, got {clean_images.shape}")
        if len(source_images) == 0 and not allow_random_color_fallback:
            raise ValueError("No anomaly source images were provided and random-color fallback is disabled")

        self.clean_images = clean_images
        self.source_images = [image.detach().cpu().clone() for image in source_images]
        self.allow_random_color_fallback = allow_random_color_fallback
        self.texture_augmentation = texture_augmentation
        self.texture_augmentation_count = texture_augmentation_count
        self.perlin_scale_min = perlin_scale_min
        self.perlin_scale_max = perlin_scale_max
        self.perlin_threshold = perlin_threshold
        self.anomaly_blend_range = anomaly_blend_range
        self.anomaly_probability = anomaly_probability
        self.rotation_degrees = rotation_degrees
        self.rotation_probability = rotation_probability
        self._rng = np.random.default_rng(random_seed)
        self._height = int(clean_images.shape[-2])
        self._width = int(clean_images.shape[-1])
        self._in_channels = int(clean_images.shape[1])

    def __len__(self) -> int:
        return int(self.clean_images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_image = self.clean_images[index].clone()
        if float(self._rng.random()) < self.rotation_probability:
            clean_image = self._rotate_image(clean_image)

        if float(self._rng.random()) > self.anomaly_probability:
            empty_mask = torch.zeros((1, self._height, self._width), dtype=clean_image.dtype)
            return clean_image, clean_image.clone(), empty_mask

        anomaly_mask = self._sample_anomaly_mask().to(dtype=clean_image.dtype)
        anomaly_texture = self._sample_anomaly_texture(dtype=clean_image.dtype)
        blend_min, blend_max = self.anomaly_blend_range
        blend_ratio = float(self._rng.uniform(blend_min, blend_max))

        anomalous_image = clean_image * (1.0 - anomaly_mask)
        anomalous_region = ((1.0 - blend_ratio) * clean_image + blend_ratio * anomaly_texture) * anomaly_mask
        anomalous_image = (anomalous_image + anomalous_region).clamp(0.0, 1.0)

        return clean_image, anomalous_image, anomaly_mask

    def _rotate_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.rotation_degrees <= 0.0:
            return image
        angle = float(self._rng.uniform(-self.rotation_degrees, self.rotation_degrees))
        return TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)

    def _sample_anomaly_mask(self) -> torch.Tensor:
        for _ in range(10):
            scale_y = min(self._height, 2 ** int(self._rng.integers(self.perlin_scale_min, self.perlin_scale_max + 1)))
            scale_x = min(self._width, 2 ** int(self._rng.integers(self.perlin_scale_min, self.perlin_scale_max + 1)))
            perlin_noise = rand_perlin_2d(
                shape=(self._height, self._width),
                resolution=(max(1, scale_y), max(1, scale_x)),
                rng=self._rng,
            )
            if self.rotation_degrees > 0.0:
                angle = float(self._rng.uniform(-self.rotation_degrees, self.rotation_degrees))
                perlin_noise = TF.rotate(
                    perlin_noise.unsqueeze(0),
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )[0]

            anomaly_mask = (perlin_noise > self.perlin_threshold).to(dtype=self.clean_images.dtype).unsqueeze(0)
            anomaly_fraction = float(anomaly_mask.mean())
            if 0.01 <= anomaly_fraction <= 0.75:
                return anomaly_mask

        return self._sample_rectangular_mask()

    def _sample_rectangular_mask(self) -> torch.Tensor:
        mask = torch.zeros((1, self._height, self._width), dtype=self.clean_images.dtype)
        rectangles = int(self._rng.integers(1, 4))

        min_h = max(1, self._height // 16)
        max_h = max(min_h + 1, self._height // 3 + 1)
        min_w = max(1, self._width // 16)
        max_w = max(min_w + 1, self._width // 3 + 1)

        for _ in range(rectangles):
            rect_h = int(self._rng.integers(min_h, max_h))
            rect_w = int(self._rng.integers(min_w, max_w))
            top = int(self._rng.integers(0, max(1, self._height - rect_h + 1)))
            left = int(self._rng.integers(0, max(1, self._width - rect_w + 1)))
            mask[:, top : top + rect_h, left : left + rect_w] = 1.0

        return mask

    def _sample_anomaly_texture(self, dtype: torch.dtype) -> torch.Tensor:
        if len(self.source_images) > 0:
            texture_image = self.source_images[int(self._rng.integers(0, len(self.source_images)))].clone()
            if self.texture_augmentation:
                texture_image = self._apply_random_augmentations(texture_image)
            return texture_image.to(dtype=dtype)

        random_color = torch.tensor(self._rng.random(self._in_channels), dtype=dtype).view(self._in_channels, 1, 1)
        return random_color.expand(-1, self._height, self._width)

    def _apply_random_augmentations(self, texture_image: torch.Tensor) -> torch.Tensor:
        if self.texture_augmentation_count == 0:
            return texture_image

        pil_image = TF.to_pil_image(texture_image)
        augmentations = [
            self._gamma_contrast,
            self._brightness,
            self._sharpness,
            self._color,
            self._solarize,
            self._posterize,
            self._invert,
            self._autocontrast,
            self._equalize,
            self._rotate_texture,
        ]
        chosen_indices = self._rng.choice(
            len(augmentations),
            size=self.texture_augmentation_count,
            replace=self.texture_augmentation_count > len(augmentations),
        )

        for augmentation_index in np.atleast_1d(chosen_indices):
            pil_image = augmentations[int(augmentation_index)](pil_image)

        augmented_tensor = TF.pil_to_tensor(pil_image).to(dtype=torch.float32) / 255.0
        augmented_tensor = self._match_channels(augmented_tensor)
        if tuple(augmented_tensor.shape[-2:]) != (self._height, self._width):
            augmented_tensor = F.interpolate(
                augmented_tensor.unsqueeze(0),
                size=(self._height, self._width),
                mode="bilinear",
                align_corners=False,
            )[0]
        return augmented_tensor.clamp(0.0, 1.0)

    def _match_channels(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[0] == self._in_channels:
            return image
        if image.shape[0] == 1 and self._in_channels > 1:
            return image.repeat(self._in_channels, 1, 1)
        if image.shape[0] > 1 and self._in_channels == 1:
            return image.mean(dim=0, keepdim=True)
        raise ValueError(f"Cannot convert source texture with channels={image.shape[0]} to {self._in_channels}")

    def _gamma_contrast(self, image: Image.Image) -> Image.Image:
        gamma = float(self._rng.uniform(0.5, 2.0))
        return TF.adjust_gamma(image, gamma=gamma)

    def _posterize(self, image: Image.Image) -> Image.Image:
        bits = int(self._rng.integers(3, 7))
        return ImageOps.posterize(image, bits)

    def _sharpness(self, image: Image.Image) -> Image.Image:
        factor = float(self._rng.uniform(0.1, 2.0))
        return ImageEnhance.Sharpness(image).enhance(factor)

    def _solarize(self, image: Image.Image) -> Image.Image:
        threshold = int(self._rng.integers(32, 224))
        return ImageOps.solarize(image, threshold)

    @staticmethod
    def _invert(image: Image.Image) -> Image.Image:
        return ImageOps.invert(image)

    @staticmethod
    def _equalize(image: Image.Image) -> Image.Image:
        return ImageOps.equalize(image)

    def _brightness(self, image: Image.Image) -> Image.Image:
        brightness_factor = float(self._rng.uniform(0.8, 1.2))
        brightened = ImageEnhance.Brightness(image).enhance(brightness_factor)
        color_shift = np.asarray(brightened, dtype=np.float32) + self._rng.uniform(-30.0, 30.0)
        color_shift = np.clip(color_shift, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(color_shift)

    def _color(self, image: Image.Image) -> Image.Image:
        factor = float(self._rng.uniform(0.0, 2.0))
        return ImageEnhance.Color(image).enhance(factor)

    def _rotate_texture(self, image: Image.Image) -> Image.Image:
        angle = float(self._rng.uniform(-45.0, 45.0))
        return image.rotate(angle, resample=Image.BILINEAR)

    @staticmethod
    def _autocontrast(image: Image.Image) -> Image.Image:
        return ImageOps.autocontrast(image)
