from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


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


def load_penalty_image_paths(source_root: Optional[str], source_paths: Optional[Sequence[str]]) -> list[str]:
    return [str(path) for path in _resolve_image_paths(source_root=source_root, source_paths=source_paths)]


class EfficientADTrainDataset(Dataset):
    def __init__(
        self,
        clean_images: torch.Tensor,
        input_size: tuple[int, int],
        penalty_image_paths: Sequence[str],
        penalty_resize_scale: float,
        penalty_grayscale_probability: float,
        random_seed: int,
    ):
        if clean_images.ndim != 4:
            raise ValueError(f"Expected clean_images to be a 4D tensor, got {clean_images.shape}")

        self.clean_images = clean_images
        self.input_size = tuple(int(v) for v in input_size)
        self.penalty_image_paths = tuple(penalty_image_paths)
        self.penalty_resize_scale = float(penalty_resize_scale)
        self.penalty_grayscale_probability = float(penalty_grayscale_probability)
        self._rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        return int(self.clean_images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_image = self.clean_images[index].clone()

        if len(self.penalty_image_paths) == 0:
            penalty_image = torch.zeros_like(clean_image)
            has_penalty = torch.tensor(False)
        else:
            penalty_path = self.penalty_image_paths[int(self._rng.integers(0, len(self.penalty_image_paths)))]
            penalty_image = self._load_penalty_image(penalty_path=penalty_path, dtype=clean_image.dtype)
            has_penalty = torch.tensor(True)

        return clean_image, penalty_image, has_penalty

    def _load_penalty_image(self, penalty_path: str, dtype: torch.dtype) -> torch.Tensor:
        with Image.open(penalty_path) as image:
            image = image.convert("RGB")
            resized_size = (
                max(self.input_size[1], int(round(self.input_size[1] * self.penalty_resize_scale))),
                max(self.input_size[0], int(round(self.input_size[0] * self.penalty_resize_scale))),
            )
            image = image.resize(resized_size, resample=Image.BILINEAR)
            if float(self._rng.random()) < self.penalty_grayscale_probability:
                image = image.convert("L").convert("RGB")
            image = TF.center_crop(image, output_size=self.input_size)
            tensor = TF.pil_to_tensor(image).to(dtype=torch.float32) / 255.0

        return tensor.to(dtype=dtype)
