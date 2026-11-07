from __future__ import annotations

import abc
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from pyclad.strategies.replaycad.config import ReplayCADConfig


class ReplayCADMaskGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, images: np.ndarray, concept_id: str) -> np.ndarray:
        """Return one binary mask for every input image."""

    def release_device_memory(self) -> None:
        """Release optional accelerator state between mask-generation calls."""


def _resize_binary(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    resized = image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


class FullFrameMaskGenerator(ReplayCADMaskGenerator):
    """Lightweight test backend; not intended for paper-result runs."""

    def generate(self, images: np.ndarray, concept_id: str) -> np.ndarray:
        del concept_id
        if len(images) == 0:
            return np.empty((0, 0, 0), dtype=np.uint8)
        height, width = np.asarray(images[0]).shape[:2]
        return np.ones((len(images), height, width), dtype=np.uint8)


class PrecomputedMaskGenerator(ReplayCADMaskGenerator):
    def __init__(self, root: Path):
        self.root = Path(root)

    def generate(self, images: np.ndarray, concept_id: str) -> np.ndarray:
        concept_root = self.root / concept_id
        if not concept_root.is_dir():
            raise FileNotFoundError(f"Precomputed ReplayCAD mask directory does not exist: {concept_root}")
        paths = sorted(
            path
            for path in concept_root.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if not paths:
            raise FileNotFoundError(f"No masks found in {concept_root}")
        if len(paths) not in {1, len(images)}:
            raise ValueError(
                f"Expected one reusable mask or {len(images)} masks for '{concept_id}', found {len(paths)}"
            )

        masks: List[np.ndarray] = []
        for index, image in enumerate(images):
            path = paths[0] if len(paths) == 1 else paths[index]
            mask = np.asarray(Image.open(path).convert("L"))
            masks.append(_resize_binary(mask, np.asarray(image).shape[:2]))
        return np.stack(masks).astype(np.uint8, copy=False)


class SAMMaskGenerator(ReplayCADMaskGenerator):
    """SAM mask extraction following ReplayCAD's two published heuristics."""

    def __init__(self, config: ReplayCADConfig):
        self.config = config
        self._sam = None
        self._predictor = None
        self._automatic = None

    def _load(self):
        if self._predictor is not None:
            self._sam.to(device=self.config.device)
            return
        try:
            from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "ReplayCAD SAM masks require segment-anything. Install the 'replaycad' extra "
                "and provide --replaycad-sam-checkpoint."
            ) from exc

        checkpoint = Path(self.config.sam_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint}")
        sam = sam_model_registry[self.config.sam_model_type](checkpoint=str(checkpoint))
        sam.to(device=self.config.device)
        sam.eval()
        self._sam = sam
        self._predictor = SamPredictor(sam)
        self._automatic = SamAutomaticMaskGenerator(sam)

    def release_device_memory(self) -> None:
        if self._sam is None or not self.config.device.startswith("cuda"):
            return
        self._predictor.reset_image()
        self._sam.to(device="cpu")
        import torch

        torch.cuda.empty_cache()

    def generate(self, images: np.ndarray, concept_id: str) -> np.ndarray:
        self._load()
        mode = self.config.mask_modes.get(concept_id, "area")
        masks = [self._generate_one(np.asarray(image), concept_id, mode) for image in images]
        return np.stack(masks).astype(np.uint8, copy=False)

    def _generate_one(self, image: np.ndarray, concept_id: str, mode: str) -> np.ndarray:
        image_rgb = self._as_rgb_uint8(image)
        height, width = image_rgb.shape[:2]
        if mode == "full-frame":
            return np.ones((height, width), dtype=np.uint8)
        if mode == "corner-background":
            inset_x = max(1, int(round(width * 0.01)))
            inset_y = max(1, int(round(height * 0.01)))
            points = np.asarray(
                [
                    [inset_x, inset_y],
                    [width - 1 - inset_x, inset_y],
                    [inset_x, height - 1 - inset_y],
                    [width - 1 - inset_x, height - 1 - inset_y],
                ],
                dtype=np.float32,
            )
            self._predictor.set_image(image_rgb)
            predicted, _, _ = self._predictor.predict(
                point_coords=points,
                point_labels=np.ones(4, dtype=np.int32),
                multimask_output=False,
            )
            return (~predicted[0].astype(bool)).astype(np.uint8)

        candidates = self._automatic.generate(image_rgb)
        if not candidates:
            raise RuntimeError(f"SAM did not produce a mask for concept '{concept_id}'")
        image_area = float(height * width)
        target_fraction = self.config.mask_target_area_fractions.get(concept_id)
        filtered = [candidate for candidate in candidates if 0.005 <= candidate["area"] / image_area <= 0.98]
        pool = filtered or candidates
        if target_fraction is None:
            selected = max(pool, key=lambda candidate: (candidate.get("predicted_iou", 0.0), candidate["area"]))
        else:
            selected = min(pool, key=lambda candidate: abs(candidate["area"] / image_area - target_fraction))
        return np.asarray(selected["segmentation"], dtype=np.uint8)

    @staticmethod
    def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.dtype != np.uint8:
            max_value = float(array.max()) if array.size else 0.0
            if max_value <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array[..., :3])


def build_mask_generator(config: ReplayCADConfig) -> ReplayCADMaskGenerator:
    if config.mask_backend == "sam":
        return SAMMaskGenerator(config)
    if config.mask_backend == "precomputed":
        return PrecomputedMaskGenerator(Path(config.precomputed_mask_root))
    if config.mask_backend == "full-frame":
        return FullFrameMaskGenerator()
    raise ValueError(f"Unsupported ReplayCAD mask backend: {config.mask_backend}")
