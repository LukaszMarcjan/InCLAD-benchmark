from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from pyclad.models.ucad.config import UCADConfig


class StructureMaskGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        images: np.ndarray,
        concept_id: Optional[str],
        task_index: int,
    ) -> Optional[np.ndarray]:
        """Return integer region labels with shape ``(N, H, W)``."""


class NoStructureMaskGenerator(StructureMaskGenerator):
    def generate(self, images: np.ndarray, concept_id: Optional[str], task_index: int) -> None:
        del images, concept_id, task_index
        return None


class PrecomputedStructureMaskGenerator(StructureMaskGenerator):
    """Load UCAD's precomputed SAM label maps in dataset order.

    The expected layout mirrors the source benchmark, for example
    ``ROOT/bottle/train/good/*.png``.  ``ROOT/task_000/*.png`` is also
    accepted for scenarios that do not expose concept names.
    """

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def generate(
        self,
        images: np.ndarray,
        concept_id: Optional[str],
        task_index: int,
    ) -> np.ndarray:
        directory = self._resolve_directory(concept_id=concept_id, task_index=task_index)
        paths = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in self._IMAGE_SUFFIXES
        )
        if len(paths) < len(images):
            raise ValueError(
                f"UCAD needs at least {len(images)} structure masks for task {task_index}, "
                f"but found {len(paths)} in {directory}"
            )

        target_height, target_width = self._spatial_size(np.asarray(images[0]))
        masks = [self._load_label_map(path, (target_height, target_width)) for path in paths[: len(images)]]
        return np.stack(masks).astype(np.int64, copy=False)

    def _resolve_directory(self, concept_id: Optional[str], task_index: int) -> Path:
        names: list[str] = []
        nested_names: list[tuple[str, str]] = []
        if concept_id:
            names.append(str(concept_id))
            if "__" in str(concept_id):
                source, category = str(concept_id).split("__", 1)
                names.append(category)
                nested_names.append((source, category))
        names.append(f"task_{task_index:03d}")

        candidates: list[Path] = []
        for source, category in nested_names:
            candidates.extend(
                [
                    self.root / source / category / "train" / "good",
                    self.root / source / category / "train" / "ok",
                    self.root / source / category,
                ]
            )
        for name in names:
            candidates.extend(
                [
                    self.root / name / "train" / "good",
                    self.root / name / "train" / "ok",
                    self.root / name,
                ]
            )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            "Could not locate precomputed UCAD structure masks. Checked: " + ", ".join(str(path) for path in candidates)
        )

    @staticmethod
    def _load_label_map(path: Path, target_size: tuple[int, int]) -> np.ndarray:
        with Image.open(path) as image:
            label_map = image.convert("L")
            if label_map.size != (target_size[1], target_size[0]):
                label_map = label_map.resize(
                    (target_size[1], target_size[0]),
                    resample=Image.Resampling.NEAREST,
                )
            return np.asarray(label_map, dtype=np.int64)

    @staticmethod
    def _spatial_size(image: np.ndarray) -> tuple[int, int]:
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D image, got {image.shape}")
        if image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
            return int(image.shape[1]), int(image.shape[2])
        return int(image.shape[0]), int(image.shape[1])


class SAMStructureMaskGenerator(StructureMaskGenerator):
    """Generate the multi-region SAM maps used by UCAD directly from arrays."""

    def __init__(self, checkpoint: str, model_type: str, device: str):
        self.checkpoint = Path(checkpoint).expanduser()
        self.model_type = model_type
        self.device = device
        self._automatic = None

    def _load(self) -> None:
        if self._automatic is not None:
            return
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "UCAD SCL with online SAM requires segment-anything. " "Install pyclad with the 'ucad' extra."
            ) from exc
        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint))
        sam.to(device=self.device)
        sam.eval()
        self._automatic = SamAutomaticMaskGenerator(sam)

    def generate(
        self,
        images: np.ndarray,
        concept_id: Optional[str],
        task_index: int,
    ) -> np.ndarray:
        del concept_id, task_index
        self._load()
        return np.stack([self._generate_one(image) for image in images]).astype(np.int64, copy=False)

    def _generate_one(self, image: np.ndarray) -> np.ndarray:
        rgb = self._as_rgb_uint8(image)
        annotations = self._automatic.generate(rgb)
        labels = np.zeros(rgb.shape[:2], dtype=np.int64)
        # Matches the official preprocessing: large masks are painted first,
        # then smaller masks overwrite them with a new region identity.
        for region_id, annotation in enumerate(
            sorted(annotations, key=lambda item: item["area"], reverse=True),
            start=1,
        ):
            labels[np.asarray(annotation["segmentation"], dtype=bool)] = region_id
        return labels

    @staticmethod
    def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.dtype != np.uint8:
            if array.size and float(array.max()) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array[..., :3])


def build_structure_mask_generator(config: UCADConfig, device: str) -> StructureMaskGenerator:
    if config.structure_mode == "none":
        return NoStructureMaskGenerator()
    if config.structure_mode == "precomputed":
        return PrecomputedStructureMaskGenerator(config.structure_mask_root)
    if config.structure_mode == "sam":
        return SAMStructureMaskGenerator(
            checkpoint=config.sam_checkpoint,
            model_type=config.sam_model_type,
            device=device,
        )
    raise ValueError(f"Unknown UCAD structure mode: {config.structure_mode}")
