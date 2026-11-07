from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


def concept_slug(concept_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(concept_id).strip()).strip("-.")
    return slug or "concept"


@dataclass(frozen=True)
class ReplayCADArtifact:
    concept_id: str
    directory: Path

    @property
    def condition_path(self) -> Path:
        return self.directory / "condition.pt"

    @property
    def masks_path(self) -> Path:
        return self.directory / "masks.npz"

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.json"

    @property
    def complete(self) -> bool:
        return self.condition_path.is_file() and self.masks_path.is_file() and self.metadata_path.is_file()

    def save_masks(self, masks: np.ndarray) -> None:
        masks_array = np.asarray(masks, dtype=np.uint8)
        if masks_array.ndim != 3:
            raise ValueError(f"ReplayCAD masks must have shape (N,H,W), got {masks_array.shape}")
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.masks_path, masks=masks_array)

    def load_masks(self) -> np.ndarray:
        with np.load(self.masks_path) as payload:
            return np.asarray(payload["masks"], dtype=np.uint8)

    def write_metadata(self, payload: Dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temp_path = self.metadata_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.metadata_path)

    def read_metadata(self) -> Dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def storage_size_bytes(self) -> int:
        # Generated images are an optional runtime cache, not part of the
        # compressed ReplayCAD representation reported by the benchmark.
        return sum(
            path.stat().st_size
            for path in (self.condition_path, self.masks_path, self.metadata_path)
            if path.is_file()
        )


def artifact_for(root: Path, concept_id: str) -> ReplayCADArtifact:
    return ReplayCADArtifact(concept_id=concept_id, directory=Path(root) / "concepts" / concept_slug(concept_id))
