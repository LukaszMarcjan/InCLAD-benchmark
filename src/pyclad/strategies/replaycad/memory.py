from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pyclad.strategies.replaycad.artifacts import ReplayCADArtifact, artifact_for
from pyclad.strategies.replaycad.backend import DiffusersReplayCADBackend, ReplayCADBackend
from pyclad.strategies.replaycad.config import ReplayCADConfig
from pyclad.strategies.replaycad.masks import ReplayCADMaskGenerator, build_mask_generator

logger = logging.getLogger(__name__)


class ReplayCADMemory:
    """Per-concept conditional memory used by ReplayCAD."""

    def __init__(
        self,
        config: ReplayCADConfig,
        *,
        backend: Optional[ReplayCADBackend] = None,
        mask_generator: Optional[ReplayCADMaskGenerator] = None,
    ):
        self.config = config
        self.root = Path(config.artifact_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._backend = backend or DiffusersReplayCADBackend(config)
        self._mask_generator = mask_generator or build_mask_generator(config)
        # Files already present on disk are reusable cache entries, not tasks
        # that the restarted continual scenario has seen.  Activating them all
        # here would replay future concepts at the first training step.
        self._stored_artifacts: List[ReplayCADArtifact] = self._load_existing_artifacts()
        self._artifacts: List[ReplayCADArtifact] = []
        self._generated_samples = 0
        self._generation_seconds = 0.0
        self._compression_seconds = 0.0
        self._write_run_config()

    @property
    def artifacts(self) -> tuple[ReplayCADArtifact, ...]:
        return tuple(self._artifacts)

    def compress(self, concept_id: str, images: np.ndarray) -> ReplayCADArtifact:
        existing = self._artifact_by_concept(concept_id)
        if existing is not None and existing.complete:
            self._activate(existing)
            logger.info("ReplayCAD condition for '%s' already exists; reusing %s", concept_id, existing.directory)
            return existing

        artifact = existing or artifact_for(self.root, concept_id)
        registered_index = next(
            (index for index, item in enumerate(self._stored_artifacts) if item.concept_id == concept_id),
            None,
        )
        masks = self._mask_generator.generate(np.asarray(images), concept_id)
        if len(masks) != len(images):
            raise ValueError(
                f"ReplayCAD mask generator returned {len(masks)} masks for {len(images)} images of '{concept_id}'"
            )
        selected_masks = self._select_stored_masks(masks, concept_id)
        artifact.save_masks(selected_masks)
        # SAM and diffusion compression run sequentially.  Keeping SAM ViT-H
        # on CUDA while loading the diffusion model only wastes GPU memory.
        self._mask_generator.release_device_memory()

        started = time.perf_counter()
        backend_metadata = self._backend.compress(
            concept_id=concept_id,
            images=np.asarray(images),
            masks=masks,
            artifact=artifact,
        )
        elapsed = time.perf_counter() - started
        self._compression_seconds += elapsed
        metadata = {
            "schema_version": 1,
            "concept_id": concept_id,
            "concept_index": registered_index if registered_index is not None else len(self._stored_artifacts),
            "preset": self.config.preset,
            "benchmark": self.config.benchmark,
            "model_id": self.config.model_id,
            "diffusion_family": self.config.diffusion_family,
            "resolution": self.config.resolution,
            "training_samples": int(len(images)),
            "stored_masks": int(len(selected_masks)),
            **backend_metadata,
        }
        artifact.write_metadata(metadata)
        if registered_index is None:
            self._stored_artifacts.append(artifact)
        self._activate(artifact)
        self._write_manifest()
        return artifact

    def generate_previous(self, seed: Optional[int] = None) -> np.ndarray:
        if not self._artifacts:
            return np.empty((0, self.config.resolution, self.config.resolution, 3), dtype=np.uint8)

        base_seed = self.config.seed if seed is None else int(seed)
        batches = []
        started = time.perf_counter()
        for artifact in self._artifacts:
            artifact_seed = self._concept_seed(base_seed, artifact.concept_id)
            cached = self._load_generated_cache(artifact, artifact_seed)
            if cached is None:
                generated = self._backend.generate(
                    artifact=artifact,
                    count=self.config.replay_samples_per_concept,
                    seed=artifact_seed,
                )
                self._save_generated_cache(artifact, artifact_seed, generated)
            else:
                generated = cached
            if len(generated) != self.config.replay_samples_per_concept:
                raise RuntimeError(
                    f"ReplayCAD backend generated {len(generated)} samples for '{artifact.concept_id}', "
                    f"expected {self.config.replay_samples_per_concept}"
                )
            batches.append(np.asarray(generated, dtype=np.uint8))

        elapsed = time.perf_counter() - started
        self._generation_seconds += elapsed
        generated_count = sum(len(batch) for batch in batches)
        self._generated_samples += generated_count
        return np.concatenate(batches, axis=0)

    def release_device_memory(self) -> None:
        """Release backend accelerator memory until the next backend call."""
        self._backend.release_device_memory()
        self._mask_generator.release_device_memory()

    def storage_size_bytes(self) -> int:
        return sum(artifact.storage_size_bytes() for artifact in self._artifacts)

    def info(self) -> Dict[str, Any]:
        return {
            "name": "ReplayCADMemory",
            "preset": self.config.preset,
            "benchmark": self.config.benchmark,
            "diffusion_family": self.config.diffusion_family,
            "model_id": self.config.model_id,
            "resolution": self.config.resolution,
            "semantic_tokens": self.config.semantic_tokens,
            "spatial_tokens": self.config.spatial_tokens,
            "masks_per_concept": self.config.masks_per_concept,
            "replay_samples_per_concept": self.config.replay_samples_per_concept,
            "concepts": [artifact.concept_id for artifact in self._artifacts],
            "concept_count": len(self._artifacts),
            "conditional_storage_bytes": self.storage_size_bytes(),
            "generated_samples_total": self._generated_samples,
            "generation_seconds": self._generation_seconds,
            "compression_seconds": self._compression_seconds,
            "artifact_dir": str(self.root),
            "mask_backend": self.config.mask_backend,
        }

    def _load_existing_artifacts(self) -> List[ReplayCADArtifact]:
        concepts_root = self.root / "concepts"
        if not concepts_root.is_dir():
            return []
        artifacts = []
        for directory in concepts_root.iterdir():
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                continue
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            artifact = ReplayCADArtifact(concept_id=str(payload["concept_id"]), directory=directory)
            if artifact.complete:
                artifacts.append((int(payload.get("concept_index", len(artifacts))), artifact))
        return [artifact for _, artifact in sorted(artifacts, key=lambda item: item[0])]

    def _artifact_by_concept(self, concept_id: str) -> Optional[ReplayCADArtifact]:
        for artifact in self._stored_artifacts:
            if artifact.concept_id == concept_id:
                return artifact
        candidate = artifact_for(self.root, concept_id)
        if candidate.directory.exists():
            if candidate.metadata_path.is_file():
                stored_concept = str(candidate.read_metadata().get("concept_id"))
                if stored_concept != concept_id:
                    raise ValueError(
                        f"ReplayCAD concept names '{stored_concept}' and '{concept_id}' resolve to the same "
                        f"artifact directory: {candidate.directory}"
                    )
            return candidate
        return None

    def _activate(self, artifact: ReplayCADArtifact) -> None:
        if not any(item.concept_id == artifact.concept_id for item in self._artifacts):
            self._artifacts.append(artifact)

    def _select_stored_masks(self, masks: np.ndarray, concept_id: str) -> np.ndarray:
        count = min(self.config.masks_per_concept, len(masks))
        if count <= 0:
            raise ValueError(f"No masks are available for concept '{concept_id}'")
        rng = np.random.default_rng(self._concept_seed(self.config.seed, concept_id))
        indices = rng.choice(len(masks), size=count, replace=False)
        return np.asarray(masks[indices], dtype=np.uint8)

    @staticmethod
    def _concept_seed(base_seed: int, concept_id: str) -> int:
        digest = hashlib.sha256(f"{base_seed}:{concept_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    def _cache_path(self, artifact: ReplayCADArtifact, seed: int) -> Path:
        signature = self._generation_cache_signature()
        return (
            artifact.directory
            / "generated"
            / f"n{self.config.replay_samples_per_concept}_seed{seed}_g{signature}.npz"
        )

    def _generation_cache_signature(self) -> str:
        """Identify every setting that can change cached replay images."""
        payload = {
            "model_id": self.config.model_id,
            "resolution": self.config.resolution,
            "replay_samples_per_concept": self.config.replay_samples_per_concept,
            "inference_steps": self.config.inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "generation_batch_size": self.config.generation_batch_size,
            "negative_prompt": self.config.negative_prompt,
            "augment_masks": self.config.augment_masks,
            "max_mask_rotation_degrees": self.config.max_mask_rotation_degrees,
            "max_mask_shift_fraction": self.config.max_mask_shift_fraction,
            "torch_dtype": self.config.torch_dtype,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def _load_generated_cache(self, artifact: ReplayCADArtifact, seed: int) -> Optional[np.ndarray]:
        if not self.config.cache_generated:
            return None
        path = self._cache_path(artifact, seed)
        if not path.is_file():
            return None
        with np.load(path) as payload:
            return np.asarray(payload["images"], dtype=np.uint8)

    def _save_generated_cache(self, artifact: ReplayCADArtifact, seed: int, images: np.ndarray) -> None:
        if not self.config.cache_generated:
            return
        path = self._cache_path(artifact, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, images=np.asarray(images, dtype=np.uint8))

    def _write_run_config(self) -> None:
        path = self.root / "replaycad_config.json"
        payload = self.config.model_dump(mode="json")
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            comparable_keys = (
                "benchmark",
                "preset",
                "diffusion_family",
                "model_id",
                "resolution",
                "condition_dim",
                "latent_scaling_factor",
                "semantic_tokens",
                "mask_group_width",
                "mask_projection_width",
                "masks_per_concept",
                "compression_steps",
                "compression_batch_size",
                "semantic_learning_rate",
                "spatial_learning_rate",
                "timestep_sampling",
                "prompt_prefix",
                "initializer_word",
                "mask_backend",
                "sam_model_type",
                "torch_dtype",
                "seed",
            )
            mismatches = [key for key in comparable_keys if existing.get(key) != payload.get(key)]
            if mismatches:
                raise ValueError(
                    f"ReplayCAD artifact directory {self.root} contains an incompatible configuration; "
                    f"different fields: {mismatches}"
                )
            return
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_manifest(self) -> None:
        path = self.root / "manifest.json"
        payload = {
            "schema_version": 1,
            "concepts": [artifact.read_metadata() for artifact in self._stored_artifacts],
            "conditional_storage_bytes": sum(
                artifact.storage_size_bytes() for artifact in self._stored_artifacts
            ),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
