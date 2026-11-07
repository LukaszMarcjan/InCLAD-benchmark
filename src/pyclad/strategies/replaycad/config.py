from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ReplayCADPreset = Literal["paper-mvtec", "paper-visa", "inclad-ldm-256"]
ReplayCADPreset = Literal["paper-mvtec", "paper-visa", "inclad-ldm-256", "inclad-md-ldm-256"]
MaskBackendName = Literal["sam", "precomputed", "full-frame"]


class ReplayCADConfig(BaseModel):
    """Configuration of the faithful ReplayCAD mechanism.

    The two paper presets preserve the diffusion family, resolution and
    conditioning dimensions described by Hu et al.  Runtime values such as the
    number of compression steps remain explicit because the released research
    scripts used category-specific checkpoint steps.
    """

    # benchmark: Literal["btech", "dagm", "mpdd", "mvtec", "visa"]
    benchmark: Literal["btech", "dagm", "mpdd", "mvtec", "visa", "inclad-md"]
    preset: ReplayCADPreset
    artifact_dir: Path

    diffusion_family: Literal["ldm", "stable-diffusion"]
    model_id: str
    resolution: int = Field(gt=0)
    condition_dim: int = Field(gt=0)
    latent_scaling_factor: float = Field(default=0.18215, gt=0.0)

    semantic_tokens: int = Field(default=20, gt=0)
    mask_group_width: int = Field(gt=0)
    mask_projection_width: int = Field(gt=0)
    masks_per_concept: int = Field(default=5, ge=1, le=10)
    replay_samples_per_concept: int = Field(default=800, gt=0)

    compression_steps: int = Field(gt=0)
    compression_batch_size: int = Field(default=1, gt=0)
    semantic_learning_rate: float = Field(default=5e-3, gt=0.0)
    spatial_learning_rate: float = Field(default=5e-5, gt=0.0)
    timestep_sampling: Literal["uniform", "cubic"] = "cubic"

    inference_steps: int = Field(default=50, gt=0)
    guidance_scale: float = Field(default=10.0, ge=1.0)
    generation_batch_size: int = Field(default=8, gt=0)
    prompt_prefix: str = "a photo of a"
    initializer_word: str = "object"
    negative_prompt: str = ""

    mask_backend: MaskBackendName = "sam"
    sam_checkpoint: Optional[Path] = None
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_h"
    precomputed_mask_root: Optional[Path] = None
    mask_modes: Dict[str, Literal["corner-background", "area", "full-frame"]] = Field(default_factory=dict)
    mask_target_area_fractions: Dict[str, float] = Field(default_factory=dict)

    augment_masks: bool = True
    max_mask_rotation_degrees: float = Field(default=180.0, ge=0.0)
    max_mask_shift_fraction: float = Field(default=0.20, ge=0.0, le=1.0)

    device: str = "cuda"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    local_files_only: bool = False
    seed: int = 42
    cache_generated: bool = False

    @model_validator(mode="after")
    def validate_runtime_files(self):
        if self.mask_backend == "sam" and self.sam_checkpoint is None:
            raise ValueError("sam_checkpoint is required when mask_backend='sam'")
        if self.mask_backend == "precomputed" and self.precomputed_mask_root is None:
            raise ValueError("precomputed_mask_root is required when mask_backend='precomputed'")

        latent_values = self.resolution * self.resolution // 16
        if latent_values % self.mask_group_width != 0:
            raise ValueError(
                "The flattened four-channel latent mask must be divisible by mask_group_width: "
                f"{latent_values} % {self.mask_group_width} != 0"
            )
        projected_values = (latent_values // self.mask_group_width) * self.mask_projection_width
        if projected_values % self.condition_dim != 0:
            raise ValueError(
                "Projected mask features cannot be reshaped into conditioning tokens: "
                f"{projected_values} % {self.condition_dim} != 0"
            )
        return self

    @property
    def spatial_tokens(self) -> int:
        latent_values = self.resolution * self.resolution // 16
        projected_values = (latent_values // self.mask_group_width) * self.mask_projection_width
        return projected_values // self.condition_dim

    @classmethod
    def paper_preset(
        cls,
        benchmark: str,
        artifact_dir: Path,
        *,
        sam_checkpoint: Optional[Path] = None,
        precomputed_mask_root: Optional[Path] = None,
        mask_backend: MaskBackendName = "sam",
        device: str = "cuda",
        seed: int = 42,
        compression_steps: Optional[int] = None,
        replay_samples_per_concept: int = 800,
        masks_per_concept: int = 5,
        local_files_only: bool = False,
    ) -> "ReplayCADConfig":
        benchmark_key = benchmark.lower()
        if benchmark_key == "mvtec":
            return cls(
                benchmark="mvtec",
                preset="paper-mvtec",
                artifact_dir=artifact_dir,
                diffusion_family="ldm",
                model_id="CompVis/ldm-text2im-large-256",
                resolution=256,
                condition_dim=1280,
                mask_group_width=128,
                mask_projection_width=200,
                compression_steps=20_000 if compression_steps is None else compression_steps,
                compression_batch_size=16,
                generation_batch_size=8,
                replay_samples_per_concept=replay_samples_per_concept,
                masks_per_concept=masks_per_concept,
                initializer_word="screw",
                timestep_sampling="uniform",
                mask_backend=mask_backend,
                sam_checkpoint=sam_checkpoint,
                precomputed_mask_root=precomputed_mask_root,
                mask_modes={
                    "carpet": "full-frame",
                    "grid": "full-frame",
                    "leather": "full-frame",
                    "metal_nut": "corner-background",
                    "screw": "corner-background",
                    "tile": "full-frame",
                    "wood": "full-frame",
                },
                device=device,
                seed=seed,
                local_files_only=local_files_only,
            )
        if benchmark_key == "visa":
            return cls(
                benchmark="visa",
                preset="paper-visa",
                artifact_dir=artifact_dir,
                diffusion_family="stable-diffusion",
                model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                resolution=512,
                condition_dim=768,
                # The paper prints 128 -> 196, which cannot be reshaped to
                # 768-dimensional tokens. Its released main SD variant uses
                # 128 -> 192; that executable architecture is used here.
                mask_group_width=128,
                mask_projection_width=192,
                compression_steps=30_000 if compression_steps is None else compression_steps,
                compression_batch_size=2,
                generation_batch_size=2,
                replay_samples_per_concept=replay_samples_per_concept,
                masks_per_concept=masks_per_concept,
                initializer_word="sculpture",
                mask_backend=mask_backend,
                sam_checkpoint=sam_checkpoint,
                precomputed_mask_root=precomputed_mask_root,
                device=device,
                seed=seed,
                local_files_only=local_files_only,
            )
        if benchmark_key in {"btech", "dagm", "mpdd"}:
            # ReplayCAD publishes dataset-specific configurations only for
            # MVTec AD and VisA.  The remaining InCLAD datasets use one
            # explicit, shared adaptation: the paper's MVTec LDM-256
            # architecture with a neutral initializer and no MVTec-specific
            # mask-mode overrides.
            base = cls.paper_preset(
                benchmark="mvtec",
                artifact_dir=artifact_dir,
                sam_checkpoint=sam_checkpoint,
                precomputed_mask_root=precomputed_mask_root,
                mask_backend=mask_backend,
                device=device,
                seed=seed,
                compression_steps=compression_steps,
                replay_samples_per_concept=replay_samples_per_concept,
                masks_per_concept=masks_per_concept,
                local_files_only=local_files_only,
            )
            payload = base.model_dump(mode="python")
            payload.update(
                {
                    "benchmark": benchmark_key,
                    "preset": "inclad-ldm-256",
                    "initializer_word": "object",
                    # SAM returns no automatic mask for DAGM Class8.  Use the
                    # same explicit full-frame policy as the texture classes
                    # in the MVTec preset instead of aborting the benchmark.
                    "mask_modes": {"Class8": "full-frame"} if benchmark_key == "dagm" else {},
                }
            )
            return cls.model_validate(payload)
        if benchmark_key in {"inclad-md", "inclad_md", "multidataset", "multidataset_seq_v1"}:
            # InCLAD-MD streams 27 concepts from all five datasets, named
            # "<dataset>__<category>". One diffusion model serves the whole stream,
            # so it uses the shared InCLAD adaptation of the btech/dagm/mpdd
            # branch (MVTec LDM-256 architecture, neutral initializer); the VisA
            # concepts therefore do not get the SD-1.5/512 setup of the VisA preset.
            # The per-concept mask policy of the single-dataset runs is kept by
            # prefixing it with the source dataset.
            base = cls.paper_preset(
                benchmark="dagm",
                artifact_dir=artifact_dir,
                sam_checkpoint=sam_checkpoint,
                precomputed_mask_root=precomputed_mask_root,
                mask_backend=mask_backend,
                device=device,
                seed=seed,
                compression_steps=compression_steps,
                replay_samples_per_concept=replay_samples_per_concept,
                masks_per_concept=masks_per_concept,
                local_files_only=local_files_only,
            )
            mvtec_modes = cls.paper_preset(
                benchmark="mvtec",
                artifact_dir=artifact_dir,
                sam_checkpoint=sam_checkpoint,
                precomputed_mask_root=precomputed_mask_root,
                mask_backend=mask_backend,
                device=device,
                seed=seed,
            ).mask_modes
            mask_modes = {f"mvtec__{concept}": mode for concept, mode in mvtec_modes.items()}
            mask_modes.update({f"dagm__{concept}": mode for concept, mode in base.mask_modes.items()})
            payload = base.model_dump(mode="python")
            payload.update({"benchmark": "inclad-md", "preset": "inclad-md-ldm-256", "mask_modes": mask_modes})
            return cls.model_validate(payload)
        raise ValueError(
            "ReplayCAD presets support InCLAD benchmarks: btech, dagm, mpdd, mvtec, visa, inclad-md"
        )
