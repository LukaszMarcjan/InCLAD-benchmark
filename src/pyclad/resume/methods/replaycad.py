"""Checkpointable ReplayCAD: a resumable PatchCore detector plus the strategy
hooks that put the generative memory back the way an uninterrupted run left it.

ReplayCAD is the method that makes this whole layer worth building. Measured on
MVTec with a resnet18 backbone, its per-concept fit grows from 2,872 s at the
first concept to 13,308 s at the seventh, because PatchCore's greedy coreset is
quadratic in an ever-growing replay-union; concepts 11-15 alone project to about
65 h. No single 48 h job finishes that, so the run has to survive a restart.

Two halves of state have to come back, and only one of them lives in the model:

* the **detector**: the coreset memory bank, the feature-map size and the
  training-quantile threshold. The bank is the expensive part (1.86 GB at seven
  concepts, 453,017 x 1024 float32) and is the only thing here that cannot be
  recomputed without re-running the entire replay pipeline;
* the **memory activation**: ``ReplayCADMemory`` deliberately separates
  ``_stored_artifacts`` (every condition on disk) from ``_artifacts`` (the
  concepts *this* scenario has learned), because activating everything on disk
  would replay future concepts at the first training step. That separation is
  precisely the hook resume needs — but it also means that after a restart
  ``generate_previous()`` returns **nothing** unless the completed concepts are
  re-activated explicitly, in order. A resumed run would then train concept *k*
  with no replay at all and report it as a ReplayCAD number.

The third thing is subtler and lives in the diffusion backend's tokenizer.
``compress()`` reseeds torch globally and then calls ``_install_tokens``, which
grows the text-encoder embedding matrix only when the vocabulary actually grew.
How many random draws that consumes therefore depends on how many concepts'
tokens were installed *in this process*: 20*(k-1) for an uninterrupted run,
zero for a resumed run with a warm generation cache (the cache short-circuits
``generate``, so ``_install_tokens`` never runs for the earlier concepts). A
different draw count means a different mask-projection init and different noise
for all ~18k optimisation steps. :meth:`ResumableReplayCADStrategy.install_prior_tokens`
replays those installs before the first ``compress()``, which is what makes a
resumed LDM run bit-identical rather than merely similar. Restoring a saved
global RNG state would *not* fix it, because ``compress`` discards the global
state at its own ``torch.manual_seed`` call.

What is still not bit-identical: on the stable-diffusion family (VisA), the VAE
encode path calls ``latent_dist.sample()`` with no generator, drawing from the
global stream inside library code pyCLAD does not own. Those runs are recorded
as ``statistical`` in the envelope rather than being quietly claimed as exact.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from pyclad.models.patchcore.config import PatchCoreConfig
from pyclad.models.patchcore.patchcore import PatchCore
from pyclad.resume.fingerprint import fingerprint_from_config
from pyclad.resume.protocol import (
    DETERMINISM_BIT_IDENTICAL,
    DETERMINISM_STATISTICAL,
    SIDECAR_KEY,
    CheckpointCorruptionError,
    CheckpointPayloadError,
    ResumeError,
    require_payload_keys,
)
from pyclad.strategies.replaycad.artifacts import ReplayCADArtifact
from pyclad.strategies.replaycad.config import ReplayCADConfig
from pyclad.strategies.replaycad.memory import ReplayCADMemory
from pyclad.strategies.replaycad.strategy import ReplayCADStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointedPatchCore",
    "ResumableReplayCADStrategy",
    "replaycad_determinism",
    "PATCHCORE_FINGERPRINT_FIELDS",
    "REPLAYCAD_FINGERPRINT_FIELDS",
]

#: PatchCore config fields that change the numbers a restored bank produces
#: while leaving the restored state structurally valid. ``device`` is
#: deliberately absent: resuming on a different node is the point of the layer.
#: ``batch_size`` is absent too — the frozen extractor is evaluated per image,
#: so batching moves nothing but floating-point association order.
PATCHCORE_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "backbone_name",
    "backbone_return_nodes",
    "pretrained_backbone",
    "input_size",
    "pretrain_embed_dimension",
    "target_embed_dimension",
    "patchsize",
    "patchstride",
    "coreset_sampling_ratio",
    "coreset_projection_dimension",
    "coreset_starting_points",
    "n_neighbors",
    "threshold",
    "threshold_quantile",
    "normalize_mean",
    "normalize_std",
    "smoothing_sigma",
    "random_seed",
)

#: ReplayCAD config fields that change the replay images, hence the bank.
#: ``ReplayCADMemory._write_run_config`` already guards most of the compression
#: half against the artifact directory; this list exists because that guard
#: knows nothing about the *detector* and raises much later, with a message that
#: does not mention resuming.
REPLAYCAD_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "seed",
    "model_id",
    "diffusion_family",
    "resolution",
    "replay_samples_per_concept",
    "semantic_tokens",
    "condition_dim",
    "torch_dtype",
    "cache_generated",
    "artifact_dir",
)


def _torch_load(path: Path) -> Any:
    """Load a torch file with ``weights_only`` when the runtime supports it.

    A seven-line copy of the shim in ``strategies/replaycad/backend.py`` rather
    than an import of that module's private helper: copying is additive, while
    depending on another subpackage's underscore name is a coupling that breaks
    silently when it moves.
    """
    parameters = inspect.signature(torch.load).parameters
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def _module_signature(module: torch.nn.Module) -> str:
    """Hash a frozen module's weights so a swapped backbone cannot go unnoticed.

    The memory bank is a set of coordinates in *this* feature extractor's
    embedding space. Restore the bank against a differently initialised
    backbone and every nearest-neighbour distance is meaningless — but nothing
    raises, the shapes all agree, and the run produces a full metric matrix.
    """
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().to("cpu")
        if array.dtype in (torch.bfloat16, torch.float16):
            array = array.float()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.contiguous().numpy().tobytes())
    return digest.hexdigest()


def replaycad_determinism(config: Optional[ReplayCADConfig]) -> Dict[str, Any]:
    """Describe honestly how reproducible a resumed run of this config is.

    The envelope carries this and the runner mirrors it into the result JSON, so
    a run that resumed across a merely-statistical boundary can never later be
    mistaken for an exactly reproducible one.
    """
    caveats = ["cudnn_nondeterminism"]
    if config is None:
        return {"guarantee": DETERMINISM_BIT_IDENTICAL, "caveats": caveats}

    guarantee = DETERMINISM_BIT_IDENTICAL
    if config.diffusion_family == "stable-diffusion":
        # AutoencoderKL.encode(...).latent_dist.sample() takes no generator and
        # draws from the global stream inside diffusers.
        caveats.append("vae_sample_unseeded")
        guarantee = DETERMINISM_STATISTICAL
    if not config.cache_generated:
        # Without the cache every step regenerates its replay images, and on SD
        # they are not reproducible byte-for-byte.
        caveats.append("replay_regenerated_without_cache")
        guarantee = DETERMINISM_STATISTICAL
    caveats.append("prior_tokens_reinstalled_on_resume")
    return {"guarantee": guarantee, "caveats": caveats}


class CheckpointedPatchCore(PatchCore):
    """PatchCore that can be written to disk and read back exactly.

    Only three attributes are persisted inline, plus the memory bank as a
    sidecar. Everything else PatchCore owns is either rebuilt from the config
    (the extractor stack, the segmentor, the coreset sampler — all stateless or
    config-derived) or is a cache that must *not* survive (see
    :meth:`_invalidate_prediction_cache`).

    The one addition beyond the bank is the backbone. ``PatchCoreConfig``
    defaults ``pretrained_backbone`` to ``False``, in which case torchvision
    draws the weights from the global RNG at construction time. Whether a second
    process reproduces those weights depends on exactly what consumed the global
    stream between ``set_global_seeds`` and the constructor — a property no part
    of this layer can guarantee. So the weights are persisted in that case, and
    a signature is recorded and checked in every case.
    """

    STATE_KIND: ClassVar[str] = "replaycad"
    STATE_VERSION: ClassVar[int] = 1

    def __init__(
        self,
        config: Optional[PatchCoreConfig] = None,
        *,
        replaycad_config: Optional[ReplayCADConfig] = None,
    ):
        super().__init__(config)
        self._replaycad_config = replaycad_config
        self._replaycad_memory: Optional[ReplayCADMemory] = None

    # ---------------------------------------------------------------- wiring

    def attach_replaycad_memory(self, memory: ReplayCADMemory) -> None:
        """Let the detector's checkpoint carry the generative memory's identity.

        The two halves of ReplayCAD's state belong in one envelope: a bank built
        from replay images of concepts A,B is only valid next to a memory that
        has A and B activated, in that order. Wiring happens once, in
        :class:`ResumableReplayCADStrategy`, so callers never have to remember it.
        """
        self._replaycad_memory = memory
        if self._replaycad_config is None:
            self._replaycad_config = memory.config

    @property
    def replaycad_config(self) -> Optional[ReplayCADConfig]:
        return self._replaycad_config

    def determinism(self) -> Dict[str, Any]:
        return replaycad_determinism(self._replaycad_config)

    def _invalidate_prediction_cache(self) -> None:
        """Drop the identity-keyed score-map cache.

        ``score_maps`` reuses ``_last_prediction_maps`` when ``data is
        self._last_prediction_data``. After a restore the cache can only alias a
        reused array from a previous process's object graph, so keeping it can
        only hand back stale maps. ``PatchCore`` has no such method; this is the
        hook ``pyclad.resume.plan.apply_resume`` looks for.
        """
        self._last_prediction_data = None
        self._last_prediction_maps = None

    # ------------------------------------------------------------ save/load

    def state_fingerprint(self) -> Dict[str, Any]:
        # fingerprint_from_config raises when a listed field is absent, so a
        # config rename cannot quietly shrink what the fingerprint protects.
        fingerprint: Dict[str, Any] = {
            f"patchcore.{key}": value
            for key, value in fingerprint_from_config(self.config, PATCHCORE_FINGERPRINT_FIELDS).items()
        }
        if self._replaycad_config is not None:
            fingerprint.update(
                {
                    f"replaycad.{key}": value
                    for key, value in fingerprint_from_config(
                        self._replaycad_config, REPLAYCAD_FINGERPRINT_FIELDS
                    ).items()
                }
            )
        return fingerprint

    def save_state(self) -> Dict[str, Any]:
        if self._memory_bank is None or self._feature_map_size is None:
            raise ResumeError(
                "CheckpointedPatchCore.save_state() was called before fit(); there is no memory bank to persist. "
                "The state callback must run at after_training, never before."
            )

        bank = np.ascontiguousarray(self._memory_bank, dtype=np.float32)
        payload: Dict[str, Any] = {
            "feature_map_size": [int(value) for value in self._feature_map_size],
            # Present-but-null is meaningful: PatchCore.fit() sets 0.0 when it
            # cannot score, and require_payload_keys distinguishes "absent"
            # (a broken payload) from "deliberately unset".
            "threshold": None if self._threshold is None else float(self._threshold),
            "memory_bank_shape": [int(dim) for dim in bank.shape],
            "memory_bank_dtype": str(bank.dtype),
            "backbone_signature": _module_signature(self.module),
            "backbone_state": self._backbone_state(),
            # np.save, uncompressed, on purpose: zlib over gigabytes of float32
            # embeddings saves almost nothing and costs minutes per concept.
            SIDECAR_KEY: {"memory_bank": bank},
        }
        payload["replaycad"] = self._replaycad_state()
        return payload

    def load_state(self, payload: Mapping[str, Any]) -> None:
        require_payload_keys(
            payload,
            ("feature_map_size", "threshold", "backbone_signature"),
            state_kind=self.STATE_KIND,
        )
        sidecars = payload.get(SIDECAR_KEY) or {}
        require_payload_keys(sidecars, ("memory_bank",), state_kind=self.STATE_KIND)

        # ---- validate everything into locals; mutate nothing until the end.
        feature_map_size = tuple(int(value) for value in payload["feature_map_size"])
        if len(feature_map_size) != 2 or any(value <= 0 for value in feature_map_size):
            raise CheckpointPayloadError(
                f"'{self.STATE_KIND}' checkpoint has an invalid feature_map_size {feature_map_size}; "
                "expected two positive integers"
            )

        threshold = payload["threshold"]
        if threshold is not None:
            threshold = float(threshold)

        bank = np.ascontiguousarray(np.asarray(sidecars["memory_bank"]), dtype=np.float32)
        if bank.ndim != 2 or bank.shape[0] == 0:
            raise CheckpointPayloadError(
                f"'{self.STATE_KIND}' memory bank sidecar has shape {bank.shape}; expected a non-empty (N, D) matrix"
            )
        expected_shape = payload.get("memory_bank_shape")
        if expected_shape is not None and tuple(int(dim) for dim in expected_shape) != bank.shape:
            raise CheckpointPayloadError(
                f"'{self.STATE_KIND}' memory bank sidecar has shape {bank.shape} but the payload recorded "
                f"{tuple(int(dim) for dim in expected_shape)}"
            )
        if bank.shape[1] != int(self.config.target_embed_dimension):
            raise CheckpointPayloadError(
                f"'{self.STATE_KIND}' memory bank has {bank.shape[1]} dimensions but this run's PatchCore "
                f"produces {self.config.target_embed_dimension}. The checkpoint belongs to a different "
                "feature configuration."
            )

        backbone_state = self._validated_backbone_state(payload)

        # ---- commit.
        if backbone_state is not None:
            self.module.load_state_dict(backbone_state, strict=True)
        else:
            self._assert_backbone_signature(str(payload["backbone_signature"]))

        self._memory_bank = bank
        self._feature_map_size = feature_map_size
        self._threshold = threshold
        self._invalidate_prediction_cache()
        # sklearn's index is a pure function of the bank, so it is rebuilt
        # rather than pickled — pickling an estimator would force the whole
        # checkpoint to be read back with weights_only=False.
        self._nn_index = None
        self._fit_nn_index()

        logger.info(
            "Restored PatchCore state: memory bank %s, feature map %s, threshold %s",
            bank.shape,
            feature_map_size,
            threshold,
        )

    # -------------------------------------------------------------- internals

    def _backbone_state(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.config.pretrained_backbone:
            # torchvision's published weights are a fixed artifact; persisting
            # ~45 MB of them next to every checkpoint would be pure waste.
            return None
        return {key: tensor.detach().to("cpu").clone() for key, tensor in self.module.state_dict().items()}

    def _replaycad_state(self) -> Dict[str, Any]:
        memory = self._replaycad_memory
        if memory is None:
            return {}
        return {
            # The activation order *is* the replay order, and generate_previous
            # iterates this list, so it round-trips as a list and never a set.
            "activated_concepts": [artifact.concept_id for artifact in memory.artifacts],
            "generation_cache_signature": memory._generation_cache_signature(),
            "artifact_dir": str(memory.root),
            "counters": {
                # Accounting only, but it is published accounting: without it a
                # resumed run reports the compression cost of its last segment
                # as if it were the cost of the whole run.
                "generated_samples": int(memory._generated_samples),
                "generation_seconds": float(memory._generation_seconds),
                "compression_seconds": float(memory._compression_seconds),
            },
        }

    def _validated_backbone_state(self, payload: Mapping[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        raw = payload.get("backbone_state")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise CheckpointPayloadError(
                f"'{self.STATE_KIND}' backbone_state must be a mapping of parameter name to tensor, "
                f"got {type(raw).__name__}"
            )
        live = self.module.state_dict()
        missing = sorted(set(live) - set(raw))
        unexpected = sorted(set(raw) - set(live))
        if missing or unexpected:
            raise CheckpointPayloadError(
                f"'{self.STATE_KIND}' backbone_state does not match this run's feature extractor; "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        restored: Dict[str, torch.Tensor] = {}
        for key, value in raw.items():
            tensor = torch.as_tensor(value)
            if tuple(tensor.shape) != tuple(live[key].shape):
                raise CheckpointPayloadError(
                    f"'{self.STATE_KIND}' backbone_state['{key}'] has shape {tuple(tensor.shape)}, "
                    f"this run's backbone expects {tuple(live[key].shape)}"
                )
            restored[key] = tensor
        return restored

    def _assert_backbone_signature(self, expected: str) -> None:
        actual = _module_signature(self.module)
        if actual == expected:
            return
        raise CheckpointPayloadError(
            f"'{self.STATE_KIND}' checkpoint was written against a different feature extractor "
            f"(signature {expected[:12]}..., this run's is {actual[:12]}...). The restored memory bank holds "
            "coordinates in the old extractor's embedding space, so every nearest-neighbour distance would be "
            "meaningless while the run still produced a full metric matrix. Rebuild the backbone identically, "
            "or start a fresh run."
        )


class ResumableReplayCADStrategy(ReplayCADStrategy):
    """ReplayCAD strategy that can be put back together after a restart.

    ``name()`` is *not* overridden. It stays ``"ReplayCAD"`` because it is the
    join key in the results-aggregation scripts, and renaming it would silently
    drop every resumed run out of the comparison against the published
    benchmark. The v2-ness is recorded in ``additional_info()["resume"]``.
    """

    def __init__(
        self,
        model: CheckpointedPatchCore,
        memory: ReplayCADMemory,
        *,
        validate_caches: bool = True,
    ):
        super().__init__(model, memory)
        self._validate_caches = bool(validate_caches)
        self._resume_info: Optional[Mapping[str, Any]] = None
        attach = getattr(model, "attach_replaycad_memory", None)
        if callable(attach):
            attach(memory)
        else:
            logger.warning(
                "Model %s is not a CheckpointedPatchCore; its checkpoints will not carry the ReplayCAD "
                "memory activation and cannot be resumed",
                type(model).__name__,
            )

    # -------------------------------------------------------------- reporting

    def set_resume_info(self, info: Mapping[str, Any]) -> None:
        self._resume_info = dict(info)

    def additional_info(self) -> Dict[str, Any]:
        payload = super().additional_info()
        if self._resume_info is not None:
            payload["resume"] = dict(self._resume_info)
        return payload

    # ---------------------------------------------------------------- resume

    def resume_restore(self, payload: Mapping[str, Any], plan: Any) -> None:
        """Restore step 6: everything the strategy owns, in dependency order.

        Cheap checks first, then the cache scan, then the diffusion pipeline —
        loading a multi-gigabyte pipeline only to discover that the replay cache
        is black images wastes the very minutes this layer exists to save.
        """
        block = payload.get("replaycad")
        if not isinstance(block, Mapping) or "activated_concepts" not in block:
            raise CheckpointPayloadError(
                "The ReplayCAD checkpoint carries no 'replaycad' block, so the generative memory cannot be "
                "re-activated. Without it generate_previous() returns nothing and the resumed run would train "
                "with no replay while still reporting itself as ReplayCAD."
            )

        if not self._memory.config.cache_generated:
            raise ResumeError(
                "Resuming ReplayCAD requires --replaycad-cache-generated. Without the cache every step "
                "regenerates its replay images from the diffusion pipeline, which costs hours per concept and, "
                "on the stable-diffusion family, does not reproduce byte-for-byte."
            )

        activated = [str(name) for name in block["activated_concepts"]]
        self._assert_plan_agrees(activated, plan)
        self.activate_completed(activated)
        self._restore_memory_counters(block.get("counters"))
        self._check_cache_signature(block.get("generation_cache_signature"))

        if self._validate_caches:
            self.validate_generated_caches()

        self.install_prior_tokens()
        # Mirror the strategy's own discipline: the detector, not the pipeline,
        # owns the GPU until the next generation call.
        self._memory.release_device_memory()

    def activate_completed(self, concept_ids: Sequence[str]) -> None:
        """Move completed concepts from ``_stored_artifacts`` into ``_artifacts``.

        ``ReplayCADMemory`` keeps the two apart so that a warm artifact
        directory does not replay concepts the scenario has not reached yet.
        That is correct, and it is also why a resumed run replays nothing until
        this runs.
        """
        memory = self._memory
        wanted = [str(name) for name in concept_ids]
        stored = {artifact.concept_id: artifact for artifact in self._artifact_pool(memory)}

        already = [artifact.concept_id for artifact in memory.artifacts]
        extra = [name for name in already if name not in set(wanted)]
        if extra:
            # A pre-split build activates everything it finds on disk. A run that
            # died *inside* concept k leaves k's artifact directory behind, so the
            # deployed memory comes up replaying a concept the scenario never
            # completed. Dropping it here is the whole point: replay must contain
            # the completed concepts and nothing else.
            logger.warning(
                "ReplayCAD memory came up with %s activated, which the checkpoint does not list as completed; "
                "these will be dropped from the replay set (artifacts of a concept that never finished)",
                extra,
            )

        active = {artifact.concept_id for artifact in memory.artifacts}
        for concept_id in wanted:
            artifact = stored.get(concept_id)
            if artifact is None:
                raise CheckpointCorruptionError(
                    f"No ReplayCAD condition for completed concept '{concept_id}' under {memory.root}. "
                    "The checkpoint and the artifact directory belong to different runs, or the artifact "
                    "directory was written inside a per-job output directory and did not survive the restart."
                )
            if not artifact.complete:
                raise CheckpointCorruptionError(
                    f"ReplayCAD condition for '{concept_id}' in {artifact.directory} is incomplete "
                    "(condition.pt, masks.npz and metadata.json must all exist)"
                )
            if concept_id in active:
                # Single-list build: construction already activated everything it
                # found on disk, so there is nothing left to move.
                continue
            activate = getattr(memory, "_activate", None)
            if activate is not None:
                activate(artifact)
            else:
                memory._artifacts.append(artifact)

        self._enforce_replay_order(wanted)

    @staticmethod
    def _artifact_pool(memory: Any) -> list:
        """Every artifact the memory knows about, on either build of ReplayCADMemory.

        Newer builds split the pool: ``_stored_artifacts`` is everything found on
        disk, ``_artifacts`` only what the scenario has actually reached. Older
        builds keep one list and activate it wholesale at construction. Resume has
        to cope with whichever is deployed, because this file and
        ``strategies/replaycad/memory.py`` reach the cluster as separate copies and
        do drift apart -- assuming the split exists is what made a resumed run die
        with a bare AttributeError instead of resuming.
        """
        pool = getattr(memory, "_stored_artifacts", None)
        if pool is not None:
            return list(pool)

        # No split: re-read the artifact directory rather than trusting the active
        # list, which on a lazily-activating build holds only what the scenario has
        # reached so far. Reporting "no condition for concept X" while X sits on
        # disk would send the reader hunting for a missing directory that is there.
        rescan = getattr(memory, "_load_existing_artifacts", None)
        found = list(memory.artifacts)
        if callable(rescan):
            try:
                seen = {artifact.concept_id for artifact in found}
                found += [a for a in rescan() if a.concept_id not in seen]
            except Exception:  # noqa: BLE001 - a failed rescan must not mask the real error
                logger.warning("Could not re-scan the ReplayCAD artifact directory; using the active list only")
        return found

    def _enforce_replay_order(self, wanted: Sequence[str]) -> None:
        """Make the active list read exactly ``wanted``, in that order.

        Replay order is load-bearing -- ``generate_previous()`` concatenates in it
        -- and a single-list build orders artifacts by what it found on disk, which
        is not the order the scenario completed them in. Reordering is safe because
        the artifacts are the same objects; only the sequence changes.
        """
        memory = self._memory
        wanted = list(wanted)
        activated = [artifact.concept_id for artifact in memory.artifacts]
        if activated == wanted:
            return

        by_id = {artifact.concept_id: artifact for artifact in memory.artifacts}
        missing = [name for name in wanted if name not in by_id]
        if missing:
            raise ResumeError(
                f"ReplayCAD replay set is missing {missing} after activation; expected {wanted}, got {activated}. "
                "Replay order is load-bearing: generate_previous() concatenates in this order."
            )
        # Rewrite rather than append: this both fixes the order and drops any
        # concept the deployed build activated on its own (see activate_completed).
        memory._artifacts[:] = [by_id[name] for name in wanted]
        logger.info("Resume: re-activated %d ReplayCAD concepts for replay: %s", len(activated), activated)

    def install_prior_tokens(self) -> None:
        """Replay the tokenizer growth an uninterrupted process would have done.

        ``compress()`` reseeds torch and then installs the current concept's
        tokens; the number of random draws that consumes depends on the size of
        the vocabulary it starts from. With a warm generation cache the earlier
        concepts' ``generate`` calls never happen after a restart, so their
        tokens are never installed and the vocabulary is short by 20 entries per
        completed concept. Same seed, different draw count, different mask
        projection, different noise for every one of the ~18k steps.
        """
        memory = self._memory
        artifacts = memory.artifacts
        if not artifacts:
            return

        backend = getattr(memory, "_backend", None)
        if backend is None:
            # Guarding _load/_install_tokens but not the attribute that yields them
            # is the same one-level-too-shallow mistake that made the first resume
            # attempt die. Missing tokens change the RNG draw count, so the resumed
            # run is not bit-identical to an uninterrupted one -- but the replay set
            # itself is unaffected, so this warns rather than refusing to resume.
            logger.warning(
                "ReplayCAD memory exposes no backend attribute; skipping tokenizer pre-installation. "
                "The replay set is correct, but generation after this point will not reproduce an "
                "uninterrupted run bit-for-bit."
            )
            return
        load = getattr(backend, "_load", None)
        install = getattr(backend, "_install_tokens", None)
        if not callable(load) or not callable(install):
            logger.info(
                "ReplayCAD backend %s exposes no tokenizer; nothing to pre-install before the first compress()",
                type(backend).__name__,
            )
            return

        # Mandatory: the tokenizer is assigned inside _load(), so calling
        # _install_tokens on a cold backend raises AttributeError on None.
        load()
        installed = 0
        for artifact in artifacts:
            condition = _torch_load(artifact.condition_path)
            token_names = [str(name) for name in condition["token_names"]]
            try:
                install(token_names, initialize=False)
            except TypeError:
                # Older backends take `initialize` positionally.
                install(token_names, False)
            installed += len(token_names)
        logger.info(
            "Resume: pre-installed %d semantic tokens from %d completed concepts so the next compress() starts "
            "from the vocabulary an uninterrupted run would have had",
            installed,
            len(artifacts),
        )

    def validate_generated_caches(self) -> None:
        """Refuse to rebuild the detector on black images.

        Verified real, not hypothetical: every VisA replay cache in the observed
        runs is 800x512x512x3 of exact zeros (611 KB compressed, against 113 MB
        for a healthy MVTec cache), produced by float16 SD1.5.
        ``generate_previous`` only checks the *count* of cached samples, never
        their content, so without this a resume happily fits PatchCore on a
        uniform batch and reports the result as a ReplayCAD score.
        """
        memory = self._memory
        if not memory.config.cache_generated:
            return

        cache_path = getattr(memory, "_cache_path", None)
        concept_seed = getattr(memory, "_concept_seed", None)
        if not callable(cache_path) or not callable(concept_seed):
            raise ResumeError(
                "This ReplayCADMemory exposes no _cache_path/_concept_seed, so the replay caches cannot be "
                "checked before the detector is refit on them. That check is not optional: every observed VisA "
                "cache was a batch of exact zeros, and fitting on it yields a plausible-looking but meaningless "
                "score. Sync src/pyclad/strategies/replaycad/memory.py, or pass validate_caches=False to accept "
                "the risk deliberately."
            )

        checked = 0
        skipped = 0
        for artifact in memory.artifacts:
            # Bound, not unbound on the class: this form works whether the deployed
            # build declares _concept_seed as a @staticmethod or a plain method, and
            # it respects a subclass override instead of silently computing a cache
            # path that does not exist.
            seed = concept_seed(memory.config.seed, artifact.concept_id)
            path = cache_path(artifact, seed)
            if not path.is_file():
                self._warn_legacy_cache(artifact, path)
                skipped += 1
                continue
            checked += 1
            with np.load(path) as handle:
                images = np.asarray(handle["images"])
            try:
                if images.size == 0:
                    raise CheckpointCorruptionError(f"ReplayCAD replay cache {path} is empty")
                minimum = int(images.min())
                maximum = int(images.max())
                if minimum == maximum:
                    raise CheckpointCorruptionError(
                        f"ReplayCAD replay cache {path} is a constant image batch (every pixel == {minimum}). "
                        "Fitting the detector on it would produce a plausible-looking but meaningless score; "
                        "delete the cache and regenerate, or switch torch_dtype away from float16."
                    )
            finally:
                del images

        if skipped and not checked:
            raise ResumeError(
                f"None of the {skipped} replay caches could be located, so not one was checked before the "
                "detector is refit on them. Either the artifact directory belongs to a different run, or the "
                "deployed _cache_path uses a different filename convention than this adapter expects. Silently "
                "skipping every check is how an all-zero cache reaches a published metric matrix; refusing "
                "instead. Pass validate_caches=False to override deliberately."
            )

    # -------------------------------------------------------------- internals

    def _assert_plan_agrees(self, activated: Sequence[str], plan: Any) -> None:
        """Cross-check the checkpoint's replay set against the scenario's plan.

        The two are written by different mechanisms — the memory's activation
        list and the manifest's committed index — and a disagreement means one
        of them is describing a different run.
        """
        skip = [str(name) for name in getattr(plan, "skip_concepts", ())]
        order = [str(name) for name in getattr(plan, "concept_order", ())]
        pending = bool(getattr(plan, "evaluate_pending_concept", False))

        expected = list(skip)
        if pending:
            # The pending checkpoint was written after training concept k, so
            # compress(k) already ran and k is activated; the scenario will only
            # re-run k's evaluation sweep.
            index = int(getattr(plan, "resume_index", len(skip)))
            if index < len(order):
                expected.append(order[index])

        if list(activated) != expected:
            raise ResumeError(
                "The ReplayCAD checkpoint's replay set disagrees with the resume plan.\n"
                f"  checkpoint: {list(activated)}\n"
                f"  plan:       {expected}"
            )

    def _restore_memory_counters(self, counters: Any) -> None:
        if not isinstance(counters, Mapping):
            return
        memory = self._memory
        memory._generated_samples = int(counters.get("generated_samples", memory._generated_samples))
        memory._generation_seconds = float(counters.get("generation_seconds", memory._generation_seconds))
        memory._compression_seconds = float(counters.get("compression_seconds", memory._compression_seconds))

    def _check_cache_signature(self, recorded: Any) -> None:
        """Warn when every cached replay batch has just been orphaned.

        The signature hashes ``generation_batch_size`` among other things, so
        relaunching the chain on a GPU that wants a different batch size makes
        every cache file unreachable by name. The run still completes — it just
        regenerates hours of images nobody asked it to.
        """
        if not recorded:
            return
        current = self._memory._generation_cache_signature()
        if str(recorded) == current:
            return
        logger.warning(
            "ReplayCAD generation cache signature changed (%s -> %s); every cached replay batch under %s is now "
            "unreachable and will be regenerated. Check --replaycad-generation-batch-size, inference steps, "
            "guidance scale and torch_dtype against the original launch.",
            recorded,
            current,
            self._memory.root,
        )

    @staticmethod
    def _warn_legacy_cache(artifact: ReplayCADArtifact, expected: Path) -> None:
        legacy = expected.parent / f"{expected.name.split('_g')[0]}.npz"
        if legacy != expected and legacy.is_file():
            logger.warning(
                "ReplayCAD concept '%s' has a pre-signature cache at %s which current code will never find "
                "(it looks for %s); those images will be regenerated",
                artifact.concept_id,
                legacy,
                expected.name,
            )
