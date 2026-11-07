"""Checkpointable One-for-More.

One-for-More is the odd one out among the three resumable methods: it has no
strategy of its own. It runs as a bare :class:`~pyclad.models.model.Model` under
``NaiveStrategy`` through the *concept-incremental* scenario, so everything that
has to survive a restart lives on the model, and the strategy contributes
nothing but provenance and a wiring tripwire.

What has to be persisted, and why each item is not optional:

* **the trainable weights** of the conditioner and the denoiser - the whole
  point;
* **the GPM bases** held by :class:`IterativeSubspaceProjector`. This is the
  only state in the method that is genuinely *unrecoverable*.
  ``_collect_projection_subspaces`` is called with the current concept's data
  only, so rebuilding a basis needs every previous concept's training set -
  which a continual benchmark has by definition already discarded. Weights can
  be retrained; a lost basis cannot be recomputed at any price. That asymmetry
  is why :meth:`ResumableOneForMore.save_variants` also emits a small, never-pruned
  basis-only snapshot at every concept;
* **the projector's update counter**. It looks like bookkeeping, but
  ``_sample_columns`` seeds on ``random_seed + _updates`` (projection.py:55) and
  at paper defaults four of the basis-bearing layers exceed ``max_columns`` and
  are subsampled. Reset the counter and those four bases silently diverge;
* **the task counter** ``_tasks_seen``. It selects the epoch budget, offsets the
  dataloader-shuffle and patch-corruption seeds, and gates gradient projection
  entirely (``if self._tasks_seen > 0``);
* **the calibrated threshold**. Losing it degrades only F1 - ``predict`` falls
  back to a quantile of the *test* batch it was handed - while ROC-AUC and AP,
  computed from raw scores, still look perfect. That is a resume bug which reads
  as a modelling result;
* **the global torch RNG state**. Unlike UCAD, One-for-More does not derive all
  of its randomness from the task index: ``training_loss`` draws diffusion
  timesteps with ``torch.randint`` and noise with ``torch.randn_like``
  (architecture.py:321-327, neither takes a ``generator=``), and the conditioner
  applies dropout. Those consume the *global* stream, so bit-identical
  continuation requires carrying it across the restart.

Carrying the stream is necessary but, for a *committed* resume, not sufficient -
and this is the one place where measurement contradicted the design. The
checkpoint is written at ``after_training``, and the claim was that nothing
between that point and the next ``fit()`` consumes the global stream, because
the evaluation sweep goes through ``reconstruct``, which builds its own seeded
CPU generator (architecture.py:341) and runs under ``eval()`` so dropout is the
identity. Both of those are true, and the conclusion is still wrong:
``_prepare_batches`` builds its inference loaders with ``generator=None``
(one_for_more.py:84), and ``DataLoader.__iter__`` then draws a ``_base_seed``
from the *global* CPU generator. With ``num_workers=0`` that draw is never used
for anything - it changes no value the sweep computes - but it does move the
stream. Verified in this environment: replaying the sweep after a restore, or
simply consuming the same number of raw draws, makes the next concept reproduce
bit for bit; skipping it does not.

So the guarantee depends on which checkpoint is resumed, and
:meth:`ResumableOneForMore.note_resume_phase` records which one it was:

* a **pending** resume replays the interrupted concept's evaluation sweep inside
  the scenario, which advances the stream exactly as the dead process did, and
  is bit-identical;
* a **committed** resume skips that sweep, so the next concept trains from a
  stream position the uninterrupted run never had. That is a different, equally
  valid noise sequence - statistically equivalent, not bit-identical - and it is
  reported as ``statistical`` rather than assumed away.

The deficit is not guessed at. Its size is the number of inference loaders the
sweep happened to build, which depends on the evaluation callbacks attached and
on which of their calls hit the identity-keyed prediction cache; a plausible
count that is quietly wrong would be worse than an honest downgrade. Making the
premise true instead - by handing the inference loader a local generator - would
buy bit-identity for both phases, at the price of every ``ResumableOneForMore`` run
differing from every ``OneForMore`` run. That is a change to what the method
computes, not to how it is checkpointed, so it is left to the owners of the
model rather than taken as a side effect of adding resume.

What is deliberately **not** persisted is documented at :data:`_NOT_PERSISTED`.
The most surprising omission is the optimizer: AdamW is constructed inside
``fit`` (one_for_more.py:148), so an *uninterrupted* run already resets its
moments at every concept boundary. Restoring them would make the resumed run
diverge from the run it is meant to reproduce - the exact opposite of the goal.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Final, List, Mapping, Optional

import torch

from pyclad.models.model import Model
from pyclad.models.one_for_more.config import OneForMoreConfig
from pyclad.models.one_for_more.one_for_more import OneForMore
from pyclad.resume.fingerprint import fingerprint_from_config
from pyclad.resume.protocol import (
    DETERMINISM_BIT_IDENTICAL,
    DETERMINISM_STATISTICAL,
    CheckpointPayloadError,
    ResumeError,
    require_payload_keys,
)
from pyclad.resume.scenario import ResumePlan
from pyclad.strategies.baselines.naive import NaiveStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "ONE_FOR_MORE_FINGERPRINT_FIELDS",
    "ResumableOneForMore",
    "ResumableNaiveStrategy",
]

#: Config fields that silently change future numbers while leaving restored
#: state looking perfectly valid.
#:
#: The architecture fields matter beyond their effect on arithmetic: basis keys
#: are the module names of the transient ``nn.ModuleDict`` built by the
#: ``continual_module`` property (architecture.py:284-288), so changing
#: ``conditioner_depth`` or ``base_channels`` renames and reshapes every basis
#: entry at once.
#:
#: Only three config fields are intentionally absent, because none of them can
#: change a number: ``show_training_progress`` (a tqdm bar),
#: ``vae_local_files_only`` (where the frozen VAE is fetched from), and
#: ``device``. ``device`` is excluded on purpose - a scheduler chain routinely
#: lands its next job on a different node, and refusing that would defeat the
#: feature. A device change is instead reported honestly, by downgrading the
#: determinism guarantee in :meth:`ResumableOneForMore.determinism_report`.
ONE_FOR_MORE_FINGERPRINT_FIELDS: Final[tuple[str, ...]] = (
    "random_seed",
    "in_channels",
    "input_size",
    "latent_downsample",
    "vae_model_id",
    "vae_subfolder",
    "vae_latent_channels",
    "base_channels",
    "conditioner_depth",
    "conditioner_heads",
    "conditioner_feedforward",
    "conditioner_dropout",
    "neighbor_size",
    "diffusion_steps",
    "inference_steps",
    "inference_start_step",
    "beta_start",
    "beta_end",
    "condition_loss_weight",
    "epochs",
    "incremental_epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "patch_count_min",
    "patch_count_max",
    "patch_width_fraction",
    "patch_height_fraction",
    "projection_energy_threshold",
    "projection_batches",
    "projection_max_columns",
    "freeze_unprojected_parameters",
    "backbone_name",
    "backbone_return_nodes",
    "pretrained_backbone",
    "score_smoothing_sigma",
    "score_mode",
    "normalize_mean",
    "normalize_std",
    "threshold",
    "threshold_quantile",
    "threshold_calibration_samples",
)

#: State that is intentionally left out of the payload, with the reason.
#: Written down because every one of these looks like an oversight until you
#: know why it is not, and a later "fix" would break the resume it appears to
#: improve.
_NOT_PERSISTED: Final[Mapping[str, str]] = {
    "optimizer_state": (
        "AdamW is a local inside fit(), so an uninterrupted run resets its moments at every "
        "concept boundary anyway; restoring them would make the resumed run diverge from it."
    ),
    "codec.*": "The frozen VAE is rebuilt from vae_model_id, which the fingerprint pins.",
    "feature_extractor.*": "The frozen pretrained backbone is rebuilt from backbone_name/pretrained_backbone.",
    "alphas_cumulative": (
        "Derived deterministically from beta_start/beta_end/diffusion_steps, all three fingerprinted."
    ),
    "conditioner._mask_cache": "A pure function of (height, width, device); rebuilt on first use.",
    "_last_prediction_*": (
        "An identity-keyed prediction cache. Restoring it could alias a reused array and hand back "
        "stale scores, so plan.apply_resume invalidates it instead."
    ),
    "projector hyper-parameters": (
        "energy_threshold / max_columns / random_seed are read from the config, which the fingerprint pins."
    ),
}

#: Frozen sub-modules that are rebuilt by ``__init__`` from config alone.
_FROZEN_STATE_DICT_PREFIXES: Final[tuple[str, ...]] = ("codec.", "feature_extractor.")

#: Buffers that are a deterministic function of fingerprinted config values.
_DERIVED_STATE_DICT_KEYS: Final[frozenset[str]] = frozenset({"alphas_cumulative"})

_FULL_PAYLOAD: Final[str] = "full"
_BASIS_ONLY_PAYLOAD: Final[str] = "basis_only"

_REQUIRED_FULL_KEYS: Final[tuple[str, ...]] = (
    "payload_variant",
    "tasks_seen",
    "threshold",
    "module",
    "projector_bases",
    "projector_updates",
    "rng",
)
_REQUIRED_BASIS_KEYS: Final[tuple[str, ...]] = (
    "payload_variant",
    "tasks_seen",
    "projector_bases",
    "projector_updates",
)

#: Named in the result JSON so the reason is one grep away from the numbers.
_CAVEAT_EVALUATION_SWEEP: Final[str] = "committed_resume_skips_an_evaluation_sweep_that_advances_the_global_rng"


class ResumableOneForMore(OneForMore):
    """:class:`OneForMore` that can be written to disk and picked back up.

    A pure extension: no behaviour of the base model changes, ``name()`` is
    inherited unchanged so the results-aggregation join key still matches the
    published benchmark, and the base implementation is neither copied nor
    edited.
    """

    STATE_KIND: Final[str] = "one_for_more"
    #: Bump on *any* payload schema change. Old checkpoints then become
    #: unreadable by design - there is deliberately no override flag.
    STATE_VERSION: Final[int] = 1

    def __init__(self, config: Optional[OneForMoreConfig] = None, *, keep_basis_variant: bool = True):
        super().__init__(config)
        # The basis is the only unrecoverable state, and a basis-only snapshot
        # is small (a fraction of the full checkpoint), so it is kept for every
        # concept rather than falling out of the retention window.
        self._keep_basis_variant = bool(keep_basis_variant)
        self._determinism_guarantee = DETERMINISM_BIT_IDENTICAL
        self._determinism_caveats: List[str] = self._baseline_caveats()

    # ------------------------------------------------------------------ identity

    def state_fingerprint(self) -> Dict[str, Any]:
        return fingerprint_from_config(self.config, ONE_FOR_MORE_FINGERPRINT_FIELDS)

    def determinism_report(self) -> Dict[str, Any]:
        """What a resumed run may honestly claim about its numbers.

        Recorded in the envelope and mirrored into the result JSON, so a run
        that had to give up bit-identity can never later be mistaken for one
        that did not.
        """
        return {
            "guarantee": self._determinism_guarantee,
            "caveats": sorted(set(self._determinism_caveats)),
        }

    def note_resume_phase(self, plan: ResumePlan) -> None:
        """Record which checkpoint phase this process resumed from.

        Called from :meth:`ResumableNaiveStrategy.resume_restore`, the one place the
        plan is in scope. It exists because the honest guarantee differs by
        phase: a pending resume replays the interrupted concept's evaluation
        sweep and lands on the stream position the dead process would have had,
        while a committed resume skips that sweep and starts the next concept a
        few draws early. See the module docstring for the measurement.
        """
        if not plan.resumed and not plan.evaluate_pending_concept:
            return
        if plan.evaluate_pending_concept:
            logger.info(
                "Resume: replaying concept %d's evaluation sweep, so the global RNG stream lands where the "
                "interrupted process left it.",
                plan.resume_index,
            )
            return
        self._downgrade_determinism(
            _CAVEAT_EVALUATION_SWEEP,
            f"Resuming from the committed checkpoint at concept {plan.resume_index - 1} skips that concept's "
            "evaluation sweep, whose inference DataLoaders advance the global RNG stream, so concept "
            f"{plan.resume_index} trains from a stream position the uninterrupted run never had.",
        )

    def _baseline_caveats(self) -> List[str]:
        caveats: List[str] = []
        if self._device.type == "cuda":
            # True of an uninterrupted run as well: a restored RNG state only
            # reproduces numbers on identical hardware and library versions.
            caveats.extend(["cudnn_nondeterminism", "nondeterministic_backward_atomics"])
        elif self._device.type == "mps":
            caveats.append("mps_kernel_nondeterminism")
        if self.config.backbone_name is not None and not self.config.pretrained_backbone:
            # The scoring backbone is frozen but randomly initialised, and it is
            # not in the payload. It only comes back identical because the
            # runner seeds globally and constructs the model in the same order
            # in every process.
            caveats.append("unpretrained_backbone_rebuilt_from_the_global_seed")
        return caveats

    def _downgrade_determinism(self, caveat: str, message: str) -> None:
        self._determinism_guarantee = DETERMINISM_STATISTICAL
        if caveat not in self._determinism_caveats:
            self._determinism_caveats.append(caveat)
        logger.warning("%s Determinism guarantee downgraded to '%s'.", message, DETERMINISM_STATISTICAL)

    # --------------------------------------------------------------------- save

    def save_state(self) -> Dict[str, Any]:
        """Return a payload of tensors and JSON-native values only.

        Everything is moved to CPU: a checkpoint that only loads back onto the
        same GPU count is not a checkpoint a scheduler chain can use.
        """
        return {
            "payload_variant": _FULL_PAYLOAD,
            "tasks_seen": int(self._tasks_seen),
            "threshold": None if self._threshold is None else float(self._threshold),
            "module": self._persisted_module_state(),
            "projector_bases": self._persisted_bases(),
            "projector_updates": int(self._projector._updates),
            "rng": self._capture_rng(),
            # Pure reporting, three floats: without them a pending-phase resume
            # would publish nulls for a concept that did train.
            "last_losses": {
                "total": _optional_float(self._last_loss),
                "diffusion": _optional_float(self._last_diffusion_loss),
                "condition": _optional_float(self._last_condition_loss),
            },
            # Present and null on purpose, so the omission reads as a decision
            # rather than as something nobody thought about. See _NOT_PERSISTED.
            "optimizer_state": None,
        }

    def save_variants(self) -> Dict[str, Dict[str, Any]]:
        """Extra never-pruned snapshots written alongside the full checkpoint."""
        if not self._keep_basis_variant or self._projector.summary().layers == 0:
            return {}
        return {
            _BASIS_ONLY_PAYLOAD: {
                "payload_variant": _BASIS_ONLY_PAYLOAD,
                "tasks_seen": int(self._tasks_seen),
                "projector_bases": self._persisted_bases(),
                "projector_updates": int(self._projector._updates),
            }
        }

    def _persisted_module_state(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().to("cpu").clone()
            for key, value in self.module.state_dict().items()
            if _is_persisted_module_key(key)
        }

    def _persisted_bases(self) -> Dict[str, torch.Tensor]:
        # The public property already hands back clones; ``update`` keeps every
        # basis on the CPU as float32, so this is a copy and nothing more.
        return {name: basis.detach().to("cpu").contiguous() for name, basis in self._projector.bases.items()}

    def _capture_rng(self) -> Dict[str, Any]:
        rng: Dict[str, Any] = {
            "device_type": self._device.type,
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [],
            "torch_mps": None,
        }
        if self._device.type == "cuda" and torch.cuda.is_available():
            rng["torch_cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
        elif self._device.type == "mps":
            state = _mps_rng_state()
            if state is not None:
                rng["torch_mps"] = state.clone()
        return rng

    # --------------------------------------------------------------------- load

    def load_state(self, payload: Mapping[str, Any]) -> None:
        """Restore from *payload*, validating everything before mutating anything.

        Half-restored state is strictly worse than no state: a model with
        restored weights but a stale ``_tasks_seen`` would train the next concept
        with the wrong epoch budget, the wrong corruption seeds and no gradient
        projection at all, and would report none of it.
        """
        # The variant is checked first so that handing over a basis-only
        # snapshot says so, instead of listing the keys such a snapshot was
        # never meant to carry.
        variant = payload.get("payload_variant")
        if variant is not None and variant != _FULL_PAYLOAD:
            raise CheckpointPayloadError(
                f"Checkpoint payload for '{self.STATE_KIND}' is a '{variant}' payload, not a full one. "
                f"A '{_BASIS_ONLY_PAYLOAD}' snapshot carries no weights; restore it with load_basis_only()."
            )
        require_payload_keys(payload, _REQUIRED_FULL_KEYS, state_kind=self.STATE_KIND)

        # --- validation only; nothing below this line has touched the model yet.
        module_state = self._validated_module_state(payload["module"])
        bases = _validated_bases(payload["projector_bases"], state_kind=self.STATE_KIND)
        updates = _as_count(payload["projector_updates"], "projector_updates", state_kind=self.STATE_KIND)
        tasks_seen = _as_count(payload["tasks_seen"], "tasks_seen", state_kind=self.STATE_KIND)
        threshold = _validated_threshold(payload["threshold"], state_kind=self.STATE_KIND)
        rng = _validated_rng(payload["rng"], state_kind=self.STATE_KIND)
        losses = _validated_losses(payload.get("last_losses"), state_kind=self.STATE_KIND)

        # --- mutation.
        incompatible = self.module.load_state_dict(module_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if _is_persisted_module_key(key)]
        if unexpected or missing:  # pragma: no cover - pre-validated above
            raise CheckpointPayloadError(
                f"Checkpoint payload for '{self.STATE_KIND}' did not load cleanly "
                f"(unexpected={unexpected}, missing={missing})"
            )

        self._projector._bases = {name: basis.to(dtype=torch.float32).contiguous() for name, basis in bases.items()}
        self._projector._updates = updates
        self._tasks_seen = tasks_seen
        self._threshold = threshold
        self._last_loss = losses["total"]
        self._last_diffusion_loss = losses["diffusion"]
        self._last_condition_loss = losses["condition"]
        self._invalidate_prediction_cache()

        # Last, so that nothing restored afterwards can consume a draw from the
        # stream we just positioned.
        self._restore_rng(rng)

    def load_basis_only(self, payload: Mapping[str, Any]) -> None:
        """Restore just the GPM bases from a ``basis_only`` variant snapshot.

        The rescue path for the one piece of state that cannot be recomputed:
        the full checkpoints are pruned by the retention window, the basis-only
        ones are not. It refuses to graft a basis onto a model that has seen a
        different number of concepts, because a mismatched basis projects
        gradients onto the wrong subspace and reports nothing.
        """
        require_payload_keys(payload, _REQUIRED_BASIS_KEYS, state_kind=self.STATE_KIND)
        if payload["payload_variant"] != _BASIS_ONLY_PAYLOAD:
            raise CheckpointPayloadError(
                f"load_basis_only() expects a '{_BASIS_ONLY_PAYLOAD}' payload, " f"got '{payload['payload_variant']}'"
            )
        tasks_seen = _as_count(payload["tasks_seen"], "tasks_seen", state_kind=self.STATE_KIND)
        if tasks_seen != self._tasks_seen:
            raise CheckpointPayloadError(
                f"The basis-only snapshot was written after {tasks_seen} concepts but this model has seen "
                f"{self._tasks_seen}; grafting it on would project gradients onto the wrong subspace."
            )
        bases = _validated_bases(payload["projector_bases"], state_kind=self.STATE_KIND)
        updates = _as_count(payload["projector_updates"], "projector_updates", state_kind=self.STATE_KIND)

        self._projector._bases = {name: basis.to(dtype=torch.float32).contiguous() for name, basis in bases.items()}
        self._projector._updates = updates

    def _validated_module_state(self, value: Any) -> Dict[str, torch.Tensor]:
        """Check the weight payload against the freshly built module.

        ``load_state_dict`` copies parameter by parameter and only raises at the
        end, so a shape mismatch would leave the module half-overwritten. Every
        check therefore happens here, before anything is copied.
        """
        if not isinstance(value, Mapping):
            raise CheckpointPayloadError(
                f"Checkpoint payload for '{self.STATE_KIND}': 'module' must be a mapping, "
                f"got {type(value).__name__}"
            )
        expected = {
            key: tensor.shape for key, tensor in self.module.state_dict().items() if _is_persisted_module_key(key)
        }
        missing = sorted(set(expected) - set(value))
        unexpected = sorted(set(value) - set(expected))
        if missing or unexpected:
            raise CheckpointPayloadError(
                f"Checkpoint payload for '{self.STATE_KIND}' does not describe this architecture "
                f"(missing={missing[:8]}, unexpected={unexpected[:8]}). The trainable module changed shape "
                "since the checkpoint was written."
            )

        state: Dict[str, torch.Tensor] = {}
        for key, tensor in value.items():
            if not isinstance(tensor, torch.Tensor):
                raise CheckpointPayloadError(
                    f"Checkpoint payload for '{self.STATE_KIND}': module entry '{key}' is "
                    f"{type(tensor).__name__}, expected a tensor"
                )
            if tuple(tensor.shape) != tuple(expected[key]):
                raise CheckpointPayloadError(
                    f"Checkpoint payload for '{self.STATE_KIND}': module entry '{key}' has shape "
                    f"{tuple(tensor.shape)}, this model expects {tuple(expected[key])}"
                )
            state[key] = tensor
        return state

    def _restore_rng(self, rng: Mapping[str, Any]) -> None:
        """Put the global streams back where the checkpointed process left them.

        A mismatched device layout is reported and skipped rather than forced:
        restoring a three-GPU state onto two GPUs is not "close enough", it is a
        silently different run.
        """
        torch.set_rng_state(rng["torch_cpu"].to(device="cpu", dtype=torch.uint8))

        recorded_device = str(rng.get("device_type", ""))
        if recorded_device and recorded_device != self._device.type:
            self._downgrade_determinism(
                "device_changed_across_resume",
                f"The checkpoint was written on a '{recorded_device}' device and is being resumed on "
                f"'{self._device.type}'.",
            )

        cuda_states = list(rng.get("torch_cuda") or [])
        if cuda_states:
            if not torch.cuda.is_available():
                self._downgrade_determinism(
                    "cuda_rng_state_not_restored",
                    f"The checkpoint carries {len(cuda_states)} CUDA RNG states but CUDA is unavailable here.",
                )
            elif len(cuda_states) != torch.cuda.device_count():
                self._downgrade_determinism(
                    "cuda_rng_state_not_restored",
                    f"The checkpoint carries {len(cuda_states)} CUDA RNG states but this host exposes "
                    f"{torch.cuda.device_count()} devices.",
                )
            else:
                torch.cuda.set_rng_state_all([state.to(device="cpu", dtype=torch.uint8) for state in cuda_states])

        mps_state = rng.get("torch_mps")
        if mps_state is not None:
            if _set_mps_rng_state(mps_state):
                return
            self._downgrade_determinism(
                "mps_rng_state_not_restored",
                "The checkpoint carries an MPS RNG state but this torch build cannot restore one.",
            )


class ResumableNaiveStrategy(NaiveStrategy):
    """``NaiveStrategy`` with resume provenance and a wiring tripwire.

    ``NaiveStrategy`` keeps no learned state of its own, so - unlike UCAD's and
    ReplayCAD's strategies - there is genuinely nothing here to restore. The
    class still exists for two reasons.

    The first is the tripwire. A restore that constructs the model, loads state
    into it, and then hands a *different* model instance to the strategy is a
    plausible runner bug that produces no error at all: the run simply trains
    from scratch and publishes numbers. :meth:`resume_restore` costs two
    comparisons and turns that into an immediate abort.

    The second is provenance. ``name()`` is inherited unchanged - it is the join
    key the results-aggregation scripts use, and renaming it would quietly drop
    resumed runs out of every comparison against the published benchmark - so
    the v2-ness is recorded under ``additional_info()["resume"]`` instead, where
    it lands in the result JSON without touching identity.
    """

    def __init__(self, model: Model, *, resume_info: Optional[Mapping[str, Any]] = None):
        super().__init__(model)
        self._resume_info: Optional[Dict[str, Any]] = dict(resume_info) if resume_info is not None else None

    def set_resume_info(self, info: Mapping[str, Any]) -> None:
        """Record the resume block, typically ``ResumeOutcome.info()``."""
        self._resume_info = dict(info)

    def additional_info(self) -> Dict:
        info: Dict[str, Any] = dict(super().additional_info())
        if self._resume_info is not None:
            info["resume"] = dict(self._resume_info)
        return info

    def resume_restore(self, payload: Mapping[str, Any], plan: ResumePlan) -> None:
        """Verify that the model this strategy wraps is the one that was restored."""
        require_payload_keys(payload, ("tasks_seen",), state_kind=ResumableOneForMore.STATE_KIND)
        payload_tasks_seen = _as_count(payload["tasks_seen"], "tasks_seen", state_kind=ResumableOneForMore.STATE_KIND)

        # A checkpoint is written at ``after_training``, so it has always seen
        # one more concept than its index. Resuming a committed concept starts
        # the next one; replaying a pending concept re-runs only its evaluation,
        # and the scenario skips ``learn()`` for it.
        expected = plan.resume_index + (1 if plan.evaluate_pending_concept else 0)
        if payload_tasks_seen != expected:
            raise CheckpointPayloadError(
                f"Checkpoint payload records {payload_tasks_seen} completed concepts but the resume plan "
                f"starts at index {plan.resume_index} "
                f"(evaluate_pending_concept={plan.evaluate_pending_concept}), which implies {expected}."
            )

        model_tasks_seen = getattr(self._model, "_tasks_seen", None)
        if model_tasks_seen != payload_tasks_seen:
            raise ResumeError(
                f"The model wrapped by {type(self).__name__} has seen {model_tasks_seen} concepts but the "
                f"checkpoint restored {payload_tasks_seen}. The strategy is holding a different instance "
                "than the one load_state() was called on, so the run would silently train from scratch."
            )

        # The plan is in scope here and nowhere else, and the honest determinism
        # guarantee depends on which phase was resumed.
        note_phase = getattr(self._model, "note_resume_phase", None)
        if callable(note_phase):
            note_phase(plan)


# ----------------------------------------------------------------------- helpers


def _is_persisted_module_key(key: str) -> bool:
    if key in _DERIVED_STATE_DICT_KEYS:
        return False
    return not key.startswith(_FROZEN_STATE_DICT_PREFIXES)


def _optional_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def _as_count(value: Any, field: str, *, state_kind: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointPayloadError(
            f"Checkpoint payload for '{state_kind}': '{field}' must be an int, got {type(value).__name__}"
        )
    if value < 0:
        raise CheckpointPayloadError(f"Checkpoint payload for '{state_kind}': '{field}' must not be negative")
    return int(value)


def _validated_threshold(value: Any, *, state_kind: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointPayloadError(
            f"Checkpoint payload for '{state_kind}': 'threshold' must be a number or null, "
            f"got {type(value).__name__}"
        )
    return float(value)


def _validated_bases(value: Any, *, state_kind: str) -> Dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise CheckpointPayloadError(
            f"Checkpoint payload for '{state_kind}': 'projector_bases' must be a mapping, "
            f"got {type(value).__name__}"
        )
    bases: Dict[str, torch.Tensor] = {}
    for name, basis in value.items():
        if not isinstance(basis, torch.Tensor):
            raise CheckpointPayloadError(
                f"Checkpoint payload for '{state_kind}': basis '{name}' is {type(basis).__name__}, " "expected a tensor"
            )
        if basis.ndim != 2:
            raise CheckpointPayloadError(
                f"Checkpoint payload for '{state_kind}': basis '{name}' has shape {tuple(basis.shape)}, "
                "expected a 2D (features x rank) matrix"
            )
        bases[str(name)] = basis
    return bases


def _validated_rng(value: Any, *, state_kind: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointPayloadError(
            f"Checkpoint payload for '{state_kind}': 'rng' must be a mapping, got {type(value).__name__}"
        )
    require_payload_keys(value, ("torch_cpu",), state_kind=state_kind)
    cpu_state = value["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise CheckpointPayloadError(
            f"Checkpoint payload for '{state_kind}': 'rng.torch_cpu' must be a uint8 tensor, "
            f"got {type(cpu_state).__name__}"
        )
    for index, state in enumerate(value.get("torch_cuda") or []):
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
            raise CheckpointPayloadError(
                f"Checkpoint payload for '{state_kind}': 'rng.torch_cuda[{index}]' must be a uint8 tensor"
            )
    return value


def _validated_losses(value: Any, *, state_kind: str) -> Dict[str, Optional[float]]:
    if value is None:
        return {"total": None, "diffusion": None, "condition": None}
    if not isinstance(value, Mapping):
        raise CheckpointPayloadError(
            f"Checkpoint payload for '{state_kind}': 'last_losses' must be a mapping or null, "
            f"got {type(value).__name__}"
        )
    return {
        key: _validated_threshold(value.get(key), state_kind=state_kind) for key in ("total", "diffusion", "condition")
    }


def _mps_rng_state() -> Optional[torch.Tensor]:
    getter = getattr(getattr(torch, "mps", None), "get_rng_state", None)
    if not callable(getter):  # pragma: no cover - depends on the torch build
        return None
    try:
        return getter()
    except Exception:  # pragma: no cover - defensive, MPS support varies by build
        return None


def _set_mps_rng_state(state: torch.Tensor) -> bool:
    setter = getattr(getattr(torch, "mps", None), "set_rng_state", None)
    if not callable(setter):  # pragma: no cover - depends on the torch build
        return False
    try:
        setter(state.to(device="cpu", dtype=torch.uint8))
    except Exception:  # pragma: no cover - defensive
        return False
    return True
