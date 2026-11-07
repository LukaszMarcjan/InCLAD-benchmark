"""State protocol shared by every resumable pyCLAD component.

The split enforced here is deliberate: a component owns the *semantics* of its
state (which attributes, which dtypes, which invariants) while
:mod:`pyclad.resume.store` owns *durability* (paths, atomicity, retention).
Without that split the atomic-write dance would be re-implemented once per
method, and a method class could not be unit-tested without touching a disk.

Two contracts in this module are load-bearing and easy to violate silently:

* ``save_state()`` may only return tensors and JSON-native values. Checkpoints
  are always read back with ``weights_only=True``, under which a raw
  ``numpy.ndarray`` in the payload raises ``UnpicklingError`` — hours after the
  run that wrote it has finished. :func:`validate_state_payload` moves that
  failure to write time, where it is a one-line fix. Large arrays belong in
  sidecars (see :data:`SIDECAR_KEY`), not in the payload.
* ``load_state()`` must validate before it mutates. Half-restored state is
  strictly worse than no state: a model with restored memories and a stale
  feature-map size fails at prediction time, tens of minutes later, with an
  error that points nowhere near the checkpoint.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final, Iterable, Mapping, MutableMapping, Protocol, Sequence, runtime_checkable

__all__ = [
    "ENVELOPE_VERSION",
    "MANIFEST_VERSION",
    "SIDECAR_KEY",
    "PHASE_AFTER_TRAINING",
    "PHASE_AFTER_CONCEPT_PROCESSING",
    "PHASES",
    "DETERMINISM_BIT_IDENTICAL",
    "DETERMINISM_STATISTICAL",
    "DISK_FULL_EXIT_CODE",
    "ResumableState",
    "ResumableStrategy",
    "ResumeError",
    "CheckpointEnvelopeError",
    "CheckpointKindError",
    "CheckpointVersionError",
    "CheckpointFingerprintMismatch",
    "CheckpointOrderMismatch",
    "CheckpointPayloadError",
    "CheckpointCorruptionError",
    "CheckpointDiskSpaceError",
    "CheckpointLockError",
    "CheckpointMissingError",
    "CheckpointResultMismatch",
    "CheckpointVersionDrift",
    "build_envelope",
    "normalize_metadata",
    "validate_envelope",
    "require_payload_keys",
    "validate_state_payload",
    "split_sidecars",
    "merge_sidecars",
    "describe_sequence",
]

#: Bumped when the *envelope* layout changes (not the per-method payload).
ENVELOPE_VERSION: Final[int] = 1

#: Bumped when the on-disk manifest layout changes.
MANIFEST_VERSION: Final[int] = 1

#: Reserved key inside a ``save_state()`` payload holding out-of-band arrays.
#:
#: A component that owns a multi-GB array (ReplayCAD's PatchCore memory bank)
#: returns it as ``{SIDECAR_KEY: {"memory_bank": array}}``. The store pops the
#: key, writes each array with ``np.save`` next to the checkpoint, records its
#: sha256, and re-injects the loaded arrays under the same key before
#: ``load_state()`` runs. The round trip is symmetric, so a method class never
#: has to know whether a given array lives inline or beside the file.
SIDECAR_KEY: Final[str] = "__sidecars__"

PHASE_AFTER_TRAINING: Final[str] = "after_training"
PHASE_AFTER_CONCEPT_PROCESSING: Final[str] = "after_concept_processing"
PHASES: Final[tuple[str, ...]] = (PHASE_AFTER_TRAINING, PHASE_AFTER_CONCEPT_PROCESSING)

DETERMINISM_BIT_IDENTICAL: Final[str] = "bit_identical"
DETERMINISM_STATISTICAL: Final[str] = "statistical"

#: Process exit code reserved for "the checkpoint could not be written because
#: the filesystem is full". ENOSPC, so a scheduler can branch on it and requeue
#: onto a different node rather than retrying the same doomed job.
DISK_FULL_EXIT_CODE: Final[int] = 28


@runtime_checkable
class ResumableState(Protocol):
    """A component whose learned state survives a process restart.

    ``STATE_VERSION`` is a class constant rather than a value buried in
    ``save_state()`` precisely so that a reviewer sees the bump in the diff that
    changes the payload schema.
    """

    STATE_KIND: ClassVar[str]
    STATE_VERSION: ClassVar[int]

    def save_state(self) -> dict[str, Any]:
        """Return a payload of tensors and JSON-native values only."""

    def load_state(self, payload: Mapping[str, Any]) -> None:
        """Restore from *payload*, validating every required key first."""

    def state_fingerprint(self) -> dict[str, Any]:
        """Return the config fields that silently change future numbers."""


@runtime_checkable
class ResumableStrategy(Protocol):
    """Optional strategy-side hook invoked after the model state is loaded.

    Restoring the model is not always enough. UCAD's strategy keeps a
    ``_tasks_seen`` counter that gates a one-shot ``begin_continual_run()``;
    leaving it at zero wipes the restored memories at the next concept with no
    error and no log line. ReplayCAD has to re-activate the completed concepts
    in order or replay comes back empty. Both are strategy state, so the plan
    driver calls this hook when a strategy implements it.
    """

    def resume_restore(self, payload: Mapping[str, Any], plan: Any) -> None:
        """Restore strategy-level bookkeeping from the checkpoint *payload*."""


class ResumeError(RuntimeError):
    """Base class for every failure the resume layer detects."""


class CheckpointEnvelopeError(ResumeError):
    """The envelope is structurally wrong or written by another layout."""


class CheckpointKindError(ResumeError):
    """The checkpoint belongs to a different method than the one resuming."""


class CheckpointVersionError(ResumeError):
    """The payload schema version differs. Intentionally not overridable."""


class CheckpointFingerprintMismatch(ResumeError):
    """A config field that changes future numbers differs from the run."""


class CheckpointOrderMismatch(ResumeError):
    """The resolved concept order differs from the one the run was built on."""


class CheckpointPayloadError(ResumeError):
    """A payload is missing a required key or holds an unsupported type."""


class CheckpointCorruptionError(ResumeError):
    """A checkpoint file is unreadable, truncated, or fails its sha256."""


class CheckpointDiskSpaceError(ResumeError):
    """The filesystem cannot hold the checkpoint that is about to be written."""


class CheckpointLockError(ResumeError):
    """Another live process already owns this run directory."""


class CheckpointMissingError(ResumeError):
    """``--resume require`` was asked for but no usable checkpoint exists."""


class CheckpointResultMismatch(ResumeError):
    """The partial result JSON does not pair with the checkpoint state."""


class CheckpointVersionDrift(ResumeError):
    """pyCLAD version or git commit drifted; overridable, unlike the rest."""


def build_envelope(
    *,
    state_kind: str,
    state_version: int,
    provenance: Mapping[str, Any],
    run: Mapping[str, Any],
    determinism: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap a component payload in the envelope the store persists."""
    phase = run.get("phase")
    if phase not in PHASES:
        raise CheckpointEnvelopeError(f"Unknown checkpoint phase {phase!r}; expected one of {PHASES}")
    return {
        "envelope_version": ENVELOPE_VERSION,
        "state_kind": str(state_kind),
        "state_version": int(state_version),
        "provenance": normalize_metadata(provenance),
        "run": normalize_metadata(run),
        "determinism": normalize_metadata(determinism),
        "fingerprint": normalize_metadata(fingerprint),
        "state": dict(state),
    }


def normalize_metadata(value: Any) -> Any:
    """Flatten envelope metadata to types ``weights_only=True`` will load back.

    Exact types matter, not duck types: ``torch.__version__`` is a ``str``
    *subclass*, and an ``isinstance(value, str)`` check happily lets it through
    into a payload that then cannot be unpickled. That failure is invisible
    until a later process tries to resume, so metadata is coerced on the way in.
    """
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): normalize_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [normalize_metadata(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):  # numpy scalar
        try:
            return normalize_metadata(item_method())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def validate_envelope(
    envelope: Any,
    *,
    expected_kind: str | None = None,
    expected_state_version: int | None = None,
) -> Mapping[str, Any]:
    """Check envelope structure, kind and payload version. Side-effect free.

    Deliberately no ``force`` parameter: a ``state_version`` bump means the old
    checkpoints are dead, and an override flag here is how silently-wrong
    numbers get published.
    """
    if not isinstance(envelope, Mapping):
        raise CheckpointEnvelopeError(f"Checkpoint is not a mapping but {type(envelope).__name__}")

    missing = [
        key for key in ("envelope_version", "state_kind", "state_version", "run", "state") if key not in envelope
    ]
    if missing:
        raise CheckpointEnvelopeError(f"Checkpoint envelope is missing required keys: {missing}")

    envelope_version = envelope["envelope_version"]
    if envelope_version != ENVELOPE_VERSION:
        raise CheckpointEnvelopeError(
            f"Checkpoint envelope version {envelope_version} cannot be read by this pyCLAD "
            f"(expects {ENVELOPE_VERSION}). Re-run from scratch."
        )

    if expected_kind is not None and envelope["state_kind"] != expected_kind:
        raise CheckpointKindError(
            f"Checkpoint holds state for '{envelope['state_kind']}' but the run resumes '{expected_kind}'"
        )

    if expected_state_version is not None and int(envelope["state_version"]) != int(expected_state_version):
        raise CheckpointVersionError(
            f"Checkpoint payload for '{envelope['state_kind']}' has state_version "
            f"{envelope['state_version']}, this pyCLAD writes {expected_state_version}. "
            "The payload schema changed, so the old checkpoint cannot be trusted; start a fresh run."
        )

    run = envelope["run"]
    if not isinstance(run, Mapping) or "concept_index" not in run or "phase" not in run:
        raise CheckpointEnvelopeError("Checkpoint envelope 'run' block must carry 'concept_index' and 'phase'")
    if run["phase"] not in PHASES:
        raise CheckpointEnvelopeError(f"Checkpoint envelope has unknown phase {run['phase']!r}")

    return envelope


def require_payload_keys(payload: Mapping[str, Any], required: Iterable[str], *, state_kind: str) -> None:
    """Fail before mutating anything when a payload key is missing.

    Worth the handful of lines: a UCAD checkpoint that lost ``threshold``
    degrades *only* F1 (predict() falls back to zero, and every positive
    nearest-neighbour distance then reads as anomalous) while ROC-AUC and AP,
    computed from raw scores, still look perfect. That is a resume bug which
    reads as a modelling result.
    """
    missing = [key for key in required if key not in payload]
    if missing:
        raise CheckpointPayloadError(
            f"Checkpoint payload for '{state_kind}' is missing required keys {missing}; "
            f"present keys are {sorted(payload)}"
        )


def _is_tensor(value: Any) -> bool:
    torch = _maybe_torch()
    return torch is not None and isinstance(value, torch.Tensor)


def _maybe_torch():
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an extra, not a core dep
        return None
    return torch


def validate_state_payload(payload: Mapping[str, Any], *, state_kind: str = "<unknown>", _path: str = "state") -> None:
    """Reject anything ``weights_only=True`` would refuse to load back.

    Raising here costs one traceback at write time; not raising costs the whole
    resume, discovered only when the next job tries to read the file.
    """
    if not isinstance(payload, Mapping):
        raise CheckpointPayloadError(f"{_path} must be a mapping, got {type(payload).__name__}")

    for key, value in payload.items():
        if not isinstance(key, str):
            raise CheckpointPayloadError(f"{_path}: checkpoint payload keys must be strings, got {type(key).__name__}")
        child = f"{_path}.{key}"
        if key == SIDECAR_KEY and _path == "state":
            _validate_sidecar_mapping(value, path=child)
            continue
        _validate_payload_value(value, path=child, state_kind=state_kind)


def _validate_payload_value(value: Any, *, path: str, state_kind: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if _is_tensor(value):
        return
    if isinstance(value, Mapping):
        validate_state_payload(value, state_kind=state_kind, _path=path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_payload_value(item, path=f"{path}[{index}]", state_kind=state_kind)
        return

    type_name = f"{type(value).__module__}.{type(value).__name__}"
    hint = ""
    if type_name.startswith("numpy."):
        hint = (
            " Convert it with torch.as_tensor(...) / .tolist() / int() / float(), or move a large array into "
            f"the '{SIDECAR_KEY}' block. torch.load(weights_only=True) refuses numpy objects."
        )
    raise CheckpointPayloadError(
        f"Checkpoint payload for '{state_kind}' holds an unsupported value at {path}: {type_name}."
        f"{hint} Allowed: tensors, str, bytes, bool, int, float, None, list/tuple and dict of those."
    )


def _validate_sidecar_mapping(value: Any, *, path: str) -> None:
    import numpy as np

    if not isinstance(value, Mapping):
        raise CheckpointPayloadError(f"{path} must be a mapping of name -> numpy array, got {type(value).__name__}")
    for name, array in value.items():
        if not isinstance(name, str) or not name:
            raise CheckpointPayloadError(f"{path}: sidecar names must be non-empty strings, got {name!r}")
        if any(character in name for character in "/\\" + chr(0)) or name in (".", ".."):
            raise CheckpointPayloadError(f"{path}: sidecar name {name!r} may not contain path separators")
        if not isinstance(array, np.ndarray):
            raise CheckpointPayloadError(
                f"{path}.{name} must be a numpy array (sidecars are written with np.save), "
                f"got {type(array).__name__}"
            )


def split_sidecars(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate the inline payload from its out-of-band arrays."""
    inline: MutableMapping[str, Any] = dict(payload)
    sidecars = inline.pop(SIDECAR_KEY, None) or {}
    _validate_sidecar_mapping(sidecars, path=f"state.{SIDECAR_KEY}")
    return dict(inline), dict(sidecars)


def merge_sidecars(payload: Mapping[str, Any], sidecars: Mapping[str, Any]) -> dict[str, Any]:
    """Re-attach loaded sidecar arrays so ``load_state`` sees one payload."""
    merged = dict(payload)
    if sidecars:
        merged[SIDECAR_KEY] = dict(sidecars)
    return merged


def describe_sequence(values: Sequence[str], *, limit: int = 8) -> str:
    """Render a concept order compactly for an error message."""
    values = list(values)
    if len(values) <= limit:
        return "[" + ", ".join(values) + "]"
    head = ", ".join(values[: limit - 1])
    return f"[{head}, ... (+{len(values) - limit + 1} more), {values[-1]}]"
