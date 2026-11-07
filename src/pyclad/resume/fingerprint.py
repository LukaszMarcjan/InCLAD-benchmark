"""Identity of a run, and the checks that stop a resume from changing it.

A checkpoint is only safe to load into a process that would have produced it.
Three independent things can break that: the config (a different ``key_size``
reshapes every restored tensor's meaning), the concept order (row *k* of the
metric matrix would describe a different concept), and the code (a different
commit may compute a different score from identical state).

The first two are hard errors with no override, because the failure they
prevent is a plausible-looking number in a published table rather than a crash.
Only the third is overridable, and only with an explicit flag that is then
recorded in the result JSON.

This module also derives the ``run_id`` that names the checkpoint directory.
Deriving it from the run's *content* rather than from the output directory is
what makes an unattended scheduler chain work: a requeued job lands on the same
directory as its predecessor without anyone passing a job id around.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pyclad.resume.protocol import (
    CheckpointFingerprintMismatch,
    CheckpointOrderMismatch,
    CheckpointPayloadError,
    CheckpointVersionDrift,
    describe_sequence,
)

__all__ = [
    "FieldDiff",
    "compute_run_id",
    "fingerprint_from_config",
    "json_safe",
    "compare_fingerprints",
    "assert_fingerprint_matches",
    "assert_concept_order_matches",
    "current_provenance",
    "compare_provenance",
    "assert_provenance_matches",
    "utc_now",
]

#: Provenance fields that describe *code*, hence can change numbers without
#: changing any config value. Everything else in the provenance block is pure
#: bookkeeping (hostname, pid) and is never compared.
_CODE_PROVENANCE_FIELDS = ("pyclad_version", "git_commit")


@dataclass(frozen=True)
class FieldDiff:
    key: str
    expected: Any
    actual: Any

    def __str__(self) -> str:
        return f"{self.key}: checkpoint={self.expected!r} run={self.actual!r}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Coerce a config value into something JSON and ``weights_only`` accept.

    Fingerprints are compared across processes and pyCLAD versions, so they must
    not depend on any type that pickles differently (numpy scalars, ``Path``,
    ``Enum``). Everything is flattened to a JSON-native shape once, here.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items = sorted(items, key=repr)
        return [json_safe(item) for item in items]
    if hasattr(value, "model_dump"):  # pydantic BaseModel
        return json_safe(value.model_dump(mode="json"))
    item_method = getattr(value, "item", None)
    if callable(item_method):  # numpy scalar
        try:
            return json_safe(item_method())
        except Exception:  # pragma: no cover - defensive
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):  # numpy array
        return json_safe(tolist())
    return str(value)


def fingerprint_from_config(config: Any, fields: Sequence[str]) -> dict[str, Any]:
    """Read *fields* off a config object into a JSON-safe fingerprint.

    A missing field is an error rather than a silent ``None``: it means the
    fingerprint list and the config class have drifted apart, and a fingerprint
    that quietly stops covering a field stops protecting the run.
    """
    payload: dict[str, Any] = {}
    missing: list[str] = []
    for field in fields:
        if isinstance(config, Mapping):
            if field not in config:
                missing.append(field)
                continue
            payload[field] = json_safe(config[field])
        else:
            if not hasattr(config, field):
                missing.append(field)
                continue
            payload[field] = json_safe(getattr(config, field))
    if missing:
        raise CheckpointPayloadError(
            f"Fingerprint fields {missing} are absent from {type(config).__name__}; "
            "the fingerprint list and the config have drifted apart"
        )
    return payload


def compare_fingerprints(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[FieldDiff]:
    """Return every differing key, including keys present on only one side."""
    diffs: list[FieldDiff] = []
    for key in sorted(set(expected) | set(actual)):
        expected_value = json_safe(expected.get(key, "<absent>"))
        actual_value = json_safe(actual.get(key, "<absent>"))
        if expected_value != actual_value:
            diffs.append(FieldDiff(key=key, expected=expected_value, actual=actual_value))
    return diffs


def assert_fingerprint_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    diffs = compare_fingerprints(expected, actual)
    if not diffs:
        return
    rendered = "\n  ".join(str(diff) for diff in diffs)
    raise CheckpointFingerprintMismatch(
        "The checkpoint was written with a different configuration, so resuming would mix two runs "
        f"into one result. Differing fields:\n  {rendered}\n"
        "There is no override for this: start a fresh run, or restore the original configuration."
    )


def assert_concept_order_matches(expected: Sequence[str], actual: Sequence[str]) -> None:
    """Guard the one thing every restored matrix row is keyed on.

    This single comparison catches a changed ``--concept-order``,
    ``--ordering-file``, ``--shuffle-concepts``, ``--categories``, and a
    ``--max-train`` that happened to drop a concept entirely.
    """
    expected_list = [str(name) for name in expected]
    actual_list = [str(name) for name in actual]
    if expected_list == actual_list:
        return

    first_difference = len(expected_list)
    for index, (left, right) in enumerate(zip(expected_list, actual_list)):
        if left != right:
            first_difference = index
            break

    raise CheckpointOrderMismatch(
        "The resolved concept order differs from the one the checkpoint was written with; "
        f"first difference at index {first_difference}.\n"
        f"  checkpoint: {describe_sequence(expected_list)}\n"
        f"  run:        {describe_sequence(actual_list)}"
    )


def compute_run_id(
    *,
    model: str,
    strategy: str,
    benchmark: str,
    seed: int,
    concept_order: Sequence[str],
    eval_level: str | None = None,
    resize: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Derive the checkpoint directory name from what the run *is*.

    Keyed on content, never on an output path or a job id. The ReplayCAD runs
    that motivated this layer defaulted their artifact directory inside the
    per-job output directory, so every scheduler restart got a fresh directory
    and re-ran ~40 minutes of compression per concept against artifacts it
    already owned. A content-addressed id means the resumed job finds its own
    predecessor without being told where it lived.
    """
    payload = {
        "model": str(model),
        "strategy": str(strategy),
        "benchmark": str(benchmark),
        "seed": int(seed),
        "concept_order": [str(name) for name in concept_order],
        "eval_level": eval_level,
        "resize": resize,
        "extra": json_safe(extra or {}),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _git_commit() -> str | None:
    """Best-effort commit id; ``None`` outside a checkout rather than a crash."""
    try:
        import pyclad

        repo_root = Path(pyclad.__file__).resolve().parents[2]
    except Exception:  # pragma: no cover - defensive
        repo_root = Path.cwd()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _package_version(name: str) -> str | None:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - py<3.8 only
        return None
    try:
        return metadata.version(name)
    except Exception:
        return None


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    # str(), not the object: torch.__version__ is a TorchVersion instance, and
    # torch.load(weights_only=True) refuses to unpickle it — which would make
    # every checkpoint unreadable by the very layer that wrote it.
    return None if version is None else str(version)


def current_provenance(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe the code and machine that wrote a checkpoint."""
    provenance: dict[str, Any] = {
        "pyclad_version": _package_version("pyclad"),
        "torch_version": _module_version("torch"),
        "numpy_version": _module_version("numpy"),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "written_at": utc_now(),
    }
    if extra:
        provenance.update(json_safe(extra))
    return provenance


def compare_provenance(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[FieldDiff]:
    """Compare only the fields that describe code, never host or pid."""
    diffs: list[FieldDiff] = []
    for field in _CODE_PROVENANCE_FIELDS:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if expected_value is None or actual_value is None:
            # An unknown version on either side is not evidence of drift.
            continue
        if expected_value != actual_value:
            diffs.append(FieldDiff(key=field, expected=expected_value, actual=actual_value))
    return diffs


def assert_provenance_matches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    allow_drift: bool = False,
) -> list[FieldDiff]:
    """Abort on code drift unless it was explicitly allowed.

    Returns the drift so the caller can record it in the result JSON: a run that
    resumed across a code change should say so in its published artifact, not
    only in a log file nobody keeps.
    """
    diffs = compare_provenance(expected, actual)
    if not diffs:
        return []
    if allow_drift:
        return diffs
    rendered = "\n  ".join(str(diff) for diff in diffs)
    raise CheckpointVersionDrift(
        "The checkpoint was written by a different pyCLAD build:\n  "
        f"{rendered}\n"
        "Pass --resume-allow-version-drift to accept this; the drift is then recorded "
        "in the result JSON under resume.version_drift."
    )


def fingerprint_fields(fields: Iterable[str]) -> tuple[str, ...]:
    """Normalise a fingerprint field list (sorted, de-duplicated)."""
    return tuple(sorted(set(str(field) for field in fields)))
