"""Durable checkpoint storage: one manifest, one commit point, no torn pairs.

Everything here exists to make one guarantee hold across a ``SIGKILL`` at an
arbitrary instruction: *a checkpoint the manifest names is a checkpoint that is
fully on disk.* The mechanism is a commit protocol rather than a write:

1. sidecar arrays are written to ``*.tmp-<pid>``, fsynced and renamed;
2. the envelope is written the same way, then the **directory** is fsynced —
   without that last step the rename itself can be lost on a power cut, which
   on a shared cluster filesystem is the difference between a valid checkpoint
   and one that simply is not there;
3. only then is ``manifest.json`` rewritten, by the same dance.

A file present on disk but absent from the manifest is garbage and is never
read. A kill at any point therefore leaves either the old manifest (previous
checkpoint valid) or the new one (new checkpoint valid).

The second idea is the two-phase commit. State is written at ``after_training``
but the result JSON is only written after the evaluation sweep, so a crash
during evaluation would otherwise leave state at concept *k* paired with
metrics at *k-1*. The manifest records the freshly written file as ``pending``
and promotes it to ``committed`` only once the evaluation sweep has finished
and the result JSON has been snapshotted. Resume reads ``committed``; the
``pending`` entry is what lets a run re-run only the evaluation sweep of a
concept whose (expensive) fit already succeeded.
"""

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyclad.resume.fingerprint import utc_now
from pyclad.resume.protocol import (
    MANIFEST_VERSION,
    PHASE_AFTER_TRAINING,
    CheckpointCorruptionError,
    CheckpointDiskSpaceError,
    CheckpointEnvelopeError,
    CheckpointLockError,
    ResumeError,
    split_sidecars,
    validate_state_payload,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SidecarInfo",
    "CheckpointEntry",
    "CheckpointStore",
    "ensure_resume_dir_outside_output",
]

#: Free space must exceed the estimated write by this factor. Filesystems lie
#: about free space (reservations, snapshots, compression), and a checkpoint
#: that fails halfway is the one case where the previous checkpoint has to stay
#: intact.
_DISK_HEADROOM = 1.5

#: Above this size the post-write read-back verification costs more than it is
#: worth, so ``verify_writes="auto"`` turns it off.
_VERIFY_WRITE_LIMIT_BYTES = 256 * 1024 * 1024

_HASH_CHUNK = 8 * 1024 * 1024


def _torch_load(path: Path) -> Any:
    """Load with ``weights_only=True`` where the installed torch supports it.

    Copied (rather than imported) from ``strategies/replaycad/backend.py``:
    reaching into an unrelated subpackage for a private helper would couple this
    module to ReplayCAD's import graph, and this layer must import cleanly with
    neither diffusers nor timm installed.
    """
    import torch

    parameters = inspect.signature(torch.load).parameters
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _estimate_payload_bytes(value: Any) -> int:
    """Rough byte count of a payload, used only for the disk preflight."""
    if value is None or isinstance(value, bool):
        return 8
    if isinstance(value, (int, float)):
        return 16
    if isinstance(value, str):
        return len(value.encode("utf-8")) + 16
    if isinstance(value, bytes):
        return len(value) + 16
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes + 256
    if isinstance(value, Mapping):
        return 256 + sum(_estimate_payload_bytes(key) + _estimate_payload_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return 64 + sum(_estimate_payload_bytes(item) for item in value)
    return 512


@dataclass(frozen=True)
class SidecarInfo:
    name: str
    filename: str
    sha256: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "bytes": self.bytes,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SidecarInfo":
        return cls(
            name=str(payload["name"]),
            filename=str(payload["filename"]),
            sha256=str(payload["sha256"]),
            shape=tuple(int(dim) for dim in payload.get("shape", ())),
            dtype=str(payload.get("dtype", "")),
            bytes=int(payload.get("bytes", 0)),
        )


@dataclass(frozen=True)
class CheckpointEntry:
    index: int
    concept_id: str
    phase: str
    filename: str
    sha256: str
    bytes: int
    written_at: str
    sidecars: tuple[SidecarInfo, ...] = ()
    variant: str | None = None
    result_json: str | None = None
    result_json_sha256: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "concept_id": self.concept_id,
            "phase": self.phase,
            "filename": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "written_at": self.written_at,
            "sidecars": [sidecar.to_json() for sidecar in self.sidecars],
            "variant": self.variant,
            "result_json": self.result_json,
            "result_json_sha256": self.result_json_sha256,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CheckpointEntry":
        return cls(
            index=int(payload["index"]),
            concept_id=str(payload["concept_id"]),
            phase=str(payload.get("phase", PHASE_AFTER_TRAINING)),
            filename=str(payload["filename"]),
            sha256=str(payload.get("sha256", "")),
            bytes=int(payload.get("bytes", 0)),
            written_at=str(payload.get("written_at", "")),
            sidecars=tuple(SidecarInfo.from_json(item) for item in payload.get("sidecars", ())),
            variant=payload.get("variant"),
            result_json=payload.get("result_json"),
            result_json_sha256=payload.get("result_json_sha256"),
            extra=dict(payload.get("extra", {})),
        )


def ensure_resume_dir_outside_output(resume_dir: Path, output_dir: Path) -> None:
    """Refuse a checkpoint directory nested inside the per-run output.

    This is the same bug class that cost the motivating ReplayCAD run every
    cached artifact on every restart: state stored under a per-job directory is
    state a restarted job cannot find.
    """
    resume_resolved = Path(resume_dir).expanduser().resolve()
    output_resolved = Path(output_dir).expanduser().resolve()
    if resume_resolved == output_resolved or output_resolved in resume_resolved.parents:
        raise ValueError(
            f"--resume-dir ({resume_resolved}) is inside --output-dir ({output_resolved}). "
            "Checkpoints must outlive a single job's output directory, otherwise a restarted job "
            "writes to a fresh directory and silently starts from concept 0."
        )


class CheckpointStore:
    """Owns checkpoint durability for a single ``run_id``."""

    def __init__(
        self,
        root: str | Path,
        run_id: str,
        *,
        keep: int = 2,
        verify_writes: bool | None = None,
        lock_stale_after: float = 1800.0,
    ):
        self._root = Path(root).expanduser()
        self._run_id = str(run_id)
        self._directory = self._root / self._run_id
        # keep=2 rather than 1: the manifest protocol is airtight on a local
        # POSIX filesystem, but os.replace is not guaranteed atomic across NFS,
        # and one predecessor checkpoint is cheap insurance against that.
        self._keep = int(keep)
        self._verify_writes = verify_writes
        self._lock_stale_after = float(lock_stale_after)
        self._lock_acquired = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def manifest_path(self) -> Path:
        return self._directory / "manifest.json"

    @property
    def lock_path(self) -> Path:
        return self._directory / ".lock.json"

    def exists(self) -> bool:
        return self.manifest_path.is_file()

    # ------------------------------------------------------------------ manifest

    def read_manifest(self) -> dict[str, Any] | None:
        path = self.manifest_path
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointCorruptionError(f"Checkpoint manifest {path} is unreadable: {error}") from error

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self._run_id,
            "state_kind": None,
            "state_version": None,
            "concept_order": [],
            "committed": None,
            "pending": None,
            "history": [],
            "variants": [],
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        payload = dict(manifest)
        payload["updated_at"] = utc_now()
        self._atomic_write_bytes(
            self.manifest_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8"),
        )

    def committed(self) -> CheckpointEntry | None:
        return self._entry(self.read_manifest(), "committed")

    def pending(self) -> CheckpointEntry | None:
        return self._entry(self.read_manifest(), "pending")

    def history(self) -> list[CheckpointEntry]:
        manifest = self.read_manifest() or {}
        return [CheckpointEntry.from_json(item) for item in manifest.get("history", [])]

    @staticmethod
    def _entry(manifest: Mapping[str, Any] | None, key: str) -> CheckpointEntry | None:
        if not manifest:
            return None
        payload = manifest.get(key)
        if not payload:
            return None
        return CheckpointEntry.from_json(payload)

    def path_of(self, entry: CheckpointEntry) -> Path:
        return self._directory / entry.filename

    # ------------------------------------------------------------------ writing

    def write(
        self,
        envelope: Mapping[str, Any],
        *,
        variant: str | None = None,
    ) -> CheckpointEntry:
        """Write one checkpoint and make it the manifest's ``pending`` entry.

        Any failure propagates. Callers must not swallow it: continuing after a
        failed checkpoint means burning the remaining wall clock on a job that
        can no longer be resumed, so failing loudly at concept 7 costs one
        concept while failing silently costs the run.
        """
        run = envelope.get("run")
        if not isinstance(run, Mapping):
            raise CheckpointEnvelopeError("Envelope has no 'run' block; cannot name the checkpoint file")
        index = int(run["concept_index"])
        concept_id = str(run.get("concept_id", "unknown"))
        phase = str(run.get("phase", PHASE_AFTER_TRAINING))

        state_kind = str(envelope.get("state_kind", "unknown"))
        validate_state_payload(envelope.get("state", {}), state_kind=state_kind)
        inline_state, sidecar_arrays = split_sidecars(envelope.get("state", {}))

        self._directory.mkdir(parents=True, exist_ok=True)

        stem = self._stem(index, concept_id, variant)
        estimated = _estimate_payload_bytes(inline_state) + sum(
            int(np.asarray(array).nbytes) for array in sidecar_arrays.values()
        )
        self._preflight_disk_space(estimated, stem=stem)

        sidecar_infos: list[SidecarInfo] = []
        checkpoint_path = self._directory / f"{stem}.ckpt"
        try:
            for name, array in sidecar_arrays.items():
                sidecar_infos.append(self._write_sidecar(stem=stem, name=name, array=np.asarray(array)))

            payload = dict(envelope)
            payload["state"] = inline_state
            payload["sidecars"] = [sidecar.to_json() for sidecar in sidecar_infos]

            self._atomic_write(checkpoint_path, lambda handle: self._torch_save(payload, handle))
        except BaseException:
            # Nothing written so far is referenced by the manifest, so it is
            # already unreachable — but a multi-GB orphan left behind by a
            # disk-full failure would make the next attempt fail sooner.
            self._discard_unreferenced(checkpoint_path, sidecar_infos)
            raise

        checkpoint_sha = _sha256_file(checkpoint_path)
        entry = CheckpointEntry(
            index=index,
            concept_id=concept_id,
            phase=phase,
            filename=checkpoint_path.name,
            sha256=checkpoint_sha,
            bytes=checkpoint_path.stat().st_size,
            written_at=utc_now(),
            sidecars=tuple(sidecar_infos),
            variant=variant,
        )

        if self._should_verify(entry.bytes):
            self._verify_readback(checkpoint_path, entry)

        manifest = self.read_manifest() or self._empty_manifest()
        manifest["state_kind"] = state_kind
        manifest["state_version"] = envelope.get("state_version")
        if not manifest.get("concept_order"):
            manifest["concept_order"] = [str(name) for name in run.get("concept_order", [])]
        if run.get("seed") is not None:
            manifest["seed"] = run.get("seed")
        manifest["fingerprint"] = dict(envelope.get("fingerprint", {}))
        manifest["determinism"] = dict(envelope.get("determinism", {}))

        if variant is None:
            manifest["pending"] = entry.to_json()
            manifest["history"] = self._merge_history(manifest.get("history", []), entry)
        else:
            manifest["variants"] = self._merge_variants(manifest.get("variants", []), entry)

        self._write_manifest(manifest)
        self._heartbeat()
        if variant is None:
            self._prune(manifest)
        logger.info(
            "Wrote %s checkpoint for concept %d (%s) to %s (%.1f MiB)",
            variant or phase,
            index,
            concept_id,
            checkpoint_path,
            entry.bytes / (1024 * 1024),
        )
        return entry

    def commit(
        self,
        *,
        index: int,
        concept_id: str,
        result_json: str | Path | None = None,
    ) -> CheckpointEntry:
        """Promote the pending checkpoint once its evaluation sweep finished.

        Pairing the state with the sha256 of the result JSON is what makes the
        rehydrated metric matrices trustworthy later: if the JSON has been
        rewritten by a different process in the meantime, the resume aborts
        instead of restoring a matrix with holes in it.
        """
        manifest = self.read_manifest()
        if manifest is None:
            raise CheckpointCorruptionError(f"No manifest in {self._directory}; nothing to commit")
        pending = self._entry(manifest, "pending")
        if pending is None:
            raise CheckpointCorruptionError(
                f"No pending checkpoint to commit for concept {index} ({concept_id}) in {self._directory}"
            )
        if pending.index != int(index):
            raise CheckpointCorruptionError(
                f"Pending checkpoint is for concept {pending.index} ({pending.concept_id}) but commit was "
                f"asked for concept {index} ({concept_id})"
            )

        result_path = Path(result_json).expanduser() if result_json is not None else None
        result_sha = _sha256_file(result_path) if result_path is not None and result_path.is_file() else None

        committed = CheckpointEntry(
            index=pending.index,
            concept_id=pending.concept_id,
            phase=pending.phase,
            filename=pending.filename,
            sha256=pending.sha256,
            bytes=pending.bytes,
            written_at=pending.written_at,
            sidecars=pending.sidecars,
            variant=pending.variant,
            result_json=str(result_path) if result_path is not None else None,
            result_json_sha256=result_sha,
        )

        manifest["committed"] = committed.to_json()
        manifest["pending"] = None
        manifest["history"] = self._merge_history(manifest.get("history", []), committed)
        self._write_manifest(manifest)
        self._heartbeat()
        self._prune(manifest)
        return committed

    # ------------------------------------------------------------------ reading

    def load(self, entry: CheckpointEntry) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Read one checkpoint, verifying every sha256 the manifest recorded."""
        path = self.path_of(entry)
        if not path.is_file():
            raise CheckpointCorruptionError(f"Checkpoint {path} named by the manifest does not exist")
        if entry.sha256:
            actual = _sha256_file(path)
            if actual != entry.sha256:
                raise CheckpointCorruptionError(
                    f"Checkpoint {path} failed its sha256 check "
                    f"(manifest={entry.sha256[:12]}..., file={actual[:12]}...)"
                )
        try:
            envelope = _torch_load(path)
        except Exception as error:  # torch raises a zoo of types on truncation
            raise CheckpointCorruptionError(f"Checkpoint {path} could not be read: {error}") from error
        if not isinstance(envelope, dict):
            raise CheckpointCorruptionError(f"Checkpoint {path} does not hold an envelope mapping")

        sidecars: dict[str, np.ndarray] = {}
        for sidecar in entry.sidecars:
            sidecar_path = self._directory / sidecar.filename
            if not sidecar_path.is_file():
                raise CheckpointCorruptionError(f"Sidecar {sidecar_path} named by the manifest does not exist")
            actual = _sha256_file(sidecar_path)
            if sidecar.sha256 and actual != sidecar.sha256:
                raise CheckpointCorruptionError(
                    f"Sidecar {sidecar_path} failed its sha256 check "
                    f"(manifest={sidecar.sha256[:12]}..., file={actual[:12]}...)"
                )
            array = np.load(sidecar_path, allow_pickle=False)
            expected_shape = tuple(sidecar.shape)
            if expected_shape and tuple(array.shape) != expected_shape:
                raise CheckpointCorruptionError(
                    f"Sidecar {sidecar_path} has shape {tuple(array.shape)}, manifest recorded {expected_shape}"
                )
            sidecars[sidecar.name] = array
        return envelope, sidecars

    # ------------------------------------------------------------------ locking

    def acquire_lock(self, *, force: bool = False) -> None:
        """Refuse to share a run directory with another live process.

        A double submit or a scheduler requeue race would otherwise have two
        processes interleaving manifest rewrites, and the loser's concepts
        vanish from the history with no error anywhere.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        existing = self._read_lock()
        if existing is not None and not force:
            age = time.time() - float(existing.get("heartbeat", 0.0))
            mine = existing.get("hostname") == _hostname() and int(existing.get("owner_pid", -1)) == os.getpid()
            if not mine and age <= self._lock_stale_after:
                raise CheckpointLockError(
                    f"Run {self._run_id} is already active on {existing.get('hostname')} "
                    f"pid {existing.get('owner_pid')} (heartbeat {age:.0f}s ago). "
                    "Two processes writing one checkpoint directory would corrupt the history."
                )
            if not mine:
                logger.warning(
                    "Taking over stale checkpoint lock from %s pid %s (heartbeat %.0fs ago)",
                    existing.get("hostname"),
                    existing.get("owner_pid"),
                    age,
                )
        self._write_lock()
        self._lock_acquired = True

    def release_lock(self) -> None:
        if not self._lock_acquired:
            return
        existing = self._read_lock()
        if existing is not None and int(existing.get("owner_pid", -1)) == os.getpid():
            try:
                self.lock_path.unlink()
            except OSError:  # pragma: no cover - best effort
                logger.debug("Could not remove checkpoint lock %s", self.lock_path)
        self._lock_acquired = False

    def __enter__(self) -> "CheckpointStore":
        self.acquire_lock()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release_lock()

    def _read_lock(self) -> dict[str, Any] | None:
        if not self.lock_path.is_file():
            return None
        try:
            return json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable checkpoint lock %s", self.lock_path)
            return None

    def _write_lock(self) -> None:
        payload = {
            "owner_pid": os.getpid(),
            "hostname": _hostname(),
            "run_id": self._run_id,
            "acquired_at": utc_now(),
            "heartbeat": time.time(),
        }
        self._atomic_write_bytes(self.lock_path, json.dumps(payload, indent=2).encode("utf-8"))

    def _heartbeat(self) -> None:
        if self._lock_acquired:
            self._write_lock()

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _stem(index: int, concept_id: str, variant: str | None) -> str:
        safe_concept = "".join(
            character if character.isalnum() or character in "-_" else "_" for character in concept_id
        )
        stem = f"concept_{index:03d}_{safe_concept}"
        if variant:
            safe_variant = "".join(c if c.isalnum() or c in "-_" else "_" for c in variant)
            stem = f"{stem}.{safe_variant}"
        return stem

    def _preflight_disk_space(self, estimated_bytes: int, *, stem: str) -> None:
        required = int(estimated_bytes * _DISK_HEADROOM)
        try:
            free = shutil.disk_usage(self._directory).free
        except OSError:  # pragma: no cover - defensive
            return
        if free < required:
            raise CheckpointDiskSpaceError(
                f"Cannot write checkpoint {stem}: {required / (1024 ** 3):.2f} GiB needed "
                f"(estimate x{_DISK_HEADROOM}) but only {free / (1024 ** 3):.2f} GiB free on "
                f"{self._directory}. Nothing was written; the previous checkpoint is still valid."
            )

    def _should_verify(self, size_bytes: int) -> bool:
        if self._verify_writes is None:
            return size_bytes < _VERIFY_WRITE_LIMIT_BYTES
        return bool(self._verify_writes)

    def _verify_readback(self, path: Path, entry: CheckpointEntry) -> None:
        try:
            envelope = _torch_load(path)
        except Exception as error:
            raise CheckpointCorruptionError(
                f"Checkpoint {path} was written but could not be read back: {error}"
            ) from error
        run = envelope.get("run") if isinstance(envelope, Mapping) else None
        if not isinstance(run, Mapping) or int(run.get("concept_index", -1)) != entry.index:
            raise CheckpointCorruptionError(
                f"Checkpoint {path} read back with an unexpected concept index; the write did not land correctly"
            )

    @staticmethod
    def _torch_save(payload: Mapping[str, Any], handle) -> None:
        import torch

        torch.save(dict(payload), handle)

    def _write_sidecar(self, *, stem: str, name: str, array: np.ndarray) -> SidecarInfo:
        """Write one array with ``np.save``, uncompressed and on purpose.

        zlib on gigabytes of float32 embeddings saves almost nothing and costs
        minutes per concept. (Binary masks are the opposite case, which is why
        the ReplayCAD artifacts compress those — the choice is per-payload, not
        a house style.) Keeping the array out of the torch payload also keeps
        envelope validation a fast read.
        """
        filename = f"{stem}.{name}.npy"
        path = self._directory / filename
        contiguous = np.ascontiguousarray(array)
        self._atomic_write(path, lambda handle: np.save(handle, contiguous, allow_pickle=False))
        return SidecarInfo(
            name=name,
            filename=filename,
            sha256=_sha256_file(path),
            shape=tuple(int(dim) for dim in contiguous.shape),
            dtype=str(contiguous.dtype),
            bytes=path.stat().st_size,
        )

    def _atomic_write(self, path: Path, writer) -> None:
        """tmp -> fsync -> replace -> fsync(dir), with one retry on ``OSError``."""
        last_error: OSError | None = None
        for attempt in (1, 2):
            temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
            try:
                with temp_path.open("wb") as handle:
                    writer(handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
                _fsync_directory(path.parent)
                return
            except OSError as error:
                last_error = error
                logger.warning("Checkpoint write to %s failed (attempt %d/2): %s", path, attempt, error)
            finally:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:  # pragma: no cover - best effort
                        logger.debug("Could not remove temporary checkpoint file %s", temp_path)
        assert last_error is not None
        message = (
            f"Failed to write {path} twice: {last_error}. The manifest was not touched, so the previous "
            "checkpoint remains valid."
        )
        if last_error.errno in (errno.ENOSPC, errno.EDQUOT, errno.EFBIG):
            raise CheckpointDiskSpaceError(message) from last_error
        raise ResumeError(message) from last_error

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        self._atomic_write(path, lambda handle: handle.write(payload))

    @staticmethod
    def _merge_history(history: Sequence[Mapping[str, Any]], entry: CheckpointEntry) -> list[dict[str, Any]]:
        merged = [dict(item) for item in history if item.get("filename") != entry.filename]
        merged.append(entry.to_json())
        merged.sort(key=lambda item: int(item["index"]))
        return merged

    @staticmethod
    def _merge_variants(variants: Sequence[Mapping[str, Any]], entry: CheckpointEntry) -> list[dict[str, Any]]:
        merged = [dict(item) for item in variants if item.get("filename") != entry.filename]
        merged.append(entry.to_json())
        merged.sort(key=lambda item: (int(item["index"]), str(item.get("variant") or "")))
        return merged

    def _prune(self, manifest: Mapping[str, Any]) -> None:
        """Delete superseded checkpoints — only ever after the manifest commit.

        ``keep=0`` means unlimited, which is the right default for a method
        whose per-concept state is a couple of megabytes. Variant files (e.g. a
        basis-only snapshot) are never pruned: they exist precisely because
        their content cannot be recomputed once the training data of earlier
        concepts is gone.
        """
        if self._keep <= 0:
            return
        history = [CheckpointEntry.from_json(item) for item in manifest.get("history", [])]
        protected = {
            entry.filename
            for entry in (self._entry(manifest, "committed"), self._entry(manifest, "pending"))
            if entry is not None
        }
        keepers = {entry.filename for entry in sorted(history, key=lambda item: item.index)[-self._keep :]}
        keepers |= protected

        remaining: list[dict[str, Any]] = []
        for entry in history:
            if entry.filename in keepers:
                remaining.append(entry.to_json())
                continue
            self._delete_entry_files(entry)
        if len(remaining) != len(history):
            updated = dict(manifest)
            updated["history"] = remaining
            self._write_manifest(updated)

    def _discard_unreferenced(self, checkpoint_path: Path, sidecars: Sequence[SidecarInfo]) -> None:
        for path in [checkpoint_path] + [self._directory / sidecar.filename for sidecar in sidecars]:
            try:
                if path.is_file():
                    path.unlink()
            except OSError:  # pragma: no cover - best effort
                logger.warning("Could not remove unreferenced checkpoint file %s", path)

    def _delete_entry_files(self, entry: CheckpointEntry) -> None:
        for path in [self.path_of(entry)] + [self._directory / sidecar.filename for sidecar in entry.sidecars]:
            try:
                if path.is_file():
                    path.unlink()
                    logger.debug("Pruned superseded checkpoint file %s", path)
            except OSError:  # pragma: no cover - best effort
                logger.warning("Could not prune checkpoint file %s", path)


def _hostname() -> str:
    import socket

    return socket.gethostname()
