"""Map paths recorded in a checkpoint to where the same files live now.

A checkpoint records two absolute paths that a resume compares or opens: the
partial result JSON (in the manifest) and the ReplayCAD artifact directory (in
the state fingerprint). Both are written under the home directory of the
account that ran the job. When that directory is renamed (for example
``/home/user-hpc`` -> ``/home/user``), the files are still there but the
recorded paths no longer resolve, and the fingerprint comparison fails even
though nothing about the run changed.

Setting ``PYCLAD_RESUME_PATH_MAP`` rewrites recorded paths before they are used::

    PYCLAD_RESUME_PATH_MAP="/home/user-hpc=/home/user"

Several pairs are separated by ``;``. A value is rewritten only when it equals an
old prefix or starts with that prefix followed by ``/``, so ``/home/user-hpc2``
is left alone. Without the variable every function here returns its input
unchanged, which keeps the default behaviour of the resume layer.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

ENV_VAR = "PYCLAD_RESUME_PATH_MAP"


def path_map_pairs() -> list[tuple[str, str]]:
    raw = os.environ.get(ENV_VAR, "").strip()
    pairs: list[tuple[str, str]] = []
    for item in raw.split(";"):
        if not item.strip():
            continue
        old, sep, new = item.partition("=")
        if not sep or not old.strip() or not new.strip():
            raise ValueError(f"{ENV_VAR} entries must look like OLD=NEW, got {item!r}")
        pairs.append((old.strip().rstrip("/"), new.strip().rstrip("/")))
    return pairs


def map_path(value: Any) -> Any:
    """Rewrite a recorded path string; anything else is returned unchanged."""
    if not isinstance(value, str):
        return value
    for old, new in path_map_pairs():
        if value == old:
            return new
        if value.startswith(old + "/"):
            return new + value[len(old):]
    return value


def map_values(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Apply :func:`map_path` to every top-level value of a mapping."""
    return {key: map_path(value) for key, value in mapping.items()}
