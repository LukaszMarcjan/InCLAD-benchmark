"""Camera-ready addition (CR): block-wise feature reduction for PatchCore's coreset.

Why
---
Three of the five failed ReplayCAD cells died here::

    File ".../pyclad/models/patchcore/patchcore.py", line 133, in _reduce_features
      return mapper(features.to(self.device))
    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 22.66 GiB.

``_reduce_features`` moves the *entire* patch-feature matrix to the device and
applies the projection in one call. At concept 7 of MVTec that matrix is already
453,017 x 1024, and it keeps growing with the replay union, so the single
allocation reaches 22-28 GiB. The proximate trigger was three of our own jobs
sharing one H100, but a 22 GiB spike is the reason co-tenancy was fatal at all.

Note this is *not* fixed by lowering ``coreset_sampling_ratio``: the reduction
runs on the full candidate set, before any sampling happens.

What this does
--------------
``nn.Linear`` is row-independent: output row *i* depends only on input row *i*
and the weight matrix. Slicing the input into row blocks and concatenating is
therefore the same computation, with peak memory set by the block size instead
of by the number of patches.

ADDITIVE: ``patchcore.py`` is not edited. Importing this module changes nothing;
call :func:`enable` (or set ``PYCLAD_CR_CHUNK_ROWS``) to install the patch, and
:func:`disable` to put the original method back.

Determinism
-----------
The mapper is built inside the same ``torch.random.fork_rng()`` block with the
same seed, so the projection matrix is identical, and the arithmetic is the same
row by row.

It is **not bit-identical**, and the tests pin that down rather than hoping. BLAS
picks its kernel partly from the row count, so a chunked call is under no
obligation to reproduce the single-call result exactly. Measured on CPU float32:
equal for (1000 rows, 128 per block), (5000, 997) and the unchunked path, but
~1.4e-06 apart for 4097 rows in blocks of 4096 -- a 1-row tail block.

1e-06 is far below any anomaly score that matters, but the greedy coreset selects
by argmax over distances, so a perturbation that small can still flip a near-tie
and change which patch enters the bank. Treat this as a fallback for when a run
genuinely cannot fit, not as a free speedup: prefer an exclusive GPU first
(``j_exclusive=yes``), verify end to end on BTech before trusting it on headline
numbers, and write "equivalent to within float32 noise", never "identical".

Fingerprint
-----------
Checkpoints are keyed by configuration values (model, strategy, benchmark, seed,
concept order, eval level, resize, extra) -- not by source. Enabling this does
**not** invalidate an existing checkpoint directory. It does change the recorded
git commit, so a resumed run may report code drift; pass ``--allow-code-drift``
or accept the warning. (Contrast ``coreset_sampling_ratio``, which *is* a
fingerprint field for ReplayCAD and therefore does force a fresh run.)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import nn

from pyclad.models.patchcore.patchcore import ApproximateGreedyCoresetSampler

__all__ = ["enable", "disable", "is_enabled", "DEFAULT_CHUNK_ROWS"]

logger = logging.getLogger(__name__)

#: Rows per block. 65,536 x 1024 float32 is 256 MiB in, 256 MiB out, which keeps
#: the spike flat regardless of how large the replay union grows.
DEFAULT_CHUNK_ROWS = 65_536

_ORIGINAL: Optional[object] = None


def _chunked_reduce_features(self, features: torch.Tensor) -> torch.Tensor:
    chunk_rows = getattr(self, "_cr_chunk_rows", DEFAULT_CHUNK_ROWS)

    # Identical early-out to the original: no projection needed, so nothing to
    # chunk -- a single .to() of an already-correct matrix.
    if features.shape[1] == self.dimension_to_project_features_to:
        return features.to(self.device)

    # Same construction as the original, so the projection matrix is the same
    # draw: same fork_rng scope, same seed, same shapes, same order.
    with torch.random.fork_rng():
        torch.manual_seed(self.random_seed)
        mapper = nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        ).to(self.device)

    rows = features.shape[0]
    if rows <= chunk_rows:
        return mapper(features.to(self.device))

    # Allocate from the first block rather than from `mapper.weight.dtype`: the
    # result dtype is whatever the matmul promotes to, which need not match the
    # weight (half-precision features against float32 weights, say).
    first = mapper(features[:chunk_rows].to(self.device))
    out = torch.empty(
        (rows, first.shape[1]), device=first.device, dtype=first.dtype
    )
    out[:chunk_rows] = first
    for start in range(chunk_rows, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        out[start:stop] = mapper(features[start:stop].to(self.device))
    return out


def is_enabled() -> bool:
    return _ORIGINAL is not None


def enable(chunk_rows: int = DEFAULT_CHUNK_ROWS) -> None:
    """Install the block-wise reduction. Idempotent."""
    global _ORIGINAL
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if _ORIGINAL is None:
        _ORIGINAL = ApproximateGreedyCoresetSampler._reduce_features
        ApproximateGreedyCoresetSampler._reduce_features = _chunked_reduce_features
    ApproximateGreedyCoresetSampler._cr_chunk_rows = chunk_rows
    logger.info("CR: PatchCore feature reduction chunked at %d rows per block", chunk_rows)


def disable() -> None:
    """Restore the original method. Idempotent."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        ApproximateGreedyCoresetSampler._reduce_features = _ORIGINAL
        _ORIGINAL = None
    if hasattr(ApproximateGreedyCoresetSampler, "_cr_chunk_rows"):
        delattr(ApproximateGreedyCoresetSampler, "_cr_chunk_rows")


def enable_from_env() -> bool:
    """Enable if ``PYCLAD_CR_CHUNK_ROWS`` is set to a positive integer."""
    raw = os.environ.get("PYCLAD_CR_CHUNK_ROWS")
    if not raw:
        return False
    try:
        rows = int(raw)
    except ValueError:
        logger.warning("CR: ignoring non-integer PYCLAD_CR_CHUNK_ROWS=%r", raw)
        return False
    if rows <= 0:
        return False
    enable(rows)
    return True
