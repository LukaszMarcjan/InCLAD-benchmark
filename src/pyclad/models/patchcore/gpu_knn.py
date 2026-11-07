"""Exact nearest-neighbour search for PatchCore on the GPU.

``PatchCore`` scores every test patch against its memory bank with sklearn's
``NearestNeighbors(n_jobs=1)``, which runs single-threaded on the CPU. In the
InCLAD ReplayCAD runs that search is about three quarters of the wall clock.
:class:`TorchExactNearestNeighbors` gives the same neighbours and distances on
the GPU:

1. candidates: squared distances ``|q|^2 + |b|^2 - 2 q.b`` in float32, bank in
   chunks, keeping the ``n_neighbors + CANDIDATE_MARGIN`` smallest per query;
2. exact: the distances to those candidates recomputed as ``|q - b|`` in float64,
   the ``n_neighbors`` smallest kept.

Step 1 can only misorder points whose distances agree to float32 rounding, and
the margin keeps the true neighbours among the candidates, so the result
matches sklearn's brute-force search to float32 output precision. TF32 is
switched off for the matrix product.

Nothing here changes ``PatchCore``. :func:`install` swaps the index that
``PatchCore._fit_nn_index`` builds, only for models on a CUDA device, and
returns a function that restores the original method.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional, Tuple

import numpy as np
import torch

#: Extra float32 candidates per query that are re-ranked in float64.
CANDIDATE_MARGIN = 8


@contextlib.contextmanager
def _no_tf32():
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class TorchExactNearestNeighbors:
    """Euclidean k-nearest neighbours with sklearn's ``kneighbors`` interface."""

    def __init__(
        self,
        n_neighbors: int,
        device: torch.device,
        query_chunk: int = 4096,
        bank_chunk: int = 131072,
    ):
        self.n_neighbors = int(n_neighbors)
        self.device = torch.device(device)
        self.query_chunk = int(query_chunk)
        self.bank_chunk = int(bank_chunk)
        self._bank: Optional[torch.Tensor] = None
        self._bank_squared: Optional[torch.Tensor] = None

    def fit(self, bank: np.ndarray) -> "TorchExactNearestNeighbors":
        array = np.ascontiguousarray(bank, dtype=np.float32)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError(f"memory bank must be a non-empty (N, D) matrix, got shape {array.shape}")
        self._bank = torch.from_numpy(array).to(self.device)
        self._bank_squared = (self._bank * self._bank).sum(dim=1)
        return self

    def kneighbors(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._bank is None or self._bank_squared is None:
            raise RuntimeError("TorchExactNearestNeighbors.kneighbors() called before fit()")
        array = np.ascontiguousarray(queries, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self._bank.shape[1]:
            raise ValueError(f"queries must have shape (M, {self._bank.shape[1]}), got {array.shape}")
        k = min(self.n_neighbors, len(self._bank))
        candidates = min(k + CANDIDATE_MARGIN, len(self._bank))
        distances = np.empty((len(array), k), dtype=np.float32)
        indices = np.empty((len(array), k), dtype=np.int64)
        with torch.no_grad(), _no_tf32():
            for start in range(0, len(array), self.query_chunk):
                query = torch.from_numpy(array[start : start + self.query_chunk]).to(self.device)
                candidate_index = self._candidates(query, candidates)
                chosen = self._bank[candidate_index].double()
                exact = torch.linalg.vector_norm(chosen - query.double()[:, None, :], dim=-1)
                order = torch.argsort(exact, dim=1, stable=True)[:, :k]
                stop = start + len(query)
                distances[start:stop] = torch.gather(exact, 1, order).float().cpu().numpy()
                indices[start:stop] = torch.gather(candidate_index, 1, order).cpu().numpy()
        return distances, indices

    def _candidates(self, query: torch.Tensor, count: int) -> torch.Tensor:
        query_squared = (query * query).sum(dim=1, keepdim=True)
        best_distance: Optional[torch.Tensor] = None
        best_index: Optional[torch.Tensor] = None
        for start in range(0, len(self._bank), self.bank_chunk):
            bank = self._bank[start : start + self.bank_chunk]
            squared = query_squared + self._bank_squared[start : start + len(bank)][None, :] - 2.0 * (query @ bank.T)
            take = min(count, len(bank))
            distance, index = torch.topk(squared, take, dim=1, largest=False)
            index = index + start
            if best_distance is None:
                best_distance, best_index = distance, index
                continue
            merged_distance = torch.cat([best_distance, distance], dim=1)
            merged_index = torch.cat([best_index, index], dim=1)
            keep = min(count, merged_distance.shape[1])
            best_distance, position = torch.topk(merged_distance, keep, dim=1, largest=False)
            best_index = torch.gather(merged_index, 1, position)
        return best_index


def _fit_nn_index_on_gpu(original: Callable) -> Callable:
    def fit_nn_index(self) -> None:
        if getattr(self, "_device", torch.device("cpu")).type != "cuda":
            return original(self)
        self._ensure_fitted()
        self._nn_index = TorchExactNearestNeighbors(
            n_neighbors=min(self.config.n_neighbors, len(self._memory_bank)),
            device=self._device,
        ).fit(self._memory_bank)

    fit_nn_index.__wrapped__ = original
    return fit_nn_index


def install() -> Callable[[], None]:
    """Build PatchCore's nearest-neighbour index on the GPU; returns a function that undoes it."""
    from pyclad.models.patchcore.patchcore import PatchCore

    original = PatchCore.__dict__["_fit_nn_index"]
    PatchCore._fit_nn_index = _fit_nn_index_on_gpu(original)

    def uninstall() -> None:
        PatchCore._fit_nn_index = original

    return uninstall
