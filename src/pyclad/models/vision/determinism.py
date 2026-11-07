"""Opt-in determinism flags for multi-seed experiments.

Additive module. ``set_global_seeds`` in the runner is left untouched, so no
existing configuration changes behaviour unless a caller asks for it.

``set_global_seeds`` seeds ``random``, ``numpy``, ``torch`` and CUDA, which is
necessary but not sufficient: without pinning cuDNN, convolution kernels may
accumulate non-deterministically, and a run is then only empirically repeatable
rather than guaranteed. In this project nine of ten single-family baselines
reproduced exactly on an independent rerun and one did not, which is the shape
that unpinned kernels produce.

One flag cannot be set from here at all. ``CUBLAS_WORKSPACE_CONFIG`` must be in
the environment before CUDA initialises, so this function reports whether it was
set rather than pretending to pin it:

    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export PYTHONHASHSEED=0
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


def enable_determinism(warn_only: bool = True) -> Dict[str, Any]:
    """Pin the deterministic backends and report what was and was not pinned.

    ``warn_only`` is the default because some operations have no deterministic
    implementation; a hard failure would abort a grid instead of recording a
    warning in the log.
    """
    report: Dict[str, Any] = {
        "cublas_workspace_config_env_set": "CUBLAS_WORKSPACE_CONFIG" in os.environ,
        "pythonhashseed_env_set": "PYTHONHASHSEED" in os.environ,
    }
    try:
        import torch
    except Exception:
        report["torch_available"] = False
        return report

    report["torch_available"] = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    report["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)

    try:
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        report["deterministic_algorithms"] = True
    except Exception as exc:  # older torch, or an unsupported combination
        report["deterministic_algorithms"] = False
        report["deterministic_algorithms_error"] = str(exc)
    return report


__all__: List[str] = ["enable_determinism"]
