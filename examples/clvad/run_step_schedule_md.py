"""Step-schedule runner for the InCLAD-MD 27-concept stream.

The step-schedule runner loads its dataset through
``_BASE.read_registered_visual_benchmark_dataset``, which maps a benchmark *name*
onto one of five hard-coded reader presets::

    PREDEFINED_BENCHMARK_READERS = {"btech", "dagm", "mpdd", "mvtec", "visa"}

InCLAD-MD is not one of them, so asking for it dies with

    ValueError: Unsupported visual benchmark 'inclad-md'.
                Available presets: btech, dagm, mpdd, mvtec, visa

even though a loader for it exists (``pyclad.data.datasets.inclad_md_dataset``)
and the *levels* runner already uses it. Passing ``--root`` does not help: it
short-circuits the *root* resolution, not the *reader* lookup.

Rather than edit the step-schedule runner or add a sixth preset to the library, this module
loads the step-schedule runner unchanged and swaps two functions on the module object it
already holds, so ``--benchmark inclad-md`` resolves to ``InCLADMDDataset`` and
every other benchmark behaves exactly as before. Everything else -- schedule
parsing, the model/strategy sweep, rectangular metrics, output naming -- is the
step-schedule runner's own code.

Usage is the step-schedule runner's, with the benchmark switched::

    python examples/clvad/run_step_schedule_md.py \\
        --model padim --strategy cl \\
        --benchmark inclad-md --step-schedule 26-1 \\
        --backbone resnet18 --pretrained --eval-level both

``inclad-md``, ``inclad_md``, ``multidataset`` and ``md`` are all accepted.

Two properties of this stream differ from the per-dataset benchmarks and are
enforced here rather than discovered mid-run:

* it ships **only** ``easy_to_hard`` and ``hard_to_easy``; there is no random
  split, so asking for one fails immediately with an explanation;
* concepts are named ``<dataset>__<category>`` (``mvtec__bottle``,
  ``dagm__Class4``). Nothing in the callbacks parses concept names, so the
  prefix is inert -- but it is why the result files are not directly
  interchangeable with a single-dataset run.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from typing import Any

logger = logging.getLogger(__name__)

_MD_ALIASES = {"inclad-md", "incladmd", "inclad_md", "multidataset", "multidataset-seq-v1", "md"}


def _is_md(benchmark: Any) -> bool:
    if not isinstance(benchmark, str):
        return False
    return benchmark.strip().lower().replace("_", "-") in {a.replace("_", "-") for a in _MD_ALIASES}


def _load_step_schedule_runner():
    module_path = pathlib.Path(__file__).with_name("run_continual_visual_ad_step_schedule.py")
    spec = importlib.util.spec_from_file_location("pyclad_step_schedule_for_md", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the step-schedule runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_step_schedule_runner()
_BASE = _RUNNER._BASE

_orig_resolve_root = _BASE.resolve_visual_benchmark_root
_orig_read_dataset = _BASE.read_registered_visual_benchmark_dataset


def _resolve_visual_benchmark_root(benchmark=None, root=None, registry_path=None, **kwargs):
    """For InCLAD-MD there is no single root; report the shared datasets dir.

    ``InCLADMDDataset`` resolves the five source roots itself from
    ``PYCLAD_VISUAL_DATASETS_ROOT``. The step-schedule runner still logs a root and hands it
    to the pixel-metric callback, which needs *some* directory to look for
    ground-truth masks under, so the shared parent is the right answer.
    """
    if _is_md(benchmark):
        shared = root or os.environ.get("PYCLAD_VISUAL_DATASETS_ROOT")
        if not shared:
            raise ValueError(
                "InCLAD-MD needs --root (or PYCLAD_VISUAL_DATASETS_ROOT) pointing at the "
                "directory that holds mvtec_ad/, visa/, btech/, dagm/ and mpdd/."
            )
        return pathlib.Path(shared).expanduser().resolve()
    return _orig_resolve_root(benchmark=benchmark, root=root, registry_path=registry_path, **kwargs)


def _read_registered_visual_benchmark_dataset(
    benchmark=None,
    root=None,
    registry_path=None,
    dataset_name=None,
    categories=None,
    data_mode="numpy",
    resize_to=None,
    color_mode="rgb",
    max_train_samples_per_category=None,
    max_test_samples_per_category=None,
    **kwargs,
):
    if not _is_md(benchmark):
        return _orig_read_dataset(
            benchmark=benchmark,
            root=root,
            registry_path=registry_path,
            dataset_name=dataset_name,
            categories=categories,
            data_mode=data_mode,
            resize_to=resize_to,
            color_mode=color_mode,
            max_train_samples_per_category=max_train_samples_per_category,
            max_test_samples_per_category=max_test_samples_per_category,
            **kwargs,
        )

    from pyclad.data.datasets.inclad_md_dataset import (
        INCLAD_MD_ORDERINGS,
        InCLADMDDataset,
    )

    ordering = os.environ.get("INCLAD_MD_ORDERING", "easy_to_hard")
    if ordering not in INCLAD_MD_ORDERINGS:
        raise ValueError(
            f"InCLAD-MD has no '{ordering}' ordering; the manifest ships only "
            f"{INCLAD_MD_ORDERINGS}. Set INCLAD_MD_ORDERING to one of those. "
            f"(A random split does not exist for this stream, and the published "
            f"multidataset results have none either.)"
        )

    if root:
        os.environ["PYCLAD_VISUAL_DATASETS_ROOT"] = str(root)

    logger.info("Loading InCLAD-MD manifest: ordering=%s (27 concepts)", ordering)
    return InCLADMDDataset(
        ordering=ordering,
        registry_path=registry_path,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


_BASE.resolve_visual_benchmark_root = _resolve_visual_benchmark_root
_BASE.read_registered_visual_benchmark_dataset = _read_registered_visual_benchmark_dataset


# ── pixel metrics ────────────────────────────────────────────────────────────
#
# Patching the two functions above is not enough for ``--eval-level pixel|both``.
# The pixel callbacks index ground-truth masks through a *separate* path that
# never touches the dataset reader, ending in the same reader-preset lookup:
#
#   ...VisualPixelConceptMetricCallback.__init__
#     -> visual_pixel_utils.index_registered_visual_test_samples
#        -> visual_dataset_registry.index_registered_visual_benchmark
#           -> visual_benchmark_reader.build_visual_benchmark_reader
#              -> ValueError: Unsupported visual benchmark 'inclad-md'
#
# That is why the first full grid attempt produced 255 runs which all died in
# "[3/5] Building strategy and callbacks" after ~2 seconds, with the dataset
# itself having loaded correctly, 27 concepts and grouping included.
#
# The indexer lives in _inclad_md_pixel so this runner and the resumable runner
# (UCAD / One-for-More / ReplayCAD) share one implementation -- they hit the
# identical failure from different call sites.
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _inclad_md_pixel  # noqa: E402

_inclad_md_pixel.install()



if __name__ == "__main__":
    _RUNNER.main()
