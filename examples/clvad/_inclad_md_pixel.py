"""Pixel-level metrics for the InCLAD-MD 27-concept stream.

Loading InCLAD-MD training/test data works -- the runners resolve it through
``InCLADMDDataset``. Building the *pixel* metric callbacks does not::

    [5/6] Building callbacks and restoring state
    ValueError: Unsupported visual benchmark 'inclad-md'.
                Available presets: btech, dagm, mpdd, mvtec, visa

The callbacks index ground-truth masks through a *separate* path that never
touches the dataset object::

    VisualPixelConceptMetricCallback.__init__
      -> visual_pixel_utils.index_registered_visual_test_samples
         -> visual_dataset_registry.index_registered_visual_benchmark
            -> visual_benchmark_reader.build_visual_benchmark_reader
               -> ValueError: Unsupported visual benchmark 'inclad-md'

``--root`` does not help: it short-circuits *root* resolution, not *reader*
lookup. This is the same reader-preset lookup the dataset path hits, reached
from a different call site -- which is why every InCLAD-MD run died in
"Building callbacks" after ~2 seconds with the dataset itself having loaded
correctly, 27 concepts and grouping included.

Re-indexing the five source datasets is unnecessary: the InCLAD-MD manifest
already carries ``mask_relpath`` per row, and ``_index_samples`` resolves it
against the right source root, returning ``VisualSample`` objects with
``mask_path`` populated. So the replacement builds the callback's
``{concept: [VisualSample, ...]}`` map straight from the manifest, through the
same helpers the training data goes through. That is what guarantees the concept
names match exactly (``mvtec__bottle``, ``dagm__Class4``) -- the callback looks
samples up by ``evaluated_concept.name``.

Nothing under ``src/pyclad`` is modified. Used by both InCLAD-MD runners::

    import _inclad_md_pixel
    _inclad_md_pixel.install()
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_MD_ALIASES = {"inclad-md", "incladmd", "inclad_md", "multidataset", "multidataset-seq-v1", "md"}

_test_sample_cache: dict[Any, dict[str, list]] = {}
_installed = False


def is_inclad_md(benchmark: Any) -> bool:
    if not isinstance(benchmark, str):
        return False
    return benchmark.strip().lower().replace("_", "-") in {a.replace("_", "-") for a in _MD_ALIASES}


def index_md_test_samples(
    root=None,
    registry_path=None,
    categories=None,
    max_test_samples_per_category=None,
):
    """Build ``{concept_name: [test VisualSample, ...]}`` for InCLAD-MD."""
    from datasets import load_dataset

    from pyclad.data.datasets.inclad_md_dataset import (
        INCLAD_MD_HF_CONFIG,
        INCLAD_MD_HF_REPO,
        INCLAD_MD_ORDERINGS,
        _index_samples,
        _parse_hf_manifest,
        _resolve_roots,
    )

    ordering = os.environ.get("INCLAD_MD_ORDERING", "easy_to_hard")
    if ordering not in INCLAD_MD_ORDERINGS:
        raise ValueError(
            f"InCLAD-MD has no '{ordering}' ordering; the manifest ships only "
            f"{INCLAD_MD_ORDERINGS}."
        )

    if root:
        os.environ["PYCLAD_VISUAL_DATASETS_ROOT"] = str(root)

    cache_key = (
        ordering,
        tuple(categories) if categories is not None else None,
        max_test_samples_per_category,
    )
    cached = _test_sample_cache.get(cache_key)
    if cached is not None:
        return cached

    # Each source dataset gets its own root; _resolve_roots reads the same env
    # vars / registry the single-dataset loaders use.
    source_roots = _resolve_roots(None, registry_path)
    rows = _parse_hf_manifest(
        load_dataset(INCLAD_MD_HF_REPO, INCLAD_MD_HF_CONFIG, split=ordering)
    )
    samples = _index_samples(
        rows=rows,
        source_roots=source_roots,
        categories=categories,
        max_train_per_category=None,
        max_test_per_category=max_test_samples_per_category,
    )

    grouped: dict[str, list] = {}
    for sample in samples:
        if sample.split == "test":
            grouped.setdefault(sample.category, []).append(sample)

    if not grouped:
        raise RuntimeError(
            "InCLAD-MD pixel index is empty -- no test samples were resolved. "
            "Check PYCLAD_VISUAL_DATASETS_ROOT and the five source dataset roots."
        )

    total = sum(len(v) for v in grouped.values())
    with_masks = sum(1 for v in grouped.values() for s in v if s.mask_path is not None)
    logger.info(
        "InCLAD-MD pixel index: %d concepts, %d test samples, %d with ground-truth masks",
        len(grouped),
        total,
        with_masks,
    )

    _test_sample_cache[cache_key] = grouped
    return grouped


def install() -> bool:
    """Route InCLAD-MD mask indexing away from the reader-preset lookup.

    The callbacks bind the function with ``from ... import`` at *their* import
    time, and that import is lazy -- it happens inside ``build_callbacks``,
    which runs after this is called. Rebinding the attribute on the defining
    module is therefore enough for them. Any module that already imported it is
    rebound explicitly, so the patch does not depend on import order.

    Idempotent; returns True if this call applied the patch.
    """
    global _installed
    if _installed:
        return False

    from pyclad.callbacks.evaluation import visual_pixel_utils

    original = visual_pixel_utils.index_registered_visual_test_samples

    def _patched(
        benchmark=None,
        root=None,
        registry_path=None,
        categories=None,
        max_test_samples_per_category=None,
        **kwargs,
    ):
        if is_inclad_md(benchmark):
            return index_md_test_samples(
                root=root,
                registry_path=registry_path,
                categories=categories,
                max_test_samples_per_category=max_test_samples_per_category,
            )
        return original(
            benchmark=benchmark,
            root=root,
            registry_path=registry_path,
            categories=categories,
            max_test_samples_per_category=max_test_samples_per_category,
            **kwargs,
        )

    visual_pixel_utils.index_registered_visual_test_samples = _patched

    for module in list(sys.modules.values()):
        if module is None or module is visual_pixel_utils:
            continue
        if not getattr(module, "__name__", "").startswith("pyclad.callbacks.evaluation"):
            continue
        if getattr(module, "index_registered_visual_test_samples", None) is original:
            module.index_registered_visual_test_samples = _patched

    _installed = True
    return True
