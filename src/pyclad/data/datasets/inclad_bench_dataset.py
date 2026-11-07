from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from datasets import Features, Value, load_dataset

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.visual_benchmark_manifest import (
    VISUAL_BENCHMARK_MANIFEST_FIELDNAMES,
    build_visual_benchmark_manifest_spec,
)
from pyclad.data.readers.visual_benchmark_reader import read_visual_benchmark_dataset
from pyclad.data.readers.visual_dataset_registry import resolve_visual_benchmark_root

# Anonymous-review mirror, kept for reference; its manifests are byte-identical
# to the public repository apart from line endings.
INCLAD_BENCH_HF_REPO_ANONYMOUS = "anonmllab/inclad-bench"
INCLAD_BENCH_HF_REPO = "lifelonglab/inclad-benchmark"

# Commit of the Hugging Face dataset repository whose manifests were checked
# against the camera-ready results. Pass ``revision=INCLAD_BENCH_HF_REVISION``
# to pin a load to it; the default (``revision=None``) follows the repository's
# ``main`` branch.
INCLAD_BENCH_HF_REVISION = "c73c153e26b19a69b08dc11a47d59967295658d9"

INCLAD_BENCH_CONFIGS = ("btech", "dagm", "mpdd", "mvtec", "visa")

INCLAD_BENCH_ORDERINGS = ("easy_to_hard", "hard_to_easy", "random")

InCLADBenchConfig = Literal["btech", "dagm", "mpdd", "mvtec", "visa"]
InCLADBenchOrdering = Literal["easy_to_hard", "hard_to_easy", "random"]

# Every manifest column is read as a string. Left to type inference, the
# Hugging Face CSV builder parses BTech categories '01', '02', '03' as integers
# and the leading zeros are lost before any later conversion can restore them.
INCLAD_BENCH_MANIFEST_FEATURES = Features(
    {column: Value("string") for column in VISUAL_BENCHMARK_MANIFEST_FIELDNAMES}
)


def _hf_manifest_to_csv(hf_dataset, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Hugging Face converts empty optional CSV cells to NaN in pandas. If
    # stringified directly, values such as an absent normal-image mask become
    # the literal path "nan". Restore missing manifest cells before writing.
    df = hf_dataset.to_pandas().fillna("")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VISUAL_BENCHMARK_MANIFEST_FIELDNAMES)
        writer.writeheader()
        for _, row in df.iterrows():
            writer.writerow(
                {field: str(row.get(field, "")) for field in VISUAL_BENCHMARK_MANIFEST_FIELDNAMES}
            )
    return output_path


class InCLADBenchDataset(ConceptsDataset):
    """
    Loader for the InCLAD-Bench benchmark hosted on HuggingFace
    (https://huggingface.co/datasets/lifelonglab/inclad-benchmark).

    The HuggingFace dataset contains sample manifests (metadata with image
    relative paths, labels, and category orderings) for five visual anomaly
    detection benchmarks: BTech, DAGM, MPDD, MVTec AD, and VisA.

    Actual images are *not* included in the HuggingFace repository — they
    must be available locally. The ``root`` parameter (or the pyCLAD visual
    dataset registry / environment variables) is used to locate the images
    on disk.
    """

    def __init__(
        self,
        benchmark: InCLADBenchConfig,
        ordering: InCLADBenchOrdering = "easy_to_hard",
        root: Optional[Union[str, Path]] = None,
        registry_path: Optional[Union[str, Path]] = None,
        categories: Optional[Sequence[str]] = None,
        data_mode: str = "numpy",
        resize_to: Optional[tuple[int, int]] = None,
        color_mode: str = "rgb",
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        """
        :param benchmark: One of ``"btech"``, ``"dagm"``, ``"mpdd"``, ``"mvtec"``, ``"visa"``.
        :param ordering: Category ordering strategy — ``"easy_to_hard"``,
            ``"hard_to_easy"``, or ``"random"``.
        :param root: Local path to the benchmark image directory. When ``None``,
            resolved via ``PYCLAD_VISUAL_DATASETS_ROOT``, per-benchmark env vars,
            or the pyCLAD visual dataset registry.
        :param registry_path: Optional path to the visual dataset registry JSON.
        :param categories: Subset of categories to include. ``None`` = all.
        :param data_mode: ``"numpy"`` to load images as arrays, ``"paths"`` for
            file paths only.
        :param resize_to: ``(height, width)`` to resize images. ``None`` keeps
            original size.
        :param color_mode: ``"rgb"`` or ``"grayscale"``.
        :param max_train_samples_per_category: Cap on training samples per category.
        :param max_test_samples_per_category: Cap on test samples per category.
        :param cache_dir: HuggingFace cache directory. ``None`` uses the default.
        :param revision: Hugging Face revision (commit, branch or tag) of the
            manifest repository. ``None`` uses ``main``; pass
            :data:`INCLAD_BENCH_HF_REVISION` for the camera-ready manifests.
        """
        benchmark_key = benchmark.lower()
        if benchmark_key not in INCLAD_BENCH_CONFIGS:
            raise ValueError(
                f"Unknown InCLAD-Bench benchmark '{benchmark}'. "
                f"Available: {INCLAD_BENCH_CONFIGS}"
            )
        if ordering not in INCLAD_BENCH_ORDERINGS:
            raise ValueError(
                f"Unknown ordering '{ordering}'. Available: {INCLAD_BENCH_ORDERINGS}"
            )

        resolved_root = resolve_visual_benchmark_root(
            benchmark=benchmark_key, root=root, registry_path=registry_path
        )

        hf_dataset = load_dataset(
            INCLAD_BENCH_HF_REPO,
            benchmark_key,
            split=ordering,
            cache_dir=cache_dir,
            revision=revision,
            features=INCLAD_BENCH_MANIFEST_FEATURES,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / f"{benchmark_key}_{ordering}_manifest.csv"
            _hf_manifest_to_csv(hf_dataset, csv_path)

            spec = build_visual_benchmark_manifest_spec(
                benchmark=benchmark_key, csv_path=csv_path
            )
            dataset = read_visual_benchmark_dataset(
                root=resolved_root,
                benchmark=spec,
                dataset_name=f"InCLAD-{benchmark_key}-{ordering}",
                categories=categories,
                data_mode=data_mode,
                resize_to=resize_to,
                color_mode=color_mode,
                max_train_samples_per_category=max_train_samples_per_category,
                max_test_samples_per_category=max_test_samples_per_category,
            )

        super().__init__(
            name=dataset.name(),
            train_concepts=dataset.train_concepts(),
            test_concepts=dataset.test_concepts(),
            test_samples_by_concept=dataset.test_samples_by_concept(),
        )


def load_inclad_bench(
    benchmark: InCLADBenchConfig,
    ordering: InCLADBenchOrdering = "easy_to_hard",
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> ConceptsDataset:
    """Convenience function to load an InCLAD-Bench dataset.

    See :class:`InCLADBenchDataset` for parameter documentation.
    """
    return InCLADBenchDataset(
        benchmark=benchmark,
        ordering=ordering,
        root=root,
        registry_path=registry_path,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
        cache_dir=cache_dir,
        revision=revision,
    )
