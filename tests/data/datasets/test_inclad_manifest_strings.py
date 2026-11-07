"""Regression tests for reading InCLAD manifests as strings.

The manifests are loaded through the real Hugging Face CSV builder on local
synthetic files (no network). Left to type inference, that builder parses the
BTech categories '01', '02', '03' as integers; the loaders must keep them as
the strings used by the folder readers, the ordering CSVs and the paper.
"""

import csv
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from datasets import load_dataset as hf_load_dataset
from PIL import Image

from pyclad.callbacks.evaluation.visual_pixel_utils import index_registered_visual_test_samples
from pyclad.data.datasets.inclad_bench_dataset import (
    INCLAD_BENCH_HF_REVISION,
    InCLADBenchDataset,
    load_inclad_bench,
)
from pyclad.data.datasets.inclad_md_dataset import (
    INCLAD_MD_HF_REVISION,
    INCLAD_MD_MANIFEST_FIELDNAMES,
    INCLAD_MD_SOURCE_DATASETS,
    InCLADMDDataset,
    load_inclad_md,
)
from pyclad.data.readers.visual_benchmark_manifest import VISUAL_BENCHMARK_MANIFEST_FIELDNAMES

# Category order of the btech easy_to_hard manifest.
BTECH_EASY_TO_HARD = ["03", "01", "02"]


def _write_image(path: Path, value: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8), mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(path)


def _btech_tree_and_rows(root: Path, categories=BTECH_EASY_TO_HARD) -> list[dict[str, str]]:
    """Create a BTech-layout tree and the matching manifest rows."""
    rows = []
    for order, category in enumerate(categories, start=1):
        files = [
            ("train", f"{category}/train/ok/0000.bmp", "", "0", ""),
            ("test", f"{category}/test/ok/0000.bmp", "", "0", ""),
            ("test", f"{category}/test/ko/0000.bmp", f"{category}/ground_truth/ko/0000.bmp", "1", "ko"),
        ]
        for split, image_relpath, mask_relpath, label, defect_type in files:
            _write_image(root / image_relpath)
            if mask_relpath:
                _write_mask(root / mask_relpath)
            rows.append(
                {
                    "sample_id": f"btech:{len(rows):06d}",
                    "source_dataset": "btech",
                    "category": category,
                    "category_order": str(order),
                    "split": split,
                    "image_relpath": image_relpath,
                    "mask_relpath": mask_relpath,
                    "image_label": label,
                    "defect_type": defect_type,
                    "ordering_name": "easy_to_hard",
                    "ordering_master_seed": "42",
                    "ordering_seed": "191664963",
                    "source_homepage": "https://github.com/pankajmishra000/VT-ADL",
                    "source_license": "CC BY-SA",
                }
            )
    return rows


def _write_csv(path: Path, fieldnames, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


class _LocalHub:
    """Stand-in for ``load_dataset(repo, config, ...)`` backed by a local CSV.

    It records the call and forwards ``features`` to the real CSV builder, so
    the test sees exactly the column types the loader asked for.
    """

    def __init__(self, csv_path: Path, cache_dir: Path):
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.calls: list[dict] = []

    def __call__(self, repo, config, split, cache_dir=None, revision=None, features=None, **kwargs):
        self.calls.append(
            {"repo": repo, "config": config, "split": split, "revision": revision, "features": features}
        )
        return hf_load_dataset(
            "csv",
            data_files={split: str(self.csv_path)},
            split=split,
            cache_dir=str(self.cache_dir),
            features=features,
        )


# --------------------------------------------------------------------------- #
# InCLADBenchDataset
# --------------------------------------------------------------------------- #


@pytest.fixture
def btech_hub(tmp_path: Path):
    root = tmp_path / "btech_root"
    rows = _btech_tree_and_rows(root)
    csv_path = _write_csv(
        tmp_path / "hub" / "btech_easy_to_hard_samples.csv", VISUAL_BENCHMARK_MANIFEST_FIELDNAMES, rows
    )
    return root, _LocalHub(csv_path, tmp_path / "hf_cache")


def test_btech_concepts_keep_leading_zeros(btech_hub):
    root, hub = btech_hub
    with patch("pyclad.data.datasets.inclad_bench_dataset.load_dataset", side_effect=hub):
        dataset = InCLADBenchDataset(benchmark="btech", ordering="easy_to_hard", root=root, data_mode="paths")

    assert [c.name for c in dataset.train_concepts()] == BTECH_EASY_TO_HARD
    assert [c.name for c in dataset.test_concepts()] == BTECH_EASY_TO_HARD
    samples = dataset.test_samples_by_concept()
    assert list(samples) == BTECH_EASY_TO_HARD
    for category in BTECH_EASY_TO_HARD:
        assert [s.image_label for s in samples[category]] == [0, 1]
        # Empty mask cells stay empty (no literal "nan" path).
        assert samples[category][0].mask_path is None
        assert samples[category][1].mask_path is not None
        assert "nan" not in str(samples[category][0].image_path)


def test_btech_concept_names_match_pixel_index(btech_hub):
    root, hub = btech_hub
    with patch("pyclad.data.datasets.inclad_bench_dataset.load_dataset", side_effect=hub):
        dataset = InCLADBenchDataset(benchmark="btech", root=root, data_mode="paths")

    pixel_index = index_registered_visual_test_samples(benchmark="btech", root=root)
    assert set(pixel_index) == set(dataset.test_samples_by_concept())


def test_btech_category_filter_accepts_zero_padded_name(btech_hub):
    root, hub = btech_hub
    with patch("pyclad.data.datasets.inclad_bench_dataset.load_dataset", side_effect=hub):
        dataset = InCLADBenchDataset(benchmark="btech", root=root, categories=["01"], data_mode="paths")

    assert [c.name for c in dataset.train_concepts()] == ["01"]


def test_bench_revision_defaults_to_none_and_is_forwarded(btech_hub):
    root, hub = btech_hub
    with patch("pyclad.data.datasets.inclad_bench_dataset.load_dataset", side_effect=hub):
        InCLADBenchDataset(benchmark="btech", root=root, data_mode="paths")
        load_inclad_bench(benchmark="btech", root=root, data_mode="paths", revision=INCLAD_BENCH_HF_REVISION)

    assert hub.calls[0]["revision"] is None
    assert hub.calls[1]["revision"] == INCLAD_BENCH_HF_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", INCLAD_BENCH_HF_REVISION)
    for call in hub.calls:
        assert call["features"] is not None
        assert all(feature.dtype == "string" for feature in call["features"].values())


# --------------------------------------------------------------------------- #
# InCLADMDDataset
# --------------------------------------------------------------------------- #


@pytest.fixture
def md_hub(tmp_path: Path):
    roots = {name: tmp_path / f"{name}_root" for name in INCLAD_MD_SOURCE_DATASETS}
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    # Two BTech concepts, then one MVTec concept, as in the easy_to_hard stream.
    for bench_row in _btech_tree_and_rows(roots["btech"], categories=["03", "01"]):
        rows.append(
            {
                **bench_row,
                "sample_id": f"inclad_md:{len(rows):06d}",
                "source_category": bench_row["category"],
                "category": f"btech__{bench_row['category']}",
            }
        )
    mvtec_root = roots["mvtec"]
    for split, image_relpath, mask_relpath, label, defect_type in [
        ("train", "bottle/train/good/000.png", "", "0", ""),
        ("test", "bottle/test/good/000.png", "", "0", ""),
        ("test", "bottle/test/broken_large/000.png", "bottle/ground_truth/broken_large/000_mask.png", "1", "broken_large"),
    ]:
        _write_image(mvtec_root / image_relpath)
        if mask_relpath:
            _write_mask(mvtec_root / mask_relpath)
        rows.append(
            {
                "sample_id": f"inclad_md:{len(rows):06d}",
                "source_dataset": "mvtec",
                "source_category": "bottle",
                "category": "mvtec__bottle",
                "category_order": "3",
                "split": split,
                "image_relpath": image_relpath,
                "mask_relpath": mask_relpath,
                "image_label": label,
                "defect_type": defect_type,
                "ordering_name": "easy_to_hard",
                "ordering_master_seed": "42",
                "ordering_seed": "191664963",
                "source_homepage": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
                "source_license": "CC BY-NC-SA 4.0",
            }
        )

    csv_path = _write_csv(tmp_path / "hub" / "inclad-md_easy_to_hard.csv", INCLAD_MD_MANIFEST_FIELDNAMES, rows)
    return roots, _LocalHub(csv_path, tmp_path / "hf_cache")


def test_md_forwards_test_samples_by_concept_with_masks(md_hub):
    roots, hub = md_hub
    with patch("pyclad.data.datasets.inclad_md_dataset.load_dataset", side_effect=hub):
        dataset = InCLADMDDataset(ordering="easy_to_hard", roots=roots, data_mode="paths")

    expected = ["btech__03", "btech__01", "mvtec__bottle"]
    assert [c.name for c in dataset.train_concepts()] == expected
    samples = dataset.test_samples_by_concept()
    assert samples, "InCLADMDDataset must forward its per-concept test samples"
    assert list(samples) == expected
    for category in expected:
        assert [s.image_label for s in samples[category]] == [0, 1]
        assert samples[category][0].mask_path is None
        assert samples[category][1].mask_path is not None
        assert samples[category][1].mask_path.exists()
    # Test-sample order matches the order of the test concept's data.
    test_concept = dataset.test_concepts()[0]
    assert [str(s.image_path) for s in samples[test_concept.name]] == [str(p) for p in test_concept.data]


def test_md_category_filter_and_revision(md_hub):
    roots, hub = md_hub
    with patch("pyclad.data.datasets.inclad_md_dataset.load_dataset", side_effect=hub):
        dataset = load_inclad_md(
            roots=roots, categories=["btech__01"], data_mode="paths", revision=INCLAD_MD_HF_REVISION
        )
        InCLADMDDataset(roots=roots, data_mode="paths")

    assert [c.name for c in dataset.train_concepts()] == ["btech__01"]
    assert list(dataset.test_samples_by_concept()) == ["btech__01"]
    assert hub.calls[0]["revision"] == INCLAD_MD_HF_REVISION
    assert hub.calls[1]["revision"] is None
    assert INCLAD_MD_HF_REVISION == INCLAD_BENCH_HF_REVISION
    for call in hub.calls:
        assert all(feature.dtype == "string" for feature in call["features"].values())
