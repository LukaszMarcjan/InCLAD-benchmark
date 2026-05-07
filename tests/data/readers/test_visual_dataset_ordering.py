from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyclad.data.readers.visual_dataset_ordering import (
    infer_visual_benchmark_from_ordering_path,
    load_concept_order_from_csv,
    read_ordered_registered_visual_benchmark_dataset,
)
from pyclad.data.readers.visual_dataset_registry import write_visual_dataset_registry


def _write_rgb_image(path: Path, color: tuple[int, int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((6, 5, 3), dtype=np.uint8)
    array[..., 0] = color[0]
    array[..., 1] = color[1]
    array[..., 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


@pytest.mark.parametrize(
    ("ordering_path", "expected_benchmark"),
    [
        ("examples/visual_models/ordering_v2/btech_easy_to_hard_roc-auc.csv", "btech"),
        ("examples/visual_models/ordering_v2/mvtec_easy_to_hard_roc-auc.csv", "mvtec"),
    ],
)
def test_infer_visual_benchmark_from_ordering_path(ordering_path: str, expected_benchmark: str):
    assert infer_visual_benchmark_from_ordering_path(ordering_path) == expected_benchmark


def test_load_concept_order_from_csv_reads_concept_column():
    ordering = load_concept_order_from_csv("examples/visual_models/ordering_v2/btech_easy_to_hard_roc-auc.csv")

    assert ordering == ["03", "01", "02"]


def test_read_ordered_registered_visual_benchmark_dataset_uses_ordering_file(tmp_path: Path):
    root = tmp_path / "btech_like"

    for category, base in (("01", 10), ("02", 40), ("03", 70)):
        _write_rgb_image(root / category / "train" / "ok" / "000.bmp", (base, base + 1, base + 2))
        _write_rgb_image(root / category / "test" / "ok" / "001.bmp", (base + 3, base + 4, base + 5))
        _write_rgb_image(root / category / "test" / "ko" / "002.bmp", (base + 6, base + 7, base + 8))
        _write_rgb_image(root / category / "ground_truth" / "ko" / "002.png", (255, 255, 255))

    registry_path = write_visual_dataset_registry({"btech": root}, registry_path=tmp_path / "registry.json")

    dataset = read_ordered_registered_visual_benchmark_dataset(
        ordering_path="examples/visual_models/ordering_v2/btech_easy_to_hard_roc-auc.csv",
        registry_path=registry_path,
        resize_to=(4, 4),
    )

    assert dataset.name() == "btech_easy_to_hard_roc-auc"
    assert [concept.name for concept in dataset.train_concepts()] == ["03", "01", "02"]
    assert [concept.name for concept in dataset.test_concepts()] == ["03", "01", "02"]
