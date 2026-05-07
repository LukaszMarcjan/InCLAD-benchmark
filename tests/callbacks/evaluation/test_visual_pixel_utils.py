import base64
import io
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from pyclad.callbacks.evaluation.visual_pixel_utils import (
    index_registered_visual_test_samples,
    load_ground_truth_mask,
    load_ground_truth_masks_for_samples,
)
from pyclad.data.readers.visual_benchmark_reader import VisualSample


def _write_rgb_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (6, 5)):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    array[..., 0] = color[0]
    array[..., 1] = color[1]
    array[..., 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _write_supervisely_bitmap_annotation(path: Path, *, canvas_shape: tuple[int, int], mask: np.ndarray, origin: tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)

    fragment = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    fragment.save(buffer, format="PNG")
    encoded = base64.b64encode(zlib.compress(buffer.getvalue())).decode("ascii")

    payload = {
        "description": "",
        "tags": [],
        "size": {"height": canvas_shape[0], "width": canvas_shape[1]},
        "objects": [
            {
                "geometryType": "bitmap",
                "classTitle": "defect",
                "bitmap": {"data": encoded, "origin": [origin[0], origin[1]]},
            }
        ],
    }
    path.write_text(json.dumps(payload))


def test_load_ground_truth_mask_returns_zero_mask_for_normal_samples(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    _write_rgb_image(image_path, (10, 20, 30), size=(7, 4))

    sample = VisualSample(category="widget", split="test", image_path=image_path, image_label=0, mask_path=None)

    mask = load_ground_truth_mask(sample)

    assert mask.shape == (7, 4)
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) == 0


def test_load_ground_truth_mask_resizes_bitmap_masks_with_nearest_neighbor(tmp_path: Path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    _write_rgb_image(image_path, (10, 20, 30), size=(4, 4))

    mask_array = np.zeros((4, 4), dtype=np.uint8)
    mask_array[:2, :2] = 255
    _write_mask(mask_path, mask_array)

    sample = VisualSample(category="widget", split="test", image_path=image_path, image_label=1, mask_path=mask_path)

    resized = load_ground_truth_mask(sample, resize_to=(2, 2))

    assert resized.shape == (2, 2)
    assert np.array_equal(resized, np.array([[1, 0], [0, 0]], dtype=np.uint8))


def test_load_ground_truth_mask_decodes_ksdd2_style_bitmap_annotations(tmp_path: Path):
    image_path = tmp_path / "image.png"
    annotation_path = tmp_path / "mask.json"
    _write_rgb_image(image_path, (10, 20, 30), size=(6, 5))

    fragment = np.array(
        [
            [0, 255],
            [255, 255],
        ],
        dtype=np.uint8,
    )
    _write_supervisely_bitmap_annotation(
        annotation_path,
        canvas_shape=(6, 5),
        mask=fragment,
        origin=(2, 3),
    )

    sample = VisualSample(category="ksdd2", split="test", image_path=image_path, image_label=1, mask_path=annotation_path)

    mask = load_ground_truth_mask(sample)

    expected = np.zeros((6, 5), dtype=np.uint8)
    expected[3:5, 2:4] = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    assert np.array_equal(mask, expected)


def test_load_ground_truth_masks_for_samples_skips_anomalies_without_masks(tmp_path: Path):
    image_path = tmp_path / "image.png"
    _write_rgb_image(image_path, (10, 20, 30))

    samples = [
        VisualSample(category="widget", split="test", image_path=image_path, image_label=0, mask_path=None),
        VisualSample(category="widget", split="test", image_path=image_path, image_label=1, mask_path=None),
    ]

    masks, indices = load_ground_truth_masks_for_samples(samples, resize_to=(6, 5), skip_missing_anomaly_masks=True)

    assert masks.shape == (1, 6, 5)
    assert np.array_equal(indices, np.array([0]))


def test_index_registered_visual_test_samples_keeps_reader_order(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png", (10, 20, 30))
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png", (20, 30, 40))
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png", (30, 40, 50))
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png", np.full((6, 5), 255, dtype=np.uint8))

    grouped = index_registered_visual_test_samples(benchmark="mvtec", root=root)

    assert list(grouped) == ["widget"]
    assert [sample.image_label for sample in grouped["widget"]] == [1, 0]
    assert grouped["widget"][0].mask_path == root / "widget" / "ground_truth" / "crack" / "101_mask.png"
