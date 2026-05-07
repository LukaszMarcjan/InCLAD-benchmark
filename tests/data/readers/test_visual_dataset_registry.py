import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyclad.data.readers.visual_dataset_registry import (
    VISUAL_BENCHMARK_REGISTRY_ENV,
    VISUAL_BENCHMARK_SHARED_ROOT_ENV,
    load_visual_dataset_registry,
    read_registered_visual_benchmark_dataset,
    resolve_visual_benchmark_root,
    write_visual_dataset_registry,
)


def _write_rgb_image(path: Path, color: tuple[int, int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((6, 5, 3), dtype=np.uint8)
    array[..., 0] = color[0]
    array[..., 1] = color[1]
    array[..., 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, value: int = 255):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((6, 5), value, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def test_write_and_load_visual_dataset_registry_round_trip(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    entries = {
        "mvtec_ad": tmp_path / "datasets" / "mvtec_ad",
        "visa": tmp_path / "datasets" / "visa",
    }

    written_path = write_visual_dataset_registry(entries=entries, registry_path=registry_path)
    registry = load_visual_dataset_registry(registry_path=registry_path)

    assert written_path == registry_path.resolve()
    assert registry == {
        "mvtec": str((tmp_path / "datasets" / "mvtec_ad").expanduser()),
        "visa": str((tmp_path / "datasets" / "visa").expanduser()),
    }


def test_load_visual_dataset_registry_skips_empty_entries(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"mvtec": "", "visa": str(tmp_path / "visa")}))

    registry = load_visual_dataset_registry(registry_path=registry_path)

    assert registry == {"visa": str(tmp_path / "visa")}


def test_resolve_visual_benchmark_root_uses_registry_file(tmp_path: Path):
    mvtec_root = tmp_path / "mvtec_ad"
    registry_path = write_visual_dataset_registry({"mvtec": mvtec_root}, registry_path=tmp_path / "registry.json")

    resolved = resolve_visual_benchmark_root("mvtec_ad", registry_path=registry_path)

    assert resolved == mvtec_root.resolve()


def test_resolve_visual_benchmark_root_uses_benchmark_specific_env(monkeypatch, tmp_path: Path):
    visa_root = tmp_path / "visa"
    monkeypatch.setenv("PYCLAD_VISA_ROOT", str(visa_root))

    resolved = resolve_visual_benchmark_root("visa")

    assert resolved == visa_root.resolve()


def test_resolve_visual_benchmark_root_uses_shared_root_candidates(monkeypatch, tmp_path: Path):
    mvtec_root = tmp_path / "datasets" / "mvtec_ad"
    mvtec_root.mkdir(parents=True)
    monkeypatch.setenv(VISUAL_BENCHMARK_SHARED_ROOT_ENV, str(tmp_path / "datasets"))

    resolved = resolve_visual_benchmark_root("mvtec")

    assert resolved == mvtec_root.resolve()


def test_resolve_visual_benchmark_root_prefers_explicit_root(tmp_path: Path):
    explicit_root = tmp_path / "custom_mvtec"

    resolved = resolve_visual_benchmark_root("mvtec", root=explicit_root)

    assert resolved == explicit_root.resolve()


def test_resolve_visual_benchmark_root_raises_clear_error_when_missing(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("PYCLAD_MVTEC_ROOT", raising=False)
    monkeypatch.delenv(VISUAL_BENCHMARK_SHARED_ROOT_ENV, raising=False)
    monkeypatch.setenv(VISUAL_BENCHMARK_REGISTRY_ENV, str(tmp_path / "missing.json"))

    with pytest.raises(FileNotFoundError, match="Could not resolve local root"):
        resolve_visual_benchmark_root("mvtec")


def test_read_registered_visual_benchmark_dataset_uses_registry_file_for_mvtec(tmp_path: Path):
    root = tmp_path / "mvtec_like"
    _write_rgb_image(root / "widget" / "train" / "good" / "000.png", (10, 20, 30))
    _write_rgb_image(root / "widget" / "test" / "good" / "100.png", (20, 30, 40))
    _write_rgb_image(root / "widget" / "test" / "crack" / "101.png", (30, 40, 50))
    _write_mask(root / "widget" / "ground_truth" / "crack" / "101_mask.png")
    registry_path = write_visual_dataset_registry({"mvtec": root}, registry_path=tmp_path / "registry.json")

    dataset = read_registered_visual_benchmark_dataset(
        benchmark="mvtec",
        registry_path=registry_path,
        resize_to=(8, 8),
    )

    assert len(dataset.train_concepts()) == 1
    assert len(dataset.test_concepts()) == 1
    assert dataset.train_concepts()[0].data.shape == (1, 8, 8, 3)
    assert dataset.test_concepts()[0].data.shape == (2, 8, 8, 3)
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([1, 0]))
