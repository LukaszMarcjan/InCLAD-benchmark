import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.visual_benchmark_reader import (
    PREDEFINED_BENCHMARK_ALIASES,
    VisualBenchmarkReader,
    VisualBenchmarkSpec,
    build_visual_benchmark_reader,
    index_visual_benchmark,
    read_visual_benchmark_dataset,
)

DEFAULT_VISUAL_DATASET_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "visual_datasets" / "registry.json"
)

VISUAL_BENCHMARK_ENV_VARS = {
    "mvtec": "PYCLAD_MVTEC_ROOT",
    "visa": "PYCLAD_VISA_ROOT",
}

VISUAL_BENCHMARK_DIRECTORY_CANDIDATES = {
    "btech": ("BTech_Dataset_transformed", "btech_dataset_transformed", "btech"),
    "dagm": ("DAGM_KaggleUpload", "dagm_kaggleupload", "dagm"),
    "mpdd": ("MPDD", "mpdd"),
    "mvtec": ("mvtec_ad", "mvtec", "mvtec_anomaly_detection"),
    "visa": ("visa", "VisA"),
}

VISUAL_BENCHMARK_SHARED_ROOT_ENV = "PYCLAD_VISUAL_DATASETS_ROOT"
VISUAL_BENCHMARK_REGISTRY_ENV = "PYCLAD_VISUAL_DATASETS_FILE"


def resolve_visual_dataset_registry_path(registry_path: Optional[Union[str, Path]] = None) -> Path:
    if registry_path is not None:
        return Path(registry_path).expanduser().resolve()

    env_path = os.getenv(VISUAL_BENCHMARK_REGISTRY_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return DEFAULT_VISUAL_DATASET_REGISTRY_PATH


def load_visual_dataset_registry(registry_path: Optional[Union[str, Path]] = None) -> dict[str, str]:
    path = resolve_visual_dataset_registry_path(registry_path)
    if not path.exists():
        return {}

    content = json.loads(path.read_text())
    if not isinstance(content, dict):
        raise ValueError(f"Visual dataset registry must be a JSON object, got {type(content).__name__}")

    registry = {}
    for benchmark_name, root_path in content.items():
        if not isinstance(benchmark_name, str):
            raise ValueError("Visual dataset registry keys must be strings")
        if root_path in (None, ""):
            continue
        if not isinstance(root_path, str):
            raise ValueError(f"Visual dataset registry values must be strings, got {type(root_path).__name__}")
        registry[_normalize_visual_benchmark_name(benchmark_name)] = root_path
    return registry


def write_visual_dataset_registry(
    entries: Mapping[str, Union[str, Path]],
    registry_path: Optional[Union[str, Path]] = None,
) -> Path:
    path = resolve_visual_dataset_registry_path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = {}
    for benchmark_name, root_path in entries.items():
        serialized[_normalize_visual_benchmark_name(benchmark_name)] = str(Path(root_path).expanduser())

    path.write_text(json.dumps(serialized, indent=4, sort_keys=True))
    return path


def resolve_visual_benchmark_root(
    benchmark: Union[str, VisualBenchmarkSpec],
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()

    benchmark_name = _normalize_visual_benchmark_name(benchmark)
    registry = load_visual_dataset_registry(registry_path)
    if benchmark_name in registry:
        return Path(registry[benchmark_name]).expanduser().resolve()

    benchmark_env = VISUAL_BENCHMARK_ENV_VARS.get(benchmark_name)
    if benchmark_env:
        env_root = os.getenv(benchmark_env)
        if env_root:
            return Path(env_root).expanduser().resolve()

    shared_root = os.getenv(VISUAL_BENCHMARK_SHARED_ROOT_ENV)
    if shared_root:
        base_root = Path(shared_root).expanduser().resolve()
        for dirname in VISUAL_BENCHMARK_DIRECTORY_CANDIDATES.get(benchmark_name, (benchmark_name,)):
            candidate = base_root / dirname
            if candidate.exists():
                return candidate

    registry_file = resolve_visual_dataset_registry_path(registry_path)
    raise FileNotFoundError(
        f"Could not resolve local root for visual benchmark '{benchmark_name}'. "
        f"Pass root=..., set {VISUAL_BENCHMARK_ENV_VARS.get(benchmark_name, 'a benchmark-specific env var')}, "
        f"set {VISUAL_BENCHMARK_SHARED_ROOT_ENV}, or add an entry to {registry_file}."
    )


def build_registered_visual_benchmark_reader(
    benchmark: Union[str, VisualBenchmarkSpec],
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
) -> VisualBenchmarkReader:
    resolved_root = resolve_visual_benchmark_root(benchmark=benchmark, root=root, registry_path=registry_path)
    return build_visual_benchmark_reader(root=resolved_root, benchmark=benchmark)


def read_registered_visual_benchmark_dataset(
    benchmark: Union[str, VisualBenchmarkSpec],
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    dataset_name: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    data_mode: str = "numpy",
    resize_to: Optional[tuple[int, int]] = None,
    color_mode: str = "rgb",
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
) -> ConceptsDataset:
    resolved_root = resolve_visual_benchmark_root(benchmark=benchmark, root=root, registry_path=registry_path)
    return read_visual_benchmark_dataset(
        root=resolved_root,
        benchmark=benchmark,
        dataset_name=dataset_name,
        categories=categories,
        data_mode=data_mode,
        resize_to=resize_to,
        color_mode=color_mode,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def index_registered_visual_benchmark(
    benchmark: Union[str, VisualBenchmarkSpec],
    root: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    categories: Optional[Sequence[str]] = None,
    max_train_samples_per_category: Optional[int] = None,
    max_test_samples_per_category: Optional[int] = None,
):
    resolved_root = resolve_visual_benchmark_root(benchmark=benchmark, root=root, registry_path=registry_path)
    return index_visual_benchmark(
        root=resolved_root,
        benchmark=benchmark,
        categories=categories,
        max_train_samples_per_category=max_train_samples_per_category,
        max_test_samples_per_category=max_test_samples_per_category,
    )


def _normalize_visual_benchmark_name(benchmark: Union[str, VisualBenchmarkSpec]) -> str:
    if isinstance(benchmark, str):
        key = benchmark.lower()
        return PREDEFINED_BENCHMARK_ALIASES.get(key, key)
    return PREDEFINED_BENCHMARK_ALIASES.get(benchmark.name.lower(), benchmark.name.lower())
