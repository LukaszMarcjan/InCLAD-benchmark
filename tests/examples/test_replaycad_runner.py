import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.visual_pixel_concept_metric_callback import VisualPixelConceptMetricCallback
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario

RUNNER_PATH = pathlib.Path(__file__).resolve().parents[2] / "examples" / "clvad" / "run_continual_visual_ad.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("_replaycad_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse(runner, argv):
    with patch("sys.argv", ["run_continual_visual_ad.py", *argv]):
        return runner.parse_args()


def test_existing_strategy_defaults_are_unchanged():
    runner = load_runner()
    args = parse(runner, ["--model", "patchcore", "--benchmark", "mvtec"])

    assert args.strategy == "replay"
    assert runner.normalize_replaycad_args(args) is args
    assert args.backbone is None
    assert args.resize is None
    assert args.eval_level == "image"


def test_replaycad_normalization_is_narrow_and_additive(tmp_path):
    runner = load_runner()
    args = parse(
        runner,
        [
            "--model",
            "patchcore",
            "--strategy",
            "replaycad",
            "--benchmark",
            "mvtec",
            "--inclad-bench",
            "--replaycad-mask-backend",
            "full-frame",
            "--output-dir",
            str(tmp_path),
        ],
    )
    normalized = runner.normalize_replaycad_args(args)

    assert normalized.ordering_mode == "easy_to_hard"
    assert normalized.backbone == "wide_resnet50_2"
    assert normalized.pretrained is True
    assert normalized.resize == 256


def test_replaycad_accepts_explicit_resnet18(tmp_path):
    runner = load_runner()
    args = parse(
        runner,
        [
            "--model",
            "patchcore",
            "--strategy",
            "replaycad",
            "--benchmark",
            "mvtec",
            "--inclad-bench",
            "--backbone",
            "resnet18",
            "--replaycad-mask-backend",
            "full-frame",
            "--output-dir",
            str(tmp_path),
        ],
    )

    normalized = runner.normalize_replaycad_args(args)

    assert normalized.backbone == "resnet18"
    assert normalized.pretrained is True


def test_replaycad_preserves_inclad_random_ordering(tmp_path):
    runner = load_runner()
    args = parse(
        runner,
        [
            "--model",
            "patchcore",
            "--strategy",
            "replaycad",
            "--benchmark",
            "mvtec",
            "--inclad-bench",
            "--ordering-mode",
            "random",
            "--replaycad-mask-backend",
            "full-frame",
            "--output-dir",
            str(tmp_path),
        ],
    )

    normalized = runner.normalize_replaycad_args(args)

    assert normalized.ordering_mode == "random"


def test_replaycad_accepts_inclad_hard_to_easy_ordering(tmp_path):
    runner = load_runner()
    args = parse(
        runner,
        [
            "--model",
            "patchcore",
            "--strategy",
            "replaycad",
            "--benchmark",
            "mvtec",
            "--inclad-bench",
            "--ordering-mode",
            "hard_to_easy",
            "--replaycad-mask-backend",
            "full-frame",
            "--output-dir",
            str(tmp_path),
        ],
    )

    normalized = runner.normalize_replaycad_args(args)

    assert normalized.ordering_mode == "hard_to_easy"


def test_replaycad_accepts_remaining_inclad_benchmark(tmp_path):
    runner = load_runner()
    args = parse(
        runner,
        [
            "--model",
            "patchcore",
            "--strategy",
            "replaycad",
            "--benchmark",
            "mpdd",
            "--inclad-bench",
            "--ordering-mode",
            "random",
            "--replaycad-mask-backend",
            "full-frame",
            "--output-dir",
            str(tmp_path),
        ],
    )

    normalized = runner.normalize_replaycad_args(args)
    config = runner.build_replaycad_config(
        normalized,
        seed=11,
        output_path=tmp_path / "mpdd.json",
    )

    assert config.benchmark == "mpdd"
    assert config.preset == "inclad-ldm-256"


def test_replaycad_builds_balanced_runtime_overrides(tmp_path):
    runner = load_runner()
    args = parse(
        runner,
        [
            "--model",
            "patchcore",
            "--strategy",
            "replaycad",
            "--benchmark",
            "mvtec",
            "--inclad-bench",
            "--replaycad-mask-backend",
            "full-frame",
            "--replaycad-compression-steps",
            "15000",
            "--replaycad-inference-steps",
            "30",
            "--replaycad-samples-per-concept",
            "600",
            "--output-dir",
            str(tmp_path),
        ],
    )
    args = runner.normalize_replaycad_args(args)

    config = runner.build_replaycad_config(
        args,
        seed=7,
        output_path=tmp_path / "balanced.json",
    )

    assert config.compression_steps == 15000
    assert config.inference_steps == 30
    assert config.replay_samples_per_concept == 600


def test_build_callbacks_can_enable_image_and_pixel_levels():
    runner = load_runner()
    dataset = ConceptsDataset(
        name="empty-visual",
        train_concepts=[],
        test_concepts=[],
        test_samples_by_concept={},
    )
    args = SimpleNamespace(
        eval_level="both",
        benchmark="mvtec",
        root="/unused",
        registry_path=None,
        categories=None,
        max_test=None,
        pixel_threshold=0.5,
        pixel_threshold_mode="train-quantile",
        pixel_threshold_quantile=0.995,
        pixel_skip_missing_masks=True,
    )

    callbacks = runner.build_callbacks(dataset, strategy=object(), args=args)

    assert sum(isinstance(callback, ConceptMetricCallback) for callback in callbacks) == 3
    assert sum(isinstance(callback, VisualPixelConceptMetricCallback) for callback in callbacks) == 1


def test_identity_aware_training_helpers_use_concept_aware_scenario():
    runner = load_runner()
    dataset = ConceptsDataset(name="empty", train_concepts=[], test_concepts=[])
    callbacks: list[Callback] = []
    strategy = object()

    replaycad = runner.build_scenario("replaycad", dataset, strategy, callbacks)
    ucad = runner.build_scenario("ucad", dataset, strategy, callbacks)
    existing = runner.build_scenario("replay", dataset, strategy, callbacks)

    assert isinstance(replaycad, ConceptAwareScenario)
    assert isinstance(ucad, ConceptAwareScenario)
    assert isinstance(existing, ConceptIncrementalScenario)
