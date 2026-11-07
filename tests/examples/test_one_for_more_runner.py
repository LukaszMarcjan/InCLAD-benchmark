import importlib.util
import pathlib
import sys
from unittest.mock import patch


RUNNER_PATH = pathlib.Path(__file__).resolve().parents[2] / "examples" / "clvad" / "run_continual_visual_ad.py"
LEVELS_RUNNER_PATH = RUNNER_PATH.with_name("run_continual_visual_ad_levels.py")
BATCH_RUNNER_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples"
    / "visual_models"
    / "run_all_experiments.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("_one_for_more_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_batch_runner():
    spec = importlib.util.spec_from_file_location("_one_for_more_batch_runner", BATCH_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(BATCH_RUNNER_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BATCH_RUNNER_PATH.parent))
    return module


def _load_levels_runner():
    spec = importlib.util.spec_from_file_location("_one_for_more_levels_runner", LEVELS_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_one_for_more_is_additive_and_cli_builds_its_config():
    runner = _load_runner()
    with patch(
        "sys.argv",
        [
            "run_continual_visual_ad.py",
            "--model",
            "one_for_more",
            "--strategy",
            "naive",
            "--benchmark",
            "mvtec",
            "--epochs",
            "3",
            "--incremental-epochs",
            "2",
            "--no-pretrained",
            "--config-json",
            '{"base_channels": 16, "conditioner_heads": 2, "backbone_name": null}',
        ],
    ):
        args = runner.parse_args()

    config = runner.build_model_config(args, seed=7)
    assert args.strategy == "naive"
    assert config.epochs == 3
    assert config.incremental_epochs == 2
    assert config.base_channels == 16
    assert config.backbone_name is None
    assert config.random_seed == 7


def test_existing_cli_defaults_and_model_specs_remain_available():
    runner = _load_runner()
    with patch("sys.argv", ["run_continual_visual_ad.py", "--model", "patchcore"]):
        args = runner.parse_args()

    assert args.strategy == "replay"
    assert args.incremental_epochs is None
    for model_name in (
        "cfa",
        "draem",
        "efficientad",
        "fastflow",
        "ganomaly",
        "padim",
        "paste",
        "patchcore",
        "rd4ad",
        "stfpm",
        # "unet",  # pyclad.models.unet is not part of this release
    ):
        assert model_name in runner.MODEL_SPECS


def test_batch_runner_support_is_opt_in_to_preserve_old_default_matrix(tmp_path):
    runner = _load_batch_runner()
    with patch("sys.argv", ["run_all_experiments.py", "--benchmarks", "mvtec"]):
        default_args = runner.parse_args()

    assert "one_for_more" in runner.SUPPORTED_MODELS
    assert "one_for_more" not in default_args.models

    with patch(
        "sys.argv",
        [
            "run_all_experiments.py",
            "--models",
            "one_for_more",
            "--strategies",
            "naive",
            "--benchmarks",
            "mvtec",
            "--incremental-epochs",
            "2",
        ],
    ):
        one_for_more_args = runner.parse_args()
    one_for_more_args.output_root = str(tmp_path)
    command = runner.build_command(
        "one_for_more",
        "naive",
        "mvtec",
        tmp_path / "mvtec",
        one_for_more_args,
    )

    assert one_for_more_args.models == ["one_for_more"]
    assert command[command.index("--incremental-epochs") + 1] == "2"


def test_pixel_level_batch_runner_accepts_one_for_more():
    runner = _load_levels_runner()

    assert "one_for_more" in runner.PIXEL_LEVEL_MODELS
