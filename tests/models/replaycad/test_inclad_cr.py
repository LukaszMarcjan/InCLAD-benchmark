import importlib.util
import pathlib
import sys
import typing
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import pyclad.strategies.replaycad as replaycad_package
from pyclad.callbacks.evaluation.visual_pixel_concept_metric_callback import VisualPixelConceptMetricCallback
from pyclad.models.model import Model
from pyclad.strategies.replaycad import config as config_module
from pyclad.strategies.replaycad import inclad_cr
from pyclad.strategies.replaycad.artifacts import ReplayCADArtifact
from pyclad.strategies.replaycad.backend import DiffusersReplayCADBackend, ReplayCADBackend
from pyclad.strategies.replaycad.inclad_cr import (
    PROFILE_PRESET,
    InCLADCRDiffusersBackend,
    InCLADCRReplayCADConfig,
    InCLADCRReplayCADMemory,
    InCLADCRSAMMaskGenerator,
)
from pyclad.strategies.replaycad.strategy import ReplayCADStrategy

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE_RUNNER_PATH = REPO_ROOT / "examples" / "clvad" / "run_continual_visual_ad.py"
CR_RUNNER_PATH = REPO_ROOT / "examples" / "clvad" / "run_replaycad_cr.py"
TINY_SD = "diffusers/tiny-stable-diffusion-torch"


def load_script(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecordingModel(Model):
    def __init__(self):
        self.fit_batches = []

    def fit(self, data: np.ndarray):
        self.fit_batches.append(np.asarray(data).copy())

    def predict(self, data: np.ndarray):
        return np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=np.float32)

    def name(self) -> str:
        return "RecordingModel"


class ColourBackend(ReplayCADBackend):
    """Writes a dummy condition and generates pure-red replay."""

    def compress(self, concept_id, images, masks, artifact: ReplayCADArtifact):
        artifact.condition_path.write_bytes(f"condition:{concept_id}".encode())
        return {"compression_seconds": 0.0}

    def generate(self, artifact: ReplayCADArtifact, count: int, seed: int):
        replay = np.zeros((count, 4, 4, 3), dtype=np.uint8)
        replay[..., 0] = 200
        return replay


class BankOnlyBackend(ReplayCADBackend):
    """Fails on any diffusion call: proves that a later ordering only reads the bank."""

    def compress(self, concept_id, images, masks, artifact: ReplayCADArtifact):
        raise AssertionError(f"'{concept_id}' should have been reused from the bank")

    def generate(self, artifact: ReplayCADArtifact, count: int, seed: int):
        raise AssertionError(f"replay of '{artifact.concept_id}' should come from the bank cache")


def tiny_config(tmp_path: pathlib.Path, **updates) -> InCLADCRReplayCADConfig:
    payload = dict(
        benchmark="visa",
        preset=PROFILE_PRESET,
        artifact_dir=tmp_path / "bank" / "visa",
        diffusion_family="stable-diffusion",
        model_id=TINY_SD,
        resolution=32,
        condition_dim=32,
        mask_group_width=16,
        mask_projection_width=32,
        semantic_tokens=2,
        compression_steps=4,
        compression_batch_size=2,
        semantic_learning_rate=8e-2,
        spatial_learning_rate=1e-2,
        generation_batch_size=2,
        replay_samples_per_concept=2,
        masks_per_concept=1,
        inference_steps=2,
        initializer_word="object",
        mask_backend="full-frame",
        augment_masks=False,
        device="cpu",
        local_files_only=True,
        cache_generated=True,
    )
    payload.update(updates)
    return InCLADCRReplayCADConfig(**payload)


def tiny_backend(config: InCLADCRReplayCADConfig) -> InCLADCRDiffusersBackend:
    """Backend over a randomly initialised Stable Diffusion pipeline; no weights are downloaded."""
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.CLIPTokenizer.from_pretrained(TINY_SD, subfolder="tokenizer", local_files_only=True)
    except OSError as error:
        pytest.skip(f"the {TINY_SD} tokenizer is not cached locally: {error}")

    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(
        sample_size=4,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=config.condition_dim,
        attention_head_dim=8,
    )
    # Four blocks give the usual factor-8 VAE, so latents match the config's res^2/16 values.
    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=config.resolution,
    )
    text_encoder = transformers.CLIPTextModel(
        transformers.CLIPTextConfig(
            vocab_size=len(tokenizer),
            hidden_size=config.condition_dim,
            intermediate_size=37,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=77,
        )
    )
    scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
    pipeline = diffusers.StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    for module in (text_encoder, unet, vae):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    backend = InCLADCRDiffusersBackend(config)
    backend._pipeline = pipeline
    backend._text_encoder = text_encoder
    backend._tokenizer = tokenizer
    backend._unet = unet
    backend._vae = vae
    backend._scheduler = scheduler
    backend._torch = torch
    backend._dtype = torch.float32
    return backend


def test_profile_preset_literal_matches_the_constant():
    assert typing.get_args(InCLADCRReplayCADConfig.model_fields["preset"].annotation) == (PROFILE_PRESET,)


@pytest.mark.parametrize("benchmark", inclad_cr.INCLAD_CR_BENCHMARKS)
def test_profile_is_one_ldm_256_protocol_for_every_benchmark(tmp_path, benchmark):
    config = InCLADCRReplayCADConfig.paper_preset(
        benchmark=benchmark,
        artifact_dir=tmp_path / benchmark,
        mask_backend="full-frame",
    )

    assert (config.model_id, config.resolution, config.condition_dim) == ("CompVis/ldm-text2im-large-256", 256, 1280)
    assert config.spatial_tokens == 5
    assert (config.semantic_learning_rate, config.spatial_learning_rate) == (8e-2, 8e-4)
    assert (config.compression_steps, config.compression_batch_size) == (6_000, 16)
    assert (config.inference_steps, config.guidance_scale) == (50, 10.0)
    assert config.replay_samples_per_concept == 200
    assert (config.initializer_word, config.prompt_prefix) == ("screw", "a photo of")
    assert config.augment_masks is False
    assert (config.torch_dtype, config.autocast_dtype, config.cache_generated) == ("float32", "bfloat16", True)
    assert config.weight_decay == pytest.approx(0.01)
    assert config.default_mask_mode == ("full-frame" if benchmark == "dagm" else "corner-background")
    expected_full_frame = {
        "mvtec": {"carpet", "grid", "leather", "tile", "wood"},
        "btech": {"2"},
        "inclad-md": set(inclad_cr.MD_FULL_FRAME_CONCEPTS),
    }.get(benchmark, set())
    assert set(config.mask_modes) == expected_full_frame


def test_profile_refuses_bfloat16_conditions_and_a_disabled_cache(tmp_path):
    payload = InCLADCRReplayCADConfig.paper_preset(
        benchmark="mvtec", artifact_dir=tmp_path, mask_backend="full-frame"
    ).model_dump(mode="python")

    with pytest.raises(ValueError, match="float32"):
        InCLADCRReplayCADConfig.model_validate({**payload, "torch_dtype": "bfloat16"})
    with pytest.raises(ValueError, match="cache_generated"):
        InCLADCRReplayCADConfig.model_validate({**payload, "cache_generated": False})
    with pytest.raises(ValueError):
        InCLADCRReplayCADConfig.paper_preset(benchmark="kolektor", artifact_dir=tmp_path, mask_backend="full-frame")


def test_install_routes_the_unchanged_base_runner_to_the_profile(tmp_path):
    base_runner = load_script("_inclad_cr_base_runner", BASE_RUNNER_PATH)
    cr_runner = load_script("_inclad_cr_runner", CR_RUNNER_PATH)
    originals = (
        config_module.ReplayCADConfig,
        replaycad_package.ReplayCADMemory,
        replaycad_package.ReplayCADStrategy,
        VisualPixelConceptMetricCallback.__dict__["_resolve_threshold"],
    )
    argv, bank_dir = cr_runner.build_base_argv(
        [
            "--model",
            "patchcore",
            "--strategy",
            "replaycad",
            "--benchmark",
            "dagm",
            "--inclad-bench",
            "--replaycad-mask-backend",
            "full-frame",
            "--output-dir",
            str(tmp_path / "out"),
            "--replaycad-bank-dir",
            str(tmp_path / "bank"),
        ]
    )
    uninstall = inclad_cr.install(bank_dir=bank_dir)
    try:
        with patch("sys.argv", ["run_continual_visual_ad.py", *argv]):
            args = base_runner.normalize_replaycad_args(base_runner.parse_args())
        config = base_runner.build_replaycad_config(args, seed=7, output_path=tmp_path / "out" / "run.json")

        assert isinstance(config, InCLADCRReplayCADConfig)
        assert config.artifact_dir == tmp_path / "bank" / "dagm"
        assert (config.replay_samples_per_concept, config.cache_generated, config.seed) == (200, True, 7)
        assert args.config_json == '{"threshold": 0.0}'
        assert replaycad_package.ReplayCADMemory is InCLADCRReplayCADMemory
        assert replaycad_package.ReplayCADStrategy is ReplayCADStrategy
        assert VisualPixelConceptMetricCallback.__dict__["_resolve_threshold"] is not originals[3]
    finally:
        uninstall()
    restored = (
        config_module.ReplayCADConfig,
        replaycad_package.ReplayCADMemory,
        replaycad_package.ReplayCADStrategy,
        VisualPixelConceptMetricCallback.__dict__["_resolve_threshold"],
    )
    assert restored == originals
    assert inclad_cr.current_settings().bank_dir is None


def test_fixed_pixel_threshold_skips_the_training_scan():
    def scanning_resolver(self, model, cache_key, train_concepts):
        return "scanned"

    resolve = inclad_cr._resolve_fixed_threshold_without_training_scan(scanning_resolver)
    fixed = SimpleNamespace(_threshold_mode="fixed", _fixed_threshold=0.5, _resolved_thresholds={})
    assert resolve(fixed, model=None, cache_key="bottle", train_concepts=[object()]) == 0.5
    assert fixed._resolved_thresholds == {"bottle": 0.5}
    quantile = SimpleNamespace(_threshold_mode="train-quantile", _fixed_threshold=0.5, _resolved_thresholds={})
    assert resolve(quantile, model=None, cache_key="bottle", train_concepts=[]) == "scanned"


def test_wrapper_keeps_explicit_options_and_refuses_protocol_breaks(tmp_path):
    cr_runner = load_script("_inclad_cr_runner_options", CR_RUNNER_PATH)
    base = ["--model", "patchcore", "--strategy", "replaycad", "--benchmark", "mvtec"]

    argv, _ = cr_runner.build_base_argv([*base, "--replaycad-samples-per-concept=50", "--no-replaycad-cache-generated"])
    assert argv.count("--replaycad-samples-per-concept=50") == 1
    assert "--replaycad-cache-generated" not in argv

    # Defaults go first, so an abbreviated user option given later still wins in argparse.
    argv, _ = cr_runner.build_base_argv([*base, "--replaycad-samples", "50"])
    assert argv.index("--replaycad-samples-per-concept") < argv.index("--replaycad-samples")

    for refused in (
        ["--model", "patchcore", "--strategy", "replay", "--benchmark", "mvtec"],
        [*base, "--pixel-threshold-mode", "train-quantile"],
        [*base, "--pixel-threshold-m=train-quantile"],
        [*base, "--replaycad-bank-dir", str(tmp_path), "--replaycad-artifact-dir", str(tmp_path)],
    ):
        with pytest.raises(SystemExit):
            cr_runner.build_base_argv(refused)


def test_token_names_are_lower_case_and_unknown_tokens_fail(tmp_path, monkeypatch):
    backend = InCLADCRDiffusersBackend(tiny_config(tmp_path, semantic_tokens=20))
    names = backend._semantic_token_names("Class8")
    assert names[0] == "<rc-class8-00>" and names[-1] == "<rc-class8-19>"

    class Tokenizer:
        unk_token_id = 100

    backend._tokenizer = Tokenizer()
    monkeypatch.setattr(DiffusersReplayCADBackend, "_install_tokens", lambda self, token_names, initialize: [100, 5])
    with pytest.raises(ValueError, match="unknown token"):
        backend._install_tokens(names[:2], initialize=False)
    monkeypatch.setattr(DiffusersReplayCADBackend, "_install_tokens", lambda self, token_names, initialize: [5, 5])
    with pytest.raises(ValueError, match="distinct"):
        backend._install_tokens(names[:2], initialize=False)


def test_gradient_drift_is_zero_without_learning():
    initial = torch.linspace(-1.0, 1.0, 50)
    decayed_only = initial * 0.9**10
    assert inclad_cr._relative_drift(initial, decayed_only, 0.9**10) == pytest.approx(0.0, abs=1e-6)
    assert inclad_cr._relative_drift(initial, initial.clone(), 0.9**10) == 0.0
    assert inclad_cr._relative_drift(initial, initial + 0.5, 0.9**10) > 0.1


def test_compression_trains_float32_conditions_and_fills_a_bank_every_ordering_reuses(tmp_path):
    config = tiny_config(tmp_path)
    backend = tiny_backend(config)
    rng = np.random.default_rng(0)
    first = rng.integers(0, 255, size=(3, 32, 32, 3), dtype=np.uint8)
    second = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)

    memory = InCLADCRReplayCADMemory(config, backend=backend)
    strategy = ReplayCADStrategy(RecordingModel(), memory)
    strategy.learn(first, concept_id="pcb1")
    strategy.learn(second, concept_id="pcb2")

    metadata = memory.artifacts[0].read_metadata()
    assert metadata["profile"] == PROFILE_PRESET
    assert metadata["spatial_gradient_drift"] > 0.05
    assert metadata["semantic_gradient_drift"] > 0.05
    condition = torch.load(memory.artifacts[0].condition_path, weights_only=True)
    assert condition["token_names"] == ["<rc-pcb1-00>", "<rc-pcb1-01>"]
    assert condition["projection_state"]["0.weight"].dtype == torch.float32

    # The last concept of this ordering already has its replay cached, so the bank is complete.
    cached = sorted(path.parent.parent.name for path in (tmp_path / "bank" / "visa").rglob("generated/*.npz"))
    assert cached == ["pcb1", "pcb2"]
    assert not list((tmp_path / "bank" / "visa").rglob("*.tmp.npz"))

    # A fresh process (restarted job) must install the stored tokens before generating.
    fresh = tiny_backend(config)
    assert fresh.generate(memory.artifacts[1], count=2, seed=3).shape == (2, 32, 32, 3)

    reversed_memory = InCLADCRReplayCADMemory(config, backend=BankOnlyBackend())
    reversed_strategy = ReplayCADStrategy(RecordingModel(), reversed_memory)
    reversed_strategy.learn(second, concept_id="pcb2")
    reversed_strategy.learn(first, concept_id="pcb1")
    assert [len(batch) for batch in reversed_strategy._model.fit_batches] == [4, 2 + 3]


def test_bank_refuses_a_run_with_different_profile_settings(tmp_path):
    config = tiny_config(tmp_path)
    InCLADCRReplayCADMemory(config, backend=ColourBackend())

    changed = InCLADCRReplayCADConfig.model_validate({**config.model_dump(mode="python"), "weight_decay": 0.0})
    with pytest.raises(ValueError, match="weight_decay"):
        InCLADCRReplayCADMemory(changed, backend=ColourBackend())


def test_grey_replay_follows_the_replayed_concept_not_the_current_one(tmp_path):
    memory = InCLADCRReplayCADMemory(tiny_config(tmp_path), backend=ColourBackend())
    colour = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    colour[..., 2] = 50
    grey = np.full((3, 4, 4, 3), 90, dtype=np.uint8)

    memory.compress("bottle", colour)
    memory.compress("screw", grey)
    replay = memory.generate_previous()

    samples = memory.config.replay_samples_per_concept
    bottle, screw = replay[:samples], replay[samples:]
    assert bottle[..., 0].max() == 200 and bottle[..., 1].max() == 0
    assert np.array_equal(screw[..., 0], screw[..., 1]) and np.array_equal(screw[..., 1], screw[..., 2])
    assert int(screw[0, 0, 0, 0]) == round(200 * 0.299)


def test_mask_generator_falls_back_to_full_frame_and_skips_sam_for_full_frame_concepts(tmp_path, monkeypatch):
    config = tiny_config(tmp_path, mask_backend="sam", sam_checkpoint=tmp_path / "sam.pth")
    generator = InCLADCRSAMMaskGenerator(config)
    images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    object_mask = np.zeros((8, 8), dtype=np.uint8)
    object_mask[2:6, 2:6] = 1
    produced = iter([object_mask, np.zeros((8, 8), dtype=np.uint8), np.ones((8, 8), dtype=np.uint8)])
    monkeypatch.setattr(InCLADCRSAMMaskGenerator, "_load", lambda self: None)
    monkeypatch.setattr(InCLADCRSAMMaskGenerator, "_generate_one", lambda self, image, concept_id, mode: next(produced))

    masks = generator.generate(images, "cashew")

    assert masks[0].tolist() == object_mask.tolist()
    assert masks[1].all() and masks[2].all()
    assert generator.statistics["cashew"]["mask_mode"] == "corner-background"
    assert generator.statistics["cashew"]["mask_full_frame_fallbacks"] == 2

    def out_of_memory(self, image, concept_id, mode):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(InCLADCRSAMMaskGenerator, "_generate_one", out_of_memory)
    with pytest.raises(RuntimeError, match="out of memory"):
        generator.generate(images, "cashew")

    area = InCLADCRSAMMaskGenerator(tiny_config(tmp_path, default_mask_mode="area", mask_backend="sam", sam_checkpoint=tmp_path / "sam.pth"))

    def no_proposal(self, image, concept_id, mode):
        raise RuntimeError(f"SAM did not produce a mask for concept '{concept_id}'")

    monkeypatch.setattr(InCLADCRSAMMaskGenerator, "_generate_one", no_proposal)
    assert area.generate(images, "Class8").all()
    assert area.statistics["Class8"]["mask_full_frame_fallbacks"] == 3

    dagm = InCLADCRSAMMaskGenerator(tiny_config(tmp_path, default_mask_mode="full-frame", mask_backend="sam", sam_checkpoint=tmp_path / "sam.pth"))
    monkeypatch.setattr(InCLADCRSAMMaskGenerator, "_load", lambda self: pytest.fail("SAM must not load"))
    assert dagm.generate(images, "Class8").all()
    assert dagm.statistics["Class8"]["mask_full_frame_fallbacks"] == 0
