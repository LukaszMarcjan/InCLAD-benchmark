"""Run ReplayCAD with the ``inclad-cr`` profile through the resumable runner.

``run_continual_visual_ad_resumable.py`` checkpoints after every concept and is
the only runner that loads InCLAD-MD. This wrapper installs the ``inclad-cr``
profile (see ``run_replaycad_cr.py``), lets the base ReplayCAD argument check
accept ``--benchmark inclad-md`` and then calls the resumable runner's
``main``. Neither runner is modified.

Differences from ``run_replaycad_cr.py``:

* ``--replaycad-artifact-dir`` is required and holds the artifact bank of this
  run only (one bank per job; nothing is shared between orderings);
* ``--patchcore-gpu-knn`` (default on) moves PatchCore's nearest-neighbour
  search to the GPU (``pyclad.models.patchcore.gpu_knn``); ``--no-patchcore-gpu-knn``
  keeps sklearn;
* the resumable runner's flags apply: pass ``--resume`` on every submission so
  a resubmitted job continues after its last finished concept.

Example::

    python examples/clvad/run_replaycad_cr_resumable.py \\
        --model patchcore --strategy replaycad --benchmark inclad-md --inclad-bench \\
        --root ~/pyCLAD/src/pyclad/data/datasets/visual_datasets \\
        --ordering-mode easy_to_hard --backbone resnet18 --pretrained --eval-level both \\
        --replaycad-sam-checkpoint ~/models/sam_vit_h_4b8939.pth \\
        --replaycad-artifact-dir ~/replaycad-cr-md-bank/easy_to_hard \\
        --output-dir ~/replaycad-cr-results/inclad-md/easy_to_hard \\
        --resume-dir ~/replaycad-cr-checkpoints --resume --device cuda:0 --n-runs 1
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import pathlib
import sys
from typing import Any, Callable, Sequence

from pyclad.strategies.replaycad import inclad_cr

RESUMABLE_RUNNER = pathlib.Path(__file__).with_name("run_continual_visual_ad_resumable.py")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_replaycad_cr  # noqa: E402  (shares the profile defaults and argument checks)


def build_runner_argv(argv: Sequence[str]) -> tuple[list[str], bool]:
    """Return the resumable runner's command line and whether to use the GPU kNN index."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--patchcore-gpu-knn", action=argparse.BooleanOptionalAction, default=True)
    profile_args, remaining = parser.parse_known_args(list(argv))
    if run_replaycad_cr._has_option(remaining, "--replaycad-bank-dir"):
        raise SystemExit("run_replaycad_cr_resumable.py keeps one bank per run; pass --replaycad-artifact-dir")
    if not run_replaycad_cr._has_option(remaining, "--replaycad-artifact-dir"):
        raise SystemExit("--replaycad-artifact-dir is required: the bank must outlive the job for a resume")
    remaining, _ = run_replaycad_cr.build_base_argv(remaining)
    return remaining, profile_args.patchcore_gpu_knn


def accept_inclad_md(original: Callable[[argparse.Namespace], argparse.Namespace]) -> Callable:
    """Run the base ReplayCAD argument check unchanged, with InCLAD-MD let through.

    The base check allows only the five single-dataset benchmarks. For MD it is run
    with a stand-in benchmark name, so every other rule (model, backbone, orderings,
    SAM checkpoint, defaults) still applies; the real name is put back afterwards.
    """

    def normalize(args: argparse.Namespace) -> argparse.Namespace:
        if getattr(args, "strategy", None) != "replaycad" or inclad_cr.normalize_benchmark(args.benchmark) != "inclad-md":
            return original(args)
        if args.ordering_mode not in {None, "easy_to_hard", "hard_to_easy"}:
            raise ValueError("InCLAD-MD has only the easy_to_hard and hard_to_easy orderings")
        requested = args.benchmark
        args.benchmark = "mvtec"
        try:
            args = original(args)
        finally:
            args.benchmark = requested
        args.benchmark = "inclad-md"
        return args

    normalize.__wrapped__ = original
    return normalize


def load_resumable_runner() -> Any:
    spec = importlib.util.spec_from_file_location("pyclad_visual_runner_resumable", RESUMABLE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the resumable runner from {RESUMABLE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(__doc__)
        runner_argv, gpu_knn = ["--help"], False
    else:
        runner_argv, gpu_knn = build_runner_argv(sys.argv[1:])
    inclad_cr.install(gpu_knn=gpu_knn)
    runner = load_resumable_runner()
    runner._BASE.normalize_replaycad_args = accept_inclad_md(runner._BASE.normalize_replaycad_args)
    sys.argv = [str(RESUMABLE_RUNNER), *runner_argv]
    # The same exit handling as the resumable runner's own __main__ block.
    try:
        runner.main()
    except runner.CheckpointDiskSpaceError as error:
        logging.getLogger(__name__).error("Checkpoint write failed for lack of disk space: %s", error)
        raise SystemExit(runner.DISK_FULL_EXIT_CODE)
    except runner.ResumeError as error:
        logging.getLogger(__name__).error("%s: %s", type(error).__name__, error)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
