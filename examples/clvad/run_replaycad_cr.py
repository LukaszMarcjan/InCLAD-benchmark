"""Run ReplayCAD with the InCLAD camera-ready profile (``inclad-cr``).

The base runner ``run_continual_visual_ad.py`` stays unchanged. This wrapper
switches its ReplayCAD config, memory and strategy classes to the profile in
``src/pyclad/strategies/replaycad/inclad_cr.py`` and then runs it as
``__main__`` with the same command line plus one option:

``--replaycad-bank-dir DIR``
    Keep per-concept artifacts and cached replay images in ``DIR/<benchmark>``.
    All orderings of a dataset share this bank: run ``easy_to_hard`` first (it
    compresses every concept), then the other orderings once it has finished;
    they only fit and evaluate PatchCore. Two runs must never write the same
    bank at the same time.

Unless given explicitly, the wrapper adds ``--replaycad-cache-generated``,
``--replaycad-samples-per-concept 200`` and ``--config-json '{"threshold": 0.0}'``.
The fixed PatchCore threshold skips scoring the whole training union after
every fit. It changes only the thresholded image metrics (F1), which the InCLAD
tables do not report; ROC-AUC and AP use the raw scores. ``train-quantile``
pixel thresholds are refused for the same reason.

Example::

    python examples/clvad/run_replaycad_cr.py \\
        --model patchcore --strategy replaycad --benchmark mvtec --inclad-bench \\
        --ordering-mode easy_to_hard --backbone resnet18 --eval-level both \\
        --replaycad-sam-checkpoint ~/models/sam_vit_h_4b8939.pth \\
        --replaycad-bank-dir ~/replaycad-cr-bank --device cuda --n-runs 1
"""

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys
from typing import Optional, Sequence

from pyclad.strategies.replaycad import inclad_cr

BASE_RUNNER = pathlib.Path(__file__).with_name("run_continual_visual_ad.py")

#: ``(option, values)`` put in front of the command line when neither the option nor its
#: ``--no-`` form is given; argparse keeps the last occurrence, so any user value still wins.
PROFILE_DEFAULTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("--replaycad-cache-generated", ()),
    ("--replaycad-samples-per-concept", (str(inclad_cr.DEFAULT_REPLAY_SAMPLES),)),
    ("--config-json", ('{"threshold": 0.0}',)),
)


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(argument == option or argument.startswith(option + "=") for argument in argv)


def _option_value(argv: Sequence[str], option: str, minimum_prefix: Optional[str] = None) -> Optional[str]:
    """Value of the last occurrence of *option*; with *minimum_prefix*, argparse abbreviations count too."""
    value = None
    for index, argument in enumerate(argv):
        name = argument.split("=", 1)[0]
        matches = name == option or (
            minimum_prefix is not None and name.startswith(minimum_prefix) and option.startswith(name)
        )
        if not matches:
            continue
        if "=" in argument:
            value = argument.split("=", 1)[1]
        elif index + 1 < len(argv):
            value = argv[index + 1]
    return value


def build_base_argv(argv: Sequence[str]) -> tuple[list[str], Optional[pathlib.Path]]:
    """Return the base runner's command line and the bank directory."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--replaycad-bank-dir", type=str, default=None)
    profile_args, remaining = parser.parse_known_args(list(argv))

    if _option_value(remaining, "--strategy") != "replaycad":
        raise SystemExit("run_replaycad_cr.py runs only --strategy replaycad")
    # "--pixel-threshold" itself is a different option, so abbreviations need at least "-m".
    if _option_value(remaining, "--pixel-threshold-mode", minimum_prefix="--pixel-threshold-m") == "train-quantile":
        raise SystemExit(
            "The inclad-cr profile refuses --pixel-threshold-mode train-quantile: it scores every seen "
            "real training image after each task and feeds only thresholded metrics the tables do not report"
        )
    if profile_args.replaycad_bank_dir is not None and _has_option(remaining, "--replaycad-artifact-dir"):
        raise SystemExit("Pass either --replaycad-bank-dir or --replaycad-artifact-dir, not both")

    defaults: list[str] = []
    for option, values in PROFILE_DEFAULTS:
        if not _has_option(remaining, option) and not _has_option(remaining, "--no-" + option[2:]):
            defaults.extend([option, *values])
    remaining = [*defaults, *remaining]

    bank_dir = None
    if profile_args.replaycad_bank_dir is not None:
        bank_dir = pathlib.Path(profile_args.replaycad_bank_dir).expanduser()
    return remaining, bank_dir


def main() -> None:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(__doc__)
        base_argv, bank_dir = ["--help"], None
    else:
        base_argv, bank_dir = build_base_argv(sys.argv[1:])
    inclad_cr.install(bank_dir=bank_dir)
    sys.argv = [str(BASE_RUNNER), *base_argv]
    runpy.run_path(str(BASE_RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
