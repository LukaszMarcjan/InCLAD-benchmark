#!/usr/bin/env python3
"""Turn one row of the grouped-component grid manifest into a runner command.

A manifest row has six tab-separated labels:

    dataset  backbone  model  strategy  schedule  ordering
    mvtec    rn18      padim  naive     14-1      eth

This script maps the labels to the flags every grouped run of the paper used
(as recorded in the ``cli_args`` of the result JSONs) and prints the command
for ``examples/clvad/run_continual_visual_ad_step_schedule.py``. With
``--run`` it also executes the command from the repository root.

Fixed flags (all 1692 completed grouped runs):

    --pretrained --batch-size 32 --epochs 50 --n-runs 1
    --replay-buffer-fraction 0.2 --device cuda:0 --eval-level both
    --pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95

Per-row flags:

    backbone  rn18 -> --backbone resnet18, wr50 -> --backbone wide_resnet50_2
    ordering  eth  -> --ordering-mode easy_to_hard --ordering-dir examples/visual_models/ordering --master-seed 42
              hte  -> --ordering-mode hard_to_easy --ordering-dir examples/visual_models/ordering --master-seed 42
              rand -> no ordering flag (the dataset's alphabetical category order), --master-seed 123

Master seed 42 gives per-run seed 191664963 and master seed 123 gives
33158374. The runner defaults differ (``--eval-level image``,
``--pixel-threshold-mode fixed``, ``--pixel-threshold-quantile 0.995``, model
defaults for batch size and epochs), so running a row with the defaults does
not reproduce the grid.

Results go to ``<results-root>/<dataset>/<backbone>/<ordering>/<model>/<strategy>/<schedule>/``,
the relative layout read by ``examples/clvad/camera_ready/build_grouped_tables.py``.

Examples::

    # print the command of one row, given as six labels
    python examples/clvad/grouped_grid/run_cell.py mvtec rn18 padim naive 14-1 eth

    # print the command of line 3 of the manifest (1-based)
    python examples/clvad/grouped_grid/run_cell.py --row 3

    # print one command per manifest row (e.g. to feed a job scheduler)
    python examples/clvad/grouped_grid/run_cell.py --all > commands.sh

    # run one row
    python examples/clvad/grouped_grid/run_cell.py mvtec rn18 padim naive 14-1 eth --run \\
        --datasets-root /path/to/visual_datasets
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_MANIFEST = HERE / "manifest.tsv"

RUNNER = "examples/clvad/run_continual_visual_ad_step_schedule.py"
ORDERING_DIR = "examples/visual_models/ordering"

# Axes of the grid; they mirror gen_manifest.sh.
SCHEDULES = {
    "mvtec": ("14-1", "10-5", "3x5", "10-1x5"),
    "visa": ("11-1", "8-4", "8-1x4"),
    "btech": ("2-1", "1x3"),
    "mpdd": ("5-1", "3x2", "4-1x2"),
    "dagm": ("9-1", "5-5", "2x5", "5-1x5"),
}
BACKBONES = {"rn18": "resnet18", "wr50": "wide_resnet50_2"}
MODELS = ("padim", "patchcore", "cfa", "fastflow", "stfpm", "rd4ad", "paste", "ganomaly", "efficientad")
CL_CAPABLE = ("padim", "patchcore", "cfa")

# ordering label -> (runner ordering flags without the directory, master seed)
ORDERINGS = {
    "eth": ("easy_to_hard", 42),
    "hte": ("hard_to_easy", 42),
    "rand": (None, 123),
}

# Flags shared by every grouped run.
FIXED_FLAGS = (
    "--pretrained",
    "--batch-size", "32",
    "--epochs", "50",
    "--n-runs", "1",
    "--replay-buffer-fraction", "0.2",
    "--eval-level", "both",
    "--pixel-threshold-mode", "train-quantile",
    "--pixel-threshold-quantile", "0.95",
)

COLUMNS = ("dataset", "backbone", "model", "strategy", "schedule", "ordering")


def validate_row(dataset: str, backbone: str, model: str, strategy: str, schedule: str, ordering: str) -> None:
    """Raise ValueError unless the six labels form a cell of the grid."""
    if dataset not in SCHEDULES:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(SCHEDULES)}")
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; expected one of {sorted(BACKBONES)}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {list(MODELS)}")
    allowed = ("naive", "cl") if model in CL_CAPABLE else ("naive", "replay")
    if strategy not in allowed:
        raise ValueError(f"strategy {strategy!r} is not in the grid for {model}; expected one of {list(allowed)}")
    if schedule not in SCHEDULES[dataset]:
        raise ValueError(f"schedule {schedule!r} is not in the grid for {dataset}; expected one of {list(SCHEDULES[dataset])}")
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}; expected one of {sorted(ORDERINGS)}")


def output_dir_for(results_root: str, dataset: str, backbone: str, model: str, strategy: str,
                   schedule: str, ordering: str) -> str:
    return str(pathlib.PurePosixPath(results_root, dataset, backbone, ordering, model, strategy, schedule))


def build_command(dataset: str, backbone: str, model: str, strategy: str, schedule: str, ordering: str,
                  *, results_root: str = "results_grouped_grid", device: str = "cuda:0",
                  python: str = "python") -> list[str]:
    """Return the runner command (argv list) for one manifest row.

    Paths are relative to the repository root; run the command from there.
    """
    validate_row(dataset, backbone, model, strategy, schedule, ordering)
    ordering_mode, master_seed = ORDERINGS[ordering]
    command = [
        python, RUNNER,
        "--model", model,
        "--strategy", strategy,
        "--benchmark", dataset,
        "--backbone", BACKBONES[backbone],
        *FIXED_FLAGS,
        "--device", device,
        "--step-schedule", schedule,
        "--master-seed", str(master_seed),
    ]
    if ordering_mode is not None:
        command += ["--ordering-mode", ordering_mode, "--ordering-dir", ORDERING_DIR]
    command += ["--output-dir", output_dir_for(results_root, dataset, backbone, model, strategy, schedule, ordering)]
    return command


def read_manifest(path: pathlib.Path) -> list[tuple[str, ...]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = tuple(line.split("\t"))
        if len(fields) != len(COLUMNS):
            raise ValueError(f"{path}:{number}: expected {len(COLUMNS)} tab-separated columns, got {len(fields)}")
        rows.append(fields)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="*", metavar="LABEL",
                        help="six manifest labels: dataset backbone model strategy schedule ordering")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--row", type=int, default=None,
                        help="take the labels from this line of the manifest (1-based)")
    source.add_argument("--all", action="store_true", help="print the command of every manifest row")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST,
                        help="manifest TSV (default: manifest.tsv next to this script)")
    parser.add_argument("--results-root", default="results_grouped_grid",
                        help="root of the per-cell output directories, relative to the repository root "
                             "or absolute (default: %(default)s)")
    parser.add_argument("--device", default="cuda:0", help="value of --device (default: %(default)s)")
    parser.add_argument("--python", default=None,
                        help="interpreter in the printed command (default: 'python'; with --run, this interpreter)")
    parser.add_argument("--datasets-root", default=None,
                        help="exported as PYCLAD_VISUAL_DATASETS_ROOT for the runner (optional; the runner also "
                             "reads per-dataset env vars and the dataset registry)")
    parser.add_argument("--run", action="store_true", help="execute the command instead of only printing it")
    args = parser.parse_args(argv)

    if args.all:
        if args.labels:
            parser.error("--all takes no positional labels")
        if args.run:
            parser.error("--run executes a single row; use --all only to print commands")
    elif args.row is not None:
        if args.labels:
            parser.error("--row takes no positional labels")
    elif len(args.labels) != len(COLUMNS):
        parser.error(f"expected {len(COLUMNS)} labels ({' '.join(COLUMNS)}), --row N, or --all")
    return args


def environment_prefix(datasets_root: str | None) -> str:
    env = "PYTHONPATH=src"
    if datasets_root:
        env += f" PYCLAD_VISUAL_DATASETS_ROOT={shlex.quote(datasets_root)}"
    return env


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all:
        rows = read_manifest(args.manifest)
    elif args.row is not None:
        manifest_rows = read_manifest(args.manifest)
        if not 1 <= args.row <= len(manifest_rows):
            raise SystemExit(f"--row must be between 1 and {len(manifest_rows)}")
        rows = [manifest_rows[args.row - 1]]
    else:
        rows = [tuple(args.labels)]

    python = args.python or (sys.executable if args.run else "python")
    commands = []
    for row in rows:
        try:
            commands.append(build_command(*row, results_root=args.results_root, device=args.device, python=python))
        except ValueError as exc:
            raise SystemExit(f"invalid row {' '.join(row)}: {exc}") from exc

    prefix = environment_prefix(args.datasets_root)
    for command in commands:
        print(f"{prefix} {shlex.join(command)}", flush=True)

    if not args.run:
        return 0

    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.datasets_root:
        env["PYCLAD_VISUAL_DATASETS_ROOT"] = args.datasets_root
    return subprocess.call(commands[0], cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())
