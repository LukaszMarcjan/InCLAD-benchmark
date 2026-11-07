"""Locate result files for the camera-ready table builders.

The result JSON files are not part of this repository. Every builder takes
``--results-root``, the directory that holds the result trees, and derives its
default input and output paths from it. The default is the repository root,
which keeps the original behaviour of the scripts.

Aggregate files record the absolute paths of the machine the experiments ran
on. ``locate`` maps such a path onto ``--results-root`` by its trailing tree
path: it tries ``<base>/<last k components>`` for every search base, longest
suffix first, so no machine-specific prefix has to be known.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Iterable, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Layout of the result archive below --results-root. The default input and
# output paths of the builders are built from these names.
BUNDLE_NAME = "InCLAD_paper_results_20261007"  # result bundle of 2026-10-07
OUTPUT_NAME = "InCLAD_camera_ready_20261024"  # default output directory
EXPORTS_NAME = "neur_res"  # directory with the extracted result exports
# The per-run result trees of the single-dataset grid sit below this sub-directory;
# recorded run paths are also looked up there.
RUN_TREES_SUBDIR = pathlib.Path("results_all_") / "neurips"
# MultiDataset run selection read by Table 2, and the result roots it was selected from.
MULTIDATASET_JSON = pathlib.Path("results_all_") / "fin_fin" / "multidataset_whole_average_table.json"
MULTIDATASET_ROOTS = [pathlib.Path("results_all_") / "fin_res", pathlib.Path("results_all_") / "fin_resl"]


def add_results_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPO_ROOT,
        help="Directory holding the result trees; all default input and output paths are "
             "derived from it (default: the repository root).",
    )
    parser.add_argument(
        "--search-dir",
        type=pathlib.Path,
        action="append",
        default=[],
        help="Extra directory under which recorded result paths are looked up (repeatable). "
             "Searched after --results-root and --results-root/" + RUN_TREES_SUBDIR.as_posix() + ".",
    )


def results_root_from_argv(argv: Sequence[str] | None = None) -> pathlib.Path:
    """Read --results-root before the full parser is built, so defaults can depend on it."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--results-root", type=pathlib.Path, default=REPO_ROOT)
    namespace, _ = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    return absolute(namespace.results_root)


def absolute(path: pathlib.Path) -> pathlib.Path:
    """Absolute, normalised path without resolving symbolic links."""
    return pathlib.Path(os.path.abspath(os.path.expanduser(str(path))))


def search_bases(results_root: pathlib.Path, extra: Iterable[pathlib.Path] = ()) -> list[pathlib.Path]:
    root = absolute(results_root)
    bases = [root, root / RUN_TREES_SUBDIR] + [absolute(p) for p in extra]
    unique: list[pathlib.Path] = []
    for base in bases:
        if base not in unique:
            unique.append(base)
    return unique


def locate(path_str: str | os.PathLike, results_root: pathlib.Path,
           extra: Iterable[pathlib.Path] = ()) -> pathlib.Path | None:
    """Return the existing file a recorded path refers to, or None.

    Order: the path as recorded (relative paths against --results-root first);
    then, for each trailing sub-path of at least two components (longest
    first), each search base.
    """
    path = pathlib.Path(path_str)
    bases = search_bases(results_root, extra)
    if path.is_absolute():
        if path.exists():
            return path
    else:
        for base in bases:
            candidate = base / path
            if candidate.exists():
                return candidate
        if path.exists():
            return absolute(path)
    parts = [part for part in path.parts if part not in (path.anchor, "")]
    for start in range(0, max(len(parts) - 1, 0)):
        tail = pathlib.Path(*parts[start:])
        for base in bases:
            candidate = base / tail
            if candidate.exists():
                return candidate
    return None


def require(path_str: str | os.PathLike, results_root: pathlib.Path,
            extra: Iterable[pathlib.Path] = (), what: str = "result file") -> pathlib.Path:
    found = locate(path_str, results_root, extra)
    if found is None:
        raise SystemExit(
            f"{what} not found: {path_str}\n"
            f"searched as recorded and below: {', '.join(str(b) for b in search_bases(results_root, extra))}\n"
            "pass --results-root (and --search-dir) pointing at the directory that holds the result trees"
        )
    return found


def require_input(path: pathlib.Path, flag: str) -> pathlib.Path:
    """Exit with a clear message when an input file or directory is missing."""
    if not path.exists():
        raise SystemExit(f"{flag}: {path} does not exist (set --results-root or pass {flag} explicitly)")
    return path


def source_string(path: pathlib.Path, roots: Iterable[pathlib.Path]) -> str:
    """Provenance string: relative to the first root that contains path, else absolute."""
    target = absolute(path)
    for root in roots:
        root = absolute(root)
        if target.is_relative_to(root):
            return target.relative_to(root).as_posix()
    resolved = path.resolve()
    for root in roots:
        root = pathlib.Path(root).resolve()
        if resolved.is_relative_to(root):
            return resolved.relative_to(root).as_posix()
    return target.as_posix()
