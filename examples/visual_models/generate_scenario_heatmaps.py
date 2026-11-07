"""
Generate continual-learning heatmaps for one well-defined visual scenario.

This is a focused companion to ``generate_heatmaps.py``. Instead of scanning a
directory blindly, it selects the runs that belong to a *single scenario* and
renders one heatmap per (model, dataset, metric) so it is obvious what was run:

    scenario (defaults):
        strategy            : naive          (single component per dataset:
                                               one model trained step-by-step,
                                               not a multi-expert ensemble)
        step schedule       : single-component  (one new category per step,
                                                  e.g. "1x15" for MVTec,
                                                  "1x12" for VisA)
        backbone            : resnet18       (1 backbone)
        benchmarks/datasets : mvtec, visa    (2 datasets)
        models              : padim, patchcore, cfa, fastflow, stfpm  (5 models)
        ordering            : whatever the run recorded (see --ordering-label)

The results JSONs store *rectangular* concept metrics
(``rectangular_concept_metric_callback_*`` and
``rectangular_pixel_concept_metric_callback_*``): rows are training steps,
columns are evaluated categories. We relabel each row with the category learned
at that step (single-component schedules add exactly one category per step), so
each heatmap reads as a standard continual-learning forgetting matrix.

Output layout (under --output-dir, default ``<results_root>/scenario_heatmaps``):

    <dataset>/<metric>/<model>__<schedule>__<backbone>__<strategy>.pdf
    scenario_manifest.csv      # every heatmap + the full config behind it

Usage:
    python examples/visual_models/generate_scenario_heatmaps.py results_icdm
    python examples/visual_models/generate_scenario_heatmaps.py results_icdm \
        --ordering-label Random --metrics ROC-AUC Pixel-AUPRO
    python examples/visual_models/generate_scenario_heatmaps.py results_icdm --dry-run

ALL results (ignore the scenario filter — every model/strategy/schedule,
including multi_expert; files named by JSON stem under <dataset>/<strategy>/<metric>/):

    # one results root (dedups copies across batches to the newest):
    python examples/visual_models/generate_scenario_heatmaps.py results_icdm --all
    # several roots at once (scan repo root; keep every file so none is hidden):
    python examples/visual_models/generate_scenario_heatmaps.py . --all --no-dedup \
        --output-dir all_heatmaps
"""

import argparse
import csv
import json
import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)

RECTANGULAR_PREFIXES = {
    "image": "rectangular_concept_metric_callback_",
    "pixel": "rectangular_pixel_concept_metric_callback_",
}

DEFAULT_MODELS = ("padim", "patchcore", "cfa", "fastflow", "stfpm")
DEFAULT_BENCHMARKS = ("mvtec", "visa")
DEFAULT_STRATEGIES = ("naive",)
DEFAULT_BACKBONE = "resnet18"

# Pretty names for titles
MODEL_DISPLAY = {
    "padim": "PaDiM",
    "patchcore": "PatchCore",
    "cfa": "CFA",
    "fastflow": "FastFlow",
    "stfpm": "STFPM",
}
BENCHMARK_DISPLAY = {"mvtec": "MVTec", "visa": "VisA"}

PLOTTING_DEPENDENCIES = None


@dataclass(slots=True)
class ScenarioFilter:
    models: tuple[str, ...] = DEFAULT_MODELS
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    backbone: str | None = DEFAULT_BACKBONE
    single_component_only: bool = True


@dataclass(slots=True)
class HeatmapOptions:
    annotate: bool = True
    figsize: tuple[int, int] = (10, 10)
    metrics: tuple[str, ...] = ()  # empty == all metrics found
    ordering_label: str | None = None  # None == read from each run
    overwrite: bool = False
    dry_run: bool = False
    show_title: bool = False
    show_axis_labels: bool = False
    show_tick_labels: bool = False
    show_cbar: bool = True
    use_stem_filenames: bool = False
    mask_upper_diagonal: bool = False
    annot_size: float | None = None


@dataclass(slots=True)
class RunInfo:
    path: pathlib.Path
    model: str
    benchmark: str
    strategy: str
    backbone: str
    schedule: str
    schedule_parsed: list[int]
    seed: int | None
    ordering_source: str
    data: dict = field(repr=False, default_factory=dict)


def get_plotting_dependencies():
    global PLOTTING_DEPENDENCIES
    if PLOTTING_DEPENDENCIES is None:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        from pyclad.analysis.scenario_heatmap import plot_rectangular_metric_heatmap

        PLOTTING_DEPENDENCIES = (plt, plot_rectangular_metric_heatmap)
    return PLOTTING_DEPENDENCIES


def load_run_info(json_path: pathlib.Path) -> RunInfo | None:
    try:
        with json_path.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug(f"Cannot read {json_path}: {exc}")
        return None

    rc = data.get("run_config") or {}
    cli = rc.get("cli_args") or {}
    ordering = rc.get("ordering_metadata") or {}
    schedule_parsed = rc.get("step_schedule_parsed") or []

    return RunInfo(
        path=json_path,
        model=rc.get("model"),
        benchmark=rc.get("benchmark"),
        strategy=rc.get("strategy"),
        backbone=cli.get("backbone"),
        schedule=rc.get("step_schedule"),
        schedule_parsed=list(schedule_parsed),
        seed=rc.get("seed"),
        ordering_source=ordering.get("source", "unknown"),
        data=data,
    )


def matches_scenario(run: RunInfo, scenario: ScenarioFilter) -> bool:
    if run.model not in scenario.models:
        return False
    if run.benchmark not in scenario.benchmarks:
        return False
    if run.strategy not in scenario.strategies:
        return False
    if scenario.backbone is not None and run.backbone != scenario.backbone:
        return False
    if scenario.single_component_only:
        if not run.schedule_parsed or any(step != 1 for step in run.schedule_parsed):
            return False
    return True


def discover_runs(results_root: pathlib.Path, scenario: ScenarioFilter) -> list[RunInfo]:
    """Find scenario runs, de-duplicating identical configs to the newest file."""
    best: dict[tuple, RunInfo] = {}
    for json_path in results_root.rglob("*_seed_*.json"):
        run = load_run_info(json_path)
        if run is None or not matches_scenario(run, scenario):
            continue
        # Key on the filename (it encodes model, schedule, and any variant
        # tokens such as multi-expert router / train mode) plus the benchmark,
        # so distinct variants are never collapsed. Identical logical runs that
        # were copied across batches share a filename and dedup to the newest
        # path (timestamped dir names sort chronologically).
        key = (run.benchmark, json_path.name)
        if key not in best or json_path.as_posix() > best[key].path.as_posix():
            best[key] = run
    return sorted(best.values(), key=lambda r: (r.benchmark, r.model, r.schedule))


def discover_all_runs(results_root: pathlib.Path, dedup: bool = True) -> list[RunInfo]:
    """Every run under results_root, with no scenario filtering.

    With ``dedup`` (default), identical runs copied across batches/roots are
    collapsed by (benchmark, filename) keeping the lexicographically-newest path.
    Multi-expert variants keep distinct filenames, so they are never merged.
    Pass ``dedup=False`` when scanning several result roots at once so that no
    file is hidden (path-name ordering is not chronological across roots).
    """
    runs: list[RunInfo] = []
    for json_path in results_root.rglob("*_seed_*.json"):
        run = load_run_info(json_path)
        if run is not None:
            runs.append(run)
    if dedup:
        best: dict[tuple, RunInfo] = {}
        for run in runs:
            key = (run.benchmark, run.path.name)
            if key not in best or run.path.as_posix() > best[key].path.as_posix():
                best[key] = run
        runs = list(best.values())
    return sorted(runs, key=lambda r: (str(r.benchmark), str(r.strategy),
                                       str(r.model), str(r.schedule), r.path.as_posix()))


def find_rectangular_metrics(data: dict) -> list[tuple[str, str, dict]]:
    """Return (metric_kind, metric_name, payload) for each rectangular metric."""
    found = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        for metric_kind, prefix in RECTANGULAR_PREFIXES.items():
            if key.startswith(prefix):
                name = value.get("base_metric_name", key[len(prefix):])
                found.append((metric_kind, name, value))
                break
    return found


def build_row_labels(run: RunInfo) -> dict[str, str]:
    """Label each train step with the category/categories first seen at it.

    For single-component schedules (one category per step) the row reads
    ``"1. bottle"``; for grouped schedules (e.g. ``10-5``) a step introduces
    several categories at once, so it is labelled ``"1. [10 cats]"`` instead of
    silently picking one.
    """
    schedule_meta = (run.data.get("run_config") or {}).get("schedule_metadata") or {}
    first_seen = schedule_meta.get("first_seen_step") or {}
    step_to_categories: dict[int, list[str]] = {}
    for category, step_index in first_seen.items():
        step_to_categories.setdefault(int(step_index), []).append(category)
    labels: dict[str, str] = {}
    for step_index, categories in step_to_categories.items():
        if len(categories) == 1:
            labels[f"step_{step_index}"] = f"{step_index + 1}. {categories[0]}"
        else:
            labels[f"step_{step_index}"] = f"{step_index + 1}. [{len(categories)} cats]"
    return labels


def safe(text: str) -> str:
    return str(text).lower().replace(" ", "_").replace("/", "-")


def schedule_tag(run: RunInfo) -> str:
    """'single-component' when every step adds one category, else 'grouped-steps'."""
    parsed = run.schedule_parsed
    if parsed and all(step == 1 for step in parsed):
        return "single-component"
    return "grouped-steps"


def build_title(run: RunInfo, metric_name: str, ordering_label: str) -> str:
    model = MODEL_DISPLAY.get(run.model, run.model)
    dataset = BENCHMARK_DISPLAY.get(run.benchmark, run.benchmark)
    router = (run.data.get("run_config") or {}).get("router")
    router_part = f" · router: {router}" if router else ""
    return (
        f"{metric_name}  —  {model} · {dataset}\n"
        f"{run.strategy}{router_part} · {schedule_tag(run)} ({run.schedule}) · "
        f"{run.backbone} · ordering: {ordering_label} · seed {run.seed}"
    )


def process_run(run: RunInfo, output_dir: pathlib.Path, options: HeatmapOptions) -> list[dict]:
    metrics = find_rectangular_metrics(run.data)
    if not metrics:
        logger.warning(f"  No rectangular metrics in {run.path.name}, skipping")
        return []

    wanted = {m.lower() for m in options.metrics}
    row_labels = build_row_labels(run)
    ordering_label = options.ordering_label or run.ordering_source
    manifest_rows: list[dict] = []

    for metric_kind, metric_name, payload in metrics:
        if wanted and metric_name.lower() not in wanted:
            continue
        matrix = payload.get("metric_matrix")
        train_order = payload.get("train_order")
        test_order = payload.get("test_order")
        if not matrix or not train_order or not test_order:
            logger.debug(f"  Skipping '{metric_name}': incomplete matrix")
            continue

        if options.use_stem_filenames:
            # All-mode: many strategies/variants share model+schedule, so name by
            # the source JSON stem (encodes router / train mode / seed) to stay unique.
            out_path = (
                output_dir
                / run.benchmark
                / run.strategy
                / safe(metric_name)
                / f"{run.path.stem}.pdf"
            )
        else:
            out_path = (
                output_dir
                / run.benchmark
                / safe(metric_name)
                / f"{run.model}__{run.schedule}__{run.backbone}__{run.strategy}.pdf"
            )

        manifest_rows.append(
            {
                "dataset": run.benchmark,
                "model": run.model,
                "strategy": run.strategy,
                "backbone": run.backbone,
                "schedule": run.schedule,
                "ordering": ordering_label,
                "metric_kind": metric_kind,
                "metric": metric_name,
                "seed": run.seed,
                "source_json": run.path.as_posix(),
                "heatmap": out_path.as_posix(),
            }
        )

        if out_path.exists() and not options.overwrite:
            logger.info(f"  [skip] {out_path.relative_to(output_dir)} exists (use --overwrite)")
            continue
        if options.dry_run:
            logger.info(f"  [dry-run] would write {out_path.relative_to(output_dir)}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt, plot_rectangular_metric_heatmap = get_plotting_dependencies()
            plt.close("all")
            plot_rectangular_metric_heatmap(
                matrix=matrix,
                train_order=train_order,
                test_order=test_order,
                row_labels=row_labels,
                xlabel="Evaluated on category" if options.show_axis_labels else "",
                ylabel="After training on category (step order ↓)" if options.show_axis_labels else "",
                title=build_title(run, metric_name, ordering_label) if options.show_title else "",
                annotate=options.annotate,
                figsize=options.figsize,
                cbar_label=metric_name if options.show_axis_labels else None,
                show_tick_labels=options.show_tick_labels,
                show_cbar=options.show_cbar,
                mask_upper_diagonal=options.mask_upper_diagonal,
                annot_size=options.annot_size,
                output_path=out_path,
            )
            plt.close("all")
            logger.info(f"  ✓ {out_path.relative_to(output_dir)}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ✗ Failed '{metric_name}' for {run.path.name}: {exc}")

    return manifest_rows


def write_manifest(manifest_rows: list[dict], output_dir: pathlib.Path) -> None:
    if not manifest_rows:
        return
    manifest_path = output_dir / "scenario_manifest.csv"
    fieldnames = list(manifest_rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    logger.info(f"Manifest written: {manifest_path} ({len(manifest_rows)} entries)")


def generate(
    results_root: pathlib.Path,
    scenario: ScenarioFilter,
    options: HeatmapOptions,
    output_dir: pathlib.Path | None = None,
    process_all: bool = False,
    dedup: bool = True,
) -> None:
    if process_all:
        runs = discover_all_runs(results_root, dedup=dedup)
        default_subdir = "all_heatmaps"
    else:
        runs = discover_runs(results_root, scenario)
        default_subdir = "scenario_heatmaps"
    if not runs:
        logger.warning(f"No runs matched under {results_root}")
        return

    output_dir = output_dir or (results_root / default_subdir)
    logger.info(f"Runs selected: {len(runs)}"
                f"{' (all, scenario filters off)' if process_all else ' (scenario)'}")
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        by_dataset[run.benchmark].append(run.model)
    for dataset, models in sorted(by_dataset.items()):
        logger.info(f"  {dataset}: {len(models)} run(s)")
    logger.info(f"Output dir: {output_dir}")
    logger.info("")

    manifest_rows: list[dict] = []
    for run in runs:
        logger.info(f"Processing: {run.path.relative_to(results_root)}")
        manifest_rows.extend(process_run(run, output_dir, options))

    if not options.dry_run:
        write_manifest(manifest_rows, output_dir)
    logger.info("")
    logger.info(
        f"Done — {len([r for r in manifest_rows])} heatmap(s) "
        f"{'planned' if options.dry_run else 'recorded'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("results_root", help="Results directory to scan (e.g. results_icdm)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write heatmaps (default: <results_root>/scenario_heatmaps)")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE,
                        help="Required backbone (use 'any' to disable the filter)")
    parser.add_argument("--allow-multi-step", action="store_true",
                        help="Include schedules that add more than one category per step")
    parser.add_argument("--all", dest="process_all", action="store_true",
                        help="Generate heatmaps for EVERY run found (ignore scenario filters: "
                             "all models / strategies / schedules / backbones, incl. multi_expert). "
                             "Files are named by the source JSON stem and grouped by "
                             "<dataset>/<strategy>/<metric>/.")
    parser.add_argument("--no-dedup", action="store_true",
                        help="In --all mode, render one heatmap per JSON file (do not collapse "
                             "identical runs copied across batches/roots)")
    parser.add_argument("--metrics", nargs="+", default=[],
                        help="Restrict to these base metric names (default: all found)")
    parser.add_argument("--ordering-label", default=None,
                        help="Label to print for the concept ordering (default: read from run)")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 10], metavar=("W", "H"))
    parser.add_argument("--mask-upper-diagonal", dest="mask_upper_diagonal", action="store_true",
                        help="Blank out cells above the diagonal (future, not-yet-trained "
                             "categories). Only applies to single-component schedules where "
                             "the matrix is square (rows and columns aligned).")
    parser.add_argument("--annot-size", type=float, default=None, metavar="PT",
                        help="Font size (points) for the numbers inside cells, e.g. 12 "
                             "(default: seaborn auto)")
    parser.add_argument("--annotate", dest="annotate", action="store_true", default=True,
                        help="Write the numeric value inside each cell (default on)")
    parser.add_argument("--no-annotate", dest="annotate", action="store_false",
                        help="Drop the numeric values inside cells")
    parser.add_argument("--no-title", dest="show_title", action="store_false", default=True,
                        help="Drop the figure title (the line describing the run)")
    parser.add_argument("--no-axis-labels", dest="show_axis_labels", action="store_false", default=True,
                        help="Drop the x/y axis labels and the colorbar label")
    parser.add_argument("--no-tick-labels", dest="show_tick_labels", action="store_false", default=True,
                        help="Drop the category tick labels on both axes")
    parser.add_argument("--no-cbar", dest="show_cbar", action="store_false", default=True,
                        help="Drop the colorbar")
    parser.add_argument("--bare", action="store_true",
                        help="Fully naked heatmap: no title, axis labels, tick labels, cell "
                             "values or colorbar (implies all of the --no-* text flags)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    results_root = pathlib.Path(args.results_root).expanduser()
    if not results_root.is_dir():
        logger.error(f"Results root does not exist: {results_root}")
        raise SystemExit(1)

    scenario = ScenarioFilter(
        models=tuple(args.models),
        benchmarks=tuple(args.benchmarks),
        strategies=tuple(args.strategies),
        backbone=None if str(args.backbone).lower() == "any" else args.backbone,
        single_component_only=not args.allow_multi_step,
    )
    options = HeatmapOptions(
        annotate=args.annotate and not args.bare,
        figsize=tuple(args.figsize),
        metrics=tuple(args.metrics),
        ordering_label=args.ordering_label,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        show_title=args.show_title and not args.bare,
        show_axis_labels=args.show_axis_labels and not args.bare,
        show_tick_labels=args.show_tick_labels and not args.bare,
        show_cbar=args.show_cbar and not args.bare,
        use_stem_filenames=args.process_all,
        mask_upper_diagonal=args.mask_upper_diagonal,
        annot_size=args.annot_size,
    )
    output_dir = pathlib.Path(args.output_dir).expanduser() if args.output_dir else None
    generate(results_root, scenario, options, output_dir=output_dir,
             process_all=args.process_all, dedup=not args.no_dedup)
