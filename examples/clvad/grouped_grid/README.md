# Grouped-component grid

This folder lists the grouped-component (step-schedule) runs of the paper
(Appendix "Grouped-Component Scenario Results") and turns each row into the
runner command that produced it.

| File | Purpose |
|------|---------|
| `manifest.tsv` | The 1728 cells of the grid, one per line |
| `gen_manifest.sh` | Regenerates `manifest.tsv` (`bash gen_manifest.sh > manifest.tsv`) |
| `run_cell.py` | Prints (or runs) the command for one row, a manifest line, or every row |
| `test_run_cell.py` | Checks that the commands parse to the arguments recorded in the result files |

## Manifest columns

Each line has six tab-separated labels, for example `mvtec  rn18  padim  naive  14-1  eth`.

| Column | Values | Runner flag |
|--------|--------|-------------|
| dataset | `mvtec`, `visa`, `btech`, `mpdd`, `dagm` | `--benchmark` |
| backbone | `rn18`, `wr50` | `--backbone resnet18`, `--backbone wide_resnet50_2` |
| model | `padim`, `patchcore`, `cfa`, `fastflow`, `stfpm`, `rd4ad`, `paste`, `ganomaly`, `efficientad` | `--model` |
| strategy | `naive` for every model; `cl` for PaDiM, PatchCore and CFA; `replay` for the others | `--strategy` |
| schedule | MVTec AD `14-1 10-5 3x5 10-1x5`; VisA `11-1 8-4 8-1x4`; BTAD `2-1 1x3`; MPDD `5-1 3x2 4-1x2`; DAGM `9-1 5-5 2x5 5-1x5` | `--step-schedule` |
| ordering | `eth`, `hte`, `rand` | see below |

Schedules use the runner syntax: `10-1x5` (10+1x5 in the paper) learns ten components
in the first step and then one per step for five steps, and `3x5` learns five groups of three.
In the tables `naive` is NCL and `cl`/`replay` is CL.

Orderings:

- `eth` (E2H): `--ordering-mode easy_to_hard --ordering-dir examples/visual_models/ordering --master-seed 42`
- `hte` (H2E): `--ordering-mode hard_to_easy --ordering-dir examples/visual_models/ordering --master-seed 42`
- `rand` (Alph. in the paper): no ordering flag and no `--shuffle-concepts`, `--master-seed 123`.
  The categories keep the order of the source dataset, i.e. sorted by name
  (DAGM: `Class1, Class10, Class2, ...`). Despite the label, this is not a shuffle.

With `--n-runs 1`, master seed 42 gives the per-run seed 191664963 and master seed 123 gives
33158374; the seed is part of the result file name.

## Flags shared by every run

Unlike the main grid, all nine models used batch size 32 and 50 epochs here.

All 1692 completed runs used by the grouped tables record the same values in
`run_config.cli_args`:

```
--pretrained --batch-size 32 --epochs 50 --n-runs 1 --replay-buffer-fraction 0.2
--device cuda:0 --eval-level both
--pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95
```

Everything else was left at the runner default (for example `--pixel-threshold 0.5`,
`--pixel-skip-missing-masks`, `--ordering-metric roc-auc`, no `--resize`, no
`--max-train`/`--max-test`, no `--group-test`). For PatchCore CL the runner derives the
memory-bank size from `--replay-buffer-fraction` and stores it in the result file.

The runner defaults are not the grid settings: `--eval-level image` produces no pixel-level
matrix (the pixel columns of the grouped tables need one), `--pixel-threshold-mode fixed` and
`--pixel-threshold-quantile 0.995` change the pixel threshold, batch size and epochs fall back
to model defaults, and `--master-seed 42` would be wrong for the Alph. cells. Use `run_cell.py`
rather than the defaults.

The runs read the E2H/H2E curricula from copies of the CSV files in
`examples/visual_models/ordering/`; the category order stored in every E2H and H2E result
(`test_order`) is the order in those files.

The grid was run with `examples/clvad/run_continual_visual_ad_step_schedule.py`.

## Running a cell

Run from the repository root:

```bash
# print the command for one row given as six labels
python examples/clvad/grouped_grid/run_cell.py mvtec rn18 padim naive 14-1 eth

# print the command for line 3 of the manifest (1-based)
python examples/clvad/grouped_grid/run_cell.py --row 3

# one command per manifest row, e.g. to submit to a job scheduler
python examples/clvad/grouped_grid/run_cell.py --all > grouped_commands.sh

# run one row
python examples/clvad/grouped_grid/run_cell.py mvtec rn18 padim naive 14-1 eth --run \
    --datasets-root /path/to/visual_datasets
```

The first command prints:

```
PYTHONPATH=src python examples/clvad/run_continual_visual_ad_step_schedule.py --model padim --strategy naive --benchmark mvtec --backbone resnet18 --pretrained --batch-size 32 --epochs 50 --n-runs 1 --replay-buffer-fraction 0.2 --eval-level both --pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95 --device cuda:0 --step-schedule 14-1 --master-seed 42 --ordering-mode easy_to_hard --ordering-dir examples/visual_models/ordering --output-dir results_grouped_grid/mvtec/rn18/eth/padim/naive/14-1
```

Options: `--results-root` (default `results_grouped_grid`), `--device` (default `cuda:0`),
`--python`, `--datasets-root` (exported as `PYCLAD_VISUAL_DATASETS_ROOT`; the runner also
reads per-dataset variables and the dataset registry), `--manifest`. Rows outside the grid
(for example `stfpm cl`) are rejected.

Each cell writes to `<results-root>/<dataset>/<backbone>/<ordering>/<model>/<strategy>/<schedule>/`,
the relative layout read by `examples/clvad/camera_ready/build_grouped_tables.py`.

## Metrics

The runner stores, per level, the rectangular steps x components matrix (`metric_matrix`,
`train_order`, `test_order`, `first_seen_step`) together with `FinalStepAverage`,
`RectangularForgettingMeasure`, `RectangularForwardTransfer` and
`RectangularNewTaskAcquisition`. The ROC and BWT of the grouped tables are not stored; the
table builder recomputes them from the matrix with the appendix definitions, where f(c) is
the step that introduced component c:

- ROC: mean of R[s][c] over all s >= f(c);
- BWT: mean of R[s][c] - R[f(c)][c] over all s > f(c) (0 when there is no such pair).

The library classes `RectangularContinualAverage` and `RectangularBackwardTransfer`
(`pyclad.metrics.continual.rectangular_continual_average`,
`pyclad.metrics.continual.rectangular_backward_transfer`) compute the same values, and
`pyclad.metrics.continual.rectangular_matrix.matrix_from_result_block` reads a result block
into their inputs:

```python
from pyclad.metrics.continual.rectangular_backward_transfer import RectangularBackwardTransfer
from pyclad.metrics.continual.rectangular_continual_average import RectangularContinualAverage
from pyclad.metrics.continual.rectangular_matrix import matrix_from_result_block

matrix, first_seen = matrix_from_result_block(result["rectangular_concept_metric_callback_ROC-AUC"])
roc = RectangularContinualAverage().compute(matrix, first_seen)
bwt = RectangularBackwardTransfer().compute(matrix, first_seen)
```

`ContinualAverage` and `BackwardTransfer` index columns by position and assume a square
matrix with one component per step; do not apply them to these matrices.

## Coverage

The grouped tables average over the 1692 cells that have a completed run. The other 36
cells have only failed runs: VisA WR50 STFPM (all 18), DAGM WR50 STFPM (8 naive, 7 replay)
and DAGM WR50 PatchCore (2 cl, 1 naive).
