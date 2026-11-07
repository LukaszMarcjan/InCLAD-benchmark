# InCLAD-MD Dataset

InCLAD-MD is a multi-dataset sequential benchmark for continual visual anomaly detection.
It combines selected categories from all five InCLAD-Bench source datasets (BTech, DAGM,
MPDD, MVTec AD, VisA) into a single 27-task curriculum.

The benchmark manifests are published on HuggingFace as the `inclad-md` config of
[`lifelonglab/inclad-benchmark`](https://huggingface.co/datasets/lifelonglab/inclad-benchmark),
with two splits, `easy_to_hard` and `hard_to_easy`.

> **Important:** The HuggingFace repository contains only manifests (metadata). Images
> must be downloaded from each source dataset separately. See the
> [InCLAD-Bench dataset guide](inclad-bench-dataset.md) for download instructions.

---

## Category selection

InCLAD-MD draws 27 categories from five source datasets using the following rule:
**2 easiest + 2 median + 2 hardest** per dataset (based on STE ROC-AUC rankings),
except for BTech which has only 3 categories and uses all three.

| Source dataset | Selected categories |
|----------------|---------------------|
| BTech (3)  | `btech__03`, `btech__01`, `btech__02` |
| DAGM (6)   | `dagm__Class4`, `dagm__Class9`, `dagm__Class7`, `dagm__Class3`, `dagm__Class5`, `dagm__Class8` |
| MPDD (6)   | `mpdd__connector`, `mpdd__bracket_brown`, `mpdd__metal_plate`, `mpdd__bracket_white`, `mpdd__tubes`, `mpdd__bracket_black` |
| MVTec (6)  | `mvtec__bottle`, `mvtec__leather`, `mvtec__toothbrush`, `mvtec__carpet`, `mvtec__screw`, `mvtec__pill` |
| VisA (6)   | `visa__pcb4`, `visa__chewinggum`, `visa__fryum`, `visa__pcb2`, `visa__macaroni2`, `visa__capsules` |

Category names are prefixed with the source dataset name (`<dataset>__<category>`) to
avoid collisions. The selection and both curricula are defined in
`examples/visual_models/neurips_utilities/multidataset_ordering/`.

## Orderings

InCLAD-MD has two orderings (there is no `random` split):

| Split | Description |
|-------|-------------|
| `easy_to_hard` | Dataset blocks ordered easiest-first, categories within each block easiest-first |
| `hard_to_easy` | Exact reverse of `easy_to_hard` (block order and order within blocks) |

Dataset block order in `easy_to_hard`: BTech → MPDD → DAGM → VisA → MVTec

---

## Manifest schema

The InCLAD-Bench manifests have 14 columns; InCLAD-MD adds `source_category`,
for 15 columns:

| Column | Description |
|--------|-------------|
| `sample_id` | `inclad_md:000042` |
| `source_dataset` | Source benchmark: `btech`, `dagm`, `mpdd`, `mvtec`, `visa` |
| `source_category` | Original category name in the source dataset, e.g. `bottle` |
| `category` | Prefixed name used in InCLAD-MD, e.g. `mvtec__bottle` |
| `category_order` | 1-based position in the ordering |
| `split` | `train` or `test` |
| `image_relpath` | Path relative to the **source dataset's** root |
| `mask_relpath` | Mask path relative to the source dataset root (empty for normal) |
| `image_label` | `0` = normal, `1` = anomalous |
| `defect_type` | Defect class name (empty for normal) |
| `ordering_name` | `easy_to_hard` or `hard_to_easy` |
| `ordering_master_seed` | Master seed (`42`) |
| `ordering_seed` | Per-run seed derived from the master seed (run index 0: `191664963`, the seed of every InCLAD-MD paper run); informational, the curricula are fixed |
| `source_homepage` | URL of the source dataset |
| `source_license` | License of the source dataset |

`image_relpath` is **relative to the source dataset's root**, not a single shared root.
The loader resolves each image as `roots[source_dataset] / image_relpath`.

`ordering_seed` is informational: `inclad-md_hard_to_easy.csv` records `1662057957`,
the second seed derived from 42, while the paper runs used `191664963` for both
orderings. The curricula are fixed, so the column does not affect loading. BTech rows
list the license as `CC BY-SA`; it is CC BY-SA 4.0.

---

## Rebuilding the manifests

```bash
python examples/visual_models/neurips_utilities/build_inclad_md_manifest.py \
    --from-hf --hf-filenames --output-dir /tmp/inclad_md_hf
```

The builder reads the selection and the two curricula from
`examples/visual_models/neurips_utilities/multidataset_ordering/` and the
per-benchmark InCLAD-Bench manifests (`<benchmark>_easy_to_hard_samples.csv`). Those
manifests are not shipped in this repository: `--from-hf` downloads them from
`lifelonglab/inclad-benchmark`, or pass `--manifests-dir <dir>` pointing at a folder
that holds them (for example a local snapshot of the HF repository). It writes:

```
/tmp/inclad_md_hf/
  inclad-md_easy_to_hard.csv     (easy_to_hard.csv without --hf-filenames)
  inclad-md_hard_to_easy.csv     (hard_to_easy.csv without --hf-filenames)
```

By default both files get `ordering_master_seed=42`, `ordering_seed=191664963` (the
seed of the paper runs) and `source_license="CC BY-SA 4.0"` for BTech rows. To
rebuild the published files byte for byte, add
`--seed-mode published --keep-source-license`.

## Using InCLAD-MD with pyCLAD

### Quick start

```python
from pyclad.data.datasets.inclad_md_dataset import InCLADMDDataset

dataset = InCLADMDDataset(
    ordering="easy_to_hard",          # or "hard_to_easy"
    roots={
        "mvtec": "/data/mvtec_ad",
        "btech": "/data/BTech_Dataset_transformed",
        "dagm":  "/data/DAGM_KaggleUpload",
        "mpdd":  "/data/MPDD",
        "visa":  "/data/VisA",
    },
    resize_to=(224, 224),
)
```

### Root resolution

Each source dataset root is resolved by `resolve_visual_benchmark_root`, in this order:

1. `roots={"mvtec": "/data/mvtec_ad", ...}` parameter
2. Registry JSON: `registry_path=` argument, else the file named by
   `PYCLAD_VISUAL_DATASETS_FILE`, else `src/pyclad/data/datasets/visual_datasets/registry.json`.
   The file is not shipped (that directory is gitignored); create it by hand or with
   `pyclad.data.readers.visual_dataset_registry.write_visual_dataset_registry({...})`.
3. Per-benchmark environment variables, which exist only for MVTec and VisA:
   `PYCLAD_MVTEC_ROOT`, `PYCLAD_VISA_ROOT`
4. Shared `PYCLAD_VISUAL_DATASETS_ROOT` with auto-detected subfolders (see the
   [InCLAD-Bench guide](inclad-bench-dataset.md#step-2--register-dataset-locations))

A registry entry therefore overrides the environment variables. The CLI runners pass
`--root` to InCLAD-MD by exporting it as `PYCLAD_VISUAL_DATASETS_ROOT` (step 4), so a
registry entry also takes priority over `--root`.

### CLI

`--root` is the parent directory that holds the five source dataset folders.
Non-CL strategies (`naive`, `replay`) use `run_continual_visual_ad.py`; the `cl`
strategy uses `run_continual_visual_ad_cl.py`:

```bash
PYTHONPATH=src python examples/clvad/run_continual_visual_ad.py \
    --model paste --strategy replay \
    --benchmark inclad-md --inclad-bench --ordering-mode easy_to_hard \
    --root /data --backbone resnet18 --pretrained \
    --eval-level both --pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95 \
    --replay-buffer-fraction 0.2 --n-runs 1 --master-seed 42

PYTHONPATH=src python examples/clvad/run_continual_visual_ad_cl.py \
    --model cfa --strategy cl \
    --benchmark inclad-md --inclad-bench --ordering-mode hard_to_easy \
    --root /data --backbone wide_resnet50_2 --pretrained --batch-size 32 --epochs 50 \
    --eval-level both --pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95 \
    --replay-buffer-fraction 0.2 --n-runs 1 --master-seed 42
```

The checkpointed runner `examples/clvad/run_continual_visual_ad_resumable.py`
(One-for-More, ReplayCAD, UCAD only) accepts the same `--benchmark inclad-md
--inclad-bench` flags.

### Settings of the paper's InCLAD-MD runs

The MultiDataset rows of Table 2 use CFA, PaDiM and PaSTe with these settings:

| Method | Strategies | Backbones | Orderings | Model flags |
|--------|-----------|-----------|-----------|-------------|
| CFA    | `naive`, `cl` | `resnet18`, `wide_resnet50_2` | both | `--pretrained --batch-size 32 --epochs 50` |
| PaDiM  | `cl` (both backbones), `naive` (`wide_resnet50_2` only) | `resnet18`, `wide_resnet50_2` | both | `--pretrained --batch-size 32 --epochs 50` |
| PaSTe  | `naive`, `replay` | `resnet18`, `wide_resnet50_2` | both | `--pretrained` (model-default batch size and epochs) |

All runs: `--eval-level both --pixel-threshold 0.5 --pixel-threshold-mode train-quantile
--pixel-threshold-quantile 0.95 --replay-buffer-fraction 0.2 --n-runs 1 --master-seed 42`
(per-run seed `191664963`), `--device cuda:0`. CFA and PaDiM runs selected the
curriculum with `--ordering-mode`, PaSTe runs with `--ordering-file`; with
`--inclad-bench` use `--ordering-mode` for all of them.

### More examples

```python
from pyclad.data.datasets.inclad_md_dataset import load_inclad_md

# Only BTech and DAGM categories. An explicit category list also sets the task order.
dataset = load_inclad_md(
    ordering="easy_to_hard",
    categories=[
        "btech__03", "btech__01", "btech__02",
        "dagm__Class4", "dagm__Class9",
    ],
    resize_to=(256, 256),
)

# Paths only (no image loading)
dataset = load_inclad_md(ordering="hard_to_easy", data_mode="paths")
```
