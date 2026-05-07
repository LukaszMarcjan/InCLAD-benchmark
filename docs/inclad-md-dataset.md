# InCLAD-MD Dataset

> **Anonymous review note:** author and maintainer metadata has been redacted
> in this artifact for double-blind review. Required open-source license notices
> are retained, and full attribution will be restored in the de-anonymized
> public release.

InCLAD-MD is a multi-dataset sequential benchmark for continual visual anomaly detection.
It combines selected categories from all five InCLAD-Bench source datasets (BTech, DAGM,
MPDD, MVTec AD, VisA) into a single curriculum sequence.

The benchmark manifests are published on HuggingFace:
[`anonmllab/inclad-md`](https://huggingface.co/datasets/anonmllab/inclad-md)

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
avoid collisions.

## Orderings

| Split | Description |
|-------|-------------|
| `easy_to_hard` | Dataset blocks ordered easiest-first, categories within each block easiest-first |
| `hard_to_easy` | Exact reverse of `easy_to_hard` |
| `random`       | Uniformly shuffled, seeded for reproducibility |

Dataset block order in `easy_to_hard`: BTech → MPDD → DAGM → VisA → MVTec

---

## Manifest schema

Same 15-column schema as InCLAD-Bench, extended with `source_category`:

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
| `ordering_name` | `easy_to_hard`, `hard_to_easy`, or `random` |
| `ordering_master_seed` | Master seed |
| `ordering_seed` | Per-split derived seed |
| `source_homepage` | URL of the source dataset |
| `source_license` | License of the source dataset |

`image_relpath` is **relative to the source dataset's root**, not a single shared root.
The loader resolves each image as `roots[source_dataset] / image_relpath`.

---

## Generating and uploading manifests to HuggingFace

### Step 1 — Generate the manifests

```bash
python examples/visual_models/build_inclad_md_manifest.py \
    --output-dir /tmp/inclad_md_hf
```

This reads the ordering files from `examples/visual_models/multidataset_ordering_v1/`
and the per-benchmark manifests from
`src/pyclad/data/datasets/visual_datasets/manifests/`, and writes:

```
/tmp/inclad_md_hf/
  easy_to_hard.csv
  hard_to_easy.csv
  random.csv
```

### Step 2 — Create the HuggingFace repository

Create a new dataset repository at https://huggingface.co/new-dataset:

- **Owner**: `anonmllab`
- **Name**: `inclad-md`
- **License**: MIT

### Step 3 — Write the dataset card (README.md)

Create `README.md` in the repo root with the YAML front matter that defines
the splits:

```yaml
---
license: mit
task_categories:
  - image-classification
tags:
  - anomaly-detection
  - continual-learning
  - industrial-anomaly-detection
configs:
  - config_name: default
    data_files:
      - split: easy_to_hard
        path: easy_to_hard.csv
      - split: hard_to_easy
        path: hard_to_easy.csv
      - split: random
        path: random.csv
---
```

### Step 4 — Upload via huggingface_hub

```bash
pip install huggingface_hub
huggingface-cli login
```

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/tmp/inclad_md_hf",
    repo_id="anonmllab/inclad-md",
    repo_type="dataset",
)
```

Or push the CSV files individually:

```python
for split in ("easy_to_hard", "hard_to_easy", "random"):
    api.upload_file(
        path_or_fileobj=f"/tmp/inclad_md_hf/{split}.csv",
        path_in_repo=f"{split}.csv",
        repo_id="anonmllab/inclad-md",
        repo_type="dataset",
    )
```

---

## Using InCLAD-MD with pyCLAD

### Quick start

```python
from pyclad.data.datasets.inclad_md_dataset import InCLADMDDataset

dataset = InCLADMDDataset(
    ordering="easy_to_hard",
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

Roots for each source dataset are resolved in the same priority order as InCLAD-Bench:

1. `roots={"mvtec": "/data/mvtec_ad", ...}` parameter
2. Per-benchmark env vars: `PYCLAD_MVTEC_ROOT`, `PYCLAD_VISA_ROOT`
3. Shared env var `PYCLAD_VISUAL_DATASETS_ROOT` with auto-detected subfolders
4. Registry JSON `src/pyclad/data/datasets/visual_datasets/registry.json`

### CLI

```bash
python examples/clvad/run_continual_visual_ad.py \
    --model patchcore \
    --strategy naive \
    --benchmark inclad_md \
    --inclad-bench \
    --ordering-mode easy_to_hard \
    --resize 224
```

### More examples

```python
from pyclad.data.datasets.inclad_md_dataset import load_inclad_md

# Only BTech and DAGM categories
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

# Random ordering
dataset = load_inclad_md(ordering="random", resize_to=(224, 224))
```
