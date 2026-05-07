# InCLAD-Bench Dataset

InCLAD-Bench is a continual learning benchmark for visual anomaly detection.
It aggregates five widely-used industrial anomaly detection datasets into a
unified evaluation protocol with reproducible category orderings.

The benchmark manifests are published on HuggingFace:
[`anonmllab/inclad-bench`](https://huggingface.co/datasets/anonmllab/inclad-bench)

> **Important:** The HuggingFace repository contains only manifests (metadata: file
> paths, labels, category orderings). **Image files are not included** and must be
> downloaded separately from each dataset's original source. For each benchmark you
> intend to use, download the images, extract the archive, and place the resulting
> folder in a location pyCLAD can find — see [Setup: downloading images](#setup-downloading-images) below.

---

## Overview

The benchmark covers five source datasets spanning industrial inspection
scenarios with diverse object categories and defect types:

| Dataset  | Categories | Train samples | Test samples | Normal (test) | Anomalous (test) | License         |
|----------|:----------:|:-------------:|:------------:|:-------------:|:----------------:|-----------------|
| MVTec AD | 15         | 3,629         | 1,725        | 467           | 1,258            | CC BY-NC-SA 4.0 |
| BTech    | 3          | 1,799         | 741          | 451           | 290              | CC BY-SA        |
| DAGM     | 10         | 7,004         | 8,050        | 6,996         | 1,054            | CC BY 4.0       |
| MPDD     | 6          | 888           | 458          | 176           | 282              | CC BY-NC-SA 4.0 |
| VisA     | 12         | 8,659         | 2,162        | 962           | 1,200            | CC BY 4.0       |

Each benchmark is provided with three category orderings:

- **`easy_to_hard`** — categories ordered from the easiest to the hardest (based on single-task anomaly detection performance)
- **`hard_to_easy`** — reverse of the above
- **`random`** — randomly shuffled, seeded for reproducibility

---

## Benchmark Details

### MVTec AD

**Download:** https://www.mvtec.com/company/research/datasets/mvtec-ad
**License:** CC BY-NC-SA 4.0

15 categories of industrial objects and textures. Each category contains
normal training images and a test set with both normal and defective samples,
accompanied by pixel-level ground-truth masks for all anomalous images.

Categories: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
`leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`,
`wood`, `zipper`

Expected directory structure after download:

```
mvtec_ad/
  bottle/
    train/good/
    test/good/
    test/broken_large/
    test/broken_small/
    test/contamination/
    ground_truth/broken_large/
    ground_truth/broken_small/
    ground_truth/contamination/
  cable/
  ...
  zipper/
```

---

### BTech

**Download:** https://github.com/pankajmishra000/VT-ADL (Dataset Download section)
**License:** CC BY-SA

3 categories of industrial components (numbered `01`, `02`, `03`).
Images are stored in `.bmp` format. Normal samples are labelled `ok`,
defective samples `ko`.

Categories: `01`, `02`, `03`

Expected directory structure after download:

```
BTech_Dataset_transformed/
  01/
    train/ok/
    test/ok/
    test/ko/
    ground_truth/ko/
  02/
  03/
```

---

### DAGM

**Download:** https://zenodo.org/records/12750201
**License:** CC BY 4.0

10 synthetic texture classes with artificially generated defects.
Classes are named `Class1` through `Class10`. Ground-truth masks for
defective samples are stored in a `Label/` subdirectory within each
test split, with filenames suffixed `_label`.

Categories: `Class1`, `Class2`, ..., `Class10`

Expected directory structure after download:

```
DAGM_KaggleUpload/
  Class1/
    Train/
    Test/
    Test/Label/
  Class2/
  ...
  Class10/
```

---

### MPDD

**Download:** https://github.com/stepanje/MPDD
**License:** CC BY-NC-SA 4.0

6 categories of metal parts with various surface defects. Follows the
same MVTec-style directory layout with `train/good/`, `test/`, and
`ground_truth/` subdirectories.

Categories: `bracket_black`, `bracket_brown`, `bracket_white`,
`connector`, `metal_plate`, `tubes`

Expected directory structure after download:

```
MPDD/
  connector/
    train/good/
    test/good/
    test/parts_mismatch/
    test/bend_and_parts_mismatch/
    ground_truth/parts_mismatch/
    ground_truth/bend_and_parts_mismatch/
  bracket_black/
  ...
  tubes/
```

---

### VisA

**Download:** https://github.com/amazon-science/spot-diff
**License:** CC BY 4.0

12 categories of printed circuit boards and food products. VisA uses a
different directory layout from MVTec — images are stored under
`Data/Images/Normal/` and `Data/Images/Anomaly/`, with masks under
`Data/Masks/Anomaly/`. A split CSV file (`split_csv/1cls.csv`) is
included in the download and must be present in the root directory.

Categories: `candle`, `capsules`, `cashew`, `chewinggum`, `fryum`,
`macaroni1`, `macaroni2`, `pcb1`, `pcb2`, `pcb3`, `pcb4`, `pipe_fryum`

Expected directory structure after download:

```
VisA/
  split_csv/
    1cls.csv
  candle/
    Data/Images/Normal/
    Data/Images/Anomaly/
    Data/Masks/Anomaly/
  capsules/
  ...
  pipe_fryum/
```

---

## Setup: downloading images

For each benchmark you want to use:

1. Download the archive from the link in the benchmark's section above.
2. Extract it — the resulting folder (e.g. `mvtec_ad/`, `BTech_Dataset_transformed/`) must match the expected directory structure shown in that section.
3. Tell pyCLAD where to find it — see Step 2 below.

The benchmark manifests (category orderings, labels, relative file paths) are
downloaded automatically from HuggingFace at runtime; no manual action is needed
for them.

---

## Using InCLAD-Bench with pyCLAD

### Step 1 — Download and place the source images

Download the archives from the links above, extract them, and place the resulting
folders somewhere on your filesystem. The folder structure must match the layout
described in each benchmark's section above — pyCLAD resolves image files using
the relative paths stored in the HuggingFace manifest.

### Step 2 — Register dataset locations

Choose one of the following methods (checked in this order):

**Option A — explicit root path:**

```python
from pyclad.data.datasets.inclad_bench_dataset import InCLADBenchDataset

dataset = InCLADBenchDataset(benchmark="mvtec", root="/data/mvtec_ad")
```

**Option B — per-benchmark environment variable:**

```bash
export PYCLAD_MVTEC_ROOT=/data/mvtec_ad
export PYCLAD_VISA_ROOT=/data/VisA
```

**Option C — shared root with auto-detection:**

pyCLAD looks for known subdirectory names inside `PYCLAD_VISUAL_DATASETS_ROOT`:

```bash
export PYCLAD_VISUAL_DATASETS_ROOT=/data
# pyCLAD will look for /data/mvtec_ad, /data/mvtec, /data/mvtec_anomaly_detection, etc.
```

Recognised subdirectory names per benchmark:

| Benchmark | Recognised subdirectory names |
|-----------|-------------------------------|
| `mvtec`   | `mvtec_ad`, `mvtec`, `mvtec_anomaly_detection` |
| `btech`   | `BTech_Dataset_transformed`, `btech_dataset_transformed`, `btech` |
| `dagm`    | `DAGM_KaggleUpload`, `dagm_kaggleupload`, `dagm` |
| `mpdd`    | `MPDD`, `mpdd` |
| `visa`    | `visa`, `VisA` |

**Option D — registry JSON:**

Edit `src/pyclad/data/datasets/visual_datasets/registry.json`:

```json
{
    "mvtec": "/data/mvtec_ad",
    "btech": "/data/BTech_Dataset_transformed",
    "dagm":  "/data/DAGM_KaggleUpload",
    "mpdd":  "/data/MPDD",
    "visa":  "/data/VisA"
}
```

### Step 3 — Load the dataset

```python
from pyclad.data.datasets.inclad_bench_dataset import InCLADBenchDataset

dataset = InCLADBenchDataset(
    benchmark="mvtec",          # one of: btech, dagm, mpdd, mvtec, visa
    ordering="easy_to_hard",    # one of: easy_to_hard, hard_to_easy, random
    root="/data/mvtec_ad",
    resize_to=(224, 224),
    color_mode="rgb",
)

# dataset is a ConceptsDataset — ready to pass to any pyCLAD scenario
print(dataset.name())
print([c.name for c in dataset.train_concepts()])
```

### Running experiments via the CLI

```bash
python examples/clvad/run_continual_visual_ad.py \
    --model patchcore \
    --strategy naive \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --inclad-bench \
    --ordering-mode easy_to_hard \
    --resize 224
```

---

## Manifest format

Each HuggingFace split is a flat table with one row per image sample.
The schema is identical to the local manifest CSV files in
`src/pyclad/data/datasets/visual_datasets/manifests/`:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique identifier, e.g. `mvtec:000042` |
| `source_dataset` | string | Benchmark name: `mvtec`, `btech`, etc. |
| `category` | string | Object/texture category within the benchmark |
| `category_order` | int | 1-based position of the category in the ordering |
| `split` | string | `train` or `test` |
| `image_relpath` | string | Path to the image, relative to the dataset root |
| `mask_relpath` | string | Path to the ground-truth mask (empty for normal samples) |
| `image_label` | int | `0` = normal, `1` = anomalous |
| `defect_type` | string | Defect class name (empty for normal samples) |
| `ordering_name` | string | `easy_to_hard`, `hard_to_easy`, or `random` |
| `ordering_master_seed` | int | Master seed used to derive the ordering |
| `ordering_seed` | int | Per-run seed |
| `source_homepage` | string | URL of the original dataset |
| `source_license` | string | License of the original dataset |
