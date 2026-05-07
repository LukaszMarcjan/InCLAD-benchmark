# InCLAD-Bench Dataset

InCLAD-Bench is a continual learning benchmark for visual anomaly detection.
It aggregates five widely-used industrial anomaly detection datasets into a
unified evaluation protocol with reproducible category orderings.

We leverage the pyCLAD library for continual scenario orchestration. The source code is available in the repository. Install it directly:

```bash
pip install -e .
```

#### Optional dependencies

The core pyCLAD installation covers tabular and time-series anomaly detection.
For visual models, install the vision stack separately:

```bash
pip install torch torchvision pytorch-lightning
```

### Getting started

pyCLAD is built upon a few core concepts:

- **Scenario**: a continual scenario defines the data stream so that it reflects different real-life conditions and what
  are the challenges faced by continual strategy.
- **Strategy**: a strategy is a way to manage model updates. Continual strategy is responsible for how, when, and with
  which data models should be updated. Its aim is to introduce knowledge retention while keeping the ability to adapt.
- **Model**: a model is a machine learning model used for anomaly detection. Models are often leveraged by continual
  strategies that add an additional layer of managing model updates.
- **Dataset**: a dataset is a collection of data used for training and evaluation of the model.
- **Metrics**: a metric is a way to evaluate the performance of the model.
- **Callbacks**: a callback is a function that is called at specific points during the scenario. Callbacks are
  useful for monitoring the process, calculating metrics, and more.

## Visual Anomaly Detection

We implement visual anomaly detection methods in pyCLAD — image-based models, pixel-level metrics, continual learning strategies for vision, and ready-to-use visual benchmarks.

### Supported models

| Model | CL variant | Description |
|-------|:----------:|-------------|
| PatchCore | PatchCore-CL | Memory-bank approach using deep features from a pretrained backbone |
| PaDiM | PaDiM-CL | Patch-level Gaussian distribution modelling |
| EfficientAD | — | Lightweight student-teacher network |
| FastFlow | — | Normalizing flow on deep features |
| CFA | CFA-CL | Coupled hypersphere-based feature adaptation |
| GANomaly | — | GAN-based reconstruction and encoding model |
| STFPM | — | Student-teacher feature pyramid matching |
| RD4AD | — | Reverse distillation for anomaly detection |
| PaSTe | — | Patch-based student-teacher with multi-layer distillation |

CL variants (e.g. PatchCore-CL, PaDiM-CL) extend the base model with memory management for continual learning and can be used with vision-specific continual strategies.

### Strategies for visual anomaly detection

Beyond the general-purpose strategies (Naive, Cumulative, Replay), pyCLAD provides:

| Strategy | Description |
|----------|-------------|
| `SingleTaskExpertStrategy` (STE) | Trains a fresh model on each concept; used to establish single-task upper bounds |
| `PatchCoreCLStrategy` | Manages PatchCore-CL memory across concepts |
| `PaDiMCLStrategy` | Manages PaDiM-CL covariance matrices across concepts |
| `CFACLStrategy` | Manages CFA-CL memory across concepts |

Replay buffer budgeting supports three modes: `fixed` (fixed total size), `avg-concept-fraction` (fraction of average concept size), and `per-concept-budget` (fixed budget multiplied by number of concepts seen).

### Metrics

**Image-level:**

| Metric | Class |
|--------|-------|
| ROC-AUC | `RocAuc` |
| F1-Score | `F1Score` |

**Pixel-level** (require ground-truth anomaly masks):

| Metric | Class |
|--------|-------|
| Pixel ROC-AUC | `PixelRocAuc` |
| Pixel F1-Score | `PixelF1Score` |
| Pixel IoU | `PixelIoU` |
| Pixel Dice Score | `PixelDiceScore` |

**Continual metrics** (derived from the concept-level result matrix):

| Metric | Class | Description |
|--------|-------|-------------|
| Continual Average | `ContinualAverage` | Mean over all evaluated (concept, training-step) pairs |
| Diagonal Average | `DiagonalAverage` | Mean of diagonal entries — performance right after training each concept |
| Backward Transfer | `BackwardTransfer` | Change in performance on previously learned concepts |
| Forward Transfer | `ForwardTransfer` | Change in performance on future concepts before training on them |

Pixel-level evaluation is handled by `VisualPixelConceptMetricCallback`, which resolves ground-truth masks from the registered dataset and supports configurable thresholding modes (`fixed`, `quantile`).

---

## Our scenarios


| Dataset  | Categories | License         |
|----------|:----------:|-----------------|
| MVTec AD | 15         | CC BY-NC-SA 4.0 |
| BTech    | 3          | CC BY-SA        |
| DAGM     | 10         | CC BY 4.0       |
| MPDD     | 6          | CC BY-NC-SA 4.0 |
| VisA     | 12         | CC BY 4.0       |

Each dataset is provided with three category orderings: `easy_to_hard`, `hard_to_easy`, and `random` (seeded for reproducibility).
Benchmark manifests are published on HuggingFace at [`anonmllab/inclad-bench`](https://huggingface.co/datasets/anonmllab/inclad-bench) and downloaded automatically at runtime — only the raw image archives need to be downloaded manually.

### Quick start

```python
from pyclad.data.datasets.inclad_bench_dataset import InCLADBenchDataset

dataset = InCLADBenchDataset(
    benchmark="mvtec",          # btech | dagm | mpdd | mvtec | visa
    ordering="easy_to_hard",    # easy_to_hard | hard_to_easy | random
    root="/data/mvtec_ad",
    resize_to=(224, 224),
    color_mode="rgb",
)
# dataset is a ConceptsDataset — ready to pass to any pyCLAD scenario
```

Or via the CLI:

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

**[Full InCLAD-Bench documentation](docs/inclad-bench-dataset.md)** — dataset setup, directory layouts, environment variables, and manifest format.

### InCLAD-MD: multi-dataset benchmark

**InCLAD-MD** extends InCLAD-Bench into a single cross-dataset curriculum.
It combines 27 selected categories from all five source datasets into one continuous sequence,
with dataset blocks ordered by difficulty (BTech → MPDD → DAGM → VisA → MVTec).

Manifests are published on HuggingFace at [`anonmllab/inclad-bench`](https://huggingface.co/datasets/anonmllab/inclad-bench) and loaded automatically at runtime.

Categories are selected as **2 easiest + 2 median + 2 hardest** per dataset (BTech uses all 3), based on STE ROC-AUC rankings:

| Source dataset | Selected categories |
|----------------|---------------------|
| BTech (3)      | `btech__03`, `btech__01`, `btech__02` |
| DAGM (6)       | `dagm__Class4`, `dagm__Class9`, `dagm__Class7`, `dagm__Class3`, `dagm__Class5`, `dagm__Class8` |
| MPDD (6)       | `mpdd__connector`, `mpdd__bracket_brown`, `mpdd__metal_plate`, `mpdd__bracket_white`, `mpdd__tubes`, `mpdd__bracket_black` |
| MVTec (6)      | `mvtec__bottle`, `mvtec__leather`, `mvtec__toothbrush`, `mvtec__carpet`, `mvtec__screw`, `mvtec__pill` |
| VisA (6)       | `visa__pcb4`, `visa__chewinggum`, `visa__fryum`, `visa__pcb2`, `visa__macaroni2`, `visa__capsules` |

Category names are prefixed with the source dataset (`<dataset>__<category>`) to avoid collisions.

Available orderings: `easy_to_hard`, `hard_to_easy`.

```python
from pyclad.data.datasets.inclad_md_dataset import InCLADMDDataset

dataset = InCLADMDDataset(
    ordering="easy_to_hard",          # easy_to_hard | hard_to_easy
    roots={                           # one root per source dataset
        "mvtec": "/data/mvtec_ad",
        "visa":  "/data/VisA",
        "btech": "/data/BTech_Dataset_transformed",
        "dagm":  "/data/DAGM_KaggleUpload",
        "mpdd":  "/data/MPDD",
    },
    resize_to=(224, 224),
    color_mode="rgb",
)
# dataset is a ConceptsDataset — ready to pass to any pyCLAD scenario
```

Roots can also be set via environment variables or the pyCLAD visual dataset registry — see the [InCLAD-Bench documentation](docs/inclad-bench-dataset.md) for details.

**[Full InCLAD-MD documentation](docs/inclad-md-dataset.md)**
