# ReplayCAD in pyCLAD

pyCLAD contains an optional, clean-room implementation of the mechanism from
*ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection*.  The
implementation keeps existing strategies and dependencies unchanged. Diffusion
and SAM are imported only when `--strategy replaycad` is selected.

## The two ReplayCAD configurations of the InCLAD paper

The paper reports ReplayCAD runs made with two different configurations. They
use different runners, presets and hyperparameters, so their numbers differ
for the same scenario and ordering.

| | A: per-dataset presets | B: `inclad-cr` |
|---|---|---|
| Paper tables | baseline table (Table 3), per-ordering baseline table in the appendix, ReplayCAD row of the resource table | appendix "ReplayCAD Configuration and Results" (configuration and results tables) |
| Runs | 15: MVTec, VisA, BTAD, MPDD, DAGM x 3 orderings (no InCLAD-MD) | 17: the same 15 plus InCLAD-MD x 2 orderings |
| Runner | `run_continual_visual_ad_resumable.py` (MVTec, VisA, DAGM), `run_continual_visual_ad.py` (BTAD, MPDD) | `run_replaycad_cr.py` (single datasets), `run_replaycad_cr_resumable.py` (InCLAD-MD) |
| Preset (`replaycad_config.preset`) | `paper-mvtec`, `paper-visa`, `inclad-ldm-256` | `inclad-cr-v1-ldm-256` |
| Generator | LDM-256 (MVTec, BTAD, MPDD, DAGM), Stable Diffusion 1.5 at 512 px (VisA) | LDM-256 for every dataset |
| Compression steps | 18,000 | 6,000 |
| Replay images per earlier task | 700 | 200 |
| DDIM steps at generation | 40 | 50 |
| Learning rates (semantic / spatial) | 0.005 / 5e-5 | 0.08 / 8e-4 |
| Precision of the trainable conditions | bfloat16 | float32, frozen diffusion model under bfloat16 autocast |
| Mask augmentation at generation | on | off |
| Pixel threshold mode | `train-quantile` (0.995) | `fixed` (`train-quantile` is refused) |

Both use PatchCore on pretrained ResNet18 features at 256x256, 20 semantic
tokens, 5 SAM ViT-H masks per task, guidance scale 10, compression batch 16,
generation batch 8, cached replay images, `--eval-level both`, master seed 42
(per-run seed 191664963), one run per scenario and ordering.

The library defaults of the base runner (20,000 compression steps for MVTec,
30,000 for VisA, 800 replay images, 50 DDIM steps, float32) are used by
neither configuration.

## Configuration A: runs behind Table 3

### Command

The recorded `cli_args` of all 15 runs differ only in `--benchmark`,
`--root` and `--ordering-mode` (and in local paths). MVTec, VisA and DAGM were
run with the resumable runner:

```bash
python examples/clvad/run_continual_visual_ad_resumable.py \
    --model patchcore \
    --strategy replaycad \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --inclad-bench \
    --ordering-mode easy_to_hard \
    --resize 256 \
    --backbone resnet18 \
    --pretrained \
    --eval-level both \
    --pixel-threshold-mode train-quantile \
    --pixel-threshold-quantile 0.995 \
    --replaycad-mask-backend sam \
    --replaycad-sam-checkpoint /models/sam_vit_h_4b8939.pth \
    --replaycad-sam-model-type vit_h \
    --replaycad-masks-per-concept 5 \
    --replaycad-compression-steps 18000 \
    --replaycad-compression-batch-size 16 \
    --replaycad-inference-steps 40 \
    --replaycad-generation-batch-size 8 \
    --replaycad-samples-per-concept 700 \
    --replaycad-dtype bfloat16 \
    --replaycad-cache-generated \
    --replaycad-artifact-dir /artifacts/replaycad__mvtec__easy_to_hard \
    --device cuda:0 \
    --replaycad-device cuda:0 \
    --master-seed 42 \
    --n-runs 1 \
    --output-dir /results/replaycad/replaycad__mvtec__easy_to_hard \
    --resume-dir /checkpoints/replaycad__mvtec__easy_to_hard \
    --resume
```

BTAD (`--benchmark btech`) and MPDD (`--benchmark mpdd`) were run with
`examples/clvad/run_continual_visual_ad.py` and the same flags without
`--resume-dir` and `--resume`. Each run has its own `--replaycad-artifact-dir`.
With checkpointing (on by default in the resumable runner),
`--replaycad-artifact-dir` and `--replaycad-cache-generated` are required, so
that a requeued job finds the conditions and replay images of the tasks it has
already learned.

### Recorded configuration

The runner picks the preset from `--benchmark`. The `replaycad_config` blocks
recorded in the 15 result files are rebuilt exactly by
`build_replaycad_config` from the recorded `cli_args`:

- `paper-mvtec`: `CompVis/ldm-text2im-large-256` at 256 px, condition width
  1280, mask MLP `128 -> 200`, uniform timestep sampling, initializer word
  "screw"; full-frame masks for carpet, grid, leather, tile and wood,
  corner-background masks for metal_nut and screw;
- `paper-visa`: `stable-diffusion-v1-5/stable-diffusion-v1-5` at 512 px,
  condition width 768, mask MLP `128 -> 192`, cubic timestep sampling,
  initializer word "sculpture"; replay images are resized to 256 px before
  PatchCore training;
- `inclad-ldm-256` (BTAD, MPDD, DAGM): the LDM-256 setup of `paper-mvtec` with
  the initializer word "object"; full-frame masks for DAGM Class8.

Shared values: 20 semantic tokens, 5 SAM ViT-H masks per task, mask
augmentation at generation (rotation up to 180 degrees, shift up to 0.2 of the
image), 18,000 compression steps at batch 16, learning rates 0.005 (semantic)
and 5e-5 (spatial), 40 DDIM steps, guidance 10, generation batch 8, 700 replay
images per earlier task, `torch_dtype=bfloat16`, cached replay images. PatchCore:
pretrained ResNet18, 256x256 input, coreset ratio 0.1, embedding dimension
1024, one nearest neighbour.

`examples/clvad/camera_ready/build_baseline_table.py` selects ReplayCAD runs
with a ResNet18 backbone and `replaycad_compression_steps == 18000`, which
separates configuration A from `inclad-cr`.

In this configuration `--replaycad-dtype` also sets the precision of the
trainable semantic and spatial conditions. Configuration B keeps them in
float32 (see below).

## Configuration B: `inclad-cr` (appendix runs)

`inclad-cr` is defined in `src/pyclad/strategies/replaycad/inclad_cr.py` and
activated by two wrappers that leave the base runners unchanged:

- `examples/clvad/run_replaycad_cr.py` wraps `run_continual_visual_ad.py`. It
  was used for the 15 single-dataset runs. All orderings of a dataset share one
  artifact bank, `--replaycad-bank-dir DIR` -> `DIR/<benchmark>`. Run
  `easy_to_hard` first: it compresses every task of the dataset. The
  `hard_to_easy` and `random` runs then load the stored conditions and only fit
  and evaluate PatchCore. Two runs must never write the same bank at the same
  time.
- `examples/clvad/run_replaycad_cr_resumable.py` wraps
  `run_continual_visual_ad_resumable.py`, the runner that loads InCLAD-MD. It
  was used for the two InCLAD-MD runs. Each run keeps its own bank in
  `--replaycad-artifact-dir` and compresses its 27 tasks itself.
  `--patchcore-gpu-knn` (on by default in this wrapper) runs PatchCore's exact
  nearest-neighbour search on the GPU (`pyclad.models.patchcore.gpu_knn`); the
  single-dataset runs used the scikit-learn search.

Unless given explicitly, both wrappers add `--replaycad-cache-generated`,
`--replaycad-samples-per-concept 200` and `--config-json '{"threshold": 0.0}'`.
The fixed PatchCore threshold only affects thresholded image metrics (F1),
which the paper does not report.

### Commands

Single dataset (`--benchmark` is `mvtec`, `visa`, `btech`, `mpdd` or `dagm`;
run `easy_to_hard` before `hard_to_easy` and `random`):

```bash
python examples/clvad/run_replaycad_cr.py \
    --model patchcore \
    --strategy replaycad \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --inclad-bench \
    --ordering-mode easy_to_hard \
    --resize 256 \
    --backbone resnet18 \
    --pretrained \
    --eval-level both \
    --replaycad-mask-backend sam \
    --replaycad-sam-checkpoint /models/sam_vit_h_4b8939.pth \
    --replaycad-bank-dir /banks/replaycad-cr \
    --replaycad-compression-steps 6000 \
    --replaycad-local-files-only \
    --device cuda:0 \
    --replaycad-device cuda:0 \
    --master-seed 42 \
    --n-runs 1 \
    --output-dir /results/replaycad-cr/mvtec/easy_to_hard
```

InCLAD-MD (`--ordering-mode` is `easy_to_hard` or `hard_to_easy`; `--root` is
the directory that holds all five datasets):

```bash
python examples/clvad/run_replaycad_cr_resumable.py \
    --model patchcore \
    --strategy replaycad \
    --benchmark inclad-md \
    --root /data/visual_datasets \
    --inclad-bench \
    --ordering-mode easy_to_hard \
    --resize 256 \
    --backbone resnet18 \
    --pretrained \
    --eval-level both \
    --replaycad-mask-backend sam \
    --replaycad-sam-checkpoint /models/sam_vit_h_4b8939.pth \
    --replaycad-artifact-dir /banks/replaycad-cr-md/easy_to_hard \
    --replaycad-compression-steps 6000 \
    --replaycad-local-files-only \
    --patchcore-gpu-knn \
    --device cuda:0 \
    --replaycad-device cuda:0 \
    --master-seed 42 \
    --n-runs 1 \
    --output-dir /results/replaycad-cr/inclad-md/easy_to_hard \
    --resume-dir /checkpoints/replaycad-cr \
    --resume
```

`--replaycad-local-files-only` loads `CompVis/ldm-text2im-large-256` from the
local Hugging Face cache only; download it once beforehand, for example with
`python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('CompVis/ldm-text2im-large-256')"`.

### Recorded configuration

All 17 result files record `preset="inclad-cr-v1-ldm-256"` and these values:

- generator `CompVis/ldm-text2im-large-256` at 256 px for every dataset,
  condition width 1280, mask MLP `128 -> 200`, uniform timestep sampling;
- textual inversion: 6,000 steps at batch 16, AdamW with learning rates 0.08
  (semantic tokens) and 8e-4 (mask MLP), weight decay 0.01, the 81 prompt
  templates of the authors' code, initializer word "screw";
- trainable conditions in float32 (`torch_dtype="float32"`; the profile refuses
  any other value) with the frozen diffusion model under bfloat16 autocast;
- 5 SAM ViT-H masks per task, prompted with the four image corners as
  background; full-frame masks for MVTec carpet, grid, leather, tile and wood,
  BTAD 02, every DAGM class, and whenever the foreground covers less than 2 %
  or more than 98 % of the image; no mask augmentation;
- generation: 50 DDIM steps, guidance 10, batch 8, 200 replay images per
  earlier task, cached as soon as a task is compressed; replay of tasks whose
  training images are grey is converted to grey;
- PatchCore threshold 0.0, `pixel_threshold_mode="fixed"`.

InCLAD-MD uses the same values; each of its 27 tasks keeps the mask rule of its
source dataset.

## Installation

Use Linux with a CUDA-capable GPU for benchmark runs.

ReplayCAD requires PyTorch 2.6 or newer. Its MVTec LDM dependency contains a
legacy PyTorch `.bin` checkpoint, and current Transformers versions correctly
refuse to load that format with older PyTorch releases.

```bash
pip install -e ".[replaycad]"
```

Download a Segment Anything checkpoint matching the selected model type, for
example `sam_vit_h_4b8939.pth`. Diffusion weights are loaded through Hugging
Face Diffusers. Stable Diffusion weights may require accepting the model's
license and authenticating with Hugging Face.

The original ReplayCAD research repository has no declared top-level license,
so its source is not vendored here. This implementation follows the published
algorithm using Diffusers and Segment Anything APIs.

## Base runner and its defaults

The base path (`run_continual_visual_ad.py` or the resumable runner with
`--strategy replaycad`) supports:

- datasets: MVTec AD, VisA, BTAD (`btech`), MPDD and DAGM through
  `InCLADBenchDataset` (`--inclad-bench` is required);
- orderings: `easy_to_hard`, `hard_to_easy` or `random`;
- detector: PatchCore with a pretrained `wide_resnet50_2` (default, the
  released-paper PatchCore profile) or `resnet18` backbone (the InCLAD
  comparison);
- presets chosen from `--benchmark` as listed under configuration A;
- stored state: semantic embeddings, spatial MLP and 1-10 masks per concept.

Without the overrides of configuration A it uses the library defaults:
20,000 compression steps for MVTec and 30,000 for VisA, 800 replay images per
earlier concept, 50 DDIM steps and float32. A minimal MVTec call with these
defaults is:

```bash
python examples/clvad/run_continual_visual_ad.py \
    --model patchcore \
    --strategy replaycad \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --inclad-bench \
    --ordering-mode easy_to_hard \
    --replaycad-sam-checkpoint /models/sam_vit_h_4b8939.pth \
    --device cuda \
    --replaycad-device cuda \
    --n-runs 1
```

For VisA, use `--benchmark visa --root /data/VisA`. Existing strategies
continue to use `ConceptIncrementalScenario`; only ReplayCAD uses
`ConceptAwareScenario` so it can name per-concept artifacts.

Pass `--eval-level both` to write image-level ROC-AUC, F1 and AP together
with pixel-level ROC-AUC, AP, AUPRO, F1, Dice and IoU. Thresholded pixel
metrics can use seen normal training maps with
`--pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.995`, as
in configuration A. The exact test-sample metadata from the InCLAD manifest is
retained so masks remain aligned with anomaly maps.

Training the semantic and spatial conditions directly in `float16` can
overflow during long compression schedules; use `bfloat16` (configuration A)
or the float32 conditions of `inclad-cr` instead.

The VisA paper contains an inconsistent MLP size of `128 -> 196`: its output
cannot be reshaped into 768-dimensional conditioning tokens. The released
training implementation uses `128 -> 192` for its main Stable Diffusion
variant, producing 32 spatial tokens, so pyCLAD uses that executable value.

## Short infrastructure smoke test

The following is not a publishable ReplayCAD experiment. It verifies the
dataset, artifact, generation and PatchCore plumbing with only two concepts and
full-frame masks:

```bash
python examples/clvad/run_continual_visual_ad.py \
    --model patchcore \
    --strategy replaycad \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --inclad-bench \
    --categories bottle cable \
    --max-train 4 \
    --replaycad-mask-backend full-frame \
    --replaycad-compression-steps 2 \
    --replaycad-samples-per-concept 2 \
    --replaycad-masks-per-concept 1 \
    --device cuda \
    --replaycad-device cuda \
    --replaycad-dtype float16
```

The test set is intentionally not truncated: each evaluated concept must
contain both normal and anomalous images for ROC-AUC to be defined.

Never report `full-frame` results as full ReplayCAD. Use SAM or masks produced
with the published mask-selection procedure for benchmark results.

## Precomputed masks

For reproducible reruns, SAM masks can be generated once and stored as:

```text
/masks/
  bottle/
    000.png
    001.png
  cable/
    000.png
```

Use either one reusable mask or exactly one mask per training image:

```bash
--replaycad-mask-backend precomputed \
--replaycad-mask-root /masks
```

Neither configuration of the paper used precomputed masks; both ran SAM.

## Artifacts and memory accounting

Without `--replaycad-artifact-dir`, a base-runner run writes a separate
artifact directory next to its result JSON; with it, the directory is
`<artifact-dir>/<result stem>`. An `inclad-cr` single-dataset run uses the
shared bank `<bank-dir>/<benchmark>` instead. The layout is:

```text
patchcore_replaycad_..._artifacts/
  replaycad_config.json
  manifest.json
  concepts/
    bottle/
      condition.pt
      masks.npz
      metadata.json
```

`conditional_storage_bytes` counts the condition, masks, and small metadata
file. It does not count the shared frozen diffusion checkpoint. Generated
replay images are ephemeral by default. If `--replaycad-cache-generated` is
enabled, cached images remain excluded from conditional-memory reporting and
must be reported separately.

## Reproducibility note

The ReplayCAD paper specifies 20 semantic tokens, generator resolutions,
condition dimensions, 800 replay images and 50 DDIM steps. Its released scripts
select different compression checkpoints for individual categories. The base
runner therefore uses explicit, stable defaults of 20,000 compression steps for
MVTec and 30,000 for VisA. The InCLAD paper overrides them: configuration A
uses one value, 18,000 steps, with 700 replay images and 40 DDIM steps for
every dataset; `inclad-cr` uses 6,000 steps and 200 replay images. Record the
resulting protocol whenever these values are overridden.

## Tests

```bash
pytest -q \
  tests/models/replaycad/test_replaycad.py \
  tests/models/replaycad/test_inclad_cr.py \
  tests/models/replaycad/test_inclad_cr_md.py \
  tests/resume/methods/test_replaycad.py
```

The tests run on CPU with stand-in diffusion backends and no downloads. They
cover the base strategy and artifacts, the `inclad-cr` profile and both
wrappers, the InCLAD-MD path, and the checkpoint round trip of the resumable
`ResumableReplayCADStrategy`.
`test_inclad_cr_md.py::test_torch_knn_matches_sklearn_brute_force` is the
check behind the GPU nearest-neighbour statement in the appendix: on a
5,000-row test bank the exact torch search returns the same neighbour indices
as scikit-learn, with distances within 1e-5.
