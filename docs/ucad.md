# UCAD integration

This repository includes a pyCLAD-native implementation of **UCAD** from
*Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt*
(AAAI 2024). The integration keeps the existing InCLAD model contracts:

- image prediction through `predict()`;
- pixel anomaly maps through `score_maps()`;
- the standard JSON result schema and image/pixel continual metrics;
- task-agnostic inference (the test concept name is never used for routing).

## Implemented method

Each training task adds one entry to a key-prompt-knowledge memory:

1. **Key**: frozen ViT patch features, reduced by farthest-point coreset
   sampling. Keys select the most similar learned task at inference.
2. **Prompt**: learned key/value prefixes for each prompted ViT attention
   layer. Prompt parameters are the only optimized model parameters.
3. **Knowledge**: prompted normal patch features, also reduced to a fixed-size
   coreset. Nearest-neighbor distance to the selected task knowledge produces
   image scores and pixel maps.

The default profile uses ViT-B/16 pretrained on ImageNet-21K, features
after zero-based block index 5 (configured as one-based `feature_layer=6`),
and 196 key and knowledge vectors per task. The ViT is always frozen.

This is a clean-room implementation of the method and paper equations. It does
not copy the research repository's patched `timm` source or its monolithic
runner. This keeps UCAD compatible with the current pyCLAD and `timm` APIs, but
also means exact numerical identity with the authors' environment is not
expected.

## Installation

```bash
pip install -e '.[ucad]'
```

The extra installs `timm` and Segment Anything. A SAM checkpoint is needed
only for the optional online SCL mode below; the paper configuration does not
use SAM.

## Configuration used in the InCLAD paper

The UCAD columns of the baseline table (Table 3), of the per-ordering baseline
table in the appendix, and the UCAD row of the resource table come from 17
runs: MVTec AD, VisA, BTAD (`btech`), MPDD and DAGM with the `easy_to_hard`,
`hard_to_easy` and `random` orderings, and InCLAD-MD (`inclad-md`) with
`easy_to_hard` and `hard_to_easy`. One run per scenario and ordering, master
seed 42 (per-run seed 191664963).

All 17 runs used the resumable runner
`examples/clvad/run_continual_visual_ad_resumable.py`, which wraps
`run_continual_visual_ad.py` and builds the checkpointable `ResumableUCAD` /
`ResumableUCADStrategy` from `pyclad.resume.methods.ucad`. Their recorded
`cli_args` differ only in `--benchmark`, `--ordering-mode` and (for InCLAD-MD)
`--root`:

```bash
python examples/clvad/run_continual_visual_ad_resumable.py \
  --model ucad \
  --strategy ucad \
  --benchmark mvtec \
  --root /data/mvtec_ad \
  --inclad-bench \
  --ordering-mode easy_to_hard \
  --ordering-metric roc-auc \
  --resize 224 \
  --backbone vit_base_patch16_224.augreg_in21k \
  --pretrained \
  --batch-size 8 \
  --epochs 25 \
  --eval-level both \
  --pixel-threshold-mode train-quantile \
  --pixel-threshold-quantile 0.995 \
  --master-seed 42 \
  --n-runs 1 \
  --device cuda:0 \
  --output-dir /results/ucad/ucad__mvtec__easy_to_hard \
  --resume-dir /checkpoints/ucad__mvtec__easy_to_hard \
  --resume
```

- `--benchmark` takes `mvtec`, `visa`, `btech`, `mpdd`, `dagm` or `inclad-md`;
  `--ordering-mode` takes `easy_to_hard`, `hard_to_easy` or `random`
  (InCLAD-MD has no `random`).
- The single-dataset runs left `--root` unset and found the data through the
  dataset-location lookup described in
  [inclad-bench-dataset.md](inclad-bench-dataset.md); `--root` selects the
  same data explicitly. For InCLAD-MD, `--root` is the directory that holds
  all five datasets (see [inclad-md-dataset.md](inclad-md-dataset.md)).
- No `--config-json` was passed. The model therefore uses the `UCADConfig`
  defaults, which the result files record as: `structure_mode="none"`
  (CPM-only, no SCL), `feature_layer=6`, `prompt_depth=6`, `prompt_length=1`,
  `key_size=196`, `knowledge_size=196`, `target_embed_dimension=1024`,
  `learning_rate=5e-4`, `n_neighbors=1`, `smoothing_sigma=4.0`. The result
  JSON reports `"variant": "CPM-only"`.
- Checkpointing is on by default in the resumable runner. `--resume` continues
  a requeued job after its last fully evaluated concept
  (`--resume-phase committed`, the default); without it the run starts at
  concept 0 and still writes checkpoints. `--resume-dir` must lie outside
  `--output-dir`.

`run_continual_visual_ad.py` and `run_continual_visual_ad_levels.py` accept
the same flags without the checkpoint options and build the same model
configuration; they were not used for the paper's UCAD numbers.

`examples/clvad/camera_ready/build_baseline_table.py` selects these runs (UCAD
with a `vit_base_patch16_224*` backbone) and computes the table entries.

## SCL structure-mask modes

`structure_mode` controls Structure-based Contrastive Learning:

| Mode | Behavior |
|------|----------|
| `none` | Default and the setting of every UCAD run in the paper (CPM-only). Prompts are stored but are not optimized with SCL. |
| `precomputed` | Optional full UCAD. Loads integer SAM region maps from disk. Not used in the paper. |
| `sam` | Optional full UCAD. Runs `SamAutomaticMaskGenerator` directly on the normal training arrays. Not used in the paper. |

The official MVTec mask archive can be extracted so it has this layout:

```text
/data/mvtec2d-sam-b/
  bottle/train/good/000.png
  cable/train/good/000.png
  ...
```

The loader also accepts `<root>/task_000/*.png`. InCLAD-MD names such as
`mvtec__bottle` resolve both `<root>/mvtec/bottle/...` and the category suffix
`<root>/bottle/...`.

The runner uses a concept-aware scenario only to pass the training category
name to this mask loader. `UCADStrategy.predict()` discards the supplied test
name and selects the task exclusively from the learned keys.

## Optional: full UCAD with precomputed SAM maps (not used in the paper)

This profile adds SCL on top of the paper configuration. Its numbers are not
comparable with the paper's CPM-only UCAD entries.

```bash
python examples/clvad/run_continual_visual_ad_resumable.py \
  --model ucad \
  --strategy ucad \
  --benchmark mvtec \
  --root /data/mvtec_ad \
  --inclad-bench \
  --ordering-mode easy_to_hard \
  --eval-level both \
  --resize 224 \
  --backbone vit_base_patch16_224.augreg_in21k \
  --pretrained \
  --batch-size 8 \
  --epochs 25 \
  --pixel-threshold-mode train-quantile \
  --config-json '{
    "feature_layer": 6,
    "prompt_depth": 6,
    "key_size": 196,
    "knowledge_size": 196,
    "target_embed_dimension": 1024,
    "structure_mode": "precomputed",
    "structure_mask_root": "/data/mvtec2d-sam-b"
  }' \
  --output-dir /results/ucad_scl/mvtec/easy_to_hard \
  --resume-dir /checkpoints/ucad_scl__mvtec__easy_to_hard
```

For online SAM, replace the last two JSON entries with:

```json
{
  "structure_mode": "sam",
  "sam_checkpoint": "/models/sam_vit_b_01ec64.pth",
  "sam_model_type": "vit_b"
}
```

Online SAM is substantially slower because masks are generated during every
run. Precomputing them once is preferable for multi-seed experiments.

## Step-schedule runner

UCAD is also registered in the schedule-aware runner. The paper reports no
UCAD results on grouped-component schedules; this is an additional option:

```bash
python examples/clvad/run_continual_visual_ad_step_schedule.py \
  --model ucad \
  --strategy ucad \
  --benchmark mvtec \
  --root /data/mvtec_ad \
  --step-schedule 14-1 \
  --eval-level both \
  --resize 224 \
  --pretrained \
  --epochs 25 \
  --config-file /path/to/ucad_mvtec.json
```

For grouped schedule steps with SCL, store masks under `task_000`, `task_001`,
and so on, because one training step can contain several source categories.

## Tests

```bash
pytest -q \
  tests/models/test_ucad.py \
  tests/models/test_ucad_strategy.py \
  tests/resume/methods/test_ucad.py
```

The tests run on CPU without downloads. They cover SCL gradients,
key-prompt-knowledge accumulation, task-agnostic prediction, image/pixel output
contracts, prompt training, official-style precomputed mask loading, and the
checkpoint round trip of the resumable `ResumableUCAD` / `ResumableUCADStrategy` used for the
paper runs.
