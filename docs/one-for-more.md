# One-for-More (CDAD) in pyCLAD

pyCLAD includes a native benchmark adaptation of **One-for-More: Continual
Diffusion Model for Anomaly Detection** (Li et al., CVPR 2025):

- conditional denoising diffusion trained on normal images;
- the anomaly-masked network (AMN), with a local CNN and neighbor-masked
  non-local encoder;
- gradient projection memory (GPM), whose representation bases are updated
  by memory-bounded iterative SVD (iSVD);
- image-level and pixel-level anomaly scores from the difference between the
  input and its reconstructed normal image.

The implementation is exposed as a regular pyCLAD `Model`. Repeated `fit()`
calls are consecutive continual tasks. The first task uses unmodified
gradients; later tasks use the orthogonal gradient from Eq. (7) of the paper.
Only iSVD bases are retained, not previous images.

## Implementation scope

The default configuration is a **native, compact reproduction** of the method,
not a byte-for-byte vendoring of the authors' Lightning 1.9/Stable Diffusion
repository. It preserves the method's continual-learning equations and
protocol defaults while using a smaller pyCLAD U-Net and an offline
factor-eight codec. Therefore its metrics must be labelled as
`native_cdm_amn_isvd` and must not be presented as a direct reproduction of the
CDAD numbers published by Li et al. The CDAD results of the InCLAD paper use
this default configuration (see
[Configuration used in the InCLAD paper](#configuration-used-in-the-inclad-paper)).

For a stronger profile (not used in the InCLAD paper), set `vae_model_id` to a diffusers `AutoencoderKL`
checkpoint and increase `base_channels`. The VAE is frozen, as in the paper.
The exact authors' Stable Diffusion U-Net initialization is not automatically
converted by this adapter.

## Install

```bash
pip install -e '.[one-for-more]'
```

The diffusers dependency is loaded only when `vae_model_id` is set. The default
offline codec needs only PyTorch, torchvision, SciPy, and tqdm.

## Fast smoke run

This verifies the complete InCLAD path without claiming benchmark-quality
metrics:

```bash
python examples/clvad/run_continual_visual_ad.py \
  --model one_for_more \
  --strategy naive \
  --benchmark mvtec \
  --root /data/mvtec_ad \
  --inclad-bench \
  --ordering-mode easy_to_hard \
  --categories bottle cable \
  --max-train 8 \
  --max-test 8 \
  --epochs 1 \
  --incremental-epochs 1 \
  --batch-size 2 \
  --no-pretrained \
  --config-json '{"base_channels":16,"conditioner_depth":1,"conditioner_heads":2,"conditioner_feedforward":64,"diffusion_steps":20,"inference_start_step":19,"inference_steps":2,"projection_batches":1,"projection_max_columns":256,"backbone_name":null}' \
  --output-dir results/one_for_more_smoke
```

Use `--strategy naive`: continual protection is already part of the model.
Adding replay would change the One-for-More no-replay protocol.

## Configuration used in the InCLAD paper

The CDAD columns of the baseline table (Table 3) and of the per-ordering
baseline table in the appendix, and the CDAD row of the resource table, come
from 17 runs: MVTec AD, VisA, BTAD (`btech`), MPDD and DAGM with the
`easy_to_hard`, `hard_to_easy` and `random` orderings, and InCLAD-MD
(`inclad-md`) with `easy_to_hard` and `hard_to_easy`. One run per scenario and
ordering, master seed 42 (per-run seed 191664963).

All 17 runs used the resumable runner
`examples/clvad/run_continual_visual_ad_resumable.py`, which builds the
checkpointable `ResumableOneForMore` / `ResumableNaiveStrategy` from
`pyclad.resume.methods.one_for_more`. Their recorded `cli_args` differ only
in `--benchmark`, `--ordering-mode` and (for InCLAD-MD) `--root`:

```bash
python examples/clvad/run_continual_visual_ad_resumable.py \
  --model one_for_more \
  --strategy naive \
  --benchmark mvtec \
  --root /data/mvtec_ad \
  --inclad-bench \
  --ordering-mode easy_to_hard \
  --ordering-metric roc-auc \
  --resize 256 \
  --pretrained \
  --batch-size 12 \
  --epochs 400 \
  --incremental-epochs 80 \
  --no-show-training-progress \
  --eval-level both \
  --pixel-threshold-mode train-quantile \
  --pixel-threshold-quantile 0.995 \
  --master-seed 42 \
  --n-runs 1 \
  --device cuda:0 \
  --output-dir /results/one_for_more/one_for_more__mvtec__easy_to_hard \
  --resume-dir /checkpoints/one_for_more__mvtec__easy_to_hard \
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
- `--epochs 400 --incremental-epochs 80` override the model defaults of
  500/100 (the One-for-More protocol of Li et al.). The paper used 400 base
  and 80 incremental epochs.
- No `--config-json` was passed, so the remaining `OneForMoreConfig` defaults
  apply. The result files record: offline factor-eight codec
  (`vae_model_id=null`), `base_channels=64`, `conditioner_depth=8`,
  `conditioner_heads=8`, `conditioner_feedforward=1024`,
  `diffusion_steps=1000`, `inference_steps=10`, `learning_rate=1e-5`,
  `projection_energy_threshold=0.999`, `projection_batches=10`,
  `projection_max_columns=50000`, and a pretrained ResNet50 feature backbone
  (`backbone_name="resnet50"`, layers 1-3). The result JSON reports
  `"implementation": "native_cdm_amn_isvd"`.
- Checkpointing is on by default in the resumable runner. `--resume` continues
  a requeued job after its last fully evaluated concept; `--resume-dir` must
  lie outside `--output-dir`.

`run_continual_visual_ad.py` accepts the same flags without the checkpoint
options; it was not used for the paper's CDAD numbers.
`examples/clvad/camera_ready/build_baseline_table.py` selects these runs
(One-for-More with the ResNet50 backbone) and computes the table entries.

## Optional: VAE-backed profile at the scale of the original method (not used in the paper)

Li et al. train the base task for 500 epochs, each incremental task for 100
epochs, and evaluate with ten DDIM steps; 500/100 are also the model defaults.
A stronger, VAE-backed run at that scale is shown below. It was not used for
the InCLAD paper, and its numbers are not comparable with the paper's CDAD
entries:

```bash
python examples/clvad/run_continual_visual_ad.py \
  --model one_for_more \
  --strategy naive \
  --benchmark mvtec \
  --root /data/mvtec_ad \
  --inclad-bench \
  --ordering-mode easy_to_hard \
  --epochs 500 \
  --incremental-epochs 100 \
  --batch-size 12 \
  --device cuda \
  --config-json '{"vae_model_id":"stabilityai/sd-vae-ft-mse","base_channels":320,"conditioner_depth":8,"conditioner_heads":8,"conditioner_feedforward":1024,"projection_energy_threshold":0.999,"projection_batches":10,"projection_max_columns":50000}' \
  --output-dir results/one_for_more_vae/mvtec/easy_to_hard
```

For CDAD-style grouped task schedules such as `14-1`, `10-5`, `3x5`, or
`10-1x5`, use the step-schedule runner. The paper reports no CDAD results on
grouped-component schedules; the example uses the epochs of the paper's
single-component runs:

```bash
python examples/clvad/run_continual_visual_ad_step_schedule.py \
  --model one_for_more \
  --strategy naive \
  --benchmark mvtec \
  --root /data/mvtec_ad \
  --step-schedule 10-1x5 \
  --ordering-mode easy_to_hard \
  --epochs 400 \
  --incremental-epochs 80 \
  --device cuda \
  --output-dir results/one_for_more/mvtec/10-1x5
```

The runner writes the normal pyCLAD JSON result, including image-level metric
matrices, continual metrics, timing, memory, and a model section with iSVD rank
statistics. Add `--eval-level both` to a compatible runner invocation to record
pixel-level metrics as well.

To include the method in the batch benchmark matrix, select it explicitly:

```bash
python examples/visual_models/run_all_experiments.py \
  --models one_for_more \
  --strategies naive \
  --benchmarks mvtec visa \
  --inclad-bench \
  --ordering-mode easy_to_hard \
  --epochs 400 \
  --incremental-epochs 80 \
  --device cuda \
  --output-root results/one_for_more
```

It is not in the batch runner's default model list.

## Reproducibility checklist

Record the following with every reported result:

1. the exact InCLAD manifest ordering and master seed;
2. whether the offline codec or a named VAE checkpoint was used;
3. `base_channels`, AMN depth, projection threshold, and projection column cap;
4. base/incremental epochs and inference steps;
5. the pyCLAD commit and the generated JSON result file.

## Tests

```bash
pytest -q \
  tests/models/test_one_for_more.py \
  tests/resume/methods/test_one_for_more.py
```

The tests run on CPU without downloads. The second file covers the checkpoint
round trip of the resumable `ResumableOneForMore` / `ResumableNaiveStrategy` used for the
paper runs.

Reference implementation: <https://github.com/FuNz-0/One-for-More>

Paper: <https://arxiv.org/abs/2502.19848>
