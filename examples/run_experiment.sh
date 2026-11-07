#!/usr/bin/env bash
# run_experiment.sh — run continual visual anomaly detection experiments
#
# Quick start:
#   ./run_experiment.sh --benchmark mvtec --datasets-root /data --models patchcore --strategies naive
#   ./run_experiment.sh --benchmark mvtec --datasets-root /data --inclad-bench --ordering easy_to_hard
#   ./run_experiment.sh --benchmark all --datasets-root /data --models "padim patchcore" --strategies "naive cl"
#
# Only --benchmark is required. The image folders are not part of the repository:
# pass --datasets-root (or export PYCLAD_VISUAL_DATASETS_ROOT) pointing at the
# directory that holds the benchmark folders. Run with --dry-run first to preview
# the commands. Per-method settings of the paper runs are listed in README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BENCHMARK=""
MODELS="padim patchcore cfa"
STRATEGIES="naive cl"
ORDERING="easy_to_hard"
INCLAD_BENCH=0
N_RUNS=1
# MASTER_SEED=42
MASTER_SEED=""   # resolved after argument parsing (42, or 123 for random with --paper-config)
DEVICE="cuda:0"
# BATCH_SIZE=32   (was passed to every model, overriding each model's own default)
BATCH_SIZE=""
EPOCHS=""
BACKBONES=""
PRETRAINED=""
PAPER_CONFIG=1
EVAL_LEVEL="both"
PIXEL_THRESHOLD_MODE="train-quantile"
PIXEL_THRESHOLD_QUANTILE=0.95
REPLAY_BUFFER_FRACTION=0.2
OUTPUT_DIR=""
DATASETS_ROOT=""
REGISTRY_PATH=""
RESIZE=""
DRY_RUN=0
GENERATE_HEATMAPS=1

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") --benchmark BENCHMARK [OPTIONS]

Required:
  -b, --benchmark BENCHMARK      Benchmark: btech | dagm | mpdd | mvtec | visa | all
                                            inclad-md (multi-dataset; implies --inclad-bench)

Dataset:
      --inclad-bench             Load manifests from InCLAD-Bench (HuggingFace lifelonglab/inclad-benchmark).
                                 Images must still be available locally.
  -o, --ordering MODE            easy_to_hard | hard_to_easy | random | dataset_order
                                 (default: easy_to_hard; random requires --inclad-bench;
                                 inclad-md supports only easy_to_hard | hard_to_easy)
      --datasets-root DIR        Directory containing the benchmark dataset folders
                                 (default: \$PYCLAD_VISUAL_DATASETS_ROOT if set, else
                                 src/pyclad/data/datasets/visual_datasets, which is not shipped)
      --registry-path PATH       Path to visual dataset registry JSON
      --resize PIXELS            Resize images to square of this size (e.g. 224)

Models & strategies:
  -m, --models "M1 M2 ..."       Models to run (default: "padim patchcore cfa")
                                 Available: cfa draem efficientad fastflow ganomaly padim
                                            paste patchcore rd4ad stfpm
  -s, --strategies "S1 S2 ..."   Strategies (default: "naive cl")
                                 Available: naive replay cumulative ste cl

Experiment:
      --n-runs N                 Runs per job (default: 1)
      --master-seed N            Master seed (default: 42; with --paper-config and --ordering random: 123,
                                 the master seed of the paper's Random runs, per-run seed 33158374)
      --device DEVICE            PyTorch device (default: cuda:0)
      --batch-size N             Training batch size for every model (default: see --paper-config)
      --epochs N                 Training epochs for every model (default: see --paper-config)
      --backbone "B1 B2 ..."     Backbone(s); one batch run per backbone, results under
                                 <output-dir>/<backbone> (default: see --paper-config)
      --pretrained | --no-pretrained
                                 Force pretrained / random-init backbone weights
      --paper-config             (default) Settings of the paper's main grid:
                                   cfa, padim, patchcore: --pretrained --batch-size 32 --epochs 50
                                   fastflow, paste, rd4ad, stfpm: --pretrained, model-default
                                     batch size and epochs
                                   ganomaly, efficientad: model defaults, no backbone loop (they have
                                     no backbone or pretrained option; the paper runs passed
                                     --backbone resnet18 --pretrained, which these models ignore)
                                   backbones: resnet18 and wide_resnet50_2
                                   strategy cl is dropped for models other than cfa, padim, patchcore
                                 draem is not part of the paper grid and only gets --pretrained.
                                 Explicit --batch-size / --epochs / --backbone / --[no-]pretrained win.
                                 Some non-CL paper runs on BTech, MPDD and DAGM used --batch-size 16;
                                 see README.md for the per-method settings.
      --no-paper-config          Pass only the flags given explicitly (model defaults otherwise;
                                 PaDiM/PatchCore then use random-init backbones and CFA 10 epochs).
                                 --no-paper-config --batch-size 32 is the former behaviour.
      --eval-level LEVEL         image | pixel | both (default: both)
      --replay-buffer-fraction F Replay buffer fraction (default: 0.2)
      --output-dir DIR           Output directory (default: auto-timestamped)

Output:
      --no-heatmaps              Skip heatmap generation after run
      --dry-run                  Print commands without executing

  -h, --help                     Show this help

Examples (/data holds mvtec_ad/, VisA/, MPDD/, DAGM_KaggleUpload/, BTech_Dataset_transformed/):
  # Single benchmark, default models/strategies, InCLAD-Bench manifest:
  ./run_experiment.sh --benchmark mvtec --datasets-root /data --inclad-bench

  # Random ordering (seeded split from the HuggingFace manifest):
  ./run_experiment.sh --benchmark visa --datasets-root /data --inclad-bench --ordering random

  # Multi-dataset InCLAD-MD (--inclad-bench implied automatically):
  ./run_experiment.sh --benchmark inclad-md --datasets-root /data --ordering easy_to_hard

  # All five benchmarks, PatchCore only, naive strategy, one backbone:
  ./run_experiment.sh --benchmark all --datasets-root /data --models patchcore --strategies naive \\
      --backbone wide_resnet50_2

  # Preview what would run (no actual execution):
  PYCLAD_VISUAL_DATASETS_ROOT=/data ./run_experiment.sh --benchmark btech --dry-run
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--benchmark)       BENCHMARK="$2";              shift 2 ;;
    -m|--models)          MODELS="$2";                 shift 2 ;;
    -s|--strategies)      STRATEGIES="$2";             shift 2 ;;
    -o|--ordering)        ORDERING="$2";               shift 2 ;;
    --inclad-bench)       INCLAD_BENCH=1;              shift   ;;
    --n-runs)             N_RUNS="$2";                 shift 2 ;;
    --master-seed)        MASTER_SEED="$2";            shift 2 ;;
    --device)             DEVICE="$2";                 shift 2 ;;
    --batch-size)         BATCH_SIZE="$2";             shift 2 ;;
    --epochs)             EPOCHS="$2";                 shift 2 ;;
    --backbone|--backbones) BACKBONES="$2";            shift 2 ;;
    --pretrained)         PRETRAINED=1;                shift   ;;
    --no-pretrained)      PRETRAINED=0;                shift   ;;
    --paper-config)       PAPER_CONFIG=1;              shift   ;;
    --no-paper-config)    PAPER_CONFIG=0;              shift   ;;
    --eval-level)         EVAL_LEVEL="$2";             shift 2 ;;
    --replay-buffer-fraction) REPLAY_BUFFER_FRACTION="$2"; shift 2 ;;
    --output-dir)         OUTPUT_DIR="$2";             shift 2 ;;
    --datasets-root)      DATASETS_ROOT="$2";          shift 2 ;;
    --registry-path)      REGISTRY_PATH="$2";          shift 2 ;;
    --resize)             RESIZE="$2";                 shift 2 ;;
    --no-heatmaps)        GENERATE_HEATMAPS=0;         shift   ;;
    --dry-run)            DRY_RUN=1;                   shift   ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MASTER_SEED" ]]; then
  if [[ "$PAPER_CONFIG" -eq 1 && "$ORDERING" == "random" ]]; then
    MASTER_SEED=123
  else
    MASTER_SEED=42
  fi
fi

if [[ -z "$BENCHMARK" ]]; then
  echo "Error: --benchmark is required." >&2
  usage
  exit 1
fi

# ---------------------------------------------------------------------------
# Resolve Python and runner paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  PYTHON="python"
fi

RUNNER="$SCRIPT_DIR/visual_models/run_all_experiments.py"
# Previous value (directory does not exist):
# ORDERING_DIR="$SCRIPT_DIR/visual_models/neurips_utilities/ordering"
ORDERING_DIR="$SCRIPT_DIR/visual_models/ordering"

if [[ ! -f "$RUNNER" ]]; then
  echo "Error: runner not found at $RUNNER" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Expand "all" benchmark shortcut; inclad-md implies --inclad-bench
# ---------------------------------------------------------------------------
ALL_BENCHMARKS="btech dagm mpdd mvtec visa"
if [[ "$BENCHMARK" == "all" ]]; then
  BENCHMARKS_LIST="$ALL_BENCHMARKS"
elif [[ "$BENCHMARK" == "inclad-md" ]]; then
  BENCHMARKS_LIST="inclad-md"
  INCLAD_BENCH=1
  if [[ "$ORDERING" == "random" ]]; then
    echo "Error: inclad-md does not support --ordering random (use easy_to_hard or hard_to_easy)." >&2
    exit 1
  fi
else
  BENCHMARKS_LIST="$BENCHMARK"
fi

# The shipped ordering CSVs cover easy_to_hard and hard_to_easy only; the seeded
# random split is part of the HuggingFace manifests.
if [[ "$ORDERING" == "random" && "$INCLAD_BENCH" -eq 0 ]]; then
  echo "Error: --ordering random requires --inclad-bench (the seeded random split comes from the" >&2
  echo "       HuggingFace manifest; $ORDERING_DIR has only easy_to_hard / hard_to_easy files)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Datasets root: --datasets-root, else $PYCLAD_VISUAL_DATASETS_ROOT, else the
# runner's default folder, which a fresh clone does not contain.
# ---------------------------------------------------------------------------
DEFAULT_DATASETS_ROOT="$PROJECT_ROOT/src/pyclad/data/datasets/visual_datasets"
if [[ -z "$DATASETS_ROOT" && -n "${PYCLAD_VISUAL_DATASETS_ROOT:-}" ]]; then
  DATASETS_ROOT="$PYCLAD_VISUAL_DATASETS_ROOT"
fi
if [[ -n "$DATASETS_ROOT" && ! -d "$DATASETS_ROOT" ]]; then
  echo "Error: datasets root does not exist: $DATASETS_ROOT" >&2
  exit 1
fi
if [[ -z "$DATASETS_ROOT" && ! -d "$DEFAULT_DATASETS_ROOT" ]]; then
  echo "Error: no datasets root. Pass --datasets-root DIR or export PYCLAD_VISUAL_DATASETS_ROOT=DIR," >&2
  echo "       where DIR holds the benchmark folders (mvtec_ad/, VisA/, MPDD/, ...). The default" >&2
  echo "       $DEFAULT_DATASETS_ROOT is not part of the repository." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build base command
# ---------------------------------------------------------------------------
read -r -a MODEL_ARR    <<< "$MODELS"
read -r -a STRATEGY_ARR <<< "$STRATEGIES"
read -r -a BENCHMARK_ARR <<< "$BENCHMARKS_LIST"

BASE_CMD=(
  "$PYTHON"
  "$RUNNER"
  # --strategies "${STRATEGY_ARR[@]}"   (now added per model group, see below)
  --benchmarks "${BENCHMARK_ARR[@]}"
  --n-runs "$N_RUNS"
  --master-seed "$MASTER_SEED"
  --device "$DEVICE"
  # --batch-size "$BATCH_SIZE"   (previously unconditional; now only when set)
  --eval-level "$EVAL_LEVEL"
  --pixel-threshold-mode "$PIXEL_THRESHOLD_MODE"
  --pixel-threshold-quantile "$PIXEL_THRESHOLD_QUANTILE"
  --replay-buffer-fraction "$REPLAY_BUFFER_FRACTION"
)

# Optional flags
[[ -n "$DATASETS_ROOT" ]] && BASE_CMD+=(--datasets-root "$DATASETS_ROOT")
[[ -n "$REGISTRY_PATH" ]] && BASE_CMD+=(--registry-path "$REGISTRY_PATH")
[[ -n "$RESIZE" ]]        && BASE_CMD+=(--resize "$RESIZE")
[[ "$INCLAD_BENCH" -eq 1 ]] && BASE_CMD+=(--inclad-bench)
[[ "$GENERATE_HEATMAPS" -eq 0 ]] && BASE_CMD+=(--no-generate-heatmaps)
[[ "$DRY_RUN" -eq 1 ]]    && BASE_CMD+=(--dry-run)

# Ordering
case "$ORDERING" in
  easy_to_hard|hard_to_easy|random)
    BASE_CMD+=(--ordering-mode "$ORDERING")
    [[ "$INCLAD_BENCH" -eq 0 ]] && BASE_CMD+=(--ordering-dir "$ORDERING_DIR")
    ;;
  dataset_order|dataset|"")
    : # no ordering flag = dataset default order
    ;;
  *)
    echo "Unknown ordering '$ORDERING'. Use: easy_to_hard | hard_to_easy | random | dataset_order" >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Model groups. With --paper-config each group gets the flags of the paper's
# main grid; explicit --batch-size / --epochs / --backbone / --[no-]pretrained
# always take precedence. Without it there is one group and only explicit flags.
# ---------------------------------------------------------------------------
CL_TABLE_MODELS=" cfa padim patchcore "
NO_BACKBONE_MODELS=" ganomaly efficientad "
PAPER_BACKBONES="resnet18 wide_resnet50_2"

GROUP_NAMES=()
GROUP_MODELS=()
if [[ "$PAPER_CONFIG" -eq 1 ]]; then
  g_cl=""; g_bb=""; g_nobb=""
  for m in "${MODEL_ARR[@]}"; do
    if [[ "$CL_TABLE_MODELS" == *" $m "* ]]; then g_cl+="$m "
    elif [[ "$NO_BACKBONE_MODELS" == *" $m "* ]]; then g_nobb+="$m "
    else g_bb+="$m "
    fi
  done
  [[ -n "$g_cl" ]]   && { GROUP_NAMES+=(cl_table);    GROUP_MODELS+=("$g_cl"); }
  [[ -n "$g_bb" ]]   && { GROUP_NAMES+=(backbone);    GROUP_MODELS+=("$g_bb"); }
  [[ -n "$g_nobb" ]] && { GROUP_NAMES+=(no_backbone); GROUP_MODELS+=("$g_nobb"); }
else
  GROUP_NAMES+=(all)
  GROUP_MODELS+=("$MODELS")
fi

# Backbone list per group ("" = model default, no --backbone flag).
backbones_for_group() {
  local group="$1"
  if [[ -n "$BACKBONES" ]]; then echo "$BACKBONES"; return; fi
  if [[ "$PAPER_CONFIG" -eq 1 && "$group" != "no_backbone" ]]; then echo "$PAPER_BACKBONES"; return; fi
  echo ""
}

# One batch run per (group, backbone). Several backbones share model/benchmark/
# strategy result paths, so each backbone gets its own sub-directory.
N_BACKBONE_RUNS=0
for gi in "${!GROUP_NAMES[@]}"; do
  read -r -a _bbs <<< "$(backbones_for_group "${GROUP_NAMES[$gi]}")"
  (( ${#_bbs[@]} > N_BACKBONE_RUNS )) && N_BACKBONE_RUNS=${#_bbs[@]}
done
if [[ -z "$OUTPUT_DIR" && ( "$N_BACKBONE_RUNS" -gt 1 || "${#GROUP_NAMES[@]}" -gt 1 ) ]]; then
  OUTPUT_DIR="results_$(date +%Y%m%d_%H%M%S)"
fi

COMMANDS=()
COMMAND_LABELS=()
for gi in "${!GROUP_NAMES[@]}"; do
  group="${GROUP_NAMES[$gi]}"
  read -r -a group_models <<< "${GROUP_MODELS[$gi]}"
  read -r -a group_backbones <<< "$(backbones_for_group "$group")"
  [[ ${#group_backbones[@]} -eq 0 ]] && group_backbones=("")

  group_batch="$BATCH_SIZE"; group_epochs="$EPOCHS"; group_pretrained="$PRETRAINED"
  if [[ "$PAPER_CONFIG" -eq 1 ]]; then
    [[ -z "$group_pretrained" && "$group" != "no_backbone" ]] && group_pretrained=1
    if [[ "$group" == "cl_table" ]]; then
      [[ -z "$group_batch" ]]  && group_batch=32
      [[ -z "$group_epochs" ]] && group_epochs=50
    fi
  fi

  # The "cl" strategy exists only for cfa, padim and patchcore. With
  # --paper-config it is dropped for the other groups instead of failing the batch.
  group_strategies=()
  for st in "${STRATEGY_ARR[@]}"; do
    if [[ "$PAPER_CONFIG" -eq 1 && "$group" != "cl_table" && "$st" == "cl" ]]; then
      echo "Note: strategy 'cl' skipped for models [${group_models[*]}] (cl exists only for cfa, padim, patchcore)." >&2
      continue
    fi
    group_strategies+=("$st")
  done
  if [[ ${#group_strategies[@]} -eq 0 ]]; then
    echo "Note: no strategy left for models [${group_models[*]}]; skipping them." >&2
    continue
  fi

  for bb in "${group_backbones[@]}"; do
    cmd=("${BASE_CMD[@]}" --strategies "${group_strategies[@]}" --models "${group_models[@]}")
    [[ -n "$group_batch" ]]  && cmd+=(--batch-size "$group_batch")
    [[ -n "$group_epochs" ]] && cmd+=(--epochs "$group_epochs")
    [[ "$group_pretrained" == 1 ]] && cmd+=(--pretrained)
    [[ "$group_pretrained" == 0 ]] && cmd+=(--no-pretrained)
    out="$OUTPUT_DIR"
    if [[ -n "$bb" ]]; then
      cmd+=(--backbone "$bb")
      [[ -n "$out" && "$N_BACKBONE_RUNS" -gt 1 ]] && out="$out/$bb"
    fi
    [[ -n "$out" ]] && cmd+=(--output-root "$out")
    COMMANDS+=("$(printf '%q ' "${cmd[@]}")")
    COMMAND_LABELS+=("models=[${group_models[*]}] strategies=[${group_strategies[*]}] backbone=${bb:-model default} batch=${group_batch:-model default} epochs=${group_epochs:-model default}")
  done
done

# ---------------------------------------------------------------------------
# Print summary and run
# ---------------------------------------------------------------------------
echo "════════════════════════════════════════════════════════════════════════"
echo "  InCLAD-Bench experiment runner"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Benchmark(s):  $BENCHMARKS_LIST"
echo "  Models:        $MODELS"
echo "  Strategies:    $STRATEGIES"
echo "  Ordering:      $ORDERING"
echo "  InCLAD-Bench:  $([ "$INCLAD_BENCH" -eq 1 ] && echo yes || echo no)"
echo "  Eval level:    $EVAL_LEVEL"
echo "  Device:        $DEVICE"
echo "  Runs/job:      $N_RUNS  (master seed: $MASTER_SEED)"
echo "  Paper config:  $([ "$PAPER_CONFIG" -eq 1 ] && echo yes || echo no)"
echo "  Python:        $PYTHON"
[[ -n "$OUTPUT_DIR" ]]    && echo "  Output dir:    $OUTPUT_DIR"
[[ -n "$DATASETS_ROOT" ]] && echo "  Datasets root: $DATASETS_ROOT"
[[ "$DRY_RUN" -eq 1 ]]    && echo "  *** DRY RUN — no jobs will execute ***"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# PYTHONPATH="$PROJECT_ROOT/src" "${CMD[@]}"
status=0
for ci in "${!COMMANDS[@]}"; do
  echo "── batch $((ci + 1))/${#COMMANDS[@]}: ${COMMAND_LABELS[$ci]}"
  eval "cmd=(${COMMANDS[$ci]})"
  PYTHONPATH="$PROJECT_ROOT/src" "${cmd[@]}" || status=$?
done
exit "$status"
