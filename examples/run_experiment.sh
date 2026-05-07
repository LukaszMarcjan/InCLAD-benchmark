#!/usr/bin/env bash
# run_experiment.sh — run continual visual anomaly detection experiments
#
# Quick start:
#   ./run_experiment.sh --benchmark mvtec --models patchcore --strategies naive
#   ./run_experiment.sh --benchmark mvtec --inclad-bench --ordering easy_to_hard
#   ./run_experiment.sh --benchmark all --models "padim patchcore" --strategies "naive cl"
#
# All parameters have defaults — run with --dry-run first to preview commands.

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
MASTER_SEED=42
DEVICE="cuda:0"
BATCH_SIZE=32
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
      --inclad-bench             Load manifests from InCLAD-Bench (HuggingFace anonmllab/inclad-bench).
                                 Images must still be available locally.
  -o, --ordering MODE            easy_to_hard | hard_to_easy | random | dataset_order
                                 (default: easy_to_hard; inclad-md supports only easy_to_hard | hard_to_easy)
      --datasets-root DIR        Root directory containing benchmark dataset folders
      --registry-path PATH       Path to visual dataset registry JSON
      --resize PIXELS            Resize images to square of this size (e.g. 224)

Models & strategies:
  -m, --models "M1 M2 ..."       Models to run (default: "padim patchcore cfa")
                                 Available: cfa draem efficientad fastflow ganomaly padim
                                            paste patchcore rd4ad stfpm unet
  -s, --strategies "S1 S2 ..."   Strategies (default: "naive cl")
                                 Available: naive replay cumulative ste cl

Experiment:
      --n-runs N                 Runs per job (default: 1)
      --master-seed N            Master seed (default: 42)
      --device DEVICE            PyTorch device (default: cuda:0)
      --batch-size N             Training batch size (default: 32)
      --eval-level LEVEL         image | pixel | both (default: both)
      --replay-buffer-fraction F Replay buffer fraction (default: 0.2)
      --output-dir DIR           Output directory (default: auto-timestamped)

Output:
      --no-heatmaps              Skip heatmap generation after run
      --dry-run                  Print commands without executing

  -h, --help                     Show this help

Examples:
  # Single benchmark, all default models/strategies, InCLAD-Bench manifest:
  ./run_experiment.sh --benchmark mvtec --inclad-bench

  # Multi-dataset InCLAD-MD (--inclad-bench implied automatically):
  ./run_experiment.sh --benchmark inclad-md --ordering easy_to_hard

  # All five benchmarks, PatchCore only, naive strategy, easy-to-hard ordering:
  ./run_experiment.sh --benchmark all --models patchcore --strategies naive --ordering easy_to_hard

  # Preview what would run (no actual execution):
  ./run_experiment.sh --benchmark btech --dry-run
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
elif [[ -x "$HOME/.virtualenvs/pyCLAD/bin/python" ]]; then
  PYTHON="$HOME/.virtualenvs/pyCLAD/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  PYTHON="python"
fi

RUNNER="$SCRIPT_DIR/visual_models/run_all_experiments.py"
ORDERING_DIR="$SCRIPT_DIR/visual_models/neurips_utilities/ordering"

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

# ---------------------------------------------------------------------------
# Build base command
# ---------------------------------------------------------------------------
read -r -a MODEL_ARR    <<< "$MODELS"
read -r -a STRATEGY_ARR <<< "$STRATEGIES"
read -r -a BENCHMARK_ARR <<< "$BENCHMARKS_LIST"

CMD=(
  "$PYTHON"
  "$RUNNER"
  --models    "${MODEL_ARR[@]}"
  --strategies "${STRATEGY_ARR[@]}"
  --benchmarks "${BENCHMARK_ARR[@]}"
  --n-runs "$N_RUNS"
  --master-seed "$MASTER_SEED"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --eval-level "$EVAL_LEVEL"
  --pixel-threshold-mode "$PIXEL_THRESHOLD_MODE"
  --pixel-threshold-quantile "$PIXEL_THRESHOLD_QUANTILE"
  --replay-buffer-fraction "$REPLAY_BUFFER_FRACTION"
)

# Optional flags
[[ -n "$OUTPUT_DIR" ]]    && CMD+=(--output-root "$OUTPUT_DIR")
[[ -n "$DATASETS_ROOT" ]] && CMD+=(--datasets-root "$DATASETS_ROOT")
[[ -n "$REGISTRY_PATH" ]] && CMD+=(--registry-path "$REGISTRY_PATH")
[[ -n "$RESIZE" ]]        && CMD+=(--resize "$RESIZE")
[[ "$INCLAD_BENCH" -eq 1 ]] && CMD+=(--inclad-bench)
[[ "$GENERATE_HEATMAPS" -eq 0 ]] && CMD+=(--no-generate-heatmaps)
[[ "$DRY_RUN" -eq 1 ]]    && CMD+=(--dry-run)

# Ordering
case "$ORDERING" in
  easy_to_hard|hard_to_easy|random)
    CMD+=(--ordering-mode "$ORDERING")
    [[ "$INCLAD_BENCH" -eq 0 ]] && CMD+=(--ordering-dir "$ORDERING_DIR")
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
# Print summary and run
# ---------------------------------------------------------------------------
echo "════════════════════════════════════════════════════════════════════════"
echo "  pyCLAD experiment runner"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Benchmark(s):  $BENCHMARKS_LIST"
echo "  Models:        $MODELS"
echo "  Strategies:    $STRATEGIES"
echo "  Ordering:      $ORDERING"
echo "  InCLAD-Bench:  $([ "$INCLAD_BENCH" -eq 1 ] && echo yes || echo no)"
echo "  Eval level:    $EVAL_LEVEL"
echo "  Device:        $DEVICE"
echo "  Runs/job:      $N_RUNS  (master seed: $MASTER_SEED)"
echo "  Python:        $PYTHON"
[[ -n "$OUTPUT_DIR" ]]    && echo "  Output dir:    $OUTPUT_DIR"
[[ -n "$DATASETS_ROOT" ]] && echo "  Datasets root: $DATASETS_ROOT"
[[ "$DRY_RUN" -eq 1 ]]    && echo "  *** DRY RUN — no jobs will execute ***"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

PYTHONPATH="$PROJECT_ROOT/src" "${CMD[@]}"
