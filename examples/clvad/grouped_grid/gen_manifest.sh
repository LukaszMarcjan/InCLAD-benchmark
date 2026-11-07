#!/bin/bash
# Emit the grouped-component grid manifest of the paper (TSV) to stdout.
# Columns: dataset  backbone  model  strategy  schedule  ordering
#
# Axes: 9 models x {naive, continual} x schedule(dataset) x {eth,hte,rand} x 2 backbones.
#   continual = cl for padim/patchcore/cfa, else replay (always 2 strategies/model).
# Row count: per backbone 864 (mvtec 216, visa 162, btech 108, mpdd 162, dagm 216);
#   x 2 backbones = 1728 rows total.
#
set -euo pipefail

BACKBONES=(rn18 wr50)
DATASETS=(mvtec visa btech mpdd dagm)
MODELS=(padim patchcore cfa fastflow stfpm rd4ad paste ganomaly efficientad)
CL_CAPABLE=(padim patchcore cfa)
ORDERINGS=(eth hte rand)

schedules_for() {
  case "$1" in
    mvtec) echo "14-1 10-5 3x5 10-1x5" ;;
    visa)  echo "11-1 8-4 8-1x4" ;;
    btech) echo "2-1 1x3" ;;
    mpdd)  echo "5-1 3x2 4-1x2" ;;
    dagm)  echo "9-1 5-5 2x5 5-1x5" ;;
    *) echo "ERROR: unknown dataset '$1'" >&2; exit 1 ;;
  esac
}

is_cl_capable() {
  local m="$1" c
  for c in "${CL_CAPABLE[@]}"; do [ "$c" = "$m" ] && return 0; done
  return 1
}

for bb in "${BACKBONES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    read -r -a scheds <<< "$(schedules_for "$ds")"
    for model in "${MODELS[@]}"; do
      if is_cl_capable "$model"; then strategies=(naive cl); else strategies=(naive replay); fi
      for strat in "${strategies[@]}"; do
        for sched in "${scheds[@]}"; do
          for ord in "${ORDERINGS[@]}"; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ds" "$bb" "$model" "$strat" "$sched" "$ord"
          done
        done
      done
    done
  done
done
