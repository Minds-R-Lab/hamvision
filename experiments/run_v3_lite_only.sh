#!/usr/bin/env bash
#
# V3 Phase B (partial) -- runs ONLY the 6 HamVision-Lite jobs.
#
# The bandpass sweep (12 jobs, answering Q1 + Q2) has already been
# executed and its checkpoints/logs are in `outputs_v3/`. This script
# handles the remaining Lite runs one at a time so you can watch
# progress in the terminal and stop/adjust between jobs if needed.
#
#   ACDC       -- 3 seeds -> outputs_v3_lite/acdc/lite/seed_{42,43,44}/
#   ISIC 2018  -- 3 seeds -> outputs_v3_lite/isic2018/lite/seed_{42,43,44}/
#
# HamVision-Lite = HamSeg with two components dropped:
#   --lite_no_se       drops the SE attention on the energy tensor
#   --lite_no_psattn   drops phase-space attention at decoder stage d3
#
# Total budget: 6 training runs (~18 GPU-hours on a single H100).
#
# Usage:
#   bash experiments/run_v3_lite_only.sh /path/to/data_root [outputs_v3_lite]

set -euo pipefail

DATA_ROOT="${1:-./data}"
OUTPUT_ROOT="${2:-outputs_v3_lite}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: data root '$DATA_ROOT' does not exist."
    echo "Usage: bash experiments/run_v3_lite_only.sh /path/to/data_root [outputs_v3_lite]"
    exit 1
fi

SCRIPT="src/hamseg.py"
SEEDS=(42 43 44)
DATASETS=(acdc isic2018)

mkdir -p "$OUTPUT_ROOT"
echo "HamVision-Lite will write to: $OUTPUT_ROOT/{dataset}/lite/seed_{42,43,44}/"
echo "Data root:                    $DATA_ROOT"
echo

for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "==== [$(date +%T)] LITE  ds=$DS  seed=$SEED ============================="
        python "$SCRIPT" \
            --dataset "$DS" \
            --data_root "$DATA_ROOT" \
            --lite_no_se \
            --lite_no_psattn \
            --seed "$SEED" \
            --output_dir "$OUTPUT_ROOT"
    done
done

echo
echo "All 6 HamVision-Lite training runs complete."
echo
echo "Aggregate results:"
echo "  python experiments/aggregate_ablation.py --root $OUTPUT_ROOT --label lite"
