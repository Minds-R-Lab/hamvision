#!/usr/bin/env bash
#
# V3 Phase B (partial) -- runs ONLY the 6 HamVision-Lite jobs.
#
# The bandpass sweep (12 jobs, answering Q1 + Q2) has already been
# executed and its checkpoints/logs are in outputs_v3/. This script
# handles the remaining Lite runs one at a time so you can watch
# progress in the terminal and stop/adjust between jobs if needed.
#
#   ACDC       -- 3 seeds -> outputs_v3_lite/acdc/seed_{42,43,44}/
#   ISIC 2018  -- 3 seeds -> outputs_v3_lite/isic2018/seed_{42,43,44}/
#
# HamVision-Lite = HamSeg with two components dropped:
#   --lite_no_se       drops the SE attention on the energy tensor
#   --lite_no_psattn   drops phase-space attention at decoder stage d3
#
# Total budget: 6 training runs (~18 GPU-hours on a single H100).
#
# Usage (DATA_PARENT is the folder that CONTAINS ACDC/ ISIC2018/, e.g. ./data):
#   bash experiments/run_v3_lite_only.sh /path/to/data_parent [outputs_v3_lite]

set -euo pipefail

DATA_PARENT="${1:-./data}"
OUTPUT_ROOT="${2:-outputs_v3_lite}"

if [ ! -d "$DATA_PARENT" ]; then
    echo "ERROR: data parent '$DATA_PARENT' does not exist."
    echo "Usage: bash experiments/run_v3_lite_only.sh /path/to/data_parent [outputs_v3_lite]"
    exit 1
fi

SCRIPT="src/hamseg.py"
SEEDS=(42 43 44)

# ACDC is 4-class (bg + RV + Myo + LV); every other segmentation dataset
# in this suite is binary. Force the correct value per dataset so results
# are comparable with the paper's Table 5.
num_classes_for () {
    case "$1" in
        acdc)      echo 4 ;;
        *)         echo 1 ;;
    esac
}

DATASETS=(acdc isic2018)

# Resolve dataset -> subfolder. MedicalSegDataset needs the folder that
# CONTAINS the {train,test} splits, not the parent of that. Search a
# short list of common casings/variants.
resolve_data_root () {
    local ds="$1"
    local parent="$2"
    local candidates=()
    case "$ds" in
        acdc)     candidates=("ACDC" "acdc" "Acdc" "ACDC_data") ;;
        isic2018) candidates=("ISIC2018" "isic2018" "ISIC_2018" "ISIC-2018" "isic_2018") ;;
        *)        candidates=("$ds") ;;
    esac
    for c in "${candidates[@]}"; do
        if [ -d "$parent/$c" ]; then
            echo "$parent/$c"
            return 0
        fi
    done
    return 1
}

mkdir -p "$OUTPUT_ROOT"
echo "HamVision-Lite will write to: $OUTPUT_ROOT/{dataset}/seed_{42,43,44}/"
echo "Data parent:                  $DATA_PARENT"
echo

for DS in "${DATASETS[@]}"; do
    if ! DR=$(resolve_data_root "$DS" "$DATA_PARENT"); then
        echo "ERROR: could not find $DS folder under $DATA_PARENT"
        echo "       Looked for common casings (e.g. ACDC, ISIC2018)."
        echo "       Contents of $DATA_PARENT:"
        ls "$DATA_PARENT" | sed 's/^/         /'
        exit 1
    fi
    for SEED in "${SEEDS[@]}"; do
        echo "==== [$(date +%T)] LITE  ds=$DS  seed=$SEED  data=$DR ============"
        python "$SCRIPT" \
            --dataset "$DS" \
            --data_root "$DR" \
            --lite_no_se \
            --lite_no_psattn \
            --num_classes "$(num_classes_for "$DS")" \
            --seed "$SEED" \
            --output_dir "$OUTPUT_ROOT"
    done
done

echo
echo "All 6 HamVision-Lite training runs complete."
echo
echo "Aggregate results:"
echo "  python experiments/aggregate_ablation.py --root $OUTPUT_ROOT --label lite"
