#!/usr/bin/env bash
#
# V3 Phase B launcher -- runs the 12 training jobs needed to answer the
# round-3 reviewer questions Q1, Q2, and Q4.
#
#   Q1 / Q2 -- bandpass-filterbank baseline (replaces the Hamiltonian
#              SS2D with a parameter-matched Gabor filterbank). The
#              key question: is the gain from the dynamics, or from
#              frequency selectivity alone?
#
#   Q4      -- HamVision-Lite: drops the SE energy-channel attention
#              and the phase-space attention at decoder stage d3,
#              saving ~74K parameters at D=384. The key question: how
#              much of the architecture is incremental decoration vs.
#              load-bearing?
#
# Datasets: ACDC + ISIC 2018, three seeds each. Output structure
# matches the V2 ablation harness so `experiments/aggregate_ablation.py`
# picks the results up without modification.
#
# Usage:
#   bash experiments/run_v3_phase_b.sh /path/to/data_root [outputs_v3]
#
# Total budget: 12 training runs (~36 GPU-hours on a single H100 at
# the V2 recipe).

set -euo pipefail

DATA_ROOT="${1:-./data_root}"
OUTPUT_ROOT="${2:-outputs_v3}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: data root '$DATA_ROOT' does not exist."
    echo "Usage: bash experiments/run_v3_phase_b.sh /path/to/data_root [outputs_v3]"
    exit 1
fi

SCRIPT="src/hamseg.py"
SEEDS=(42 43 44)
DATASETS=(acdc isic2018)

mkdir -p "$OUTPUT_ROOT"
echo "Phase B will write to: $OUTPUT_ROOT/{dataset}/{abl_C, lite}/seed_{42,43,44}/"
echo "Data root:             $DATA_ROOT"
echo

# -- Experiment 1: bandpass-filterbank baseline (Q1 / Q2) ---------------
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "==== [$(date +%T)] BANDPASS  ds=$DS  seed=$SEED ==========="
        python "$SCRIPT" \
            --dataset "$DS" \
            --data_root "$DATA_ROOT" \
            --ablation C \
            --seed "$SEED" \
            --output_dir "$OUTPUT_ROOT"
    done
done

# -- Experiment 2: HamVision-Lite (Q4) ----------------------------------
for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "==== [$(date +%T)] LITE      ds=$DS  seed=$SEED ==========="
        python "$SCRIPT" \
            --dataset "$DS" \
            --data_root "$DATA_ROOT" \
            --lite_no_se \
            --lite_no_psattn \
            --seed "$SEED" \
            --output_dir "${OUTPUT_ROOT}_lite"
    done
done

echo
echo "All 12 training runs complete."
echo "Aggregate the bandpass results with:"
echo "  python experiments/aggregate_ablation.py --root $OUTPUT_ROOT --ablation C"
echo "Aggregate the lite results with:"
echo "  python experiments/aggregate_ablation.py --root ${OUTPUT_ROOT}_lite --label lite"
