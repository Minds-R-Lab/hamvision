# HamVision

**Hamiltonian Dynamics as Inductive Bias for Medical Image Analysis.**

HamVision is a single backbone with two task-specific heads. A shared
convolutional encoder feeds a *Hamiltonian bottleneck* whose state is governed
by a damped harmonic oscillator and produces three structured outputs:

- **position** *q* &mdash; feature embedding,
- **momentum** *p* &mdash; first-derivative-like spatial signal,
- **energy** *H* = ½(‖*q*‖² + ‖*p*‖²) &mdash; saliency.

The same bottleneck drives **HamSeg** (a U-Net-style decoder for medical image
segmentation) and **HamCls** (a Phase-Space Spectral Pooling head for medical
image classification). The classification head consumes the bottleneck's
complex signal *z = q + i p* in the frequency domain instead of pooling
spatially.

This repository contains the reference implementation, ablation harness,
multi-seed orchestrator, evaluation pipeline, and the visualization scripts
used to produce the paper figures.

---

## Repository layout

```
hamvision/
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/                                 Core models, training, evaluation
│   ├── hamseg.py                        HamSeg training script (segmentation)
│   ├── hamcls.py                        HamCls training script (classification, PSSP head)
│   ├── hamcls_utils.py                  Shared building blocks for hamcls.py (datasets, transforms, ConvNeXt, scan line)
│   ├── ablate_classifier.py             Head/bottleneck ablation harness for HamCls
│   ├── inference_tta.py                 Test-time augmentation + ensemble + threshold tuning
│   ├── eval_perclass_metrics.py         Per-class precision / recall / F1
│   ├── measure_flops.py                 FLOPs + parameter count (fvcore)
│   ├── analyze_pssp_complementarity.py  Canonical-correlation analysis of PSSP feature paths
│   ├── diagnose_hamseg.py               Layer-wise diagnostics for the segmentation network
│   └── shrink_checkpoint.py             Strip optimizer / EMA state for distribution
│
├── data/                                Dataset preparation
│   ├── prepare_data.py                  Universal downloader (ISIC, TN3K, MedMNIST, ...)
│   ├── preprocess_acdc.py               ACDC NIfTI → 2D slices
│   └── preprocess_mmotu.py              MMOTU official-split builder
│
├── experiments/                         Orchestration & aggregation
│   ├── run_multi_seed.py                3-seed orchestrator with per-dataset presets
│   ├── aggregate_results.py             Cross-seed mean ± std for the main tables
│   ├── aggregate_ablation.py            Cross-seed mean ± std for ablation runs
│   ├── find_reported_checkpoints.py     Map paper numbers back to their checkpoints
│   ├── migrate_checkpoints.py           Move legacy checkpoints into the seeded layout
│   ├── collect_results.py               Tarball the small JSON / log files for transport
│   └── smoke_test_ablation.py           3-epoch smoke test for every ablation variant
│
├── visualize/                           Figure generators
│   ├── generate_block_diagram.py        Fig. 1 — Hamiltonian bottleneck block diagram
│   ├── visualize_segmentation.py        Fig. 4 — qualitative segmentation panel
│   ├── visualize_energy_gates.py        Fig. 5 — multi-scale energy gates
│   ├── visualize_classification_interpretability.py
│   │                                    Fig. 7 — classification interpretability
│   └── visualize_hamseg_legacy.py       Older single-sample diagnostic plot
│
└── figures/                             Paper figures (PDF / PNG)
    ├── fig1_block_diagram.pdf
    ├── fig2_hamseg_architecture.pdf
    ├── fig3_hamcls_architecture.pdf
    ├── fig4_qualitative_segmentation.png
    ├── fig5_multiscale_energy_gates.png
    ├── fig6_segmentation_interpretability.png
    └── fig7_classification_interpretability.png
```

---

## Installation

```bash
git clone https://github.com/<your-org>/hamvision.git
cd hamvision
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

PyTorch with CUDA support should be installed first (follow the official
PyTorch instructions for your CUDA / OS combination). All other dependencies
are pinned in `requirements.txt`.

---

## Datasets

Fourteen benchmarks across nine imaging modalities are supported.

**Segmentation (5):** ISIC 2017, ISIC 2018, TN3K, ACDC, MMOTU.
**Classification (9):** PathMNIST, DermaMNIST, BloodMNIST, OCTMNIST,
OrganAMNIST, OrganCMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST.

All preparation is handled by `data/prepare_data.py`:

```bash
# Segmentation
python data/prepare_data.py --dataset isic2018 --out_dir ./data_root/ISIC2018
python data/prepare_data.py --dataset acdc     --out_dir ./data_root/ACDC
python data/prepare_data.py --dataset tn3k     --out_dir ./data_root/TN3K

# Classification (MedMNIST datasets stream via the medmnist package)
python data/prepare_data.py --dataset dermamnist --out_dir ./data_root/MedMNIST
```

ACDC and MMOTU have dataset-specific preprocessors (`preprocess_acdc.py`,
`preprocess_mmotu.py`) that `prepare_data.py` calls under the hood.

---

## Training

### Single seed (drop-in replacement for the training scripts)

```bash
# Segmentation — ISIC 2018, seed 42
python src/hamseg.py --dataset isic2018 --data_root ./data_root/ISIC2018 \
                     --seed 42 --epochs 200 --batch_size 8 --lr 5e-4 \
                     --output_dir ./outputs/isic2018

# Classification — DermaMNIST, seed 42
python src/hamcls.py --dataset dermamnist --seed 42 \
                     --epochs 100 --batch_size 32 --lr 1e-3 \
                     --output_dir ./outputs/dermamnist
```

### Three seeds (recommended; matches the paper)

```bash
# Single dataset, three seeds, in series:
python experiments/run_multi_seed.py seg --dataset isic2018 \
                                         --data_root ./data_root/ISIC2018 \
                                         --seeds 42 43 44

# Full segmentation suite (5 datasets × 3 seeds):
python experiments/run_multi_seed.py seg --preset all_seg --seeds 42 43 44 \
                                         --data_root_map seg_paths.json

# Full classification suite (9 datasets × 3 seeds):
python experiments/run_multi_seed.py cls --preset all_cls --seeds 42 43 44
```

`run_multi_seed.py` writes a master `INDEX.json` after every seed completes so
the experiment record is durable across crashes. Use `--resume` to skip seeds
that already wrote `test_results_final.json`.

### Aggregation

```bash
python experiments/aggregate_results.py --root outputs/dermamnist
# -> writes aggregate.json (mean ± std across seeds) and SUMMARY.txt
```

---

## Test-time augmentation, ensemble, threshold tuning

`src/inference_tta.py` runs flip-based TTA, ensembles seeds 42/43/44, and
tunes the binary threshold against the validation set. Matches the
"deployment-mode" rows of the segmentation tables.

```bash
python src/inference_tta.py --dataset isic2018 \
                            --data_root  ./data_root/ISIC2018 \
                            --output_root ./outputs \
                            --tta --ensemble --tune_threshold
```

---

## Ablations

```bash
# Classifier component ablations (e.g. --no_pssp_complex)
python experiments/smoke_test_ablation.py     # 3-epoch sanity check first

python src/ablate_classifier.py --variant full      --dataset dermamnist --seed 42
python src/ablate_classifier.py --variant no_ss2d   --dataset dermamnist --seed 42
python src/ablate_classifier.py --variant gap_head  --dataset dermamnist --seed 42

# Drop the complex-FFT branch of PSSP
python src/hamcls.py --dataset dermamnist --no_pssp_complex --seed 42

# Aggregate
python experiments/aggregate_ablation.py --root outputs_ablation
```

`analyze_pssp_complementarity.py` computes the canonical-correlation analysis
across the four PSSP feature paths used in the paper.

---

## FLOPs and parameter count

```bash
python src/measure_flops.py --out outputs/flops_summary.json
```

Reports both raw and autocast-patched counts (the patch makes `fvcore`'s tracer
see through the fp32 oscillator scan). HamCls comes out at **2.95 M params /
1.71 GFLOPs** at 224 × 224, HamSeg at **8.57 M params / 27.94 GFLOPs** at
256 × 256.

---

## Reproducing the figures

```bash
python visualize/generate_block_diagram.py  --out figures/fig1_block_diagram

python visualize/visualize_segmentation.py  --ckpt_root ./outputs \
                                            --save_dir  ./figures

python visualize/visualize_energy_gates.py  --ckpt_root ./outputs \
                                            --save_dir  ./figures

python visualize/visualize_classification_interpretability.py \
                                            --ckpt_root ./outputs \
                                            --save_dir  ./figures
```

---

## Headline numbers

- **5/5** segmentation leads on Dice (ACDC, ISIC 2018, ISIC 2017, TN3K,
  MMOTU) at 3-seed precision.
- **8/9** MedMNIST classification leads or ties at 3-seed precision.
- **4× smaller and 7× cheaper** than MedKAFormer-T at 224 × 224.
- **Single shared encoder + bottleneck** drives both heads &mdash; the
  Hamiltonian apparatus contributes between **+1.17 pp Dice** (ACDC) and
  **+2.99 pp Dice** (ISIC 2018) over a ConvNeXt-only stripped variant.

---

## Citing

If you use this repository, please cite the paper. A BibTeX entry will be
added here once the manuscript is accepted.

---

## License

Released under the MIT License (see `LICENSE`).
