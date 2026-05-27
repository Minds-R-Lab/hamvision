#!/usr/bin/env python3
"""
sensitivity_sweep.py
====================

Sensitivity analysis for the Hamiltonian bottleneck. Two passes, both
operating on already-trained checkpoints (no retraining required):

  (A) Learned-parameter analysis. For every HamiltonianScanLine module in
      the loaded model, dump the learned angular frequency omega and the
      learned baseline damping nu_bias. Save histograms per stage plus a
      single-CSV summary.

  (B) Inference-time perturbation sweep. Multiply omega (or nu_bias) by a
      list of scale factors and run validation. Save a CSV of
      (param, scale, dice, miou) and a 1 x 2 figure.

Parameterisation reminder (from hamseg.py):
    omega    = exp(log_k / 2.0)          # learnable per-channel
    nu       = clamp(softplus(x * nu_scale + nu_bias) + eps, max=damping_clamp)
    so we treat 'omega' as the directly-meaningful frequency tensor, and
    'nu_bias' as the baseline-damping knob. Scaling omega by factor s is
    achieved by log_k <- log_k + 2 * ln(s). Scaling nu_bias by s is direct.

Usage
-----

    # Binary segmentation (e.g. ISIC 2018)
    python src/sensitivity_sweep.py \\
        --ckpt        ./outputs/isic2018/seed_42/best_model.pth \\
        --dataset     isic2018 \\
        --data_root   ./data_root/ISIC2018 \\
        --out_dir     ./outputs/sensitivity_isic2018 \\
        --omega_scales 0.25 0.5 1.0 2.0 4.0 \\
        --nu_scales    0.25 0.5 1.0 2.0 4.0

    # Multi-class segmentation (e.g. ACDC, 4-class)
    python src/sensitivity_sweep.py \\
        --ckpt        ./outputs/acdc/seed_42/best_model.pth \\
        --dataset     acdc --num_classes 4 \\
        --data_root   ./data_root/ACDC \\
        --out_dir     ./outputs/sensitivity_acdc
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Pull model + dataset + metrics from the training module
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hamseg import (  # noqa: E402
    HamSeg,
    HamiltonianScanLine,
    MedicalSegDataset,
    compute_metrics,
    set_seed,
)


# ============================================================
# Argument parsing
# ============================================================

def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to a trained best_model.pth checkpoint.')
    p.add_argument('--dataset', type=str, required=True,
                   help='Dataset name (matches hamseg.py --dataset). Used only for logging.')
    p.add_argument('--data_root', type=str, required=True,
                   help='Path to the dataset root (val/test images + masks).')
    p.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                   help='Which split to evaluate on. val is faster.')
    p.add_argument('--num_classes', type=int, default=1,
                   help='1 for binary segmentation, >1 for multi-class.')
    p.add_argument('--img_size', '--size', dest='img_size', type=int, default=256,
                   help='Square input size matching the training run.')
    p.add_argument('--train_ratio', type=float, default=0.7,
                   help='Mirrors hamseg.py --train_ratio. Affects only datasets that split by ratio.')
    p.add_argument('--val_ratio', type=float, default=0.0,
                   help='Mirrors hamseg.py --val_ratio. Use 0.0 for ACDC (patient-level split).')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)

    # The two sweeps. Pass an empty list to skip one of them.
    p.add_argument('--omega_scales', type=float, nargs='*',
                   default=[0.25, 0.5, 1.0, 2.0, 4.0],
                   help='Multiplicative scales applied to learned omega for each run.')
    p.add_argument('--nu_scales', type=float, nargs='*',
                   default=[0.25, 0.5, 1.0, 2.0, 4.0],
                   help='Multiplicative scales applied to learned nu_bias for each run.')

    p.add_argument('--out_dir', type=str, default='./outputs/sensitivity',
                   help='Where the CSVs / figures get written.')

    # HamSeg config (must match the trained checkpoint)
    p.add_argument('--in_channels', type=int, default=3)
    p.add_argument('--embed_dim', type=int, default=48)
    p.add_argument('--depths', type=int, nargs=4, default=[2, 2, 2, 2])
    p.add_argument('--damping_clamp', type=float, default=5.0)
    p.add_argument('--drop_rate', type=float, default=0.1)
    p.add_argument('--head_drop', type=float, default=0.3)
    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def collect_scan_lines(model):
    """Walk the model and return a list of (stage_tag, full_name, module)."""
    pairs = []
    for name, mod in model.named_modules():
        if isinstance(mod, HamiltonianScanLine):
            parts = name.split('.')
            stage_idx = None
            for i, q in enumerate(parts):
                if q.startswith('stages') or q.startswith('bottleneck'):
                    if i + 1 < len(parts) and parts[i + 1].isdigit():
                        stage_idx = parts[i + 1]
                        break
            stage_tag = f'stage_{stage_idx}' if stage_idx is not None else 'bottleneck'
            pairs.append((stage_tag, name, mod))
    return pairs


def omega_of(scan_line):
    return torch.exp(scan_line.log_k.detach().float() / 2.0).cpu().numpy()


def nu_bias_of(scan_line):
    return scan_line.nu_bias.detach().float().cpu().numpy()


def snapshot_params(scan_lines):
    return [(sl.log_k.detach().clone(), sl.nu_bias.detach().clone())
            for _, _, sl in scan_lines]


def restore_params(scan_lines, snapshot):
    for (_, _, sl), (lk, nb) in zip(scan_lines, snapshot):
        sl.log_k.data.copy_(lk)
        sl.nu_bias.data.copy_(nb)


def apply_omega_scale(scan_lines, s):
    """omega' = s * omega  <=>  log_k' = log_k + 2 * ln(s)"""
    delta = 2.0 * math.log(s)
    for _, _, sl in scan_lines:
        sl.log_k.data.add_(delta)


def apply_nu_scale(scan_lines, s):
    for _, _, sl in scan_lines:
        sl.nu_bias.data.mul_(s)


# ============================================================
# Evaluation loop (no AMP, no TTA, no ensemble)
# ============================================================

@torch.no_grad()
def evaluate_dice(model, loader, device, num_classes=1, threshold=0.5):
    model.eval()
    agg = {'dice': 0.0, 'miou': 0.0, 'n': 0}
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        m = compute_metrics(logits, masks, threshold=threshold, num_classes=num_classes)
        bs = imgs.size(0)
        agg['dice'] += m['dice'] * bs
        agg['miou'] += m['miou'] * bs
        agg['n'] += bs
    return agg['dice'] / max(agg['n'], 1), agg['miou'] / max(agg['n'], 1)


# ============================================================
# Part (A) -- learned parameter analysis
# ============================================================

def part_a_learned_params(scan_lines, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = {}
    for stage_tag, name, sl in scan_lines:
        stages.setdefault(stage_tag, []).append((name, sl))

    summary_rows = []
    fig, axes = plt.subplots(2, max(1, len(stages)),
                             figsize=(4 * max(1, len(stages)), 6),
                             squeeze=False)
    for j, (stage_tag, members) in enumerate(sorted(stages.items())):
        ws = np.concatenate([omega_of(sl) for _, sl in members])
        ns = np.concatenate([nu_bias_of(sl) for _, sl in members])

        for arr, label in [(ws, 'omega'), (ns, 'nu_bias')]:
            summary_rows.append({
                'stage': stage_tag,
                'parameter': label,
                'n': int(arr.size),
                'min': float(arr.min()),
                'q25': float(np.quantile(arr, 0.25)),
                'median': float(np.median(arr)),
                'q75': float(np.quantile(arr, 0.75)),
                'max': float(arr.max()),
                'mean': float(arr.mean()),
                'std': float(arr.std()),
            })

        axes[0, j].hist(ws, bins=40, color='#3a7ca5', edgecolor='black', linewidth=0.3)
        axes[0, j].set_title(f'{stage_tag}: omega (rad/step)')
        axes[0, j].set_xlabel('omega = exp(log_k / 2)')
        axes[0, j].set_ylabel('count')

        axes[1, j].hist(ns, bins=40, color='#d96f6f', edgecolor='black', linewidth=0.3)
        axes[1, j].set_title(f'{stage_tag}: nu_bias (damping)')
        axes[1, j].set_xlabel('nu_bias')
        axes[1, j].set_ylabel('count')

    fig.suptitle('Learned Hamiltonian parameters', y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'learned_params_histograms.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'learned_params_histograms.pdf', bbox_inches='tight')
    plt.close(fig)

    csv_path = out_dir / 'learned_params_summary.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f'[A] wrote {csv_path.name} and learned_params_histograms.{{png,pdf}}')
    return summary_rows


# ============================================================
# Part (B) -- inference-time perturbation sweep
# ============================================================

def part_b_perturbation_sweep(model, loader, scan_lines, device, num_classes,
                              omega_scales, nu_scales, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    snapshot = snapshot_params(scan_lines)

    print('[B] baseline ...')
    dice, miou = evaluate_dice(model, loader, device, num_classes=num_classes)
    rows.append({'param': 'baseline', 'scale': 1.0, 'dice': dice, 'miou': miou})
    print(f'    baseline dice={dice:.4f} miou={miou:.4f}')

    for s in omega_scales:
        restore_params(scan_lines, snapshot)
        apply_omega_scale(scan_lines, s)
        dice, miou = evaluate_dice(model, loader, device, num_classes=num_classes)
        rows.append({'param': 'omega', 'scale': s, 'dice': dice, 'miou': miou})
        print(f'    omega x{s:>5.2f}  dice={dice:.4f}  miou={miou:.4f}')

    for s in nu_scales:
        restore_params(scan_lines, snapshot)
        apply_nu_scale(scan_lines, s)
        dice, miou = evaluate_dice(model, loader, device, num_classes=num_classes)
        rows.append({'param': 'nu_bias', 'scale': s, 'dice': dice, 'miou': miou})
        print(f'    nu_bias x{s:>5.2f}  dice={dice:.4f}  miou={miou:.4f}')

    restore_params(scan_lines, snapshot)

    csv_path = out_dir / 'sensitivity_sweep.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['param', 'scale', 'dice', 'miou'])
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    base_dice = rows[0]['dice']
    for ax, key, title, color in [
        (axes[0], 'omega',   'Dice vs omega scaling',   '#3a7ca5'),
        (axes[1], 'nu_bias', 'Dice vs nu_bias scaling', '#d96f6f'),
    ]:
        pts = [(r['scale'], r['dice']) for r in rows if r['param'] == key]
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, marker='o', color=color, lw=2)
            ax.axhline(base_dice, color='gray', lw=1, linestyle='--',
                       label=f'baseline = {base_dice:.3f}')
            ax.set_xscale('log')
            ax.set_xlabel(f'{key} scale factor')
            ax.set_title(title)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower center')
    axes[0].set_ylabel('Dice (validation)')
    fig.tight_layout()
    fig.savefig(out_dir / 'sensitivity_sweep.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'sensitivity_sweep.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f'[B] wrote {csv_path.name} and sensitivity_sweep.{{png,pdf}}')
    return rows


# ============================================================
# Main
# ============================================================

def build_dataloader(args):
    """Validation/test loader that mirrors hamseg.py's main()."""
    ds = MedicalSegDataset(args.data_root, args.split, args.img_size,
                           args.train_ratio, args.val_ratio, args.num_classes)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))

    print(f'Building HamSeg ({args.num_classes}-class) ...')
    model = HamSeg(args).to(device)

    print(f'Loading checkpoint: {args.ckpt}')
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    model.eval()

    scan_lines = collect_scan_lines(model)
    print(f'  found {len(scan_lines)} HamiltonianScanLine modules')
    if not scan_lines:
        raise RuntimeError('No HamiltonianScanLine modules found in the checkpoint.')

    print('\n[A] learned parameter analysis ...')
    part_a_learned_params(scan_lines, out_dir)

    print('\n[B] perturbation sweep ...')
    loader = build_dataloader(args)
    part_b_perturbation_sweep(model, loader, scan_lines, device,
                              num_classes=args.num_classes,
                              omega_scales=args.omega_scales,
                              nu_scales=args.nu_scales,
                              out_dir=out_dir)

    print(f'\nAll outputs written to: {out_dir}')


if __name__ == '__main__':
    main()
