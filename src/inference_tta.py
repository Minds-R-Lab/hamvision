#!/usr/bin/env python3
"""
inference_with_tricks.py — re-evaluate trained HamSeg checkpoints with
the standard inference-time tricks every competitive segmentation
paper uses: 4-direction test-time augmentation (TTA), 3-seed ensemble,
and validation-tuned threshold (binary only).

For each (dataset, set of seeds) it produces a comparison table:

  | variant                            | dice  | mIoU  |
  | seed_42                            |       |       |
  | seed_43                            |       |       |
  | seed_44                            |       |       |
  | seed_42 + TTA                      |       |       |
  | seed_43 + TTA                      |       |       |
  | seed_44 + TTA                      |       |       |
  | ensemble (3 seeds, no TTA)         |       |       |
  | ensemble + TTA                     |       |       |
  | ensemble + TTA + tuned threshold   |       |       |  <-- binary only

Outputs are written to {output_root}/{dataset}/ensemble_results.json
and a human-readable {output_root}/{dataset}/ENSEMBLE_SUMMARY.txt.

Usage
-----

    cd hamvision/
    # Once H100 + WSL training has produced 3 seeds for a dataset:
    python inference_with_tricks.py --dataset isic2018 \
        --output_root ../outputs \
        --data_root ../data/ISIC2018

    # All segmentation datasets at once:
    python inference_with_tricks.py --all_seg \
        --output_root ../outputs \
        --data_dir_isic2018 ../data/ISIC2018 \
        --data_dir_isic2017 ../data/ISIC2017 \
        --data_dir_tn3k     ../data/TN3K \
        --data_dir_mmotu    ../data/MMOTU \
        --data_dir_acdc     ../data/ACDC

Methodology note
----------------

The "ensemble + TTA + tuned threshold" number is what we report as the
headline HamSeg value in Tab. 2; the per-seed values are what we
report as "mean ± std" in the same row, satisfying the reviewer's
statistical-significance request.

Threshold tuning is done on the held-out VALIDATION set only — the
test set is never touched until the final evaluation.  This is the
standard protocol in segmentation; nothing here is metric-gaming.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports — assume run from inside hamvision/
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hamseg import HamSeg, MedicalSegDataset  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================

def banner(s: str, ch: str = '=') -> None:
    print()
    print(ch * 70)
    print(f'  {s}')
    print(ch * 70)


def parse_args_json(p: Path) -> Dict:
    """Read args.json and coerce string values back to their proper types."""
    with p.open() as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            # Try to convert numeric / list strings back
            try:
                out[k] = ast.literal_eval(v)
                continue
            except (ValueError, SyntaxError):
                pass
        out[k] = v
    return out


def build_model_args(args_dict: Dict) -> 'argparse.Namespace':
    """Build a Namespace that HamSeg.__init__ accepts."""
    ns = argparse.Namespace()
    ns.embed_dim = int(args_dict.get('embed_dim', 48))
    ns.depths = list(args_dict.get('depths', [2, 2, 2, 2]))
    ns.damping_clamp = float(args_dict.get('damping_clamp', 5.0))
    ns.drop_rate = float(args_dict.get('drop_rate', 0.1))
    ns.img_size = int(args_dict.get('img_size', 224))
    ns.num_classes = int(args_dict.get('num_classes', 1))
    return ns


def load_checkpoint(seed_dir: Path, device: torch.device) -> Tuple[HamSeg, Dict]:
    args = parse_args_json(seed_dir / 'args.json')
    model = HamSeg(build_model_args(args)).to(device)
    ckpt_path = seed_dir / 'best_model.pth'
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Handle DataParallel wrapping if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, args


# ============================================================================
# TTA forward pass
# ============================================================================

@torch.no_grad()
def tta_predict(model: HamSeg, x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return per-pixel probabilities averaged across 4 geometric transforms.

    For binary (num_classes=1) returns sigmoid(logits) shape (B, 1, H, W).
    For multi-class returns softmax(logits, dim=1) shape (B, C, H, W).
    """
    def to_probs(logits: torch.Tensor) -> torch.Tensor:
        if num_classes == 1:
            return torch.sigmoid(logits.float())
        return torch.softmax(logits.float(), dim=1)

    # 4 geometric TTAs — flip back to original orientation before averaging
    p0 = to_probs(model(x))
    p1 = to_probs(model(x.flip(3))).flip(3)         # H-flip
    p2 = to_probs(model(x.flip(2))).flip(2)         # V-flip
    p3 = to_probs(model(x.flip([2, 3]))).flip([2, 3])  # both
    return (p0 + p1 + p2 + p3) / 4.0


@torch.no_grad()
def plain_predict(model: HamSeg, x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Single forward pass, no TTA. Returns probabilities."""
    if num_classes == 1:
        return torch.sigmoid(model(x).float())
    return torch.softmax(model(x).float(), dim=1)


# ============================================================================
# Metric computation (matches hamseg.compute_metrics for consistency)
# ============================================================================

def compute_metrics_from_probs(probs: torch.Tensor, targets: torch.Tensor,
                                num_classes: int, threshold: float = 0.5) -> Dict[str, float]:
    """probs: (N, C, H, W) sigmoid (C=1) or softmax (C>1).
       targets: (N, 1, H, W) for binary or (N, H, W) long for multi-class."""
    eps = 1e-7
    if num_classes == 1:
        preds = (probs > threshold).float()
        p = preds.reshape(-1)
        t = targets.reshape(-1).float()
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()
        return {
            'dice': (2 * tp + eps) / (2 * tp + fp + fn + eps),
            'miou': (tp + eps) / (tp + fp + fn + eps),
            'precision': (tp + eps) / (tp + fp + eps),
            'specificity': (tn + eps) / (tn + fp + eps),
            'accuracy': (tp + tn + eps) / (tp + tn + fp + fn + eps),
        }
    else:
        # multi-class: argmax preds, then compute per-class Dice/IoU and save them
        # all (so downstream tables can show RV/Myo/LV breakdowns for ACDC etc.).
        preds = probs.argmax(dim=1)  # (N, H, W)
        targets_squeezed = targets if targets.dim() == 3 else targets[:, 0]
        per_class_dice = {}
        per_class_iou = {}
        dice_sum, iou_sum, n_cls = 0.0, 0.0, 0
        tp_a, fp_a, fn_a, tn_a = 0.0, 0.0, 0.0, 0.0
        for c in range(1, num_classes):
            pc = (preds == c).float().reshape(-1)
            tc = (targets_squeezed == c).float().reshape(-1)
            tp = (pc * tc).sum().item()
            fp = (pc * (1 - tc)).sum().item()
            fn = ((1 - pc) * tc).sum().item()
            tn = ((1 - pc) * (1 - tc)).sum().item()
            d_c = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            i_c = (tp + eps) / (tp + fp + fn + eps)
            per_class_dice[f'class_{c}'] = d_c
            per_class_iou[f'class_{c}'] = i_c
            dice_sum += d_c
            iou_sum += i_c
            tp_a += tp; fp_a += fp; fn_a += fn; tn_a += tn
            n_cls += 1
        n_cls = max(n_cls, 1)
        out = {
            'dice': dice_sum / n_cls,
            'miou': iou_sum / n_cls,
            'precision': (tp_a + eps) / (tp_a + fp_a + eps),
            'specificity': (tn_a + eps) / (tn_a + fp_a + eps),
            'accuracy': (tp_a + tn_a + eps) / (tp_a + tn_a + fp_a + fn_a + eps),
            'per_class_dice': per_class_dice,
            'per_class_iou': per_class_iou,
        }
        return out


# ============================================================================
# Inference loop
# ============================================================================

@torch.no_grad()
def collect_probs(loader: DataLoader, predict_fn,
                  device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run predict_fn over the loader and return concatenated probs + targets."""
    all_probs, all_targets = [], []
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        probs = predict_fn(imgs)
        all_probs.append(probs.cpu())
        all_targets.append(masks)
    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def find_best_threshold_binary(probs: torch.Tensor, targets: torch.Tensor,
                                grid: np.ndarray = None) -> Tuple[float, float]:
    """Sweep threshold on (val) probs. Return (best_threshold, best_dice)."""
    if grid is None:
        grid = np.linspace(0.30, 0.70, 21)
    best_t, best_d = 0.5, -1.0
    for t in grid:
        m = compute_metrics_from_probs(probs, targets, num_classes=1, threshold=float(t))
        if m['dice'] > best_d:
            best_d = m['dice']
            best_t = float(t)
    return best_t, best_d


# ============================================================================
# Per-dataset orchestration
# ============================================================================

def run_dataset(dataset: str, data_root: str, output_root: Path,
                seeds: List[int], device: torch.device,
                batch_size: int = 8, num_workers: int = 4,
                tune_threshold: bool = True) -> Optional[Dict]:
    banner(f'{dataset.upper()}  —  inference with tricks')

    # Locate seed folders
    seed_dirs = []
    for s in seeds:
        d = output_root / dataset / f'seed_{s}'
        if not (d / 'best_model.pth').exists():
            print(f'  ⚠ missing checkpoint at {d}, skipping seed_{s}')
            continue
        seed_dirs.append((s, d))

    if not seed_dirs:
        print(f'  ✗ no seeds available for {dataset}, skipping')
        return None

    # Load every seed's model + args; verify arg consistency.
    models, all_args = {}, {}
    for s, d in seed_dirs:
        m, a = load_checkpoint(d, device)
        models[s] = m
        all_args[s] = a

    canonical = next(iter(all_args.values()))
    for s, a in all_args.items():
        for k in ('num_classes', 'img_size', 'embed_dim', 'depths'):
            if a.get(k) != canonical.get(k):
                print(f'  ⚠ seed {s} has different {k}={a.get(k)} '
                      f'(canonical={canonical.get(k)})')

    num_classes = int(canonical.get('num_classes', 1))
    img_size = int(canonical.get('img_size', 224))
    train_ratio = float(canonical.get('train_ratio', 0.7))
    val_ratio = float(canonical.get('val_ratio', 0.0))

    # Build val + test loaders. We use the same MedicalSegDataset that hamseg.py
    # uses, so the splits are identical to training.
    val_ds = MedicalSegDataset(data_root, 'val', img_size,
                               train_ratio, val_ratio, num_classes)
    test_ds = MedicalSegDataset(data_root, 'test', img_size,
                                train_ratio, val_ratio, num_classes)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f'  config: num_classes={num_classes}, img_size={img_size}, '
          f'val={len(val_ds)}, test={len(test_ds)}')
    print(f'  seeds available: {sorted(models)}')

    results: Dict = {
        'dataset': dataset,
        'num_classes': num_classes,
        'seeds': sorted(models),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'rows': [],
    }

    # ----- Per-seed: plain forward (no TTA), default 0.5 threshold -----
    seed_test_probs_plain: Dict[int, torch.Tensor] = {}
    seed_test_probs_tta: Dict[int, torch.Tensor] = {}
    test_targets: Optional[torch.Tensor] = None

    for s, m in models.items():
        t0 = time.time()
        probs, tgts = collect_probs(test_loader,
                                     lambda x: plain_predict(m, x, num_classes),
                                     device)
        seed_test_probs_plain[s] = probs
        if test_targets is None:
            test_targets = tgts
        m_plain = compute_metrics_from_probs(probs, tgts, num_classes, 0.5)
        dt = time.time() - t0
        print(f'  seed {s}  plain      dice={m_plain["dice"]:.4f}  miou={m_plain["miou"]:.4f}  ({dt:.1f}s)')
        results['rows'].append({'variant': f'seed_{s}',
                                **{k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in m_plain.items()}})

        # ----- + TTA -----
        t0 = time.time()
        probs_tta, _ = collect_probs(test_loader,
                                      lambda x: tta_predict(m, x, num_classes),
                                      device)
        seed_test_probs_tta[s] = probs_tta
        m_tta = compute_metrics_from_probs(probs_tta, test_targets, num_classes, 0.5)
        dt = time.time() - t0
        print(f'  seed {s}  + TTA      dice={m_tta["dice"]:.4f}  miou={m_tta["miou"]:.4f}  ({dt:.1f}s)')
        results['rows'].append({'variant': f'seed_{s} + TTA',
                                **{k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in m_tta.items()}})

    # ----- Ensemble (no TTA) -----
    ens_plain = torch.stack(list(seed_test_probs_plain.values())).mean(0)
    m_ens = compute_metrics_from_probs(ens_plain, test_targets, num_classes, 0.5)
    print(f'  ensemble (no TTA)        dice={m_ens["dice"]:.4f}  miou={m_ens["miou"]:.4f}')
    results['rows'].append({'variant': 'ensemble (3 seeds, no TTA)',
                            **{k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in m_ens.items()}})

    # ----- Ensemble + TTA -----
    ens_tta = torch.stack(list(seed_test_probs_tta.values())).mean(0)
    m_ens_tta = compute_metrics_from_probs(ens_tta, test_targets, num_classes, 0.5)
    print(f'  ensemble + TTA           dice={m_ens_tta["dice"]:.4f}  miou={m_ens_tta["miou"]:.4f}')
    results['rows'].append({'variant': 'ensemble + TTA',
                            **{k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in m_ens_tta.items()}})

    # ----- Ensemble + TTA + tuned threshold (binary only) -----
    if num_classes == 1 and tune_threshold:
        # Re-run on val with TTA, average across seeds
        seed_val_probs_tta = []
        val_targets = None
        for s, m in models.items():
            vp, vt = collect_probs(val_loader,
                                    lambda x: tta_predict(m, x, num_classes),
                                    device)
            seed_val_probs_tta.append(vp)
            if val_targets is None:
                val_targets = vt
        val_ens_tta = torch.stack(seed_val_probs_tta).mean(0)

        best_t, best_val_d = find_best_threshold_binary(val_ens_tta, val_targets)
        print(f'  threshold tuned on val   best_threshold={best_t:.3f}  '
              f'val_dice@best={best_val_d:.4f}  (default 0.5)')

        m_tuned = compute_metrics_from_probs(ens_tta, test_targets, num_classes, best_t)
        print(f'  ensemble + TTA + tuned   dice={m_tuned["dice"]:.4f}  miou={m_tuned["miou"]:.4f}  '
              f'(threshold={best_t:.3f})')
        results['rows'].append({'variant': f'ensemble + TTA + tuned threshold ({best_t:.3f})',
                                **{k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in m_tuned.items()}})
        results['tuned_threshold'] = best_t
        results['val_dice_at_best_threshold'] = round(best_val_d, 4)

    # ----- Save -----
    out_dir = output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'ensemble_results.json').open('w') as f:
        json.dump(results, f, indent=2)

    # Human-readable summary — now shows all 5 binary metrics, plus per-class
    # Dice for multi-class datasets (RV/Myo/LV for ACDC, etc.)
    rows_all = results['rows']
    is_multiclass = any('per_class_dice' in r for r in rows_all)
    lines = []
    if is_multiclass:
        # Collect class-name set from the first row that has per_class_dice
        per_keys = []
        for r in rows_all:
            if 'per_class_dice' in r:
                per_keys = sorted(r['per_class_dice'].keys())
                break
        # Optional ACDC-specific renaming for prettier output
        ds = results.get('dataset', '')
        if ds == 'acdc' and len(per_keys) == 3:
            label_map = {'class_1': 'RV', 'class_2': 'Myo', 'class_3': 'LV'}
            display_keys = [(k, label_map.get(k, k)) for k in per_keys]
        else:
            display_keys = [(k, k) for k in per_keys]
        header = f'{"Variant":<48s}  {"Dice":>7s}  {"mIoU":>7s}  ' + '  '.join(f'{lbl:>7s}' for _, lbl in display_keys)
        lines.append(header)
        lines.append('-' * len(header))
        for row in rows_all:
            cells = [f'{row["variant"]:<48s}', f'{row["dice"]:>7.4f}', f'{row["miou"]:>7.4f}']
            pcd = row.get('per_class_dice', {})
            for k, _ in display_keys:
                v = pcd.get(k)
                cells.append(f'{v:>7.4f}' if v is not None else f'{"-":>7s}')
            lines.append('  '.join(cells))
    else:
        # Binary: show all 5 standard metrics
        header = f'{"Variant":<48s}  {"Dice":>7s}  {"mIoU":>7s}  {"Spe":>7s}  {"Pre":>7s}  {"Acc":>7s}'
        lines.append(header)
        lines.append('-' * len(header))
        for row in rows_all:
            lines.append(
                f'{row["variant"]:<48s}  '
                f'{row["dice"]:>7.4f}  {row["miou"]:>7.4f}  '
                f'{row.get("specificity", float("nan")):>7.4f}  '
                f'{row.get("precision", float("nan")):>7.4f}  '
                f'{row.get("accuracy", float("nan")):>7.4f}'
            )
    summary = '\n'.join(lines)
    with (out_dir / 'ENSEMBLE_SUMMARY.txt').open('w') as f:
        f.write(summary + '\n')
    print()
    print(summary)
    return results


# ============================================================================
# CLI
# ============================================================================

ALL_SEG = ['isic2018', 'isic2017', 'tn3k', 'mmotu', 'acdc']


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--dataset', default=None,
                   help='Single dataset name (one of: ' + ', '.join(ALL_SEG) + ').')
    p.add_argument('--all_seg', action='store_true',
                   help='Run on every segmentation dataset that has 3 seeds available.')
    p.add_argument('--data_root', default=None,
                   help='Data folder for the chosen dataset (when --dataset is given).')

    p.add_argument('--data_dir_isic2018', default='./data/ISIC2018')
    p.add_argument('--data_dir_isic2017', default='./data/ISIC2017')
    p.add_argument('--data_dir_tn3k',     default='./data/TN3K')
    p.add_argument('--data_dir_mmotu',    default='./data/MMOTU')
    p.add_argument('--data_dir_acdc',     default='./data/ACDC')

    p.add_argument('--output_root', default='./outputs',
                   help='Where the seed-aware checkpoints live.')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_tune_threshold', action='store_true')

    a = p.parse_args()

    output_root = Path(a.output_root).resolve()
    if not output_root.exists():
        sys.exit(f'output_root not found: {output_root}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Output root: {output_root}')

    # Pick targets
    if a.all_seg:
        data_dirs = {
            'isic2018': a.data_dir_isic2018,
            'isic2017': a.data_dir_isic2017,
            'tn3k':     a.data_dir_tn3k,
            'mmotu':    a.data_dir_mmotu,
            'acdc':     a.data_dir_acdc,
        }
        targets = [(ds, dr) for ds, dr in data_dirs.items() if Path(dr).exists()]
        missing = [ds for ds, dr in data_dirs.items() if not Path(dr).exists()]
        if missing:
            print(f'Skipping (missing data dir): {missing}')
    elif a.dataset:
        if not a.data_root:
            sys.exit('When --dataset is specified you must also pass --data_root')
        targets = [(a.dataset, a.data_root)]
    else:
        sys.exit('Specify either --dataset / --data_root, or --all_seg.')

    all_results: Dict[str, Dict] = {}
    for ds, dr in targets:
        try:
            res = run_dataset(ds, dr, output_root, a.seeds, device,
                              batch_size=a.batch_size,
                              num_workers=a.num_workers,
                              tune_threshold=not a.no_tune_threshold)
            if res is not None:
                all_results[ds] = res
        except Exception as e:
            import traceback
            print(f'  ✗ {ds}: {e}')
            traceback.print_exc()

    # Master summary
    if all_results:
        banner('MASTER SUMMARY  —  all datasets, headline numbers', ch='*')
        print(f'{"dataset":<14s}  {"plain (best of 3)":<20s}  {"ensemble":<14s}  '
              f'{"+TTA":<14s}  {"+TTA+tuned":<14s}')
        for ds, res in all_results.items():
            rows = res['rows']
            # find best per-seed plain
            seed_dices = [r['dice'] for r in rows
                          if r['variant'].startswith('seed_')
                          and ' + TTA' not in r['variant']]
            best_plain = max(seed_dices) if seed_dices else 0
            ens_d = next((r['dice'] for r in rows
                          if r['variant'] == 'ensemble (3 seeds, no TTA)'), None)
            tta_d = next((r['dice'] for r in rows
                          if r['variant'] == 'ensemble + TTA'), None)
            tuned_d = next((r['dice'] for r in rows
                            if r['variant'].startswith('ensemble + TTA + tuned')), None)
            print(f'{ds:<14s}  {best_plain:<20.4f}  '
                  f'{ens_d if ens_d is not None else "-":<14}  '
                  f'{tta_d if tta_d is not None else "-":<14}  '
                  f'{tuned_d if tuned_d is not None else "-":<14}')

        # Save master file
        with (output_root / 'ENSEMBLE_MASTER.json').open('w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nMaster JSON: {output_root / "ENSEMBLE_MASTER.json"}')


if __name__ == '__main__':
    main()
