#!/usr/bin/env python3
"""
eval_perclass_metrics.py — re-evaluate trained HamSeg checkpoints with the
COMPLETE metric panel reviewers expect:

  * Binary datasets (ISIC 2017/2018, TN3K, MMOTU): DSC, mIoU, specificity,
    precision, accuracy. Reported per seed and as 3-seed ensemble + TTA.
  * Multi-class datasets (ACDC): overall DSC/mIoU + per-class Dice
    (background omitted, foreground classes labelled e.g. RV/Myo/LV).

This is a thin orchestration layer over inference_with_tricks.py — it reuses
that file's `compute_metrics_from_probs`, `tta_predict`, `plain_predict`,
and `collect_probs`, then writes a paper-ready CSV per dataset.

Usage
-----
    cd hamvision/
    python eval_perclass_metrics.py --dataset acdc \
        --output_root ../outputs \
        --data_root ../data/ACDC

    # All segmentation datasets at once:
    python eval_perclass_metrics.py --all_seg \
        --output_root ../outputs \
        --data_dir_isic2018 ../data/ISIC2018 \
        --data_dir_isic2017 ../data/ISIC2017 \
        --data_dir_tn3k     ../data/TN3K \
        --data_dir_mmotu    ../data/MMOTU \
        --data_dir_acdc     ../data/ACDC

Outputs
-------
For each dataset under {output_root}/{dataset}/:
    metrics_extended.json  - full structured results (per-seed, ensemble,
                             per-class breakdowns, mean +/- std)
    metrics_paper.csv      - flat CSV, 1 row per (seed, variant), suitable
                             for pasting into a manuscript table builder.

A master metrics_paper_master.csv combining all datasets is written at
{output_root}/metrics_paper_master.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse all the existing helpers.
from inference_with_tricks import (
    compute_metrics_from_probs,
    tta_predict,
    plain_predict,
    collect_probs,
    parse_args_json,
    build_model_args,
    load_checkpoint,
)
from torch.utils.data import DataLoader
from hamseg import MedicalSegDataset

# --- ACDC class label mapping for nicer reporting ---
ACDC_LABELS = {'class_1': 'RV', 'class_2': 'Myo', 'class_3': 'LV'}

DEFAULT_SEEDS = [42, 43, 44]
SEG_DATASETS = ['isic2018', 'isic2017', 'tn3k', 'mmotu', 'acdc']


def class_label(dataset: str, key: str) -> str:
    if dataset == 'acdc':
        return ACDC_LABELS.get(key, key)
    return key


def aggregate_seed_metrics(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean +/- std across per-seed rows for each numeric metric."""
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    summary = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        summary[k] = {'mean': float(arr.mean()), 'std': float(arr.std(ddof=0))}
    # Also handle nested per_class_dice/per_class_iou
    for nested in ('per_class_dice', 'per_class_iou'):
        seen = {}
        for r in rows:
            d = r.get(nested) or {}
            for k, v in d.items():
                seen.setdefault(k, []).append(float(v))
        if seen:
            summary[nested] = {k: {'mean': float(np.mean(v)),
                                   'std': float(np.std(v, ddof=0))}
                               for k, v in seen.items()}
    return summary


def evaluate_dataset(dataset: str, output_root: Path, data_root: str,
                     seeds: List[int], device: torch.device,
                     img_size: int = 224, num_workers: int = 4) -> Dict:
    """Evaluate per-seed and ensemble+TTA on one dataset. Return structured dict."""
    print(f'\n=== {dataset} ===  data_root={data_root}')

    # Find seed directories
    out_dir = output_root / dataset
    seed_dirs = [out_dir / f'seed_{s}' for s in seeds]
    seed_dirs = [d for d in seed_dirs if d.exists() and (d / 'best_model.pth').exists()]
    if not seed_dirs:
        print(f'  no seed dirs with best_model.pth at {out_dir}, skipping')
        return {'dataset': dataset, 'rows': [], 'aggregate': {}}

    # Load canonical args from seed_42 (or whichever exists first)
    args_dict = parse_args_json(seed_dirs[0] / 'args.json')
    args = build_model_args(args_dict)
    num_classes = int(getattr(args, 'num_classes', 1))
    img_size = int(getattr(args, 'img_size', img_size))
    print(f'  num_classes={num_classes}, img_size={img_size}, seeds={[d.name for d in seed_dirs]}')

    # Build test loader from data_root
    test_ds = MedicalSegDataset(data_root, 'test', img_size,
                                 getattr(args, 'train_ratio', 0.7),
                                 getattr(args, 'val_ratio', 0.0),
                                 num_classes)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    rows = []
    seed_probs_tta = []
    seed_probs_plain = []
    cached_targets = None

    for seed_dir in seed_dirs:
        print(f'  evaluating {seed_dir.name}...')
        model, _ = load_checkpoint(seed_dir, device)
        # Plain
        p_plain, t_plain = collect_probs(test_loader,
                                          lambda x: plain_predict(model, x, num_classes),
                                          device)
        m_plain = compute_metrics_from_probs(p_plain, t_plain, num_classes, 0.5)
        m_plain['variant'] = f'{seed_dir.name} (plain)'
        m_plain['seed'] = seed_dir.name.replace('seed_', '')
        m_plain['source'] = 'per_seed_plain'
        rows.append(m_plain)
        # TTA
        p_tta, t_tta = collect_probs(test_loader,
                                      lambda x: tta_predict(model, x, num_classes),
                                      device)
        m_tta = compute_metrics_from_probs(p_tta, t_tta, num_classes, 0.5)
        m_tta['variant'] = f'{seed_dir.name} + TTA'
        m_tta['seed'] = seed_dir.name.replace('seed_', '')
        m_tta['source'] = 'per_seed_tta'
        rows.append(m_tta)

        seed_probs_plain.append(p_plain)
        seed_probs_tta.append(p_tta)
        cached_targets = t_plain

    # Ensemble (no TTA)
    ens_plain = torch.stack(seed_probs_plain, dim=0).mean(dim=0)
    m_ens_plain = compute_metrics_from_probs(ens_plain, cached_targets, num_classes, 0.5)
    m_ens_plain['variant'] = f'ensemble({len(seed_probs_plain)}) plain'
    m_ens_plain['source'] = 'ensemble_plain'
    rows.append(m_ens_plain)

    # Ensemble + TTA
    ens_tta = torch.stack(seed_probs_tta, dim=0).mean(dim=0)
    m_ens_tta = compute_metrics_from_probs(ens_tta, cached_targets, num_classes, 0.5)
    m_ens_tta['variant'] = f'ensemble({len(seed_probs_tta)}) + TTA'
    m_ens_tta['source'] = 'ensemble_tta'
    rows.append(m_ens_tta)

    # Aggregate per-seed (plain) for reporting alongside ensemble
    per_seed_plain = [r for r in rows if r.get('source') == 'per_seed_plain']
    aggregate = aggregate_seed_metrics(per_seed_plain)

    return {
        'dataset': dataset,
        'num_classes': num_classes,
        'rows': rows,
        'aggregate_per_seed_plain': aggregate,
    }


def write_dataset_outputs(result: Dict, output_root: Path) -> Path:
    """Write metrics_extended.json and metrics_paper.csv for one dataset."""
    ds = result['dataset']
    out_dir = output_root / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'metrics_extended.json'
    json_path.write_text(json.dumps(result, indent=2, default=str))

    # CSV: one row per variant, with per-class columns expanded
    csv_path = out_dir / 'metrics_paper.csv'
    fieldnames = ['dataset', 'variant', 'source', 'seed',
                  'dice', 'miou', 'precision', 'specificity', 'accuracy']
    # Add per-class columns if any row has them
    pc_keys = set()
    for r in result['rows']:
        pc_keys.update((r.get('per_class_dice') or {}).keys())
    pc_keys = sorted(pc_keys)
    fieldnames += [f'dice_{class_label(ds, k)}' for k in pc_keys]
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in result['rows']:
            row = {k: r.get(k) for k in fieldnames if not k.startswith('dice_') or k in ('dice',)}
            row['dataset'] = ds
            for k in pc_keys:
                row[f'dice_{class_label(ds, k)}'] = (r.get('per_class_dice') or {}).get(k)
            w.writerow(row)

    return csv_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=str, default=None,
                   help='Single dataset name (one of ' + ', '.join(SEG_DATASETS) + ')')
    p.add_argument('--all_seg', action='store_true',
                   help='Evaluate all segmentation datasets.')
    p.add_argument('--output_root', type=str, default='./outputs')
    p.add_argument('--data_root', type=str, default=None,
                   help='Path for the single --dataset')
    p.add_argument('--data_dir_isic2018', type=str, default=None)
    p.add_argument('--data_dir_isic2017', type=str, default=None)
    p.add_argument('--data_dir_tn3k', type=str, default=None)
    p.add_argument('--data_dir_mmotu', type=str, default=None)
    p.add_argument('--data_dir_acdc', type=str, default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--num_workers', type=int, default=4)
    a = p.parse_args()

    output_root = Path(a.output_root).resolve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if a.all_seg:
        targets = [
            ('isic2018', a.data_dir_isic2018),
            ('isic2017', a.data_dir_isic2017),
            ('tn3k',     a.data_dir_tn3k),
            ('mmotu',    a.data_dir_mmotu),
            ('acdc',     a.data_dir_acdc),
        ]
    elif a.dataset:
        targets = [(a.dataset, a.data_root)]
    else:
        sys.exit('Need --dataset or --all_seg')

    master_rows = []
    for ds, droot in targets:
        if not droot:
            print(f'\n[SKIP] {ds}: no data_root provided')
            continue
        result = evaluate_dataset(ds, output_root, droot,
                                  seeds=a.seeds, device=device,
                                  num_workers=a.num_workers)
        if not result['rows']:
            continue
        write_dataset_outputs(result, output_root)
        for r in result['rows']:
            r2 = dict(r)
            r2['dataset'] = ds
            master_rows.append(r2)

    if master_rows:
        master_csv = output_root / 'metrics_paper_master.csv'
        # Determine union of fieldnames
        keys = set()
        for r in master_rows:
            keys.update(r.keys())
        # Flatten per-class keys
        per_class_keys = set()
        for r in master_rows:
            for k in (r.get('per_class_dice') or {}):
                per_class_keys.add(class_label(r.get('dataset', ''), k))
        keys = (['dataset', 'variant', 'source', 'seed',
                 'dice', 'miou', 'precision', 'specificity', 'accuracy']
                + [f'dice_{k}' for k in sorted(per_class_keys)])
        with master_csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in master_rows:
                row = {k: r.get(k) for k in keys}
                pcd = r.get('per_class_dice') or {}
                ds = r.get('dataset', '')
                for k, v in pcd.items():
                    row[f'dice_{class_label(ds, k)}'] = v
                w.writerow(row)
        print(f'\nMaster CSV: {master_csv}')


if __name__ == '__main__':
    main()
