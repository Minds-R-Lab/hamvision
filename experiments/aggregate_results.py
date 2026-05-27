#!/usr/bin/env python3
"""
aggregate_results.py
====================

Walks the seed-aware output tree produced by run_multi_seed.py and computes
mean/std (and optional paired Wilcoxon p-values vs. a comma-separated baseline
table) for every (task, dataset) cell. Writes:

    {output_root}/{dataset}/aggregate.json     machine-readable per-dataset
    {output_root}/{dataset}/SUMMARY.txt        human-readable per-dataset
    {output_root}/AGGREGATE.csv                one row per (dataset, metric)
    {output_root}/AGGREGATE.md                 a Markdown table for the paper

Inputs
------
By default we look for `seed_*/test_results_final.json` under every dataset
directory; you can override the glob with --seed_pattern.

For segmentation, the metric of record is "dice" (with mIoU as secondary).
For classification, it is "accuracy" (with AUC as secondary).

Usage
-----

  # Aggregate everything we have
  python aggregate_results.py --output_root ./outputs

  # Restrict to one dataset
  python aggregate_results.py --output_root ./outputs --dataset isic2018

  # Compare against a baseline CSV (one row per dataset, columns: dataset,dice,miou,acc,auc)
  python aggregate_results.py --output_root ./outputs --baselines baselines.csv

The baseline file is optional; without it we just report mean ± std.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------
# Per-task metric semantics
# ---------------------------------------------------------------

SEG_DATASETS = {'isic2018', 'isic2017', 'tn3k', 'acdc'}
CLS_DATASETS = {'pathmnist', 'bloodmnist', 'dermamnist', 'breastmnist',
                'organsmnist', 'organamnist', 'organcmnist', 'retinamnist',
                'octmnist', 'pneumoniamnist', 'tissuemnist'}


def detect_task(dataset: str) -> str:
    if dataset in SEG_DATASETS:
        return 'seg'
    if dataset in CLS_DATASETS:
        return 'cls'
    return 'unknown'


def extract_pct(result: Dict, task: str) -> Tuple[Optional[float], Optional[float]]:
    """Pull (primary, secondary) percent values out of one test_results_final.json."""
    if task == 'seg':
        test = result.get('test') or {}
        d = test.get('dice')
        m = test.get('miou')
        if d is None or m is None:
            return None, None
        # hamseg saves decimal fractions
        return d * 100.0, m * 100.0
    elif task == 'cls':
        a = result.get('accuracy')
        u = result.get('auc')
        if a is None or u is None:
            return None, None
        # hamcls saves percentages directly
        return float(a), float(u)
    return None, None


def primary_secondary_labels(task: str) -> Tuple[str, str]:
    return ('dice', 'miou') if task == 'seg' else ('accuracy', 'auc')


# ---------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------

def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def wilcoxon(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sided paired Wilcoxon signed-rank p-value. Returns None if unavailable
    (sample too small or scipy missing)."""
    if len(a) != len(b) or len(a) < 5:
        return None
    try:
        from scipy.stats import wilcoxon as _wilcoxon
        stat = _wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(stat.pvalue)
    except (ImportError, ValueError):
        return None


def paired_t(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sided paired t-test as a fallback when n is small (n<5)."""
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    md = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) >= 2 else 0.0
    if sd == 0:
        return None
    t = md / (sd / math.sqrt(len(diffs)))
    # Approximate two-sided p-value with the survival function of t-distribution.
    try:
        from scipy.stats import t as _t
        return float(2 * (1 - _t.cdf(abs(t), df=len(diffs) - 1)))
    except ImportError:
        # Crude normal approximation
        from math import erf, sqrt
        z = abs(t)
        return float(2 * (1 - 0.5 * (1 + erf(z / sqrt(2)))))


# ---------------------------------------------------------------
# Per-dataset aggregation
# ---------------------------------------------------------------

def load_seed_results(dataset_dir: Path, seed_pattern: str) -> List[Dict]:
    """Return a list of (seed, test_results_final dict) for every seed under dataset_dir."""
    results = []
    for sd in sorted(dataset_dir.glob(seed_pattern)):
        f = sd / 'test_results_final.json'
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        # Extract seed from folder name "seed_42" -> 42
        try:
            seed = int(sd.name.split('_')[-1])
        except ValueError:
            seed = -1
        results.append({'seed': seed, 'folder': str(sd), 'data': d})
    return results


def aggregate_dataset(dataset_dir: Path, seed_pattern: str = 'seed_*',
                      baseline_pri: Optional[List[float]] = None,
                      baseline_sec: Optional[List[float]] = None) -> Optional[Dict]:
    dataset = dataset_dir.name
    task = detect_task(dataset)
    if task == 'unknown':
        return None

    seeds = load_seed_results(dataset_dir, seed_pattern)
    if not seeds:
        return None

    pri_label, sec_label = primary_secondary_labels(task)
    pri_vals, sec_vals = [], []
    rows = []
    for entry in seeds:
        pri, sec = extract_pct(entry['data'], task)
        if pri is None:
            continue
        pri_vals.append(pri)
        sec_vals.append(sec)
        rows.append({'seed': entry['seed'],
                     'folder': entry['folder'],
                     pri_label: round(pri, 4),
                     sec_label: round(sec, 4)})

    if not pri_vals:
        return None

    pri_mean, pri_std = mean_std(pri_vals)
    sec_mean, sec_std = mean_std(sec_vals)

    agg = {
        'dataset':   dataset,
        'task':      task,
        'n_seeds':   len(pri_vals),
        'seeds':     [r['seed'] for r in rows],
        pri_label:   {'mean': round(pri_mean, 4),
                      'std':  round(pri_std, 4),
                      'min':  round(min(pri_vals), 4),
                      'max':  round(max(pri_vals), 4)},
        sec_label:   {'mean': round(sec_mean, 4),
                      'std':  round(sec_std, 4),
                      'min':  round(min(sec_vals), 4),
                      'max':  round(max(sec_vals), 4)},
        'per_seed':  rows,
    }

    # Optional comparisons against a paired baseline.
    if baseline_pri is not None and len(baseline_pri) == len(pri_vals):
        pw = wilcoxon(pri_vals, baseline_pri) or paired_t(pri_vals, baseline_pri)
        agg[pri_label]['p_value_vs_baseline'] = pw
    if baseline_sec is not None and len(baseline_sec) == len(sec_vals):
        pw = wilcoxon(sec_vals, baseline_sec) or paired_t(sec_vals, baseline_sec)
        agg[sec_label]['p_value_vs_baseline'] = pw

    return agg


# ---------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------

def write_summary_txt(dataset_dir: Path, agg: Dict) -> None:
    pri_label, sec_label = primary_secondary_labels(agg['task'])
    pri = agg[pri_label]
    sec = agg[sec_label]
    lines = []
    lines.append('=' * 60)
    lines.append(f'  {agg["dataset"]}  ({agg["task"].upper()})')
    lines.append('=' * 60)
    lines.append(f'  Seeds                : {agg["seeds"]}')
    lines.append(f'  N runs aggregated    : {agg["n_seeds"]}')
    lines.append('')
    lines.append(f'  {pri_label.upper():12s}  '
                 f'mean = {pri["mean"]:6.2f}    '
                 f'std = {pri["std"]:5.3f}    '
                 f'range = [{pri["min"]:.2f}, {pri["max"]:.2f}]')
    lines.append(f'  {sec_label.upper():12s}  '
                 f'mean = {sec["mean"]:6.2f}    '
                 f'std = {sec["std"]:5.3f}    '
                 f'range = [{sec["min"]:.2f}, {sec["max"]:.2f}]')
    lines.append('')
    lines.append('  Per-seed values:')
    for row in agg['per_seed']:
        lines.append(f'    seed={row["seed"]}    '
                     f'{pri_label}={row[pri_label]:.2f}    '
                     f'{sec_label}={row[sec_label]:.2f}')
    p = dataset_dir / 'SUMMARY.txt'
    p.write_text('\n'.join(lines) + '\n')


def write_master_csv(output_root: Path, all_aggs: List[Dict]) -> None:
    rows = []
    for agg in all_aggs:
        pri_label, sec_label = primary_secondary_labels(agg['task'])
        rows.append({
            'task':     agg['task'],
            'dataset':  agg['dataset'],
            'n_seeds':  agg['n_seeds'],
            'pri_label': pri_label,
            'pri_mean': agg[pri_label]['mean'],
            'pri_std':  agg[pri_label]['std'],
            'sec_label': sec_label,
            'sec_mean': agg[sec_label]['mean'],
            'sec_std':  agg[sec_label]['std'],
        })
    p = output_root / 'AGGREGATE.csv'
    with p.open('w', newline='') as f:
        if not rows:
            f.write('')
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_master_md(output_root: Path, all_aggs: List[Dict]) -> None:
    lines = []
    lines.append(f'# Aggregate results — `{output_root.name}`\n')
    lines.append(f'Generated by `aggregate_results.py`. '
                 f'{len(all_aggs)} (dataset, task) cells.\n')

    seg = [a for a in all_aggs if a['task'] == 'seg']
    cls = [a for a in all_aggs if a['task'] == 'cls']

    if seg:
        lines.append('## Segmentation\n')
        lines.append('| Dataset | n | Dice (mean ± std) | mIoU (mean ± std) |')
        lines.append('|---|---|---|---|')
        for a in seg:
            d = a['dice']; m = a['miou']
            lines.append(f'| {a["dataset"]} | {a["n_seeds"]} | '
                         f'{d["mean"]:.2f} ± {d["std"]:.2f} | '
                         f'{m["mean"]:.2f} ± {m["std"]:.2f} |')
        lines.append('')

    if cls:
        lines.append('## Classification\n')
        lines.append('| Dataset | n | ACC (mean ± std) | AUC (mean ± std) |')
        lines.append('|---|---|---|---|')
        for a in cls:
            ac = a['accuracy']; au = a['auc']
            lines.append(f'| {a["dataset"]} | {a["n_seeds"]} | '
                         f'{ac["mean"]:.2f} ± {ac["std"]:.2f} | '
                         f'{au["mean"]:.2f} ± {au["std"]:.2f} |')
        lines.append('')

    p = output_root / 'AGGREGATE.md'
    p.write_text('\n'.join(lines))


# ---------------------------------------------------------------
# Optional baseline CSV (paired comparisons)
# ---------------------------------------------------------------

def load_baseline_csv(p: Path) -> Dict[str, Dict[str, List[float]]]:
    """A CSV with columns dataset, seed, dice, miou, accuracy, auc (any subset)."""
    out: Dict[str, Dict[str, List[float]]] = {}
    if not p.exists():
        return out
    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get('dataset', '').lower()
            if not ds:
                continue
            entry = out.setdefault(ds, {})
            for key in ('dice', 'miou', 'accuracy', 'auc'):
                if row.get(key):
                    entry.setdefault(key, []).append(float(row[key]))
    return out


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Aggregate multi-seed HamVision results into mean ± std tables.')
    p.add_argument('--output_root', required=True,
                   help='Output root used by run_multi_seed.py')
    p.add_argument('--dataset', default=None,
                   help='Restrict aggregation to one dataset.')
    p.add_argument('--seed_pattern', default='seed_*',
                   help='Glob to identify seed sub-folders. Default: seed_*')
    p.add_argument('--baselines', default=None,
                   help='Optional CSV with per-seed baseline metrics for paired tests.')
    args = p.parse_args()

    root = Path(args.output_root).resolve()
    if not root.exists():
        raise SystemExit(f'output_root not found: {root}')

    # Load optional baselines
    baselines = load_baseline_csv(Path(args.baselines)) if args.baselines else {}

    candidate_dirs = []
    if args.dataset:
        cand = root / args.dataset
        if cand.is_dir():
            candidate_dirs.append(cand)
    else:
        for child in sorted(root.iterdir()):
            if child.is_dir() and detect_task(child.name) != 'unknown':
                candidate_dirs.append(child)

    all_aggs = []
    for ds_dir in candidate_dirs:
        ds = ds_dir.name
        bl = baselines.get(ds, {})
        bl_pri = bl.get('dice') if detect_task(ds) == 'seg' else bl.get('accuracy')
        bl_sec = bl.get('miou') if detect_task(ds) == 'seg' else bl.get('auc')
        agg = aggregate_dataset(ds_dir, args.seed_pattern,
                                baseline_pri=bl_pri, baseline_sec=bl_sec)
        if agg is None:
            print(f'⚠ {ds}: no seed runs found, skipping')
            continue
        # Per-dataset outputs
        with (ds_dir / 'aggregate.json').open('w') as f:
            json.dump(agg, f, indent=2)
        write_summary_txt(ds_dir, agg)
        all_aggs.append(agg)
        pri_label, _ = primary_secondary_labels(agg['task'])
        pri = agg[pri_label]
        print(f'✓ {ds:14s}  n={agg["n_seeds"]}  '
              f'{pri_label}={pri["mean"]:.2f} ± {pri["std"]:.2f}')

    if all_aggs:
        write_master_csv(root, all_aggs)
        write_master_md(root, all_aggs)
        print()
        print(f'Wrote master CSV  : {root}/AGGREGATE.csv')
        print(f'Wrote master Md   : {root}/AGGREGATE.md')
    else:
        print('No aggregations produced.')


if __name__ == '__main__':
    main()
