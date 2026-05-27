#!/usr/bin/env python3
"""
find_reported_checkpoints.py
============================

Walks an output folder tree, reads every test_results_*.json that hamseg.py /
hamcls.py wrote, and matches the metrics against the numbers reported in
Tables 2 and 3 of the HamVision manuscript. For each dataset, prints the
checkpoint folder whose final-test JSON is closest to the reported number,
plus the next-best alternatives so you can spot ambiguities.

The script also reports the *args* the matching run was launched with
(saved as part of the JSON), so we can identify the exact training
configuration that produced the manuscript's headline numbers.

Usage
-----
# walk every folder under ./outputs
python find_reported_checkpoints.py --root ./outputs

# limit to a single dataset
python find_reported_checkpoints.py --root ./outputs --dataset acdc

# show the args dict for the best-matching run
python find_reported_checkpoints.py --root ./outputs --dataset isic2018 --show-args

# print machine-readable CSV instead of pretty tables
python find_reported_checkpoints.py --root ./outputs --csv > checkpoint_audit.csv

Notes
-----
* hamseg.py stores 'test' metrics as decimal fractions (0.8938 == 89.38 %).
* hamcls.py stores 'accuracy' / 'auc' already in percent (98.85 == 98.85 %).
* The script handles both conventions automatically.
* Match tolerance defaults to 0.05 percentage points.
"""

import argparse
import json
import os
from pathlib import Path

# ---------------------------------------------------------------
# Reported values from the submitted manuscript (Tables 2 and 3)
# ---------------------------------------------------------------
REPORTED = {
    # segmentation, decimal fractions in JSON
    'isic2018': {'task': 'seg', 'dice': 89.38, 'miou': 81.22, 'params': '8.57M'},
    'isic2017': {'task': 'seg', 'dice': 88.40, 'miou': 79.20, 'params': '8.57M'},
    'tn3k':     {'task': 'seg', 'dice': 87.05, 'miou': 76.33, 'params': '8.57M'},
    'acdc':     {'task': 'seg', 'dice': 92.40, 'miou': 86.14, 'params': '8.57M'},

    # classification, percentages in JSON
    'bloodmnist':  {'task': 'cls', 'accuracy': 98.85, 'auc': 99.93},
    'pathmnist':   {'task': 'cls', 'accuracy': 96.65, 'auc': 99.36},
    'dermamnist':  {'task': 'cls', 'accuracy': 77.96, 'auc': 93.66},
    'breastmnist': {'task': 'cls', 'accuracy': 89.60, 'auc': 89.94},
    'organsmnist': {'task': 'cls', 'accuracy': 80.96, 'auc': 98.02},
    'retinamnist': {'task': 'cls', 'accuracy': 56.75, 'auc': 76.24},
}


def detect_dataset(path: Path):
    """Pick the dataset that appears in the path string. Returns None if ambiguous."""
    s = str(path).replace('\\', '/').lower()
    hits = [ds for ds in REPORTED if ('/' + ds + '/') in s
            or s.endswith('/' + ds)
            or ('/' + ds + '_') in s
            or ('_' + ds + '/') in s]
    if len(hits) == 1:
        return hits[0]
    # fall back to longest match (organsmnist contains 'organ', 'organa' might also)
    if hits:
        return max(hits, key=len)
    return None


def load_json(path: Path):
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def extract_metrics(result: dict, task: str):
    """
    Return (primary_pct, secondary_pct) where primary is dice/accuracy and
    secondary is miou/auc, both as percentages. Returns (None, None) if the
    JSON does not contain the expected keys.
    """
    if result is None:
        return None, None

    if task == 'seg':
        # hamseg's test_results_*.json
        test = result.get('test', {})
        d = test.get('dice')
        m = test.get('miou')
        if d is None or m is None:
            return None, None
        # Decimal fraction => percent
        return d * 100.0, m * 100.0
    else:
        # hamcls's test_results_*.json (already percentages)
        a = result.get('accuracy')
        u = result.get('auc')
        if a is None or u is None:
            return None, None
        return float(a), float(u)


def score(actual_pri, actual_sec, exp_pri, exp_sec):
    """Lower is better. Sum of absolute differences in percentage points."""
    return abs(actual_pri - exp_pri) + abs(actual_sec - exp_sec)


def args_summary(result: dict):
    """One-line summary of the launch arguments saved in the JSON."""
    args = result.get('args') or {}
    if not args:
        return '(no args saved)'
    keys = ['embed_dim', 'depths', 'epochs', 'batch_size', 'lr', 'weight_decay',
            'drop_rate', 'damping_clamp', 'img_size', 'size', 'num_classes',
            'seed', 'balanced']
    parts = []
    for k in keys:
        if k in args:
            parts.append(f'{k}={args[k]}')
    return ', '.join(parts) if parts else f'{len(args)} keys: {sorted(args.keys())}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='Top-level folder to walk. Every test_results_*.json '
                         'underneath will be considered.')
    ap.add_argument('--dataset', default=None,
                    help='Restrict to one dataset (isic2018, acdc, bloodmnist, ...).')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='Match tolerance in percentage points (default 0.05).')
    ap.add_argument('--top', type=int, default=5,
                    help='Show top-N alternatives per dataset (default 5).')
    ap.add_argument('--show-args', action='store_true',
                    help='Print the launch args of every shown row.')
    ap.add_argument('--csv', action='store_true',
                    help='Emit machine-readable CSV instead of pretty tables.')
    a = ap.parse_args()

    root = Path(a.root).resolve()
    if not root.exists():
        print(f'Root does not exist: {root}')
        return

    files = sorted(root.glob('**/test_results_*.json'))
    print(f'Walked {root}\nFound {len(files)} test_results_*.json files\n')

    candidates = []  # list of dicts
    skipped_reasons = {'no_dataset': 0, 'json_load': 0, 'no_metrics': 0}

    for fp in files:
        ds = detect_dataset(fp)
        if ds is None:
            skipped_reasons['no_dataset'] += 1
            continue
        if a.dataset and ds != a.dataset:
            continue

        result = load_json(fp)
        if result is None:
            skipped_reasons['json_load'] += 1
            continue

        task = REPORTED[ds]['task']
        pri, sec = extract_metrics(result, task)
        if pri is None:
            skipped_reasons['no_metrics'] += 1
            continue

        if task == 'seg':
            exp_pri = REPORTED[ds]['dice']
            exp_sec = REPORTED[ds]['miou']
            pri_label, sec_label = 'dice', 'miou'
        else:
            exp_pri = REPORTED[ds]['accuracy']
            exp_sec = REPORTED[ds]['auc']
            pri_label, sec_label = 'acc', 'auc'

        s = score(pri, sec, exp_pri, exp_sec)
        match_pri = abs(pri - exp_pri) <= a.tol
        match_sec = abs(sec - exp_sec) <= a.tol
        match = match_pri and match_sec

        candidates.append({
            'dataset': ds,
            'file': str(fp.relative_to(root)),
            'folder': str(fp.parent.relative_to(root)),
            'tag': result.get('tag', fp.stem.replace('test_results_', '')),
            'best_epoch': result.get('best_epoch') or result.get('test', {}).get('best_epoch'),
            'pri_label': pri_label,
            'pri': pri,
            'expected_pri': exp_pri,
            'sec_label': sec_label,
            'sec': sec,
            'expected_sec': exp_sec,
            'score': s,
            'match': match,
            'params': result.get('params'),
            'args_summary': args_summary(result),
        })

    if not candidates:
        print('No matching JSON files found. Reasons skipped:', skipped_reasons)
        return

    candidates.sort(key=lambda r: (r['dataset'], r['score']))

    if a.csv:
        keys = ['dataset', 'folder', 'tag', 'best_epoch',
                'pri_label', 'pri', 'expected_pri',
                'sec_label', 'sec', 'expected_sec',
                'score', 'match', 'params', 'args_summary']
        print(','.join(keys))
        for r in candidates:
            row = []
            for k in keys:
                v = r[k]
                if v is None:
                    row.append('')
                elif isinstance(v, float):
                    row.append(f'{v:.4f}')
                else:
                    s = str(v).replace(',', ';').replace('\n', ' ')
                    row.append(s)
            print(','.join(row))
        return

    # Pretty print
    print(f'Tolerance: ±{a.tol} percentage points\n')
    print('Symbol legend:  ✓ = both metrics within tol  ~ = one within tol  · = neither\n')

    for ds in sorted(set(r['dataset'] for r in candidates)):
        ds_rows = [r for r in candidates if r['dataset'] == ds]
        if not ds_rows:
            continue

        print('=' * 78)
        target = REPORTED[ds]
        if target['task'] == 'seg':
            print(f'{ds.upper()}    target: dice={target["dice"]:.2f}  miou={target["miou"]:.2f}    '
                  f'({len(ds_rows)} candidate JSONs)')
        else:
            print(f'{ds.upper()}    target: acc={target["accuracy"]:.2f}  auc={target["auc"]:.2f}    '
                  f'({len(ds_rows)} candidate JSONs)')
        print('=' * 78)

        header_fmt = '  {sym} {score:>6s}  {pri:>6s} (Δ{dpri:>5s})  {sec:>6s} (Δ{dsec:>5s})  ' \
                     'epoch={ep:>4s}  tag={tag}'
        for i, r in enumerate(ds_rows[:a.top]):
            if r['match']:
                sym = '✓'
            elif abs(r['pri'] - r['expected_pri']) <= a.tol or \
                 abs(r['sec'] - r['expected_sec']) <= a.tol:
                sym = '~'
            else:
                sym = '·'
            print(header_fmt.format(
                sym=sym,
                score=f'{r["score"]:.3f}',
                pri=f'{r["pri"]:.2f}',
                dpri=f'{r["pri"]-r["expected_pri"]:+.2f}',
                sec=f'{r["sec"]:.2f}',
                dsec=f'{r["sec"]-r["expected_sec"]:+.2f}',
                ep=str(r['best_epoch']) if r['best_epoch'] is not None else '?',
                tag=r['tag'],
            ))
            print(f'      folder: {r["folder"]}')
            if a.show_args:
                print(f'      args:   {r["args_summary"]}')
            if r['params']:
                params_pretty = (f'{r["params"]:,}' if isinstance(r['params'], int)
                                 else str(r['params']))
                print(f'      params: {params_pretty}')

            # Look for sibling best_model.pth
            ckpt = (root / r['folder'] / 'best_model.pth')
            if ckpt.exists():
                size_mb = ckpt.stat().st_size / 1024 / 1024
                print(f'      ckpt:   best_model.pth  ({size_mb:.1f} MB)  '
                      f'mtime={ckpt.stat().st_mtime:.0f}')
            else:
                print(f'      ckpt:   (no best_model.pth in {r["folder"]})')
            print()

        if len(ds_rows) > a.top:
            print(f'  ... {len(ds_rows) - a.top} more candidates suppressed (--top {a.top})\n')

    print('=' * 78)
    print('SUMMARY')
    print('=' * 78)

    by_ds = {}
    for r in candidates:
        by_ds.setdefault(r['dataset'], []).append(r)

    for ds in sorted(by_ds):
        rows = by_ds[ds]
        best = rows[0]
        sym = '✓' if best['match'] else ('~' if best['score'] < 1.0 else '·')
        print(f'  {sym} {ds:<14s} best match: {best["folder"]}'
              f'   score={best["score"]:.3f}'
              f'   ({best["pri_label"]}={best["pri"]:.2f}/{best["expected_pri"]:.2f}, '
              f'{best["sec_label"]}={best["sec"]:.2f}/{best["expected_sec"]:.2f})')

    missing = [ds for ds in REPORTED if ds not in by_ds and (a.dataset is None or a.dataset == ds)]
    if missing:
        print('\n  Datasets in REPORTED but missing in audit:', ', '.join(missing))

    if skipped_reasons:
        print('\n  Files skipped:', skipped_reasons)


if __name__ == '__main__':
    main()
