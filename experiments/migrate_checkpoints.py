#!/usr/bin/env python3
"""
migrate_checkpoints.py
======================

Copies the round-1 exact-match checkpoints into the seed-aware layout that
the v4 pipeline (run_multi_seed.py + aggregate_results.py) expects.

After running this, the four datasets that reproduce exactly from saved
checkpoints (ISIC 2018, ACDC, DermaMNIST, OrganSMNIST) become "seed_42"
under the new outputs root. You can then launch run_multi_seed.py to add
seeds 43 and 44 without re-doing seed 42, and run the orchestrator from
scratch on the other six datasets.

What this script does
---------------------

For each of the four exact-match datasets, it:
  1. Locates the source folder under the audited checkpoints tree.
  2. Creates the destination folder {output_root}/{dataset}/seed_42/.
  3. Copies best_model.pth, last_checkpoint.pth (if present), all
     test_results_*.json files, history.json, and any train.log.
  4. Synthesises the v4 metadata files (args.json, model_info.json) by
     extracting them from the test_results_final.json, where available.
  5. Appends a record to {output_root}/INDEX.json so the orchestrator's
     resume mechanism recognises seed_42 as already complete.

Defaults match the layout of the audit folders, but every path is
configurable via flags.

Usage
-----

  # Dry-run: show what would be copied
  python migrate_checkpoints.py --src ../checkpoints --dest ../outputs --dry_run

  # Real run
  python migrate_checkpoints.py --src ../checkpoints --dest ../outputs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Hard-coded list of exact-match datasets and where to find them inside ../checkpoints.
# Each entry: (dataset_name, source_subpath_relative_to_src_root, default_seed)
EXACT_MATCHES: List[Dict] = [
    {
        'dataset': 'isic2018',
        'task': 'seg',
        'source_relpath': 'maybenew/outputs_hamseg/isic2018/isic2018',
        'fallback_relpath': 'outputs_hamseg/outputs_hamseg/isic2018',
        'seed': 42,
    },
    {
        'dataset': 'acdc',
        'task': 'seg',
        'source_relpath': 'maybenew/outputs_hamseg/acdc',
        'fallback_relpath': None,
        'seed': 42,
    },
    {
        'dataset': 'dermamnist',
        'task': 'cls',
        'source_relpath': 'maybenew/outputs_hamcls/dermamnist',
        'fallback_relpath': None,
        'seed': 42,
    },
    {
        'dataset': 'organsmnist',
        'task': 'cls',
        'source_relpath': 'maybenew/outputs_hamcls/organsmnist',
        'fallback_relpath': None,
        'seed': 42,
    },
]


# Files we copy verbatim from each source folder.
COPY_PATTERNS = [
    'best_model.pth',
    'last_checkpoint.pth',
    'history.json',
    'train.log',
    'training.log',
    'report.txt',
    'training_curves.png',
    'segmentation_results.png',
]


def find_source(src_root: Path, entry: Dict) -> Optional[Path]:
    primary = src_root / entry['source_relpath']
    if primary.is_dir():
        return primary
    if entry.get('fallback_relpath'):
        fb = src_root / entry['fallback_relpath']
        if fb.is_dir():
            return fb
    return None


def derive_args_from_test_json(test_json: Dict, dataset: str, seed: int, task: str) -> Dict:
    """Build the args.json payload, preferring values stored in test_results_final.json."""
    src_args = test_json.get('args') or {}
    # Normalise common keys.  hamseg.py stores them as strings (e.g. epochs="200").
    args = dict(src_args)
    args.setdefault('dataset', dataset)
    args.setdefault('seed', seed)
    args['migrated_from_checkpoint'] = True
    args['migration_timestamp'] = dt.datetime.utcnow().isoformat() + 'Z'
    if task == 'cls' and not src_args:
        # hamcls.py round-1 did not save args. Provide a placeholder so users know.
        args['note'] = ('Original launch arguments were not saved by hamcls.py round-1. '
                        'Recover from the run trajectory in history.json if needed.')
    return args


def derive_model_info(test_json: Dict, dataset: str, seed: int, task: str) -> Dict:
    """Build a minimal model_info.json from what we can extract."""
    return {
        'model': test_json.get('model', 'unknown'),
        'dataset': dataset,
        'seed': seed,
        'task': task,
        'params_total': test_json.get('params'),
        'params_total_mb_fp32': (test_json.get('params', 0) * 4 / 1024**2
                                  if test_json.get('params') else None),
        'flops_total': None,  # to be filled by a separate FLOPs profiling step
        'gflops_at_input': None,
        'migrated_from_checkpoint': True,
    }


def migrate_one(src_root: Path, dest_root: Path, entry: Dict, dry_run: bool = False) -> Dict:
    dataset = entry['dataset']
    seed = entry['seed']
    task = entry['task']
    source = find_source(src_root, entry)
    rec = {
        'dataset': dataset,
        'task': task,
        'seed': seed,
        'source': str(source) if source else None,
        'destination': str(dest_root / dataset / f'seed_{seed}'),
        'copied': [],
        'synthesised': [],
        'completed': False,
        'note': '',
    }
    if source is None:
        rec['note'] = 'source folder not found'
        print(f'  ✗ {dataset}: source not found under {src_root}')
        return rec

    dest = dest_root / dataset / f'seed_{seed}'
    if dest.exists() and any(dest.iterdir()) and not dry_run:
        rec['note'] = 'destination already exists and is non-empty; skipping'
        print(f'  ⏭  {dataset}: destination {dest} already populated, skipping')
        return rec

    print(f'  ▶ {dataset}: {source}  →  {dest}')
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    # 1) Copy verbatim files
    for name in COPY_PATTERNS:
        s = source / name
        if s.exists():
            t = dest / name
            if dry_run:
                print(f'    (dry-run)   copy   {s.relative_to(src_root)}')
            else:
                shutil.copy2(s, t)
            rec['copied'].append(name)

    # 2) Copy every test_results_*.json (we need them all, including periodic ones)
    for s in source.glob('test_results_*.json'):
        t = dest / s.name
        if dry_run:
            print(f'    (dry-run)   copy   {s.relative_to(src_root)}')
        else:
            shutil.copy2(s, t)
        rec['copied'].append(s.name)

    # 3) Synthesise args.json and model_info.json from test_results_final.json
    final_json = source / 'test_results_final.json'
    if final_json.exists():
        try:
            data = json.loads(final_json.read_text())
        except json.JSONDecodeError:
            data = {}
        args_blob = derive_args_from_test_json(data, dataset, seed, task)
        info_blob = derive_model_info(data, dataset, seed, task)
        if dry_run:
            print(f'    (dry-run)   synth  args.json  model_info.json')
        else:
            with (dest / 'args.json').open('w') as f:
                json.dump(args_blob, f, indent=2)
            with (dest / 'model_info.json').open('w') as f:
                json.dump(info_blob, f, indent=2)
        rec['synthesised'] = ['args.json', 'model_info.json']

    rec['completed'] = True
    return rec


def append_to_index(dest_root: Path, records: List[Dict]) -> None:
    idx_path = dest_root / 'INDEX.json'
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text())
        except json.JSONDecodeError:
            idx = {'created': dt.datetime.utcnow().isoformat() + 'Z',
                   'updated': None, 'runs': []}
    else:
        idx = {'created': dt.datetime.utcnow().isoformat() + 'Z',
               'updated': None, 'runs': []}
    for rec in records:
        if not rec.get('completed'):
            continue
        idx['runs'].append({
            'task': rec['task'],
            'dataset': rec['dataset'],
            'seed': rec['seed'],
            'output_dir': rec['destination'],
            'source_for_seed': rec['source'],
            'finished': dt.datetime.utcnow().isoformat() + 'Z',
            'completed': True,
            'note': 'imported via migrate_checkpoints.py',
        })
    idx['updated'] = dt.datetime.utcnow().isoformat() + 'Z'
    tmp = idx_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(idx, indent=2))
    tmp.replace(idx_path)


def main():
    p = argparse.ArgumentParser(
        description='Migrate exact-match checkpoints into seed_42 of the v4 layout.')
    p.add_argument('--src', default='../checkpoints',
                   help='Folder containing the round-1 audit (default: ../checkpoints).')
    p.add_argument('--dest', default='../outputs',
                   help='Output root for the v4 pipeline (default: ../outputs).')
    p.add_argument('--dry_run', action='store_true',
                   help='Print what would be copied without changing anything.')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Restrict to a subset of datasets (default: all four exact matches).')
    args = p.parse_args()

    src_root = Path(args.src).resolve()
    dest_root = Path(args.dest).resolve()
    if not src_root.exists():
        sys.exit(f'src root not found: {src_root}')

    targets = EXACT_MATCHES
    if args.datasets:
        wanted = {d.lower() for d in args.datasets}
        targets = [t for t in EXACT_MATCHES if t['dataset'] in wanted]

    print(f'Source: {src_root}')
    print(f'Dest:   {dest_root}')
    print(f'Datasets: {[t["dataset"] for t in targets]}')
    print(f'Dry run:  {args.dry_run}')
    print()

    records: List[Dict] = []
    for entry in targets:
        rec = migrate_one(src_root, dest_root, entry, dry_run=args.dry_run)
        records.append(rec)

    if not args.dry_run:
        append_to_index(dest_root, records)
        print()
        print(f'INDEX.json updated at {dest_root / "INDEX.json"}')

    n_done = sum(1 for r in records if r['completed'])
    n_skipped = sum(1 for r in records if 'skipping' in (r['note'] or ''))
    n_failed = len(records) - n_done - n_skipped
    print()
    print('=' * 60)
    print(f'Done.  imported={n_done}  skipped={n_skipped}  failed={n_failed}')
    print('=' * 60)
    print()
    print('Next step: launch the multi-seed orchestrator to add seeds 43 and 44 for the')
    print('imported datasets, and to run all 3 seeds for the remaining 6 datasets:')
    print()
    print(f'  python run_multi_seed.py all --preset all \\\n'
          f'      --seeds 42 43 44 \\\n'
          f'      --output_root {dest_root} \\\n'
          f'      --data_dir_isic2018 ./data/ISIC2018 \\\n'
          f'      --data_dir_isic2017 ./data/ISIC2017 \\\n'
          f'      --data_dir_tn3k     ./data/TN3K \\\n'
          f'      --data_dir_acdc     ./data/ACDC_npz \\\n'
          f'      --data_root_cls     ./data')
    print()
    print('Resume is on by default, so seed_42 of the four imported datasets')
    print('will be skipped automatically.')


if __name__ == '__main__':
    main()
