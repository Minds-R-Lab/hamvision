#!/usr/bin/env python3
"""
run_multi_seed.py
=================

Single entry point for HamVision experiments. Supports:
  * single-seed runs (just like calling hamseg.py / hamcls.py directly)
  * multi-seed runs across the same dataset, in series or parallel
  * a "preset" mode that runs every Tabs. 2 / 3 dataset with 3 seeds each

After every individual seed completes, we write to a master INDEX.json so the
experiment record is durable: if the next seed crashes (OOM, kernel panic,
power loss), the prior seeds are already preserved on disk.

Layout produced
---------------

    {output_root}/
    ├── INDEX.json                          (master index of all completed runs)
    ├── isic2018/
    │   ├── seed_42/  (full hamseg.py output, including args.json)
    │   ├── seed_43/
    │   ├── seed_44/
    │   ├── aggregate.json                  (written by aggregate_results.py)
    │   └── SUMMARY.txt
    ├── tn3k/
    │   └── ...
    ...

Examples
--------

  # 1) Single seed on one dataset (the simplest case):
  python run_multi_seed.py seg --dataset isic2018 --data_root ./data/ISIC2018 \
                              --seeds 42

  # 2) Three seeds on one dataset (the recommended for the revision):
  python run_multi_seed.py seg --dataset isic2018 --data_root ./data/ISIC2018 \
                              --seeds 42 43 44

  # 3) Three seeds on every classification dataset (the full Tab. 3 row):
  python run_multi_seed.py cls --preset all_cls --seeds 42 43 44

  # 4) Three seeds on every Tab. 2 + Tab. 3 dataset (the full revision):
  python run_multi_seed.py all --preset all --seeds 42 43 44

Failure handling
----------------

  * Each seed runs as a subprocess.  If a seed fails, the orchestrator logs the
    failure to INDEX.json and proceeds with the next seed/dataset.
  * --resume picks up where the last invocation left off (skips datasets/seeds
    whose seed_{seed}/test_results_final.json already exists).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------
# Per-task defaults that match the manuscript's stated configurations
# ---------------------------------------------------------------

# Segmentation: hamseg.py defaults plus per-dataset overrides.
SEG_PRESETS: Dict[str, Dict[str, str]] = {
    'isic2018': {
        '--num_classes': '1',
        '--epochs':      '200',
        '--batch_size':  '8',
        '--lr':          '5e-4',
    },
    'isic2017': {
        '--num_classes': '1',
        '--epochs':      '200',
        '--batch_size':  '16',
        '--lr':          '5e-4',
    },
    'tn3k': {
        '--num_classes': '1',
        '--epochs':      '200',
        '--batch_size':  '16',
        '--lr':          '5e-4',
        '--val_ratio':   '0.1',
    },
    'acdc': {
        '--num_classes': '4',
        '--epochs':      '200',
        '--batch_size':  '16',
        '--lr':          '5e-4',
    },
    # MMOTU added in v4 — ovarian-tumour ultrasound, 1,469 binary masks.
    # Direct head-to-head with FreqConvMamba which also reports on MMOTU.
    'mmotu': {
        '--num_classes': '1',
        '--epochs':      '200',
        '--batch_size':  '16',
        '--lr':          '5e-4',
    },
}

# Classification: hamcls.py defaults plus per-dataset overrides.
CLS_PRESETS: Dict[str, Dict[str, str]] = {
    'pathmnist':   {'--epochs': '100', '--batch_size': '64', '--lr': '1e-3'},
    'bloodmnist':  {'--epochs': '100', '--batch_size': '64', '--lr': '1e-3'},
    'dermamnist':  {'--epochs': '100', '--batch_size': '32', '--lr': '1e-3'},
    'breastmnist': {'--epochs': '200', '--batch_size': '8',  '--lr': '3e-4',
                    '--weight_decay': '0.01', '--drop_rate': '0.3',
                    '--head_drop': '0.4', '--balanced': '__flag__'},
    'organsmnist': {'--epochs': '100', '--batch_size': '64', '--lr': '1e-3'},
    'retinamnist': {'--epochs': '150', '--batch_size': '32', '--lr': '1e-3',
                    '--balanced': '__flag__'},
}

ALL_SEG = list(SEG_PRESETS)
ALL_CLS = list(CLS_PRESETS)


# ---------------------------------------------------------------
# INDEX.json helpers (durable record across runs)
# ---------------------------------------------------------------

def load_index(output_root: Path) -> Dict:
    p = output_root / 'INDEX.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {'created': dt.datetime.utcnow().isoformat() + 'Z',
            'updated': None, 'runs': []}


def save_index(output_root: Path, index: Dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    index['updated'] = dt.datetime.utcnow().isoformat() + 'Z'
    p = output_root / 'INDEX.json'
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(p)  # atomic rename so a crash mid-write cannot corrupt the index.


def append_run(output_root: Path, record: Dict) -> None:
    idx = load_index(output_root)
    idx['runs'].append(record)
    save_index(output_root, idx)


def is_complete(seed_dir: Path) -> bool:
    """A seed run counts as complete iff test_results_final.json exists and has nonzero metrics."""
    f = seed_dir / 'test_results_final.json'
    if not f.exists():
        return False
    try:
        d = json.loads(f.read_text())
        # hamseg has nested 'test'; hamcls has top-level keys.
        if 'test' in d and isinstance(d['test'], dict):
            return d['test'].get('dice', 0) > 0 or d['test'].get('miou', 0) > 0
        return d.get('accuracy', 0) > 0 or d.get('auc', 0) > 0
    except (json.JSONDecodeError, OSError):
        return False


# ---------------------------------------------------------------
# Run a single seed via subprocess
# ---------------------------------------------------------------

def build_cmd(task: str, dataset: str, data_root: Optional[str],
              seed: int, output_root: Path,
              extra_overrides: Dict[str, str],
              python: str = sys.executable) -> List[str]:
    """Construct the command line for hamseg.py or hamcls.py."""
    here = Path(__file__).resolve().parent
    src_dir = here.parent / 'src'
    if task == 'seg':
        script = src_dir / 'hamseg.py'
        preset = dict(SEG_PRESETS.get(dataset, {}))
    else:
        script = src_dir / 'hamcls.py'
        preset = dict(CLS_PRESETS.get(dataset, {}))
    preset.update(extra_overrides)

    cmd = [python, str(script),
           '--dataset', dataset,
           '--seed', str(seed),
           '--output_dir', str(output_root)]

    if task == 'seg':
        if data_root is None:
            raise ValueError('Segmentation requires --data_root')
        cmd += ['--data_root', data_root]
    else:
        if data_root is not None:
            cmd += ['--data_root', data_root]

    for k, v in preset.items():
        if v == '__flag__':
            cmd.append(k)
        else:
            cmd += [k, str(v)]
    return cmd


def run_one_seed(task: str, dataset: str, data_root: Optional[str],
                 seed: int, output_root: Path,
                 extra_overrides: Dict[str, str],
                 ablation: str = 'none',
                 dry_run: bool = False,
                 resume: bool = True) -> Dict:
    """Run a single (task, dataset, seed) combo. Returns a dict for INDEX.json."""
    if ablation != 'none':
        seed_dir = output_root / dataset / f'abl_{ablation}' / f'seed_{seed}'
        # Make sure the per-script wrapper also picks up the ablation flag.
        extra_overrides = dict(extra_overrides)
        extra_overrides['--ablation'] = ablation
    else:
        seed_dir = output_root / dataset / f'seed_{seed}'
    record = {
        'task': task,
        'dataset': dataset,
        'seed': seed,
        'ablation': ablation,
        'output_dir': str(seed_dir),
        'started': dt.datetime.utcnow().isoformat() + 'Z',
        'finished': None,
        'wallclock_seconds': None,
        'returncode': None,
        'completed': False,
    }

    if resume and is_complete(seed_dir):
        record.update(returncode=0, completed=True,
                      finished=record['started'],
                      note='skipped — previously completed')
        print(f'  ⏭  {dataset} seed={seed}  already complete in {seed_dir}, skipping')
        return record

    cmd = build_cmd(task, dataset, data_root, seed, output_root, extra_overrides)
    record['cmd'] = ' '.join(shlex.quote(c) for c in cmd)
    print('  ▶ ' + record['cmd'])

    if dry_run:
        record['note'] = 'dry-run; not executed'
        return record

    t0 = time.time()
    try:
        proc = subprocess.run(cmd)
        record['returncode'] = proc.returncode
    except KeyboardInterrupt:
        record['returncode'] = 130
        record['note'] = 'KeyboardInterrupt'
        raise
    except Exception as e:  # pragma: no cover
        record['returncode'] = -1
        record['note'] = f'subprocess raised: {e!r}'
    record['wallclock_seconds'] = round(time.time() - t0, 1)
    record['finished'] = dt.datetime.utcnow().isoformat() + 'Z'
    record['completed'] = is_complete(seed_dir)
    return record


# ---------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------

def expand_targets(task: str, args) -> List[Tuple[str, str, Optional[str]]]:
    """Yield (task, dataset, data_root) tuples."""
    if args.preset == 'all':
        out = []
        for ds in ALL_SEG:
            out.append(('seg', ds, args.data_dirs.get(ds)))
        for ds in ALL_CLS:
            out.append(('cls', ds, args.data_dirs.get(ds, args.data_root_cls)))
        return out
    if args.preset == 'all_seg':
        return [('seg', ds, args.data_dirs.get(ds)) for ds in ALL_SEG]
    if args.preset == 'all_cls':
        return [('cls', ds, args.data_dirs.get(ds, args.data_root_cls)) for ds in ALL_CLS]

    if args.dataset is None:
        raise SystemExit('Specify either --preset {all|all_seg|all_cls} or --dataset NAME')
    if task == 'seg':
        return [('seg', args.dataset, args.data_root)]
    elif task == 'cls':
        return [('cls', args.dataset, args.data_root)]
    elif task == 'all':
        # task=all but a single dataset chosen — auto-detect
        if args.dataset in ALL_SEG:
            return [('seg', args.dataset, args.data_root)]
        elif args.dataset in ALL_CLS:
            return [('cls', args.dataset, args.data_root)]
        raise SystemExit(f'Unknown dataset: {args.dataset!r}')
    raise SystemExit(f'Unknown task: {task!r}')


def main():
    p = argparse.ArgumentParser(
        description='Run HamVision experiments with single-seed or multi-seed support.')
    p.add_argument('task', choices=['seg', 'cls', 'all'],
                   help='Which model family to launch (seg=HamSeg, cls=HamCls, all=both).')

    # Target selection
    p.add_argument('--dataset', default=None,
                   help='Single dataset name (e.g., isic2018, bloodmnist).')
    p.add_argument('--preset', default=None,
                   choices=['all', 'all_seg', 'all_cls'],
                   help='Run a fixed batch of datasets. Overrides --dataset.')
    p.add_argument('--data_root', default=None,
                   help='Data folder for the chosen dataset. For segmentation: '
                        'path to the dataset directory. For MedMNIST: the parent '
                        'folder where the .npz lives (medmnist auto-downloads here).')
    p.add_argument('--data_root_cls', default='./data',
                   help='(For preset=all_cls or preset=all only) parent folder for MedMNIST '
                        'datasets.  Default: ./data')
    p.add_argument('--data_dir_isic2018', default='./data/ISIC2018')
    p.add_argument('--data_dir_isic2017', default='./data/ISIC2017')
    p.add_argument('--data_dir_tn3k',     default='./data/TN3K')
    p.add_argument('--data_dir_mmotu',    default='./data/MMOTU')
    p.add_argument('--data_dir_acdc',     default='./data/ACDC')

    # Seed selection
    p.add_argument('--seeds', type=int, nargs='+', default=[42],
                   help='One or more random seeds. Default: [42] (single-seed run).')

    # Orchestration
    p.add_argument('--output_root', default='./outputs',
                   help='Top-level output directory. Default: ./outputs')
    p.add_argument('--resume', action='store_true', default=True,
                   help='Skip seeds whose final test JSON already exists. (default)')
    p.add_argument('--no_resume', dest='resume', action='store_false')
    p.add_argument('--dry_run', action='store_true',
                   help='Print every command but do not execute.')
    p.add_argument('--continue_on_error', action='store_true', default=True,
                   help='Continue with remaining seeds/datasets after a failure.')
    p.add_argument('--no_continue', dest='continue_on_error', action='store_false')

    # Pass-throughs
    p.add_argument('--ablation', type=str, default='none', choices=['none', 'A', 'B'],
                   help='Ablation variant of HamSeg/HamCls. "none" (default) = full model. '
                        '"A" = ConvNeXt-only. "B" = Oscillator-only. When set, outputs go to '
                        '{output_root}/{dataset}/abl_{A|B}/seed_{N}/.')
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                   help='Anything after --extra is forwarded verbatim to hamseg.py/hamcls.py. '
                        'Example: --extra --no_amp --num_workers 8')

    args = p.parse_args()

    # Per-dataset data_dirs map for preset=all
    args.data_dirs = {
        'isic2018': args.data_dir_isic2018,
        'isic2017': args.data_dir_isic2017,
        'tn3k':     args.data_dir_tn3k,
        'mmotu':    args.data_dir_mmotu,
        'acdc':     args.data_dir_acdc,
    }

    targets = expand_targets(args.task, args)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Convert any --extra K V pairs into a dict for downstream use
    extra_overrides: Dict[str, str] = {}
    it = iter(args.extra)
    for tok in it:
        if not tok.startswith('--'):
            continue
        # Peek the next token; if it doesn't start with '--', it's the value;
        # otherwise treat the current as a flag.
        try:
            nxt = next(it)
        except StopIteration:
            extra_overrides[tok] = '__flag__'
            break
        if nxt.startswith('--'):
            extra_overrides[tok] = '__flag__'
            # push back the peeked token
            it = iter([nxt] + list(it))
        else:
            extra_overrides[tok] = nxt

    print(f'Targets ({len(targets)}):')
    for task, ds, dr in targets:
        print(f'  {task}  {ds}  data_root={dr}')
    print(f'Seeds: {args.seeds}')
    print(f'Output root: {output_root}')
    print(f'Resume: {args.resume}    Dry run: {args.dry_run}')
    print()

    # Subdirectory layout: output_root has separate seg/cls roots so INDEX is shared
    overall_started = time.time()
    n_done, n_failed, n_skipped = 0, 0, 0

    for task, dataset, data_root in targets:
        dataset_root = output_root if args.task == 'all' else output_root
        # We use one INDEX at output_root for the entire orchestration.
        for seed in args.seeds:
            print(f'━━━ {task}  {dataset}  seed={seed} ━━━')
            try:
                rec = run_one_seed(task, dataset, data_root, seed, dataset_root,
                                   extra_overrides, ablation=args.ablation, dry_run=args.dry_run,
                                   resume=args.resume)
                append_run(output_root, rec)
                if rec.get('completed'):
                    if rec.get('note', '').startswith('skipped'):
                        n_skipped += 1
                    else:
                        n_done += 1
                else:
                    n_failed += 1
                    if not args.continue_on_error and not args.dry_run:
                        print('Aborting on first failure (use --continue_on_error to disable).')
                        break
            except KeyboardInterrupt:
                rec_partial = {
                    'task': task, 'dataset': dataset, 'seed': seed,
                    'completed': False, 'note': 'KeyboardInterrupt',
                    'finished': dt.datetime.utcnow().isoformat() + 'Z'}
                append_run(output_root, rec_partial)
                print('Interrupted by user.')
                raise
        else:
            continue
        break

    total_h = (time.time() - overall_started) / 3600.0
    print()
    print('=' * 60)
    print(f'Done.  completed={n_done}  skipped={n_skipped}  failed={n_failed}    '
          f'wallclock={total_h:.2f} h')
    print('Index file:', output_root / 'INDEX.json')
    if n_failed and not args.dry_run:
        sys.exit(1)
