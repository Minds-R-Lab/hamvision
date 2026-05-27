#!/usr/bin/env python3
"""
collect_results.py — bundle the small result files under outputs/
(or any directory you point it at) into a single .tar.gz you can scp from a
remote machine and unpack locally.

What gets INCLUDED (small, structured artefacts):
  test_results_final.json, test_results_*.json   (per-seed final metrics)
  args.json, model_info.json                      (training config + FLOPs)
  history.json, report.txt                        (training curves + summary)
  ensemble_results.json, ENSEMBLE_SUMMARY*.txt    (inference_with_tricks output)
  metrics_extended.json, metrics_paper.csv        (eval_perclass_metrics output)
  INDEX.json, AGGREGATE.md                        (orchestrator artefacts)
  loss_curve.png, dice_curve.png                  (training-curve plots)
  any *.csv or *.md at the root of <root>

What gets EXCLUDED (too big):
  best_model.pth, last_model.pth, any *.pth/*.pt/*.bin
  *.npz, *.npy
  test sample prediction images (sample_*.png)

Use --include_log to also pack training.log / train.log files.
Use --include_samples to also pack sample_*.png files.

Usage:
    cd hamvision/
    # Default: bundle everything under ../outputs
    python collect_results.py --root ../outputs

    # Custom output path:
    python collect_results.py --root ../outputs \\
        --out /tmp/h100_results.tar.gz

    # Include training logs (text, can be a few MB total):
    python collect_results.py --root ../outputs --include_log

After download, extract on your laptop in the directory ABOVE outputs:
    cd ~/path/to/HamVision_V2
    tar -xzf outputs_results.tar.gz
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import tarfile
from pathlib import Path
from typing import List

# Files to ALWAYS include (small, structured outputs).
INCLUDE_PATTERNS = [
    "test_results_final.json",
    "test_results_*.json",
    "args.json",
    "model_info.json",
    "history.json",
    "report.txt",
    "ensemble_results.json",
    "ENSEMBLE_SUMMARY*",
    "metrics_extended.json",
    "metrics_paper.csv",
    "INDEX.json",
    "AGGREGATE.md",
    "checkpoint_audit.md",
    # Master CSVs / docs at the root
    "*.csv",
    "*.md",
]

# Files to ALWAYS exclude (model weights, raw arrays, sample images, etc.)
EXCLUDE_PATTERNS = [
    "*.pth",
    "*.pt",
    "*.bin",
    "*.npz",
    "*.npy",
    "*.jpg",
    "*.jpeg",
]

# PNG handling: exclude prediction sample PNGs by default but keep training-curve
# plots (loss_curve.png, dice_curve.png).
KEEP_PNG_NAMES = {
    "loss_curve.png",
    "dice_curve.png",
    "lr_curve.png",
    "metric_curve.png",
}


def matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def collect(root: Path, include_log: bool, include_samples: bool) -> List[Path]:
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        for fname in filenames:
            full = d / fname
            # Hard excludes take precedence
            if matches_any(fname, EXCLUDE_PATTERNS):
                continue
            if fname.endswith('.png'):
                # Keep training curves; skip sample PNGs unless --include_samples
                if fname in KEEP_PNG_NAMES:
                    files.append(full)
                elif include_samples and fname.startswith('sample_'):
                    files.append(full)
                continue
            if matches_any(fname, INCLUDE_PATTERNS):
                files.append(full)
                continue
            if include_log and fname in ('training.log', 'train.log'):
                files.append(full)
                continue
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='outputs',
                    help='Directory to collect from (default: outputs)')
    ap.add_argument('--out', default=None,
                    help='Output tar.gz path (default: <root>_results.tar.gz '
                         'in the current working directory)')
    ap.add_argument('--include_log', action='store_true',
                    help='Also include training.log / train.log files (text, '
                         'usually a few MB total)')
    ap.add_argument('--include_samples', action='store_true',
                    help='Also include sample_*.png prediction images (can be '
                         'tens of MB)')
    ap.add_argument('--dry_run', action='store_true',
                    help='List files that would be packed without creating the '
                         'tarball.')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f'FATAL: {root} does not exist')
    if not root.is_dir():
        raise SystemExit(f'FATAL: {root} is not a directory')

    out = Path(args.out) if args.out else Path(f'{root.name}_results.tar.gz')

    files = collect(root, args.include_log, args.include_samples)
    if not files:
        raise SystemExit('No matching files found. Wrong --root?')

    total_bytes = sum(f.stat().st_size for f in files)
    print(f'Found {len(files)} files under {root}')
    print(f'Total uncompressed: {total_bytes / 1e6:.2f} MB')

    # Show a per-extension breakdown so the user can sanity-check what is in
    # the bundle before sending it.
    by_ext: dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower() or '(no ext)'
        by_ext[ext] = by_ext.get(ext, 0) + 1
    print('Breakdown by extension:')
    for ext, n in sorted(by_ext.items(), key=lambda kv: -kv[1]):
        print(f'  {ext:10s}  {n}')

    if args.dry_run:
        print('\n--- DRY RUN: nothing written. First 30 files: ---')
        for f in files[:30]:
            try:
                rel = f.relative_to(root.parent)
            except ValueError:
                rel = f.name
            print(f'  {rel}')
        if len(files) > 30:
            print(f'  ... and {len(files) - 30} more')
        return

    # Make the archive paths relative to root.parent so the bundle preserves the
    # same directory structure (outputs/dataset/seed_X/...).
    parent = root.parent
    print(f'\nWriting {out} ...')
    with tarfile.open(out, 'w:gz') as tar:
        for f in files:
            try:
                arcname = f.relative_to(parent)
            except ValueError:
                arcname = Path(root.name) / f.name
            tar.add(f, arcname=str(arcname))

    sz = out.stat().st_size
    print(f'\nDone: {out}')
    print(f'  compressed:   {sz / 1e6:.2f} MB')
    print(f'  uncompressed: {total_bytes / 1e6:.2f} MB')
    print(f'  ratio:        {sz / max(total_bytes, 1):.2%}')
    print()
    print('Download from the H100, then on your laptop:')
    print(f'  cd ~/path/to/HamVision_V2')
    print(f'  tar -xzf {out.name}')


if __name__ == '__main__':
    main()
