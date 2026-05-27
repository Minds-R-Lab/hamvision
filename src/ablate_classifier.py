#!/usr/bin/env python3
"""
ablate_hamcls.py — Classification ablation study for HamCls

Tests the necessity of each architectural component by training ablated
variants on a single dataset (recommended: DermaMNIST -- small enough for fast
iteration, the full model wins by large margins so there's room to ablate without losing
the win).

Architectural variants (selected via --variant):
  full          - complete HamCls (baseline)
  no_ss2d       - replace SS2D bottleneck with depth-matched ConvNeXt block,
                  head becomes plain GAP. Tests "is the Hamiltonian apparatus
                  necessary?"
  gap_head      - keep SS2D, replace PSSP with global average pooling.
                  Tests "is frequency-domain pooling necessary?"
  pssa_head     - keep SS2D, replace PSSP with energy-weighted spatial
                  attention pooling (PSSA). Tests "is FFT specifically the
                  right pooling, or does any phase-space-aware pool work?"

Component variants (use existing hamcls.py CLI flags directly):
  --no_pssp_complex          PSSP without complex FFT (magnitude only)
  --no_pssp_cross            PSSP without <q*p>, <q^2+p^2> features
  --no_pssp_use_ss2d_energy  Row-attention from |Z|^2 instead of SS2D energy
  --n_scan_dirs 1            Single-direction scan (no bidirectional merge)

USAGE
-----
  # Architectural ablation (this script):
  python ablate_hamcls.py --variant full      --dataset dermamnist --seed 42 ...
  python ablate_hamcls.py --variant no_ss2d   --dataset dermamnist --seed 42 ...
  python ablate_hamcls.py --variant gap_head  --dataset dermamnist --seed 42 ...
  python ablate_hamcls.py --variant pssa_head --dataset dermamnist --seed 42 ...

  # Component ablation (use main script):
  python hamcls.py --no_pssp_complex --dataset dermamnist --seed 42 ...

See run_classification_ablation.sh for a full sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hamcls import (  # noqa: E402
    MedMNISTDataset,
    load_medmnist,
    set_seed,
    setup_logging,
    build_transforms,
    build_transforms_for_dataset,
    plot_curves,
    ConvNeXtBlock,
    ORIENTATION_SENSITIVE,
)
from hamcls import (  # noqa: E402
    HamCls,
    HamiltonianSS2D,
    PhaseSpaceSpectralPooling,
    Trainer,
)


# ============================================================
# Ablation heads (replacements for PSSP)
# ============================================================
class GAPHead(nn.Module):
    """Global average pooling head — ignores the q/p/energy phase-space
    structure and treats the bottleneck output as a generic CNN feature map.
    This is the natural baseline: 'what if we throw away everything the
    Hamiltonian apparatus tells us and just average-pool?'.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def output_feature_dim(self):
        return self.dim

    def forward(self, q, p=None, energy_map=None):
        # GAP on q (the position field) — equivalent to a standard CNN classifier
        feat = q.mean(dim=(-1, -2))
        return self.norm(feat)


class PSSAHead(nn.Module):
    """Phase-Space Saliency Attention head (v1's design, no FFT).

    Pools q via a softmax weighting derived from the energy map (if available)
    or from |q|^2 + |p|^2. Adds <q*p> and <q^2+p^2> auxiliary features and a
    GAP path. Tests whether the FFT-pooling specifically is what matters, or
    whether any phase-space-aware pool would do.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
        self.orbital_norm = nn.LayerNorm(dim, eps=1e-6)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def output_feature_dim(self):
        # gap + saliency-pool + cross + orbital  =  4 * C
        return 4 * self.dim

    def forward(self, q, p, energy_map=None):
        B, C, H, W = q.shape
        # 1) GAP path (stability)
        gap_feat = q.mean(dim=(-1, -2))
        gap_feat = self.norm(gap_feat)

        # 2) Saliency-pool: prefer SS2D energy, fall back to |q|^2 + |p|^2
        with torch.cuda.amp.autocast(enabled=False):
            if energy_map is not None:
                E = energy_map.float()
            else:
                E = q.float().pow(2) + p.float().pow(2)
            # Subtract spatial mean so the softmax can use full dynamic range
            E_centered = E - E.mean(dim=(-1, -2), keepdim=True)
            attn = (
                E_centered.flatten(-2)
                .div(self.log_temp.exp().clamp(min=1e-3))
                .softmax(dim=-1)
                .reshape(B, C, H, W)
            )
        sal_feat = (q * attn.to(q.dtype)).sum(dim=(-1, -2))

        # 3) Cross + orbital
        cross_feat = (q * p).mean(dim=(-1, -2))
        orbital_feat = (q.pow(2) + p.pow(2)).mean(dim=(-1, -2))
        cross_feat = self.cross_norm(cross_feat)
        orbital_feat = self.orbital_norm(orbital_feat)

        return torch.cat([gap_feat, sal_feat, cross_feat, orbital_feat], dim=-1)


# ============================================================
# Ablation model
# ============================================================
class _ConvNeXtBottleneck(nn.Module):
    """Depth-matched stand-in for SS2D in the no_ss2d variant.
    Returns the conv output as `q` and zero tensors for `p` and `energy_map`
    so the GAPHead can ignore them. The downstream head MUST be GAP for this
    variant since p / energy carry no signal.
    """
    def __init__(self, dim, n_blocks=2):
        super().__init__()
        self.dim = dim
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        feat = self.blocks(x)
        zero = torch.zeros_like(feat)
        # Return a (q, p, energy_map) triple so the upstream forward stays uniform
        return feat, zero, zero


class HamCls_Ablation(HamCls):
    """HamCls with selectable architectural ablations.

    Reuses everything from HamCls (stem, encoder stages) and replaces the
    bottleneck and/or head based on `args.variant`.
    """
    VARIANT_CHOICES = ('full', 'no_ss2d', 'gap_head', 'pssa_head')

    def __init__(self, args):
        super().__init__(args)  # builds the full classifier architecture first
        variant = getattr(args, 'variant', 'full')
        if variant not in self.VARIANT_CHOICES:
            raise ValueError(
                f"Unknown variant '{variant}'. Choices: {self.VARIANT_CHOICES}"
            )
        self.variant = variant
        bottleneck_dim = self.pssp.dim

        if variant == 'no_ss2d':
            # Replace SS2D bottleneck with ConvNeXt blocks.
            # 2 blocks ≈ depth-matched to SS2D's 2 scans + 3 merge convs.
            self.ss2d = _ConvNeXtBottleneck(bottleneck_dim, n_blocks=2)
            # GAP head is mandatory here: p/energy carry no signal.
            self.pssp = GAPHead(bottleneck_dim)
            self._rebuild_classifier(args, feat_dim=bottleneck_dim)

        elif variant == 'gap_head':
            self.pssp = GAPHead(bottleneck_dim)
            self._rebuild_classifier(args, feat_dim=bottleneck_dim)

        elif variant == 'pssa_head':
            self.pssp = PSSAHead(bottleneck_dim)
            self._rebuild_classifier(args, feat_dim=4 * bottleneck_dim)

        # Re-init weights ONLY for variants that replaced modules.
        # The parent ctor already inited everything for 'full'; calling
        # _init_weights again would consume different randoms after the
        # parent's pass and produce weights that don't match the main
        # HamCls path used to generate Tab. 3. Skipping for 'full'
        # keeps the ablation 'full' baseline numerically identical to the
        # main script, which is what we want for clean apples-to-apples.
        if variant != 'full':
            self._init_weights()

    def _rebuild_classifier(self, args, feat_dim):
        bottleneck_dim = self.pssp.dim
        head_drop = float(getattr(args, 'head_drop', 0.3))
        n_cls = args.num_classes
        hidden = bottleneck_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(head_drop),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(head_drop * 0.5),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x_n = self.bottleneck_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.variant == 'no_ss2d':
            # ConvNeXt bottleneck path (no fp32 autocast guard needed)
            q, p, energy_map = self.ss2d(x_n)
            features = self.pssp(q, p, energy_map=energy_map)
        else:
            # Standard SS2D bottleneck (keep fp32 for the oscillator scan)
            with torch.cuda.amp.autocast(enabled=False):
                q, p, energy_map = self.ss2d(x_n.float())
            q = q.to(x.dtype)
            p = p.to(x.dtype)
            energy_map = energy_map.to(x.dtype)
            features = self.pssp(q, p, energy_map=energy_map)

        return self.classifier(features)


# ============================================================
# Argument parsing — extends hamcls.get_args with --variant
# ============================================================
def get_args_ablation():
    p = argparse.ArgumentParser(
        description='HamCls ablation study (architectural variants)'
    )
    # Variant selector (the only new flag)
    p.add_argument(
        '--variant', type=str, default='full',
        choices=HamCls_Ablation.VARIANT_CHOICES,
        help='Architectural ablation variant (default: full).',
    )
    # All other flags mirror hamcls.get_args() exactly.
    p.add_argument('--dataset', type=str, default='dermamnist')
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--num_classes', type=int, default=None)
    p.add_argument('--in_channels', type=int, default=3)
    p.add_argument('--embed_dim', type=int, default=96)
    p.add_argument('--depths', type=int, nargs='+', default=[3, 3])
    p.add_argument('--damping_clamp', type=float, default=5.0)
    p.add_argument('--drop_rate', type=float, default=0.2)
    p.add_argument('--head_drop', type=float, default=0.3)
    p.add_argument('--pssp_K', type=int, default=12)
    p.add_argument('--pssp_complex', action='store_true', default=True)
    p.add_argument('--no_pssp_complex', dest='pssp_complex', action='store_false')
    p.add_argument('--pssp_cross', action='store_true', default=True)
    p.add_argument('--no_pssp_cross', dest='pssp_cross', action='store_false')
    p.add_argument('--pssp_use_ss2d_energy', action='store_true', default=True)
    p.add_argument('--no_pssp_use_ss2d_energy', dest='pssp_use_ss2d_energy',
                   action='store_false')
    p.add_argument('--n_scan_dirs', type=int, default=2, choices=[1, 2])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'])
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--balanced', action='store_true')
    p.add_argument('--use_ema', action='store_true', default=True)
    p.add_argument('--no_ema', dest='use_ema', action='store_false')
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--output_dir', type=str, default='./outputs_ablation_cls')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_seed_subdir', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--test_only', action='store_true')
    p.add_argument('--test_every', type=int, default=30)

    a = p.parse_args()
    if a.no_amp:
        a.use_amp = False
    a.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Output dir layout: {output_dir}/{variant}/{dataset}/seed_{seed}/
    a.output_dir = os.path.join(a.output_dir, a.variant, a.dataset)
    if not a.no_seed_subdir:
        a.output_dir = os.path.join(a.output_dir, f'seed_{a.seed}')
    return a


# ============================================================
# Main
# ============================================================
def main():
    args = get_args_ablation()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info('=' * 60)
    logger.info('  HamCls ablation study')
    logger.info(f'  Variant: {args.variant}')
    logger.info(f'  Dataset: {args.dataset}    Seed: {args.seed}')
    logger.info(f'  Output:  {args.output_dir}')
    logger.info('=' * 60)

    args_dict = {
        k: (str(v) if not isinstance(v, (int, float, str, bool, list, type(None))) else v)
        for k, v in vars(args).items()
    }
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    logger.info('Loading data...')
    (train_imgs, train_lbls,
     val_imgs, val_lbls,
     test_imgs, test_lbls,
     n_classes) = load_medmnist(args.dataset, size=args.size, data_root=args.data_root)
    if args.num_classes is None:
        args.num_classes = n_classes
    logger.info(f'  Classes: {args.num_classes}')
    logger.info(f'  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}')

    # Dataset-aware augmentation: orientation-sensitive datasets (organa/c) get
    # hflip+vflip disabled to avoid label corruption. Critical for OrganAMNIST
    # ablation -- without this fix the corrupting flip policy would dominate
    # any architectural signal.
    train_tf = build_transforms_for_dataset(args.dataset, args.size, is_train=True)
    eval_tf = build_transforms_for_dataset(args.dataset, args.size, is_train=False)
    if args.dataset.lower() in ORIENTATION_SENSITIVE:
        logger.info('  Augmentation: ORIENTATION-SENSITIVE policy (no hflip/vflip)')
    else:
        logger.info('  Augmentation: default policy (with hflip+vflip)')
    train_ds = MedMNISTDataset(train_imgs, train_lbls, transform=train_tf)
    val_ds = MedMNISTDataset(val_imgs, val_lbls, transform=eval_tf)
    test_ds = MedMNISTDataset(test_imgs, test_lbls, transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    logger.info(f'Building HamCls ablation [{args.variant}]...')
    for k in ['device', 'num_classes', 'embed_dim', 'depths', 'size', 'batch_size',
              'lr', 'weight_decay', 'drop_rate', 'head_drop', 'pssp_K',
              'pssp_complex', 'pssp_cross', 'pssp_use_ss2d_energy', 'n_scan_dirs',
              'loss_type', 'focal_gamma', 'label_smoothing', 'balanced',
              'use_ema', 'ema_decay']:
        logger.info(f'  {k:22s}: {getattr(args, k)}')
    model = HamCls_Ablation(args).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Parameters: {n_params:,} ({n_params * 4 / 1024**2:.1f} MB)')

    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.zeros(1, args.in_channels, args.size, args.size, device=args.device)
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.tracer_warnings('none')
        logger.info(f'  GFLOPs ({args.size}x{args.size}): {flops.total() / 1e9:.2f}')
    except Exception as e:
        logger.info(f'  (fvcore unavailable, skipping FLOPs: {e})')

    class_weights = None
    if args.balanced:
        train_labels_flat = train_lbls.flatten()
        counts = np.bincount(train_labels_flat, minlength=args.num_classes).astype(float)
        counts = np.maximum(counts, 1.0)
        weights = len(train_labels_flat) / (args.num_classes * counts)
        class_weights = (weights / weights.sum() * args.num_classes).tolist()
        logger.info(f'  Class counts: {counts.astype(int).tolist()}')
        logger.info(f'  Class weights: {[f"{w:.3f}" for w in class_weights]}')

    trainer = Trainer(args, model, train_loader, val_loader, test_loader, logger,
                         class_weights=class_weights)
    if not args.test_only:
        trainer.train()
    results = trainer.test(use_ema=args.use_ema)

    logger.info('=' * 50)
    logger.info(f'  ABLATION TEST RESULTS [{args.variant}]')
    logger.info('=' * 50)
    logger.info(f'  Accuracy       : {results["accuracy"] * 100:.2f}%')
    logger.info(f'  AUC (macro)    : {results["auc"] * 100:.2f}%')
    logger.info(f'  F1 (macro)     : {results["f1_macro"] * 100:.2f}%')
    logger.info(f'  F1 (weighted)  : {results["f1_weighted"] * 100:.2f}%')
    logger.info(f'  Best val acc   : {results["best_val_acc"]:.2f}%')
    logger.info(f'  Best epoch     : {results["best_epoch"]}')
    logger.info(f'  Used EMA       : {results["used_ema"]}')
    logger.info(f'  Variant        : {args.variant}')
    logger.info(f'  Parameters     : {n_params:,}')
    for c in results['per_class']:
        logger.info(f'  Class {c["class"]:>2}  acc={c["acc"] * 100:.1f}%  f1={c["f1"] * 100:.1f}%')

    # Persist a small summary keyed by variant for the aggregator
    summary = {
        'variant': args.variant,
        'dataset': args.dataset,
        'seed': args.seed,
        'params': n_params,
        'accuracy': float(results['accuracy']),
        'auc': float(results['auc']),
        'f1_macro': float(results['f1_macro']),
        'f1_weighted': float(results['f1_weighted']),
        'per_class': results['per_class'],
    }
    with open(os.path.join(args.output_dir, 'ablation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if trainer.history['train_loss']:
        plot_curves(trainer.history, args.output_dir)
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(trainer.history, f, indent=2)
    logger.info('Done!')


if __name__ == '__main__':
    main()
