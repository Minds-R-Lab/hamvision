#!/usr/bin/env python3
"""
HamCls: Phase-Space Spectral Classification (PSSC)
======================================================

Complete redesign of the HamCls classification head and training recipe,
motivated by the observation that the natural feature representation of a
damped harmonic oscillator is in the FREQUENCY domain (paper Section 3.3),
not the spatial domain. v1 pooled (q, p) spatially via GAP/PSSA and threw
away the trajectory the SS2D scan creates. v2 keeps it via FFT.

ARCHITECTURE CHANGES vs v1:
  (a) Lighter encoder: stride-4 stem + 2 ConvNeXt stages.
  (b) Single-direction SS2D bottleneck: horizontal scan only (4-direction
      v1 scan adds rotational equivariance segmentation needs but
      classification does not -- translation-invariance is gained for
      free from FFT magnitude in the head).
  (c) Phase-Space Spectral Pooling (PSSP) head: complex signal z = q + i*p,
      FFT along the scan axis, magnitude spectrum truncated to K dominant
      frequency bins, energy-weighted attention pool over rows, plus an
      auxiliary GAP path for stability.

TRAINING-RECIPE CHANGES vs v1:
  (d) Focal loss (handles imbalanced datasets like RetinaMNIST class 4 = 0%).
  (e) ModelEMA: exponential moving average of model weights for stable
      test-time predictions.
  (f) Optional ImageNet-pretrained encoder via timm (deferred to follow-up).

Usage:
    python hamcls.py --dataset dermamnist --size 224 --epochs 100 \\
        --data_root $DATA_CLS --seed 42 --use_ema

    # For imbalanced datasets:
    python hamcls.py --dataset retinamnist --size 224 --epochs 150 \\
        --data_root $DATA_CLS --seed 42 \\
        --loss_type focal --focal_gamma 2.0 --balanced --use_ema
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# Shared utilities + dataset/transform helpers live in hamcls_utils.py.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hamcls_utils import (  # noqa: E402
    MedMNISTDataset,
    load_medmnist,
    set_seed,
    setup_logging,
    build_transforms,
    build_transforms_for_dataset,
    plot_curves,
    ConvNeXtBlock,
    HamiltonianScanLine,
    ORIENTATION_SENSITIVE,
)

warnings.filterwarnings('ignore')
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


# ============================================================
# 1. Single-direction SS2D bottleneck
# ============================================================
class HamiltonianSS2D(nn.Module):
    """Hamiltonian scan with selectable direction count (1 or 2).

    n_dirs=1: single horizontal scan (v2 original).
    n_dirs=2: horizontal + vertical scans, q/p/energy merged via 1x1 conv.

    The vertical scan provides orientation diversity that single-direction
    misses; the merge is learnable.
    """
    def __init__(self, dim, damping_clamp=5.0, n_dirs=1):
        super().__init__()
        assert n_dirs in (1, 2)
        self.dim = dim
        self.n_dirs = n_dirs
        self.scans = nn.ModuleList([
            HamiltonianScanLine(dim, damping_clamp) for _ in range(n_dirs)
        ])
        if n_dirs > 1:
            # Merge q/p/energy across directions via 1x1 conv
            self.q_merge = nn.Conv2d(n_dirs * dim, dim, 1, bias=False)
            self.p_merge = nn.Conv2d(n_dirs * dim, dim, 1, bias=False)
            self.e_merge = nn.Conv2d(n_dirs * dim, dim, 1, bias=False)

    def _scan_one_direction(self, x, d):
        """d=0: horizontal (along W). d=1: vertical (along H)."""
        B, C, H, W = x.shape
        if d == 0:
            lines = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
            q, p, e = self.scans[d](lines)
            q = q.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            p = p.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            e = e.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            # Vertical: scan along H. Build sequence per column.
            lines = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
            q, p, e = self.scans[d](lines)
            q = q.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()
            p = p.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()
            e = e.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        return q, p, e

    def forward(self, x):
        """x: (B, C, H, W). Returns (q, p, energy_map) each (B, C, H, W).

        For bidirectional scan, q/p/energy are merged across directions.
        For single-direction, returns the horizontal scan directly.

        Note: the returned q is from the HORIZONTAL scan (the canonical
        scan axis used by PSSP's FFT). For n_dirs > 1 the merged q is
        a learnable combination of horizontal and vertical scans.
        """
        if self.n_dirs == 1:
            q, p, e = self._scan_one_direction(x, 0)
            return q, p, e
        # Bidirectional
        q_list, p_list, e_list = [], [], []
        for d in range(self.n_dirs):
            qd, pd, ed = self._scan_one_direction(x, d)
            q_list.append(qd); p_list.append(pd); e_list.append(ed)
        q = self.q_merge(torch.cat(q_list, dim=1))
        p = self.p_merge(torch.cat(p_list, dim=1))
        e = self.e_merge(torch.cat(e_list, dim=1))
        return q, p, e


# ============================================================
# 2. Phase-Space Spectral Pooling (PSSP) head
# ============================================================
class PhaseSpaceSpectralPooling(nn.Module):
    """Phase-Space Spectral Pooling.

    Combines:
      * Complex-FFT pooling along the scan axis (real + imag bins),
        preserving phase information (where in the row a frequency
        component peaks). With magnitude included this gives 3*K features
        per channel (real, imag, mag).
      * Phase-space cross features <q*p> and <q^2 + p^2> globally pooled
        per channel (the joint statistics PSSA used to win on rare classes).
      * Optional SS2D-energy-driven attention: use the SS2D's actual
        per-channel energy map for row-pooling, instead of computing it from
        |Z|^2 (this is the "energy correlates with discriminative regions"
        signal the paper makes much of).
      * Auxiliary GAP path on q (training stability, prevents trajectory
        collapse the way it did in v2).

    Configuration via flags (default = full enrichment):
      use_complex   : keep real+imag (True) or magnitude-only (False)
      use_cross     : add <q*p>, <q^2+p^2> features (True) or skip (False)
      use_ss2d_energy: drive row-attention from SS2D energy map (True) or
                      from |Z|^2 (False)

    Output dim per channel:
      with use_complex=True: 3*K bins (real, imag, magnitude)
      with use_complex=False: K bins (magnitude only)
    Plus auxiliary features:
      always: GAP feature (1*C)
      use_cross=True: + 2*C (cross, orbital energy)
    """
    def __init__(self, dim, K_freq_bins=12,
                 use_complex=True, use_cross=True, use_ss2d_energy=True):
        super().__init__()
        self.dim = dim
        self.K = K_freq_bins
        self.use_complex = bool(use_complex)
        self.use_cross = bool(use_cross)
        self.use_ss2d_energy = bool(use_ss2d_energy)
        # Learnable temperature for the row-attention softmax
        self.log_temp = nn.Parameter(torch.zeros(1))
        # Per-channel learnable mixing of frequency bins. Output channels = 3K
        # for complex (real, imag, mag) or K for magnitude-only.
        out_per_ch = (3 if self.use_complex else 1) * K_freq_bins
        self.freq_proj = nn.Conv1d(dim, dim, 1, bias=False)  # acts on per-bin dim
        self.spec_dim_per_channel = out_per_ch
        if self.use_cross:
            # Small LayerNorm to scale-stabilize the cross / orbital energy paths
            self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
            self.orbital_norm = nn.LayerNorm(dim, eps=1e-6)

    def output_feature_dim(self):
        """Total feature vector dimensionality returned by forward()."""
        feat = self.dim * self.spec_dim_per_channel + self.dim  # spectrum + GAP
        if self.use_cross:
            feat += 2 * self.dim                                  # cross + orbital
        return feat

    def forward(self, q, p, energy_map=None):
        """q, p: (B, C, H, W). energy_map: (B, C, H, W) or None.
        Returns concatenated (B, output_feature_dim()) features."""
        B, C, H, W = q.shape
        with torch.cuda.amp.autocast(enabled=False):
            z = torch.complex(q.float(), p.float())
            Z = torch.fft.fft(z, dim=-1)
            K = min(self.K, Z.size(-1))
            Z_low = Z[..., :K]
            mag = Z_low.abs()

            # Row-attention source: prefer SS2D energy map if provided & enabled
            if self.use_ss2d_energy and energy_map is not None:
                # Per-channel mean energy per row -> (B, C, H)
                row_energy = energy_map.float().pow(2).mean(dim=-1)  # avg across W
            else:
                row_energy = mag.pow(2).sum(dim=-1)  # (B, C, H)
            attn = F.softmax(
                row_energy / self.log_temp.exp().clamp(min=1e-3),
                dim=-1,
            )

            # Pool spectrum
            mag_pool = (mag * attn.unsqueeze(-1)).sum(dim=-2)  # (B, C, K)
            if self.use_complex:
                real_pool = (Z_low.real * attn.unsqueeze(-1)).sum(dim=-2)
                imag_pool = (Z_low.imag * attn.unsqueeze(-1)).sum(dim=-2)
                spec_features = torch.cat([real_pool, imag_pool, mag_pool], dim=-1)
            else:
                spec_features = mag_pool

        spec_features = spec_features.to(q.dtype)
        spec_features = self.freq_proj(spec_features)        # (B, C, K or 3K)

        # Auxiliary GAP path
        gap_feat = q.mean(dim=(-1, -2))                       # (B, C)

        outputs = [spec_features.flatten(1), gap_feat]

        if self.use_cross:
            cross_feat = (q * p).mean(dim=(-1, -2))           # phase-space corr
            orbital_feat = (q.pow(2) + p.pow(2)).mean(dim=(-1, -2))  # orbital E
            cross_feat = self.cross_norm(cross_feat)
            orbital_feat = self.orbital_norm(orbital_feat)
            outputs.extend([cross_feat, orbital_feat])

        return torch.cat(outputs, dim=-1)


# ============================================================
# 3. HamCls main model
# ============================================================
class HamCls(nn.Module):
    """HamCls -- Phase-Space Spectral Classification.

    Encoder: stride-4 ConvNeXt stem + 2 ConvNeXt stages (lighter than v1).
    Bottleneck: single-direction SS2D (horizontal).
    Head: PSSP (frequency-domain features + auxiliary GAP).
    """
    def __init__(self, args):
        super().__init__()
        C = args.embed_dim
        depths = list(args.depths)
        if len(depths) < 2:
            depths = [3, 3]
        else:
            depths = depths[:2]
        dc = args.damping_clamp
        in_ch = getattr(args, 'in_channels', 3)
        n_cls = args.num_classes

        # Stride-4 ConvNeXt-style stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(1, C, eps=1e-6),
        )
        # Stage 1 at C @ 56x56 (224/4)
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(C) for _ in range(depths[0])])
        # Stride-2 downsample: C -> 2C, spatial 56 -> 28
        self.down = nn.Sequential(
            nn.GroupNorm(1, C, eps=1e-6),
            nn.Conv2d(C, C * 2, kernel_size=2, stride=2),
        )
        # Stage 2 at 2C @ 28x28
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(C * 2) for _ in range(depths[1])])

        # Bottleneck: configurable-direction SS2D (1 or 2)
        bottleneck_dim = C * 2
        n_dirs = int(getattr(args, 'n_scan_dirs', 1))
        self.n_scan_dirs = n_dirs
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim, eps=1e-6)
        self.ss2d = HamiltonianSS2D(bottleneck_dim, dc, n_dirs=n_dirs)

        # PSSP head with optional complex / cross / SS2D-energy attention
        self.pssp_K = int(getattr(args, 'pssp_K', 12))
        self.pssp_complex = bool(getattr(args, 'pssp_complex', True))
        self.pssp_cross = bool(getattr(args, 'pssp_cross', True))
        self.pssp_use_ss2d_energy = bool(getattr(args, 'pssp_use_ss2d_energy', True))
        self.pssp = PhaseSpaceSpectralPooling(
            bottleneck_dim,
            K_freq_bins=self.pssp_K,
            use_complex=self.pssp_complex,
            use_cross=self.pssp_cross,
            use_ss2d_energy=self.pssp_use_ss2d_energy,
        )

        # Classifier MLP
        head_drop = float(getattr(args, 'head_drop', 0.3))
        feat_dim = self.pssp.output_feature_dim()
        hidden = bottleneck_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(head_drop),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(head_drop * 0.5),
            nn.Linear(hidden, n_cls),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)               # (B, C, 56, 56)  for 224 input
        x = self.stage1(x)             # (B, C, 56, 56)
        x = self.down(x)               # (B, 2C, 28, 28)
        x = self.stage2(x)             # (B, 2C, 28, 28)

        # Bottleneck SS2D in fp32 for stability
        x_n = self.bottleneck_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        with torch.cuda.amp.autocast(enabled=False):
            q, p, energy_map = self.ss2d(x_n.float())
        q = q.to(x.dtype)
        p = p.to(x.dtype)
        energy_map = energy_map.to(x.dtype)

        # PSSP head: spectral + cross + auxiliary GAP features
        # PSSP returns one concatenated feature vector now.
        features = self.pssp(q, p, energy_map=energy_map)
        return self.classifier(features)


# ============================================================
# 4. Focal Loss (with optional class weights and label smoothing)
# ============================================================
class FocalLoss(nn.Module):
    """Focal loss = -(1-p_t)^gamma * log(p_t), optionally class-weighted.

    alpha: per-class weights tensor of shape (num_classes,), or None.
    gamma: focusing exponent (default 2.0; gamma=0 reduces to weighted CE).
    label_smoothing: target smoothing eps in [0, 1).
    """
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        n_cls = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if self.label_smoothing > 0.0:
            targets_oh = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1.0)
            targets_smooth = (1.0 - self.label_smoothing) * targets_oh + \
                             self.label_smoothing / n_cls
        else:
            targets_smooth = F.one_hot(targets, num_classes=n_cls).float()

        # Predicted prob for the true class (per sample)
        p_t = (probs * targets_smooth).sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets].unsqueeze(-1)
            focal_weight = focal_weight * alpha_t

        loss = -(focal_weight * targets_smooth * log_probs).sum(dim=-1).mean()
        return loss


# ============================================================
# 5. ModelEMA (exponential moving average of model weights)
# ============================================================
class ModelEMA:
    """Maintain a shadow copy of model parameters as an exponential moving
    average. Apply via apply_shadow(model) (saves current weights to backup);
    restore with restore(model)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


# ============================================================
# 6. CLI args
# ============================================================
def get_args():
    p = argparse.ArgumentParser(
        description='HamCls: Phase-Space Spectral Classification')
    # Dataset
    p.add_argument('--dataset', type=str, default='pathmnist')
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--num_classes', type=int, default=None)
    p.add_argument('--in_channels', type=int, default=3)
    # Model
    p.add_argument('--embed_dim', type=int, default=48)
    p.add_argument('--depths', type=int, nargs='+', default=[3, 3])
    p.add_argument('--damping_clamp', type=float, default=5.0)
    p.add_argument('--drop_rate', type=float, default=0.2)
    p.add_argument('--head_drop', type=float, default=0.3)
    p.add_argument('--pssp_K', type=int, default=12,
                   help='Number of frequency bins kept in PSSP (default 12).')
    # PSSP enrichment flags
    p.add_argument('--pssp_complex', action='store_true', default=True,
                   help='Keep real+imag FFT bins (preserves phase info). Default ON.')
    p.add_argument('--no_pssp_complex', dest='pssp_complex', action='store_false',
                   help='Use magnitude-only FFT (no phase).')
    p.add_argument('--pssp_cross', action='store_true', default=True,
                   help='Add phase-space cross features <q*p>, <q^2+p^2>. Default ON.')
    p.add_argument('--no_pssp_cross', dest='pssp_cross', action='store_false',
                   help='Skip phase-space cross features.')
    p.add_argument('--pssp_use_ss2d_energy', action='store_true', default=True,
                   help='Use SS2D energy map for row-attention (vs |Z|^2). Default ON.')
    p.add_argument('--no_pssp_use_ss2d_energy', dest='pssp_use_ss2d_energy',
                   action='store_false',
                   help='Use |Z|^2 magnitude for row-attention instead.')
    # Bottleneck scan directions
    p.add_argument('--n_scan_dirs', type=int, default=1, choices=[1, 2],
                   help='Number of SS2D scan directions (1=horizontal only, 2=horizontal+vertical). Default 1 for v2 back-compat.')
    # Training
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
    # Loss
    p.add_argument('--loss_type', type=str, default='ce',
                   choices=['ce', 'focal'])
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--balanced', action='store_true')
    # EMA
    p.add_argument('--use_ema', action='store_true',
                   help='Enable EMA of model weights for test-time evaluation.')
    p.add_argument('--ema_decay', type=float, default=0.999)
    # System
    p.add_argument('--output_dir', type=str, default='./outputs_hamcls')
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
    a.output_dir = os.path.join(a.output_dir, a.dataset)
    if not a.no_seed_subdir:
        a.output_dir = os.path.join(a.output_dir, f'seed_{a.seed}')
    return a


# ============================================================
# 7. Trainer (with EMA + focal-loss support)
# ============================================================
class Trainer:
    def __init__(self, args, model, train_loader, val_loader, test_loader, logger,
                 class_weights=None):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger

        alpha = None
        if class_weights is not None:
            alpha = torch.tensor(class_weights, dtype=torch.float32, device=args.device)
        if args.loss_type == 'focal':
            self.criterion = FocalLoss(
                gamma=args.focal_gamma,
                alpha=alpha,
                label_smoothing=args.label_smoothing,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=alpha,
                label_smoothing=args.label_smoothing,
            )

        # Param-group split: no weight decay for norms, biases, scale parameters
        no_decay_keys = ('bias', 'norm', 'gamma', 'log_freq', 'log_damping', 'log_temp')
        decay_params, no_decay_params = [], []
        for n, par in model.named_parameters():
            if not par.requires_grad:
                continue
            if any(k in n.lower() for k in no_decay_keys):
                no_decay_params.append(par)
            else:
                decay_params.append(par)
        self.optim = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': args.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=args.lr,
            betas=(0.9, 0.999),
        )
        self.scaler = GradScaler(enabled=args.use_amp)

        # Scheduler: cosine annealing with linear warmup
        steps_per_epoch = max(1, len(train_loader))
        total_iters = steps_per_epoch * args.epochs
        warmup_iters = steps_per_epoch * args.warmup_epochs
        min_lr_ratio = max(1e-8, args.min_lr / args.lr)

        def lr_lambda(it):
            if it < warmup_iters:
                return (it + 1) / max(1, warmup_iters)
            progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return cosine * (1.0 - min_lr_ratio) + min_lr_ratio

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)

        self.ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_cnt = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_correct, total_seen = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f'Train {epoch+1}/{self.args.epochs}',
                    leave=False, ncols=120)
        for imgs, labels in pbar:
            imgs = imgs.to(self.args.device, non_blocking=True)
            labels = labels.to(self.args.device, non_blocking=True)
            self.optim.zero_grad(set_to_none=True)
            with autocast(enabled=self.args.use_amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optim.step()
            self.sched.step()
            if self.ema is not None:
                self.ema.update(self.model)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_seen += labels.size(0)
            try:
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 acc=f'{100 * correct / max(1, labels.size(0)):.1f}%')
            except Exception:
                pass
        avg_loss = total_loss / max(1, total_seen)
        avg_acc = 100.0 * total_correct / max(1, total_seen)
        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(self, loader, use_ema=False):
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
        self.model.eval()
        total_loss, total_correct, total_seen = 0.0, 0, 0
        for imgs, labels in loader:
            imgs = imgs.to(self.args.device, non_blocking=True)
            labels = labels.to(self.args.device, non_blocking=True)
            with autocast(enabled=self.args.use_amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)
        avg_loss = total_loss / max(1, total_seen)
        avg_acc = 100.0 * total_correct / max(1, total_seen)
        return avg_loss, avg_acc

    @torch.no_grad()
    def test(self, use_ema=True):
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        for imgs, labels in self.test_loader:
            imgs = imgs.to(self.args.device, non_blocking=True)
            with autocast(enabled=self.args.use_amp):
                logits = self.model(imgs)
            probs = F.softmax(logits.float(), dim=-1)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                                     classification_report)
        acc = accuracy_score(all_labels, all_preds)
        # AUC: for binary, use the positive-class probability column directly;
        # multi_class='ovr' on a 2-column matrix can degenerate to NaN.
        try:
            n_cls = all_probs.shape[1]
            if n_cls == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs,
                                    multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        n_cls = all_probs.shape[1]
        per_class = []
        for c in range(n_cls):
            mask = all_labels == c
            cls_acc = float((all_preds[mask] == c).mean()) if mask.any() else 0.0
            cls_f1 = float(f1_score(all_labels == c, all_preds == c, zero_division=0))
            per_class.append({'class': int(c), 'acc': cls_acc, 'f1': cls_f1})

        results = {
            'accuracy': float(acc),
            'auc': float(auc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'best_val_acc': float(self.best_val_acc),
            'best_epoch': int(self.best_epoch),
            'per_class': per_class,
            'used_ema': bool(use_ema and self.ema is not None),
        }
        out_dir = self.args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'test_results_final.json'), 'w') as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
            f.write('HamCls (PSSC) -- test report\n')
            f.write(f'Dataset: {self.args.dataset}  Seed: {self.args.seed}\n')
            f.write(f'EMA at test: {results["used_ema"]}\n')
            f.write('=' * 50 + '\n')
            f.write(f'Accuracy       : {acc * 100:.2f}%\n')
            f.write(f'AUC (macro)    : {auc * 100:.2f}%\n')
            f.write(f'F1 (macro)     : {f1_macro * 100:.2f}%\n')
            f.write(f'F1 (weighted)  : {f1_weighted * 100:.2f}%\n')
            f.write(f'Best val acc   : {self.best_val_acc:.2f}%\n')
            f.write(f'Best epoch     : {self.best_epoch}\n')
            for c in per_class:
                f.write(f'Class {c["class"]:>2}  acc={c["acc"] * 100:.1f}%  f1={c["f1"] * 100:.1f}%\n')
            f.write('\n' + classification_report(all_labels, all_preds, zero_division=0))
        return results

    def save_best(self):
        out_dir = self.args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
            torch.save(self.model.state_dict(), os.path.join(out_dir, 'best_model_ema.pth'))
            self.ema.restore(self.model)

    def train(self):
        for epoch in range(self.args.epochs):
            tr_loss, tr_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader, use_ema=self.args.use_ema)
            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optim.param_groups[0]['lr'])
            best_marker = ''
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_cnt = 0
                best_marker = 'BEST'
                self.save_best()
            else:
                self.patience_cnt += 1
            self.logger.info(
                f'Ep {epoch + 1:>3d}/{self.args.epochs} | '
                f'L:{tr_loss:.4f}/{val_loss:.4f} | '
                f'Acc:{tr_acc:.2f}/{val_acc:.2f} | '
                f'LR:{self.optim.param_groups[0]["lr"]:.6f} {best_marker}'
            )
            if self.patience_cnt >= self.args.patience:
                self.logger.info(f'Early stopping at epoch {epoch + 1}')
                break


# ============================================================
# 8. Main
# ============================================================
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info('=' * 60)
    logger.info('  HamCls: Phase-Space Spectral Classification')
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

    # Dataset-aware augmentation policy: orientation-sensitive datasets
    # (organa/c/s, retina) get hflip+vflip disabled to avoid label corruption.
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

    logger.info('Building HamCls...')
    for k in ['device', 'num_classes', 'embed_dim', 'depths', 'size', 'batch_size',
              'lr', 'weight_decay', 'drop_rate', 'head_drop', 'pssp_K',
              'loss_type', 'focal_gamma', 'label_smoothing', 'balanced',
              'use_ema', 'ema_decay']:
        logger.info(f'  {k:15s}: {getattr(args, k)}')
    model = HamCls(args).to(args.device)
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
    logger.info('  HAMCLS v2 TEST RESULTS')
    logger.info('=' * 50)
    logger.info(f'  Accuracy       : {results["accuracy"] * 100:.2f}%')
    logger.info(f'  AUC (macro)    : {results["auc"] * 100:.2f}%')
    logger.info(f'  F1 (macro)     : {results["f1_macro"] * 100:.2f}%')
    logger.info(f'  F1 (weighted)  : {results["f1_weighted"] * 100:.2f}%')
    logger.info(f'  Best val acc   : {results["best_val_acc"]:.2f}%')
    logger.info(f'  Best epoch     : {results["best_epoch"]}')
    logger.info(f'  Used EMA       : {results["used_ema"]}')
    for c in results['per_class']:
        logger.info(f'  Class {c["class"]:>2}  acc={c["acc"] * 100:.1f}%  f1={c["f1"] * 100:.1f}%')

    if trainer.history['train_loss']:
        plot_curves(trainer.history, args.output_dir)
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(trainer.history, f, indent=2)
    logger.info('Done!')


if __name__ == '__main__':
    main()
