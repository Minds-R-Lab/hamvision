#!/usr/bin/env python3
"""
hamcls_utils
============

Utility functions and shared building blocks used by hamcls.py:

  * set_seed, setup_logging, plot_curves
  * MedMNISTDataset, load_medmnist
  * build_transforms, build_transforms_for_dataset, ORIENTATION_SENSITIVE
  * ConvNeXtBlock, HamiltonianScanLine

These were factored out so the HamCls training script and any auxiliary
script (ablation harness, FLOPs measurement, interpretability visualisation)
can pull from a single, well-defined module.
"""

import math
import os
import sys
import time
import json
import random
import warnings
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(os.path.join(output_dir, 'train.log')),
                  logging.StreamHandler()])
    return logging.getLogger(__name__)


# ============================================================
# 2. DATASET — MedMNIST npz loading
# ============================================================
class MedMNISTDataset(Dataset):
    """Load MedMNIST .npz files. Works with any resolution (28, 64, 128, 224)."""

    def __init__(self, images, labels, transform=None):
        self.images = images    # (N, H, W, C) uint8 or (N, H, W)
        self.labels = labels    # (N,) or (N, 1) int
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx].flatten()[0])

        # Convert to PIL
        if img.ndim == 2:
            img = Image.fromarray(img, mode='L').convert('RGB')
        elif img.shape[-1] == 1:
            img = Image.fromarray(img[:, :, 0], mode='L').convert('RGB')
        else:
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def load_medmnist(dataset_name, size=224, data_root=None):
    """Load MedMNIST dataset. Tries medmnist package first, falls back to npz."""
    info = None
    n_classes = None

    # Try medmnist package
    try:
        import medmnist
        from medmnist import INFO
        info = INFO[dataset_name]
        n_classes = len(info['label'])
        n_channels = info['n_channels']

        DataClass = getattr(medmnist, info['python_class'])
        download = True
        root = data_root or './data'

        train_ds_raw = DataClass(split='train', download=download, root=root, size=size)
        val_ds_raw = DataClass(split='val', download=download, root=root, size=size)
        test_ds_raw = DataClass(split='test', download=download, root=root, size=size)

        train_imgs = train_ds_raw.imgs
        train_lbls = train_ds_raw.labels
        val_imgs = val_ds_raw.imgs
        val_lbls = val_ds_raw.labels
        test_imgs = test_ds_raw.imgs
        test_lbls = test_ds_raw.labels

    except (ImportError, KeyError):
        # Fall back to manual npz loading
        if data_root and os.path.exists(data_root):
            npz_path = data_root
        else:
            # Try standard MedMNIST npz location
            npz_name = f'{dataset_name}.npz' if size == 28 else f'{dataset_name}_{size}.npz'
            npz_path = os.path.join(data_root or './data', npz_name)

        if not os.path.exists(npz_path):
            print(f"Cannot find {npz_path}")
            print(f"Please install medmnist (pip install medmnist) or download the npz file.")
            print(f"Download from: https://medmnist.com/")
            sys.exit(1)

        data = np.load(npz_path)
        train_imgs = data['train_images']
        train_lbls = data['train_labels']
        val_imgs = data['val_images']
        val_lbls = data['val_labels']
        test_imgs = data['test_images']
        test_lbls = data['test_labels']

    # Infer num_classes
    all_labels = np.concatenate([train_lbls.flatten(), val_lbls.flatten(), test_lbls.flatten()])
    if n_classes is None:
        n_classes = int(all_labels.max()) + 1

    return train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls, n_classes


def build_transforms(size, is_train=True, hflip=True, vflip=True,
                     rotation_deg=15, color_jitter=True, affine_translate=0.05):
    """Standard classification transforms with augmentation.

    For datasets where anatomical orientation is the primary discriminator
    (OrganA/C/SMNIST, RetinaMNIST), call with hflip=False, vflip=False,
    rotation_deg=0 -- the default flips are silently corrupting labels
    (e.g. RandomHorizontalFlip turns kidney-left into kidney-right while
    keeping the label).
    """
    if is_train:
        ops = [transforms.Resize((size, size))]
        if hflip:
            ops.append(transforms.RandomHorizontalFlip())
        if vflip:
            ops.append(transforms.RandomVerticalFlip())
        if rotation_deg and rotation_deg > 0:
            ops.append(transforms.RandomRotation(rotation_deg))
        if color_jitter:
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
        if affine_translate and affine_translate > 0:
            ops.append(transforms.RandomAffine(degrees=0,
                                               translate=(affine_translate, affine_translate)))
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(ops)
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# Datasets where the image plane encodes patient left/right.
# Hflip swaps L vs R anatomy here, so it corrupts labels for paired organs
# (femur-L/R, kidney-L/R). Empirically: removing hflip jumps OrganAMNIST
# F1 by +15.7 pts (77.58 -> 93.28).
#
# OrganSMNIST is the SAGITTAL view: patient L/R is encoded by which slice
# was taken, NOT by the image's horizontal axis (which is anterior-posterior).
# Hflip there is a legitimate augmentation and removing it HURTS by ~3 pts
# AUC due to lost regularization on the smaller (14K) training set.
ORIENTATION_SENSITIVE = {
    'organamnist',   # axial:    image horizontal axis = patient L/R
    'organcmnist',   # coronal:  image horizontal axis = patient L/R
    # NOTE: 'organsmnist' (sagittal) is intentionally NOT in this set --
    # see comment above. Hflip is a legitimate augmentation for sagittal CT.
}


def build_transforms_for_dataset(dataset_name, size, is_train=True):
    """Wrapper that picks the right augmentation policy per dataset."""
    name = (dataset_name or '').lower()
    if name in ORIENTATION_SENSITIVE:
        # Anatomy-sensitive: keep small rotation + translation, drop flips
        return build_transforms(size, is_train=is_train,
                                hflip=False, vflip=False,
                                rotation_deg=10, color_jitter=False,
                                affine_translate=0.05)
    # Default: full augmentation policy that won the SOTA datasets
    return build_transforms(size, is_train=is_train)


# ============================================================
# 3. MODEL COMPONENTS (reused from HamSeg)
# ============================================================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class HamiltonianScanLine(nn.Module):
    """Damped harmonic oscillator — parallel scan over rows/columns."""
    def __init__(self, d_model, damping_clamp=5.0):
        super().__init__()
        self.damping_clamp = damping_clamp
        self.log_k = nn.Parameter(torch.linspace(-1, 3, d_model))
        self.nu_scale = nn.Parameter(torch.ones(d_model))
        self.nu_bias = nn.Parameter(torch.ones(d_model) * 1.0)
        self.dt_scale = nn.Parameter(torch.ones(d_model) * 0.3)
        self.dt_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        B, L, D = x.shape
        x_f = x.float()
        omega = torch.exp(self.log_k.float() / 2.0)

        nu = torch.clamp(F.softplus(x_f * self.nu_scale + self.nu_bias) + 1e-6,
                         max=self.damping_clamp)
        dt = F.softplus(x_f * self.dt_scale + self.dt_bias) + 1e-6

        log_decay = -nu * dt
        angle = omega.unsqueeze(0).unsqueeze(0) * dt

        L_re = torch.cumsum(log_decay, dim=1).clamp(-5, 0)
        L_im = torch.cumsum(angle, dim=1)

        scale = torch.exp(-L_re)
        cos_neg = torch.cos(-L_im)
        sin_neg = torch.sin(-L_im)

        rot_re = x_f * scale * cos_neg
        rot_im = x_f * scale * sin_neg

        acc_re = torch.cumsum(rot_re, dim=1)
        acc_im = torch.cumsum(rot_im, dim=1)

        unscale = torch.exp(L_re)
        cos_L = torch.cos(L_im)
        sin_L = torch.sin(L_im)

        q = unscale * (cos_L * acc_re - sin_L * acc_im)
        p = unscale * (sin_L * acc_re + cos_L * acc_im)

        q = q.clamp(-50, 50)
        p = p.clamp(-50, 50)
        energy = 0.5 * (q * q + p * p)

        return q.to(x.dtype), p.to(x.dtype), energy.to(x.dtype)




def plot_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('HamCls Training', fontsize=14, fontweight='bold')

    axes[0].plot(epochs, history['train_loss'], 'b-', lw=2, label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', lw=2, label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [100 * a for a in history['train_acc']], 'b-', lw=2, label='Train')
    axes[1].plot(epochs, [100 * a for a in history['val_acc']], 'r-', lw=2, label='Val')
    axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history['lr'], 'g-', lw=2)
    axes[2].set_title('Learning Rate'); axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 7. MAIN
# ============================================================
