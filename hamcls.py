#!/usr/bin/env python3
"""
HamCls: Hamiltonian Classification Network
============================================

Adapts the Hamiltonian oscillator inductive bias from HamSeg to image
classification. The encoder + Hamiltonian bottleneck remain identical;
the U-Net decoder is replaced by a phase-space pooling head that aggregates
position (features), momentum (spatial gradients), and energy (saliency)
into a compact classification vector.

Architecture:
  Encoder: ConvNeXt stem + 3 stages (C → 2C → 4C → 8C)
  Bottleneck: ConvNeXt || Oscillator Bank → gated fusion
  Head: Phase-space pooling (features + momentum + energy) → MLP → classes

Supports MedMNIST datasets (npz format) at any resolution.

Usage:
    # Install medmnist: pip install medmnist
    # Requires: pip install scikit-learn (for AUC, F1 metrics)
    python hamcls.py --dataset pathmnist --size 224 --epochs 100
    python hamcls.py --dataset dermamnist --size 224 --epochs 100
    python hamcls.py --dataset bloodmnist --size 224 --epochs 100
    python hamcls.py --dataset organamnist --size 224 --epochs 100

    # Or with local npz:
    python hamcls.py --dataset custom --data_root ./data/my_dataset.npz --num_classes 9

Author: Mohamed Mabrok
"""

import os, sys, math, time, json, random, warnings, argparse, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================
# 1. ARGUMENTS
# ============================================================
def get_args():
    p = argparse.ArgumentParser(description='HamCls: Hamiltonian Classification')
    # Dataset
    p.add_argument('--dataset', type=str, default='pathmnist',
                   help='MedMNIST dataset name or "custom"')
    p.add_argument('--data_root', type=str, default=None,
                   help='Path to npz file (for custom datasets)')
    p.add_argument('--size', type=int, default=224,
                   help='Image size (28, 64, 128, 224)')
    p.add_argument('--num_classes', type=int, default=None,
                   help='Override number of classes')
    p.add_argument('--in_channels', type=int, default=3)
    # Model
    p.add_argument('--embed_dim', type=int, default=48)
    p.add_argument('--depths', type=int, nargs='+', default=[2, 2, 2, 2])
    p.add_argument('--damping_clamp', type=float, default=5.0)
    p.add_argument('--drop_rate', type=float, default=0.2)
    p.add_argument('--head_drop', type=float, default=0.3)
    # Training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_true')
    # System
    p.add_argument('--output_dir', type=str, default='./outputs_hamcls')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--test_only', action='store_true')
    p.add_argument('--test_every', type=int, default=30,
                   help='Run test evaluation every N epochs (0=disabled)')
    p.add_argument('--balanced', action='store_true',
                   help='Use inverse-frequency class weights for imbalanced datasets')

    a = p.parse_args()
    if a.no_amp: a.use_amp = False
    a.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a.output_dir = os.path.join(a.output_dir, a.dataset)
    return a


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


def build_transforms(size, is_train=True):
    """Standard classification transforms with augmentation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


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


class HamiltonianSS2D(nn.Module):
    """4-direction row/column oscillator scan for 2D feature maps."""
    def __init__(self, dim, damping_clamp=5.0):
        super().__init__()
        self.scans = nn.ModuleList([HamiltonianScanLine(dim, damping_clamp) for _ in range(4)])
        self.pos_merge = nn.Linear(dim * 4, dim, bias=False)
        self.mom_merge = nn.Linear(dim * 4, dim, bias=False)

    def _to_lines(self, x, d):
        B, C, H, W = x.shape
        if d == 0:
            return x.permute(0, 2, 1, 3).reshape(B * H, C, W).permute(0, 2, 1), H, W
        elif d == 1:
            return x.permute(0, 2, 1, 3).reshape(B * H, C, W).flip(1).permute(0, 2, 1), H, W
        elif d == 2:
            return x.permute(0, 3, 1, 2).reshape(B * W, C, H).permute(0, 2, 1), H, W
        else:
            return x.permute(0, 3, 1, 2).reshape(B * W, C, H).flip(1).permute(0, 2, 1), H, W

    def _to_2d(self, s, B, H, W, d):
        C = s.shape[2]
        if d == 0:
            return s.permute(0, 2, 1).reshape(B, H, C, W).permute(0, 2, 1, 3)
        elif d == 1:
            return s.flip(1).permute(0, 2, 1).reshape(B, H, C, W).permute(0, 2, 1, 3)
        elif d == 2:
            return s.permute(0, 2, 1).reshape(B, W, C, H).permute(0, 2, 3, 1)
        else:
            return s.flip(1).permute(0, 2, 1).reshape(B, W, C, H).permute(0, 2, 3, 1)

    def forward(self, x):
        B, C, h, w = x.shape
        pos_l, mom_l, eng_l = [], [], []
        for d in range(4):
            lines, h_, w_ = self._to_lines(x, d)
            q, p, e = self.scans[d](lines)
            pos_l.append(self._to_2d(q, B, h_, w_, d))
            mom_l.append(self._to_2d(p, B, h_, w_, d))
            eng_l.append(self._to_2d(e, B, h_, w_, d))
        pos = self.pos_merge(torch.cat(pos_l, 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        mom = self.mom_merge(torch.cat(mom_l, 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        energy = torch.stack(eng_l, 0).mean(0)
        return pos, mom, energy


class HamiltonianBottleneck(nn.Module):
    """ConvNeXt + oscillator bank with gated fusion."""
    def __init__(self, dim, damping_clamp=5.0, drop_rate=0.1):
        super().__init__()
        self.conv_block = ConvNeXtBlock(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ss2d = HamiltonianSS2D(dim, damping_clamp)
        self.pos_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[0].bias, 2.0)
        self.drop = nn.Dropout2d(drop_rate)
        self.energy_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_block(x)
        x_n = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        with torch.cuda.amp.autocast(enabled=False):
            pos, mom, energy_raw = self.ss2d(x_n.float())
            ham_out = self.pos_proj(pos)
            g = self.gate(torch.cat([conv_out.float(), ham_out], 1))
            out = conv_out.float() * g + ham_out * (1 - g)
        out = self.drop(out.to(x.dtype))
        mom = mom.to(x.dtype)
        energy_f = energy_raw.to(x.dtype)
        ch_weights = self.energy_attn(energy_f)
        ch_weights = ch_weights.unsqueeze(-1).unsqueeze(-1)
        energy_map = (energy_f * ch_weights).mean(dim=1, keepdim=True)
        return out, mom, energy_map


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 2, stride=2, bias=False),
            nn.BatchNorm2d(dim * 2))

    def forward(self, x):
        return self.reduction(x)


# ============================================================
# 4. HAMCLS — Classification model
# ============================================================
class HamCls(nn.Module):
    """Hamiltonian Classification Network.

    Same encoder + bottleneck as HamSeg, but replaces the U-Net decoder with
    a phase-space pooling head that aggregates features, momentum, and energy
    into a classification vector.
    """
    def __init__(self, args):
        super().__init__()
        C = args.embed_dim
        depths = args.depths
        dc = args.damping_clamp
        in_ch = getattr(args, 'in_channels', 3)
        n_cls = args.num_classes

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.GELU())

        self.enc1 = nn.Sequential(*[ConvNeXtBlock(C) for _ in range(depths[0])])
        self.down1 = PatchMerging(C)
        self.enc2 = nn.Sequential(*[ConvNeXtBlock(C * 2) for _ in range(depths[1])])
        self.down2 = PatchMerging(C * 2)
        self.enc3 = nn.Sequential(*[ConvNeXtBlock(C * 4) for _ in range(depths[2])])
        self.down3 = PatchMerging(C * 4)

        # Hamiltonian bottleneck
        drop_rate = getattr(args, 'drop_rate', 0.1)
        self.bottleneck = nn.ModuleList([
            HamiltonianBottleneck(C * 8, dc, drop_rate) for _ in range(depths[3])])

        # Phase-space pooling head
        # Feature branch: GAP on bottleneck features
        feat_dim = C * 8

        # Momentum branch: L2 norm across channels → GAP
        mom_dim = C * 8

        # Energy branch: already 1-channel from bottleneck → GAP → scalar features
        energy_feat_dim = 16  # small MLP on energy statistics

        # Energy statistics extractor
        self.energy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, energy_feat_dim),
            nn.GELU(),
        )

        # Momentum statistics extractor
        self.mom_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Combined classification head
        head_drop = getattr(args, 'head_drop', 0.3)
        total_dim = feat_dim + mom_dim + energy_feat_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(head_drop),
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Dropout(head_drop * 0.5),
            nn.Linear(total_dim // 2, n_cls),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.down3(e3)

        # Hamiltonian bottleneck
        momentum, energy = None, None
        for blk in self.bottleneck:
            e4, momentum, energy = blk(e4)

        # Phase-space pooling
        # Features: global average pool
        feat = F.adaptive_avg_pool2d(e4, 1).flatten(1)  # (B, 8C)

        # Momentum: pool magnitude
        mom_pool = self.mom_head(momentum)  # (B, 8C)

        # Energy: pool scalar energy map
        en_pool = self.energy_head(energy)  # (B, energy_feat_dim)

        # Concatenate and classify
        combined = torch.cat([feat, mom_pool, en_pool], dim=1)
        return self.classifier(combined)


# ============================================================
# 5. TRAINING
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
        if class_weights is not None:
            w = torch.FloatTensor(class_weights).to(args.device)
            self.criterion = nn.CrossEntropyLoss(weight=w)
            logger.info(f'  Class weights: {[f"{x:.2f}" for x in class_weights]}')
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
        self.scaler = GradScaler(enabled=args.use_amp)
        self.history = {k: [] for k in [
            'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']}
        self.best_val_acc = 0
        self.best_epoch = 0
        self.start_epoch = 0
        self.no_improve = 0

    def _warmup_lr(self, epoch, bi, total):
        if epoch < self.args.warmup_epochs:
            prog = (epoch * total + bi) / (self.args.warmup_epochs * total)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.args.lr * max(prog, 0.01)

    def train_epoch(self, epoch):
        self.model.train()
        loss_sum, correct, total_samples = 0, 0, 0
        pbar = tqdm(self.train_loader, desc=f'Train {epoch + 1}/{self.args.epochs}',
                    leave=False, dynamic_ncols=True)
        for bi, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(self.args.device, non_blocking=True)
            labels = labels.to(self.args.device, non_blocking=True)
            self._warmup_lr(epoch, bi, len(self.train_loader))
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.args.use_amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total_samples += imgs.size(0)
            if bi % 20 == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                acc=f'{100 * correct / max(total_samples, 1):.1f}%')
        return loss_sum / max(total_samples, 1), correct / max(total_samples, 1)

    @torch.no_grad()
    def evaluate(self, loader, desc='Val'):
        self.model.eval()
        loss_sum, correct, total_samples = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        for imgs, labels in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            imgs = imgs.to(self.args.device, non_blocking=True)
            labels = labels.to(self.args.device, non_blocking=True)
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss_sum += loss.item() * imgs.size(0)
            probs = torch.softmax(logits.float(), dim=1)
            preds = probs.argmax(1)
            correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        acc = correct / max(total_samples, 1)
        avg_loss = loss_sum / max(total_samples, 1)
        all_probs = np.concatenate(all_probs, axis=0)
        return avg_loss, acc, np.array(all_preds), np.array(all_labels), all_probs

    @staticmethod
    def compute_metrics(labels, preds, probs, n_classes):
        """Compute ACC, AUC (macro OvR), F1 (macro & weighted)."""
        from sklearn.metrics import (accuracy_score, roc_auc_score,
                                     f1_score, classification_report,
                                     confusion_matrix)
        metrics = {}
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)

        # AUC — one-vs-rest, macro average
        try:
            if n_classes == 2:
                metrics['auc'] = roc_auc_score(labels, probs[:, 1])
            else:
                # One-hot encode labels for multi-class AUC
                from sklearn.preprocessing import label_binarize
                lb = label_binarize(labels, classes=list(range(n_classes)))
                # Handle classes not present in labels
                if lb.shape[1] == 1 and n_classes == 2:
                    lb = np.hstack([1 - lb, lb])
                metrics['auc'] = roc_auc_score(lb, probs, average='macro',
                                               multi_class='ovr')
        except (ValueError, IndexError):
            metrics['auc'] = 0.0

        # Per-class F1
        metrics['per_class_f1'] = f1_score(labels, preds, average=None, zero_division=0).tolist()

        # Per-class accuracy
        per_class_acc = []
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                per_class_acc.append(float((preds[mask] == c).mean()))
            else:
                per_class_acc.append(0.0)
        metrics['per_class_acc'] = per_class_acc

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()

        # Classification report string
        metrics['report_str'] = classification_report(
            labels, preds, digits=4, zero_division=0)

        return metrics

    def save_ckpt(self, epoch, is_best=False):
        torch.save({
            'epoch': epoch, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }, os.path.join(self.args.output_dir, 'last_checkpoint.pth'))
        if is_best:
            torch.save(self.model.state_dict(),
                       os.path.join(self.args.output_dir, 'best_model.pth'))

    def load_ckpt(self):
        path = os.path.join(self.args.output_dir, 'last_checkpoint.pth')
        if not os.path.exists(path): return
        ckpt = torch.load(path, map_location=self.args.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_acc = ckpt['best_val_acc']
        self.best_epoch = ckpt['best_epoch']
        self.history = ckpt['history']
        self.logger.info(f'Resumed from epoch {self.start_epoch}')

    def train(self):
        if self.args.resume:
            self.load_ckpt()
        self.logger.info(f'Training {self.args.epochs} epochs from {self.start_epoch}')

        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()
            tr_loss, tr_acc = self.train_epoch(epoch)

            if epoch >= self.args.warmup_epochs:
                self.scheduler.step()

            vl_loss, vl_acc, _, _, _ = self.evaluate(self.val_loader)
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(vl_acc)
            self.history['lr'].append(lr)

            is_best = vl_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = vl_acc
                self.best_epoch = epoch + 1
                self.no_improve = 0
            else:
                self.no_improve += 1

            self.save_ckpt(epoch, is_best)
            self.logger.info(
                f'Ep {epoch + 1:3d}/{self.args.epochs} | '
                f'L:{tr_loss:.4f}/{vl_loss:.4f} | '
                f'Acc:{100 * tr_acc:.2f}/{100 * vl_acc:.2f} | '
                f'LR:{lr:.6f} | {time.time() - t0:.0f}s'
                f'{" BEST" if is_best else ""}')

            # Periodic testing — test current model (not best)
            if self.args.test_every > 0 and (epoch + 1) % self.args.test_every == 0:
                self.logger.info(f'--- Periodic test at epoch {epoch + 1} ---')
                metrics = self._run_test(tag=f'epoch{epoch + 1:03d}', load_best=False)
                self.logger.info(
                    f'  Test ACC: {100 * metrics["accuracy"]:.2f}%  '
                    f'AUC: {100 * metrics["auc"]:.2f}%  '
                    f'F1-macro: {100 * metrics["f1_macro"]:.2f}%')

            if self.args.patience > 0 and self.no_improve >= self.args.patience:
                self.logger.info(f'Early stopping at epoch {epoch + 1}')
                break

    def _run_test(self, tag='final', load_best=True):
        """Run test evaluation, compute full metrics, save with tag."""
        if load_best:
            best = os.path.join(self.args.output_dir, 'best_model.pth')
            if os.path.exists(best):
                self.model.load_state_dict(
                    torch.load(best, map_location=self.args.device, weights_only=True))

        loss, acc, preds, labels, probs = self.evaluate(self.test_loader, f'Test-{tag}')
        metrics = self.compute_metrics(labels, preds, probs, self.args.num_classes)
        metrics['loss'] = round(loss, 4)
        metrics['tag'] = tag
        metrics['best_val_acc'] = round(100 * self.best_val_acc, 2)
        metrics['best_epoch'] = self.best_epoch

        # Save tagged result
        fname = f'test_results_{tag}.json'
        save_metrics = {k: v for k, v in metrics.items()
                       if k != 'report_str' and k != 'confusion_matrix'}
        save_metrics['accuracy'] = round(100 * metrics['accuracy'], 2)
        save_metrics['auc'] = round(100 * metrics['auc'], 2)
        save_metrics['f1_macro'] = round(100 * metrics['f1_macro'], 2)
        save_metrics['f1_weighted'] = round(100 * metrics['f1_weighted'], 2)
        save_metrics['per_class_acc'] = [round(100 * a, 2) for a in metrics['per_class_acc']]
        save_metrics['per_class_f1'] = [round(100 * f, 2) for f in metrics['per_class_f1']]
        save_metrics['params'] = sum(p.numel() for p in self.model.parameters())
        with open(os.path.join(self.args.output_dir, fname), 'w') as f:
            json.dump(save_metrics, f, indent=2)

        return metrics

    def test(self):
        metrics = self._run_test(tag='final', load_best=True)

        self.logger.info('=' * 50)
        self.logger.info('  HAMCLS TEST RESULTS')
        self.logger.info('=' * 50)
        self.logger.info(f'  Accuracy       : {100 * metrics["accuracy"]:.2f}%')
        self.logger.info(f'  AUC (macro)    : {100 * metrics["auc"]:.2f}%')
        self.logger.info(f'  F1 (macro)     : {100 * metrics["f1_macro"]:.2f}%')
        self.logger.info(f'  F1 (weighted)  : {100 * metrics["f1_weighted"]:.2f}%')
        self.logger.info(f'  Loss           : {metrics["loss"]:.4f}')
        self.logger.info(f'  Best val acc   : {100 * self.best_val_acc:.2f}%')
        self.logger.info(f'  Best epoch     : {self.best_epoch}')

        n_cls = self.args.num_classes
        for c in range(n_cls):
            ca = metrics['per_class_acc'][c] if c < len(metrics['per_class_acc']) else 0
            cf = metrics['per_class_f1'][c] if c < len(metrics['per_class_f1']) else 0
            self.logger.info(f'  Class {c:2d}  acc={100*ca:.1f}%  f1={100*cf:.1f}%')

        self._save_report(metrics)
        return metrics['accuracy']

    def _save_report(self, metrics):
        lines = []
        lines.append('=' * 60)
        lines.append(f'  HamCls — Results Report')
        lines.append(f'  Dataset: {self.args.dataset}')
        lines.append(f'  Date: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append('=' * 60)
        lines.append('')
        lines.append('--- Configuration ---')
        for k in ['dataset', 'size', 'num_classes', 'embed_dim', 'depths',
                   'epochs', 'batch_size', 'lr', 'weight_decay', 'drop_rate', 'head_drop']:
            lines.append(f'  {k:20s}: {getattr(self.args, k)}')
        total = sum(p.numel() for p in self.model.parameters())
        lines.append(f'  {"parameters":20s}: {total:,} ({total * 4 / 1024 ** 2:.1f} MB)')
        lines.append('')
        lines.append('--- Test Results ---')
        lines.append(f'  Accuracy        : {100 * metrics["accuracy"]:.2f}%')
        lines.append(f'  AUC (macro OvR) : {100 * metrics["auc"]:.2f}%')
        lines.append(f'  F1 (macro)      : {100 * metrics["f1_macro"]:.2f}%')
        lines.append(f'  F1 (weighted)   : {100 * metrics["f1_weighted"]:.2f}%')
        lines.append(f'  Loss            : {metrics["loss"]:.4f}')
        lines.append(f'  Best val acc    : {100 * self.best_val_acc:.2f}%')
        lines.append(f'  Best epoch      : {self.best_epoch}')
        lines.append('')
        lines.append('--- Per-Class Results ---')
        n_cls = self.args.num_classes
        lines.append(f'  {"Class":>8s}  {"Accuracy":>10s}  {"F1":>10s}')
        lines.append(f'  {"-"*8}  {"-"*10}  {"-"*10}')
        for c in range(n_cls):
            ca = metrics['per_class_acc'][c] if c < len(metrics['per_class_acc']) else 0
            cf = metrics['per_class_f1'][c] if c < len(metrics['per_class_f1']) else 0
            lines.append(f'  {c:>8d}  {100*ca:>9.2f}%  {100*cf:>9.2f}%')
        lines.append('')
        lines.append('--- Classification Report (sklearn) ---')
        lines.append(metrics.get('report_str', ''))
        lines.append('')
        # Periodic test results
        lines.append('--- Periodic Test Results ---')
        for f in sorted(Path(self.args.output_dir).glob('test_results_epoch*.json')):
            with open(f) as fh:
                r = json.load(fh)
            tag = r.get('tag', '?')
            lines.append(
                f'  {tag:12s}: ACC={r.get("accuracy",0):.2f}%  '
                f'AUC={r.get("auc",0):.2f}%  '
                f'F1={r.get("f1_macro",0):.2f}%')
        lines.append('')
        lines.append('--- Training Summary ---')
        if self.history['train_loss']:
            lines.append(f'  Total epochs run  : {len(self.history["train_loss"])}')
            lines.append(f'  Final train loss  : {self.history["train_loss"][-1]:.4f}')
            lines.append(f'  Final val loss    : {self.history["val_loss"][-1]:.4f}')
            lines.append(f'  Final train acc   : {100*self.history["train_acc"][-1]:.2f}%')
            lines.append(f'  Final val acc     : {100*self.history["val_acc"][-1]:.2f}%')
        lines.append('')
        lines.append('=' * 60)

        report_path = os.path.join(self.args.output_dir, 'report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        self.logger.info(f'Report saved to {report_path}')


# ============================================================
# 6. VISUALIZATION
# ============================================================
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
def main():
    args = get_args()
    set_seed(args.seed)
    logger = setup_logging(args.output_dir)

    logger.info('=' * 60)
    logger.info(f'  HamCls: Hamiltonian Classification Network')
    logger.info(f'  Dataset: {args.dataset} → {args.output_dir}')
    logger.info('=' * 60)

    # Load data
    logger.info('Loading data...')
    train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls, n_classes = \
        load_medmnist(args.dataset, args.size, args.data_root)

    if args.num_classes is None:
        args.num_classes = n_classes
    logger.info(f'  Classes: {args.num_classes}')
    logger.info(f'  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}')
    logger.info(f'  Image shape: {train_imgs.shape[1:]}')

    # Build dataloaders
    train_tf = build_transforms(args.size, is_train=True)
    eval_tf = build_transforms(args.size, is_train=False)

    train_ds = MedMNISTDataset(train_imgs, train_lbls, train_tf)
    val_ds = MedMNISTDataset(val_imgs, val_lbls, eval_tf)
    test_ds = MedMNISTDataset(test_imgs, test_lbls, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Build model
    logger.info('Building HamCls...')
    for k in ['device', 'num_classes', 'embed_dim', 'depths', 'size',
              'batch_size', 'lr', 'weight_decay', 'drop_rate', 'head_drop']:
        logger.info(f'  {k:15s}: {getattr(args, k)}')

    model = HamCls(args).to(args.device)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f'  Parameters: {total:,} ({total * 4 / 1024 ** 2:.1f} MB)')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Compute class weights for imbalanced datasets
    class_weights = None
    if args.balanced:
        train_labels_flat = train_lbls.flatten()
        counts = np.bincount(train_labels_flat, minlength=args.num_classes).astype(float)
        counts = np.maximum(counts, 1.0)
        weights = len(train_labels_flat) / (args.num_classes * counts)
        class_weights = (weights / weights.sum() * args.num_classes).tolist()
        logger.info(f'  Class counts: {counts.astype(int).tolist()}')
        logger.info(f'  Class weights: {[f"{w:.3f}" for w in class_weights]}')

    # Train
    trainer = Trainer(args, model, train_loader, val_loader, test_loader, logger,
                      class_weights=class_weights)
    if not args.test_only:
        trainer.train()
    trainer.test()

    if trainer.history['train_loss']:
        plot_curves(trainer.history, args.output_dir)
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(trainer.history, f, indent=2)
    logger.info('Done!')


if __name__ == '__main__':
    main()
