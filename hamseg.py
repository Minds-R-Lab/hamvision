#!/usr/bin/env python3
"""
HamSeg v3: Hamiltonian Segmentation Network
=============================================

Strategy: Proven CNN backbone + Hamiltonian innovations where they matter.

Architecture:
  - Encoder stages 1-4: ConvNeXt-style blocks (DW Conv 7x7 + LN + PW + GELU)
  - Bottleneck: HamiltonianBottleneck (ConvNeXt + SS2D gated fusion + dropout)
  - Skip connections: Energy-gated + momentum injection at all levels
  - Decoder stage 3: Phase-Space Attention (centered energy + momentum)
  - Decoder: ConvNeXt blocks

v3 improvements over v2:
  1. Phase-Space Attention uses centered energy (fixes saturation at 1.0)
  2. Momentum flows to ALL 3 decoder levels (was d3 only)
  3. Dropout2d in bottleneck (reduces overfitting)
  4. Learned channel attention for energy map (improves boundary detection)
  5. Per-dataset output folders, periodic testing, comprehensive reports

Usage:
    python hamseg.py --dataset isic2018 --data_root ./data/ISIC2018

Author: Mohamed Mabrok
"""

import os, sys, math, time, json, random, warnings, argparse, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
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
    p = argparse.ArgumentParser(description='HamSeg v3 Training')
    p.add_argument('--dataset', type=str, default='isic2018')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--num_classes', type=int, default=1)
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--val_ratio', type=float, default=0.0)
    p.add_argument('--embed_dim', type=int, default=48)
    p.add_argument('--depths', type=int, nargs='+', default=[2, 2, 2, 2])
    p.add_argument('--damping_clamp', type=float, default=5.0)
    p.add_argument('--drop_rate', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--min_lr', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=80)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--output_dir', type=str, default='./outputs_hamseg')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--test_only', action='store_true')
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--test_every', type=int, default=50,
                   help='Run test evaluation every N epochs (0=disabled)')
    a = p.parse_args()
    if a.no_amp: a.use_amp = False
    a.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Auto dataset subfolder: outputs_hamseg/isic2018/, outputs_hamseg/tn3k/, etc.
    a.output_dir = os.path.join(a.output_dir, a.dataset)
    return a


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def setup_logging(d):
    os.makedirs(d, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(os.path.join(d, 'training.log')),
                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)


# ============================================================
# 2. DATASET
# ============================================================
class JointTransform:
    def __init__(self, img_size=224, is_train=True, num_classes=1):
        self.img_size = img_size
        self.is_train = is_train
        self.num_classes = num_classes

    def __call__(self, image, mask):
        image = TF.resize(image, [self.img_size, self.img_size],
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST)
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image); mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image); mask = TF.vflip(mask)
            if random.random() > 0.5:
                a = random.uniform(-30, 30)
                image = TF.rotate(image, a, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, a, interpolation=TF.InterpolationMode.NEAREST)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask)).float()
        if mask.ndim == 3: mask = mask[:, :, 0]  # (H, W, C) → (H, W)
        if self.num_classes > 1:
            # Multi-class: keep integer labels as (H, W) long tensor
            mask = mask.long()
        else:
            # Binary: threshold and add channel dim
            if mask.ndim == 2: mask = mask.unsqueeze(0)
            mask = (mask > 0.5).float()
        return image, mask


class MedicalSegDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=224,
                 train_ratio=0.7, val_ratio=0.0, num_classes=1):
        self.data_root = Path(data_root)
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_classes = num_classes
        self.transform = JointTransform(img_size, is_train=(split == 'train'),
                                        num_classes=num_classes)
        self.npz_mode = False  # may be set True by _find_paths
        self.image_paths, self.mask_paths = self._find_paths()
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {data_root} for '{split}'.")

    def _find_paths(self):
        root = self.data_root
        # Check for .npz format (ACDC style: train/*.npz, test/*.npz)
        has_npz = any((root / s).exists() and list((root / s).glob('*.npz'))
                      for s in ['train', 'test', 'val'])
        if has_npz:
            self.npz_mode = True
            if self.val_ratio > 0:
                # Use the current split directory directly
                npz_dir = root / self.split
                if npz_dir.exists():
                    files = sorted([str(f) for f in npz_dir.glob('*.npz')])
                    if files:
                        return self._split_npz_by_patient(files)
            else:
                # val_ratio=0: use preprocessor's patient-level split as-is
                # Train/val come from train/ dir ONLY (patient-level val split)
                # Test comes from test/ dir ONLY (no leakage)
                if self.split == 'test':
                    test_dir = root / 'test'
                    if test_dir.exists():
                        files = sorted([str(f) for f in test_dir.glob('*.npz')])
                        if files:
                            return files, []
                else:
                    # split is 'train' or 'val' — use ONLY train/ dir
                    train_dir = root / 'train'
                    if train_dir.exists():
                        files = sorted([str(f) for f in train_dir.glob('*.npz')])
                        if files:
                            return self._split_npz_by_patient(files)
        self.npz_mode = False
        # Fall back to images/masks directory structure
        img_dir = root / self.split / 'images'
        mask_dir = root / self.split / 'masks'
        if img_dir.exists() and mask_dir.exists():
            if self.val_ratio > 0:
                return self._pair(img_dir, mask_dir)
            else:
                all_i, all_m = [], []
                for s in ['train', 'val', 'test']:
                    si, sm = root / s / 'images', root / s / 'masks'
                    if si.exists() and sm.exists():
                        i, m = self._pair(si, sm)
                        all_i.extend(i); all_m.extend(m)
                if all_i: return self._split(all_i, all_m)
        for iname in ['images', 'image', 'imgs']:
            for mname in ['masks', 'mask', 'labels', 'gt', 'annotations']:
                id_, md_ = root / iname, root / mname
                if id_.exists() and md_.exists():
                    i, m = self._pair(id_, md_)
                    if i: return self._split(i, m)
        known_img = ['ISIC2018_Task1-2_Training_Input', 'ISIC-2017_Training_Data']
        known_msk = ['ISIC2018_Task1_Training_GroundTruth', 'ISIC-2017_Training_Part1_GroundTruth']
        for sd in [root] + [d for d in root.iterdir() if d.is_dir()]:
            for iname in known_img:
                for mname in known_msk:
                    id_, md_ = sd / iname, sd / mname
                    if id_.exists() and md_.exists():
                        i, m = self._pair(id_, md_)
                        if i: return self._split(i, m)
        return [], []

    def _pair(self, img_dir, mask_dir):
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        sfx = ['_segmentation', '_Segmentation', '_mask', '_seg', '_label', '_gt']
        ml = {}
        for p in mask_dir.iterdir():
            if p.suffix.lower() in exts:
                ml[p.stem] = p
                for s in sfx:
                    if p.stem.endswith(s): ml[p.stem[:-len(s)]] = p
        imgs, msks = [], []
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in exts: continue
            m = ml.get(p.stem)
            if not m:
                for s in sfx:
                    m = ml.get(p.stem + s)
                    if m: break
            if m: imgs.append(str(p)); msks.append(str(m))
        return imgs, msks

    def _split(self, imgs, msks):
        n = len(imgs)
        idx = list(range(n))
        random.Random(42).shuffle(idx)
        tr, vr = self.train_ratio, self.val_ratio
        if vr > 0:
            nt, nv = int(n * tr), int(n * vr)
            if self.split == 'train': sel = idx[:nt]
            elif self.split == 'val': sel = idx[nt:nt + nv]
            else: sel = idx[nt + nv:]
        else:
            ntv = int(n * tr)
            nv = max(1, int(ntv * 0.1))
            nta = ntv - nv
            if self.split == 'train': sel = idx[:nta]
            elif self.split == 'val': sel = idx[nta:ntv]
            else: sel = idx[ntv:]
        return [imgs[i] for i in sel], [msks[i] for i in sel]

    def _split_npz_by_patient(self, files):
        """Split npz files ensuring all slices from the same patient stay together.
        Patient ID extracted from filename: 'patient001_frame01_s005.npz' → 'patient001'
        """
        import re
        # Extract patient ID from each filename
        patient_to_files = {}
        for f in files:
            fname = Path(f).stem
            # Match 'patientXXX' pattern anywhere in filename
            m = re.search(r'(patient\d+)', fname)
            pid = m.group(1) if m else fname  # fallback: whole name
            if pid not in patient_to_files:
                patient_to_files[pid] = []
            patient_to_files[pid].append(f)

        # Shuffle and split at patient level
        patients = sorted(patient_to_files.keys())
        random.Random(42).shuffle(patients)
        n_patients = len(patients)

        tr, vr = self.train_ratio, self.val_ratio
        if vr > 0:
            nt = int(n_patients * tr)
            nv = int(n_patients * vr)
            if self.split == 'train': sel_patients = patients[:nt]
            elif self.split == 'val': sel_patients = patients[nt:nt + nv]
            else: sel_patients = patients[nt + nv:]
        else:
            # Carve 10% of patients for validation
            nv = max(1, int(n_patients * 0.1))
            nta = n_patients - nv
            if self.split == 'train': sel_patients = patients[:nta]
            elif self.split == 'val': sel_patients = patients[nta:]
            else: sel_patients = patients  # shouldn't happen for npz train/val

        # Collect all files for selected patients
        sel_files = []
        for pid in sel_patients:
            sel_files.extend(patient_to_files[pid])
        return sorted(sel_files), []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.npz_mode:
            data = np.load(self.image_paths[idx])
            img_np = data['image']  # float32 (H, W) in [0, 1]
            msk_np = data['mask']   # uint8 (H, W) class labels
            # Convert grayscale to RGB PIL for transforms
            img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_uint8).convert('RGB')
            msk = Image.fromarray(msk_np)
        else:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            msk = Image.open(self.mask_paths[idx]).convert('L')
        return self.transform(img, msk)


# ============================================================
# 3. MODEL COMPONENTS
# ============================================================

# --- 3.1 ConvNeXt Block (proven, standard) ---
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



# --- 3.2 Hamiltonian Scan Line (NOVEL) ---
class HamiltonianScanLine(nn.Module):
    """Damped harmonic oscillator as SSM via PARALLEL SCAN.
    
    Key insight: scan individual rows/columns (length 28) instead of the
    flattened 2D map (length 784). This means:
    - cumsum over 28 steps: max |value| ≈ 28 → exp(28) ≈ 1e12 (fine in float32)
    - No overflow, no clamps, no sequential loop
    - Each row/column is an independent oscillator (physically correct)
    - All rows/columns batched together (GPU parallel)
    """
    def __init__(self, d_model, damping_clamp=5.0):
        super().__init__()
        self.damping_clamp = damping_clamp
        self.log_k = nn.Parameter(torch.linspace(-1, 3, d_model))
        self.nu_scale = nn.Parameter(torch.ones(d_model))
        self.nu_bias = nn.Parameter(torch.ones(d_model) * 1.0)   # stronger init damping
        self.dt_scale = nn.Parameter(torch.ones(d_model) * 0.3)  # smaller init timestep
        self.dt_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (B*num_lines, line_length, D) where line_length = H or W (typically 28)
        B, L, D = x.shape
        x_f = x.float()
        omega = torch.exp(self.log_k.float() / 2.0)

        nu = torch.clamp(F.softplus(x_f * self.nu_scale + self.nu_bias) + 1e-6,
                         max=self.damping_clamp)
        dt = F.softplus(x_f * self.dt_scale + self.dt_bias) + 1e-6

        # Transition coefficients
        log_decay = -nu * dt                                    # (B, L, D)
        angle = omega.unsqueeze(0).unsqueeze(0) * dt            # (B, L, D)

        # Parallel scan: cumulative log-coefficients
        # With L=28: cumsum can reach ~14 → exp(14)=1.2M → large intermediates
        # Clamp to [-5, 0] keeps exp in [1, 148] — very stable intermediates
        L_re = torch.cumsum(log_decay, dim=1).clamp(-5, 0)      # always ≤ 0
        L_im = torch.cumsum(angle, dim=1)                        # phase, no clamp needed

        # Rescale input by inverse cumulative decay, accumulate, then unscale
        scale = torch.exp(-L_re)                                  # ∈ [1, 1100]
        cos_neg = torch.cos(-L_im)
        sin_neg = torch.sin(-L_im)

        # Rotate input into the "unscaled" frame
        rot_re = x_f * scale * cos_neg
        rot_im = x_f * scale * sin_neg

        # Cumulative sum in the rescaled frame
        acc_re = torch.cumsum(rot_re, dim=1)
        acc_im = torch.cumsum(rot_im, dim=1)

        # Unscale back: multiply by exp(L_re) * (cos(L_im) + i*sin(L_im))
        unscale = torch.exp(L_re)
        cos_L = torch.cos(L_im)
        sin_L = torch.sin(L_im)

        q = unscale * (cos_L * acc_re - sin_L * acc_im)
        p = unscale * (sin_L * acc_re + cos_L * acc_im)

        # Mild safety clamp — with L=28 and clamped cumsum, values should
        # naturally stay in [-50, 50] but clamp prevents rare outliers
        q = q.clamp(-50, 50)
        p = p.clamp(-50, 50)
        energy = 0.5 * (q * q + p * p)

        return q.to(x.dtype), p.to(x.dtype), energy.to(x.dtype)


# --- 3.3 Hamiltonian SS2D (NOVEL) ---
class HamiltonianSS2D(nn.Module):
    """4-direction Hamiltonian scan on 2D feature maps.
    
    Scans ROWS and COLUMNS independently (not the flattened sequence):
    - d=0: each row left→right  (B*H sequences of length W)
    - d=1: each row right→left
    - d=2: each col top→bottom  (B*W sequences of length H)
    - d=3: each col bottom→top
    
    This is faster (L=28 not 784) and more correct (each line is independent).
    """
    def __init__(self, d_model, damping_clamp=5.0):
        super().__init__()
        self.scans = nn.ModuleList([
            HamiltonianScanLine(d_model, damping_clamp) for _ in range(4)
        ])
        self.pos_merge = nn.Linear(d_model * 4, d_model)
        self.mom_merge = nn.Linear(d_model * 4, d_model)

    def _to_lines(self, x, d):
        """Convert 2D feature map to batched 1D lines for scanning."""
        B, C, H, W = x.shape
        if d == 0:    # rows left→right: (B, C, H, W) → (B*H, W, C)
            return x.permute(0, 2, 1, 3).reshape(B * H, C, W).permute(0, 2, 1), H, W
        elif d == 1:  # rows right→left
            return x.permute(0, 2, 1, 3).reshape(B * H, C, W).flip(1).permute(0, 2, 1), H, W
        elif d == 2:  # cols top→bottom: (B, C, H, W) → (B*W, H, C)
            return x.permute(0, 3, 1, 2).reshape(B * W, C, H).permute(0, 2, 1), H, W
        else:         # cols bottom→top
            return x.permute(0, 3, 1, 2).reshape(B * W, C, H).flip(1).permute(0, 2, 1), H, W

    def _to_2d(self, s, B, H, W, d):
        """Convert batched 1D scan output back to 2D feature map."""
        # s: (B*num_lines, line_length, C)
        C = s.shape[2]
        if d == 0:    # (B*H, W, C) → (B, C, H, W)
            return s.permute(0, 2, 1).reshape(B, H, C, W).permute(0, 2, 1, 3)
        elif d == 1:  # flip back then same as d=0
            return s.flip(1).permute(0, 2, 1).reshape(B, H, C, W).permute(0, 2, 1, 3)
        elif d == 2:  # (B*W, H, C) → reshape to (B, W, C, H) → permute to (B, C, H, W)
            return s.permute(0, 2, 1).reshape(B, W, C, H).permute(0, 2, 3, 1)
        else:         # flip back then same as d=2
            return s.flip(1).permute(0, 2, 1).reshape(B, W, C, H).permute(0, 2, 3, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        pos_l, mom_l, eng_l = [], [], []
        for d in range(4):
            lines, h, w = self._to_lines(x, d)
            q, p, e = self.scans[d](lines)
            pos_l.append(self._to_2d(q, B, h, w, d))
            mom_l.append(self._to_2d(p, B, h, w, d))
            eng_l.append(self._to_2d(e, B, h, w, d))
        pos = self.pos_merge(torch.cat(pos_l, 1).permute(0,2,3,1)).permute(0,3,1,2)
        mom = self.mom_merge(torch.cat(mom_l, 1).permute(0,2,3,1)).permute(0,3,1,2)
        energy = torch.stack(eng_l, 0).mean(0)
        return pos, mom, energy


# --- 3.4 Hamiltonian Bottleneck (NOVEL: ConvNeXt + SS2D gated fusion) ---
class HamiltonianBottleneck(nn.Module):
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
        # FIX 3: Dropout after fusion — reduces overfitting gap
        self.drop = nn.Dropout2d(drop_rate)
        # FIX 4: Learned energy channel attention — weight which channels
        # contribute most to energy map (currently weak B/I ratio 1.03-1.11)
        self.energy_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (B, C, 1, 1)
            nn.Flatten(),                   # (B, C)
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_block(x)
        x_n = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        with torch.cuda.amp.autocast(enabled=False):
            pos, mom, energy_raw = self.ss2d(x_n.float())
            ham_out = self.pos_proj(pos)
            g = self.gate(torch.cat([conv_out.float(), ham_out], 1))
            out = conv_out.float() * g + ham_out * (1 - g)
        out = self.drop(out.to(x.dtype))  # FIX 3: dropout
        mom = mom.to(x.dtype)
        # FIX 4: Learned channel attention for energy — weight channels by
        # boundary-relevance instead of plain mean (improves B/I ratio)
        energy_f = energy_raw.to(x.dtype)
        ch_weights = self.energy_attn(energy_f)  # (B, C)
        ch_weights = ch_weights.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        energy_map = (energy_f * ch_weights).mean(dim=1, keepdim=True)  # weighted channel mean
        return out, mom, energy_map


# --- 3.5 Phase-Space Attention (NOVEL) ---
class PhaseSpaceAttention(nn.Module):
    """Combines energy spatial attention with momentum boundary features.
    
    FIX: energy_proj(energy) saturated at 1.0 on both ISIC2018 and TN3K.
    Now uses centered energy (like skip gate, which works: range [0.11, 0.97]).
    Also adds momentum to feature modulation, not just concatenation.
    """
    def __init__(self, dim):
        super().__init__()
        # Energy attention via centering (same principle as working skip gates)
        self.energy_gamma = nn.Parameter(torch.ones(1))
        # Momentum projection
        self.momentum_proj = nn.Conv2d(dim, dim, 1, bias=False)
        # Fuse: energy-attended features + momentum features
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim), nn.GELU())

    def forward(self, features, momentum, energy):
        # Centered energy attention (proven to work in skip gates)
        e_centered = energy - energy.mean(dim=(2, 3), keepdim=True)
        energy_attn = torch.sigmoid(self.energy_gamma * e_centered)  # varies around 0.5
        mom_feat = self.momentum_proj(momentum)
        attended = features * energy_attn
        return self.fuse(torch.cat([attended, mom_feat], dim=1))


# --- 3.6 Energy-Gated Skip Connection (NOVEL) ---
class EnergyGatedSkip(nn.Module):
    """Energy-gated skip with optional momentum injection.
    
    FIX 2: Momentum is the strongest physics signal (B/E ratio 1.17)
    but previously only went to d3. Now optionally injected at all levels.
    """
    def __init__(self, enc_ch, dec_ch, use_momentum=False):
        super().__init__()
        in_ch = enc_ch + dec_ch + (dec_ch if use_momentum else 0)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, dec_ch, 1, bias=False),
            nn.BatchNorm2d(dec_ch), nn.GELU())
        self.energy_gamma = nn.Parameter(torch.ones(1))
        self.use_momentum = use_momentum
        if use_momentum:
            self.mom_proj = nn.Conv2d(dec_ch, dec_ch, 1, bias=False)

    def forward(self, dec_feat, enc_feat, energy=None, momentum=None):
        if energy is not None:
            e_centered = energy - energy.mean(dim=(2, 3), keepdim=True)
            gate = torch.sigmoid(self.energy_gamma * e_centered)
            enc_feat = enc_feat * gate
        parts = [dec_feat, enc_feat]
        if self.use_momentum and momentum is not None:
            mom_proj = self.mom_proj(momentum)
            parts.append(mom_proj)
        return self.reduce(torch.cat(parts, dim=1))


# --- 3.7 Down/Up ---
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim*2, 2, stride=2, bias=False), nn.BatchNorm2d(dim*2))
    def forward(self, x): return self.reduction(x)


class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, 2, stride=2, bias=False), nn.BatchNorm2d(dim//2))
    def forward(self, x): return self.expand(x)


# ============================================================
# 4. HAMSEG v2
# ============================================================
class HamSeg(nn.Module):
    def __init__(self, args):
        super().__init__()
        C = args.embed_dim
        depths = args.depths
        dc = args.damping_clamp

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.GELU())

        self.enc1 = nn.Sequential(*[ConvNeXtBlock(C) for _ in range(depths[0])])
        self.down1 = PatchMerging(C)
        self.enc2 = nn.Sequential(*[ConvNeXtBlock(C*2) for _ in range(depths[1])])
        self.down2 = PatchMerging(C*2)
        self.enc3 = nn.Sequential(*[ConvNeXtBlock(C*4) for _ in range(depths[2])])
        self.down3 = PatchMerging(C*4)

        # Bottleneck: Hamiltonian SS2D here (28x28 = 784 tokens)
        drop_rate = getattr(args, 'drop_rate', 0.1)
        self.bottleneck = nn.ModuleList([
            HamiltonianBottleneck(C*8, dc, drop_rate) for _ in range(depths[3])])

        self.up3 = PatchExpanding(C*8)
        self.skip3 = EnergyGatedSkip(C*4, C*4, use_momentum=True)  # FIX 2: momentum at d3
        self.ps_attn = PhaseSpaceAttention(C*4)
        self.dec3 = nn.Sequential(*[ConvNeXtBlock(C*4) for _ in range(depths[2])])

        self.up2 = PatchExpanding(C*4)
        self.skip2 = EnergyGatedSkip(C*2, C*2, use_momentum=True)  # FIX 2: momentum at d2
        self.dec2 = nn.Sequential(*[ConvNeXtBlock(C*2) for _ in range(depths[1])])

        self.up1 = PatchExpanding(C*2)
        self.skip1 = EnergyGatedSkip(C, C, use_momentum=True)      # FIX 2: momentum at d1
        self.dec1 = nn.Sequential(*[ConvNeXtBlock(C) for _ in range(depths[0])])

        self.seg_head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C, args.num_classes, 1))

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
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.down3(e3)

        momentum, energy = None, None
        for blk in self.bottleneck:
            e4, momentum, energy = blk(e4)

        d3 = self.up3(e4)
        en3 = F.interpolate(energy, d3.shape[2:], mode='bilinear', align_corners=False)
        mom3 = F.interpolate(momentum, d3.shape[2:], mode='bilinear', align_corners=False)
        mom3 = mom3[:, :d3.shape[1]]
        d3 = self.skip3(d3, e3, en3, mom3)       # energy + momentum
        d3 = self.ps_attn(d3, mom3, en3)           # PS attention
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        en2 = F.interpolate(energy, d2.shape[2:], mode='bilinear', align_corners=False)
        mom2 = F.interpolate(momentum, d2.shape[2:], mode='bilinear', align_corners=False)
        mom2 = mom2[:, :d2.shape[1]]
        d2 = self.skip2(d2, e2, en2, mom2)         # FIX 2: momentum at d2
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        en1 = F.interpolate(energy, d1.shape[2:], mode='bilinear', align_corners=False)
        mom1 = F.interpolate(momentum, d1.shape[2:], mode='bilinear', align_corners=False)
        mom1 = mom1[:, :d1.shape[1]]
        d1 = self.skip1(d1, e1, en1, mom1)         # FIX 2: momentum at d1
        d1 = self.dec1(d1)

        return self.seg_head(d1)


# ============================================================
# 5. LOSS & METRICS
# ============================================================
class DiceBCELoss(nn.Module):
    """Handles both binary (num_classes=1) and multi-class segmentation."""
    def __init__(self, smooth=1.0, num_classes=1):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        if num_classes > 1:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if self.num_classes > 1:
            # Multi-class: logits (B, C, H, W), targets (B, H, W) long
            ce = self.ce(logits, targets)
            # Per-class Dice
            probs = torch.softmax(logits, dim=1)
            dice_sum = 0
            for c in range(1, self.num_classes):  # skip background
                pc = probs[:, c].reshape(-1)
                tc = (targets == c).float().reshape(-1)
                inter = (pc * tc).sum()
                dice_sum += (2*inter + self.smooth) / (pc.sum() + tc.sum() + self.smooth)
            dice = dice_sum / max(self.num_classes - 1, 1)
            return ce + (1 - dice)
        else:
            # Binary
            bce = self.bce(logits, targets)
            probs = torch.sigmoid(logits)
            pf, tf = probs.reshape(-1), targets.reshape(-1)
            inter = (pf * tf).sum()
            dice = (2*inter + self.smooth) / (pf.sum() + tf.sum() + self.smooth)
            return bce + (1 - dice)


def compute_metrics(logits, targets, threshold=0.5, num_classes=1):
    eps = 1e-7
    if num_classes > 1:
        # Multi-class: average Dice/IoU across foreground classes
        preds = logits.argmax(dim=1)  # (B, H, W)
        dice_sum, iou_sum, n_cls = 0, 0, 0
        tp_all, fp_all, fn_all, tn_all = 0, 0, 0, 0
        for c in range(1, num_classes):
            pc = (preds == c).float().reshape(-1)
            tc = (targets == c).float().reshape(-1)
            tp = (pc * tc).sum().item()
            fp = (pc * (1 - tc)).sum().item()
            fn = ((1 - pc) * tc).sum().item()
            tn = ((1 - pc) * (1 - tc)).sum().item()
            dice_sum += (2*tp+eps)/(2*tp+fp+fn+eps)
            iou_sum += (tp+eps)/(tp+fp+fn+eps)
            tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn
            n_cls += 1
        n_cls = max(n_cls, 1)
        total = tp_all + tn_all + fp_all + fn_all + eps
        return {
            'dice': dice_sum / n_cls,
            'miou': iou_sum / n_cls,
            'precision': (tp_all+eps)/(tp_all+fp_all+eps),
            'specificity': (tn_all+eps)/(tn_all+fp_all+eps),
            'accuracy': (tp_all+tn_all+eps)/total}
    else:
        # Binary
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        p, t = preds.reshape(-1), targets.reshape(-1)
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()
        return {
            'dice': (2*tp+eps)/(2*tp+fp+fn+eps),
            'miou': (tp+eps)/(tp+fp+fn+eps),
            'precision': (tp+eps)/(tp+fp+eps),
            'specificity': (tn+eps)/(tn+fp+eps),
            'accuracy': (tp+tn+eps)/(tp+tn+fp+fn+eps)}


# ============================================================
# 6. TRAINING
# ============================================================
class Trainer:
    def __init__(self, args, model, train_loader, val_loader, test_loader, logger):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger
        self.criterion = DiceBCELoss(num_classes=args.num_classes)
        self.num_classes = args.num_classes
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
        self.scaler = GradScaler(enabled=args.use_amp)
        self.history = {k: [] for k in [
            'train_loss','val_loss','train_dice','val_dice',
            'train_miou','val_miou','train_precision','val_precision',
            'train_specificity','val_specificity','train_accuracy','val_accuracy','lr']}
        self.best_val_dice = 0
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
        loss_sum, n = 0, 0
        met = {k: 0 for k in ['dice','miou','precision','specificity','accuracy']}
        total = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f'Train {epoch+1}/{self.args.epochs}',
                    leave=False, dynamic_ncols=True)
        for bi, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.args.device, non_blocking=True)
            masks = masks.to(self.args.device, non_blocking=True)
            self._warmup_lr(epoch, bi, total)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.args.use_amp):
                logits = self.model(imgs)
                logits = torch.clamp(logits, -20, 20)
                loss = self.criterion(logits, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad(set_to_none=True); continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum += loss.item(); n += 1
            m = compute_metrics(logits.detach().float(), masks, num_classes=self.num_classes)
            for k in met: met[k] += m[k]
            if bi % 10 == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}', dice=f'{m["dice"]:.4f}')
        if n == 0: return 0, {k: 0 for k in met}
        return loss_sum/n, {k: v/n for k, v in met.items()}

    @torch.no_grad()
    def evaluate(self, loader, desc='Val'):
        self.model.eval()
        loss_sum, n = 0, 0
        met = {k: 0 for k in ['dice','miou','precision','specificity','accuracy']}
        for imgs, masks in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            imgs = imgs.to(self.args.device, non_blocking=True)
            masks = masks.to(self.args.device, non_blocking=True)
            # NO AMP in eval — float16 overflow causes intermittent NaN
            logits = self.model(imgs)
            logits = torch.clamp(logits.float(), -20, 20)
            loss = self.criterion(logits, masks)
            lv = loss.item()
            if not (math.isnan(lv) or math.isinf(lv)):
                loss_sum += lv
            n += 1
            m = compute_metrics(logits.float(), masks, num_classes=self.num_classes)
            for k in met: met[k] += m[k]
        if n == 0: return 0, {k: 0 for k in met}
        return loss_sum/n, {k: v/n for k, v in met.items()}

    def save_ckpt(self, epoch, is_best=False):
        torch.save({
            'epoch': epoch, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
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
        self.best_val_dice = ckpt['best_val_dice']
        self.best_epoch = ckpt['best_epoch']
        self.history = ckpt['history']
        self.logger.info(f'Resumed from epoch {self.start_epoch}')

    def train(self):
        if self.args.resume: self.load_ckpt()
        self.logger.info(f'Training {self.args.epochs} epochs from {self.start_epoch}')
        nan_strikes = 0
        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()
            tr_loss, tr_met = self.train_epoch(epoch)

            # NaN in training
            if tr_loss == 0 and tr_met['dice'] == 0:
                nan_strikes += 1
                self.logger.info(f'Train NaN (strike {nan_strikes}/3). Recovering...')
                best_path = os.path.join(self.args.output_dir, 'best_model.pth')
                if os.path.exists(best_path):
                    self.model.load_state_dict(
                        torch.load(best_path, map_location=self.args.device, weights_only=True))
                    self.logger.info('  Reloaded best model checkpoint')
                for pg in self.optimizer.param_groups:
                    pg['lr'] = pg['lr'] * 0.5
                self.logger.info(f'  Halved LR to {self.optimizer.param_groups[0]["lr"]:.6f}')
                if nan_strikes >= 3:
                    self.logger.info('3 NaN strikes. Stopping.'); break
                continue

            if epoch >= self.args.warmup_epochs:
                self.scheduler.step()
            vl_loss, vl_met = self.evaluate(self.val_loader)

            # NaN in validation
            if vl_met['dice'] < 0.01 and vl_loss == 0:
                nan_strikes += 1
                self.logger.info(f'Eval NaN (strike {nan_strikes}/3). Recovering...')
                best_path = os.path.join(self.args.output_dir, 'best_model.pth')
                if os.path.exists(best_path):
                    self.model.load_state_dict(
                        torch.load(best_path, map_location=self.args.device, weights_only=True))
                    self.logger.info('  Reloaded best model checkpoint')
                for pg in self.optimizer.param_groups:
                    pg['lr'] = pg['lr'] * 0.5
                self.logger.info(f'  Halved LR to {self.optimizer.param_groups[0]["lr"]:.6f}')
                if nan_strikes >= 3:
                    self.logger.info('3 NaN strikes. Stopping.'); break
                continue

            nan_strikes = 0
            lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['lr'].append(lr)
            for k in ['dice','miou','precision','specificity','accuracy']:
                self.history[f'train_{k}'].append(tr_met[k])
                self.history[f'val_{k}'].append(vl_met[k])
            is_best = vl_met['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = vl_met['dice']
                self.best_epoch = epoch + 1
                self.no_improve = 0
            else:
                self.no_improve += 1
            if (epoch+1) % self.args.save_every == 0 or is_best:
                self.save_ckpt(epoch, is_best)
            self.logger.info(
                f'Ep {epoch+1:3d}/{self.args.epochs} | '
                f'L:{tr_loss:.4f}/{vl_loss:.4f} | '
                f'D:{tr_met["dice"]:.4f}/{vl_met["dice"]:.4f} | '
                f'IoU:{tr_met["miou"]:.4f}/{vl_met["miou"]:.4f} | '
                f'LR:{lr:.6f} | {time.time()-t0:.0f}s'
                f'{" BEST" if is_best else ""}')

            # Periodic testing every N epochs
            if self.args.test_every > 0 and (epoch+1) % self.args.test_every == 0:
                self.logger.info(f'--- Periodic test at epoch {epoch+1} ---')
                test_met = self._run_test(tag=f'epoch{epoch+1:03d}', load_best=False)
                self.logger.info(f'  Test Dice: {test_met["dice"]:.4f}  '
                                f'mIoU: {test_met["miou"]:.4f}  '
                                f'Acc: {test_met["accuracy"]:.4f}')

            if self.args.patience > 0 and self.no_improve >= self.args.patience:
                self.logger.info(f'Early stopping at epoch {epoch+1}'); break
        if self.start_epoch < self.args.epochs:
            self.save_ckpt(epoch)

    def _run_test(self, tag='final', load_best=True):
        """Run test evaluation and save results with a tag."""
        if load_best:
            best = os.path.join(self.args.output_dir, 'best_model.pth')
            if os.path.exists(best):
                self.model.load_state_dict(
                    torch.load(best, map_location=self.args.device, weights_only=True))
        loss, met = self.evaluate(self.test_loader, f'Test-{tag}')
        results = {
            'tag': tag,
            'dataset': self.args.dataset,
            'model': 'HamSeg_v3',
            'test': {k: round(v, 4) for k, v in met.items()},
            'test_loss': round(loss, 4),
            'best_val_dice': round(self.best_val_dice, 4),
            'best_epoch': self.best_epoch,
            'params': sum(p.numel() for p in self.model.parameters()),
            'args': {k: str(v) for k, v in vars(self.args).items()}
        }
        # Save tagged result
        fname = f'test_results_{tag}.json'
        with open(os.path.join(self.args.output_dir, fname), 'w') as f:
            json.dump(results, f, indent=2)
        return met

    def test(self):
        met = self._run_test(tag='final', load_best=True)
        self.logger.info('='*50)
        self.logger.info('  HAMSEG v3 TEST RESULTS')
        self.logger.info('='*50)
        for k, v in met.items():
            self.logger.info(f'  {k:15s}: {v:.4f}')
        # Generate comprehensive report
        self._save_report(met)
        return met

    def _save_report(self, test_met):
        """Save a comprehensive results report."""
        report_lines = []
        report_lines.append('=' * 60)
        report_lines.append(f'  HamSeg v3 — Results Report')
        report_lines.append(f'  Dataset: {self.args.dataset}')
        report_lines.append(f'  Date: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        report_lines.append('=' * 60)
        report_lines.append('')
        report_lines.append('--- Configuration ---')
        for k in ['dataset','data_root','img_size','num_classes','embed_dim',
                   'depths','epochs','batch_size','lr','weight_decay',
                   'warmup_epochs','patience','drop_rate','damping_clamp']:
            report_lines.append(f'  {k:20s}: {getattr(self.args, k)}')
        total_params = sum(p.numel() for p in self.model.parameters())
        report_lines.append(f'  {"parameters":20s}: {total_params:,} ({total_params*4/1024**2:.1f} MB)')
        report_lines.append('')
        report_lines.append('--- Test Results ---')
        for k, v in test_met.items():
            report_lines.append(f'  {k:20s}: {v:.4f}')
        report_lines.append('')
        report_lines.append('--- Training Summary ---')
        report_lines.append(f'  Best val dice     : {self.best_val_dice:.4f}')
        report_lines.append(f'  Best epoch        : {self.best_epoch}')
        if self.history['train_loss']:
            report_lines.append(f'  Final train loss  : {self.history["train_loss"][-1]:.4f}')
            report_lines.append(f'  Final val loss    : {self.history["val_loss"][-1]:.4f}')
            report_lines.append(f'  Total epochs run  : {len(self.history["train_loss"])}')
        report_lines.append('')
        # Check for periodic test results
        report_lines.append('--- Periodic Test Results ---')
        for f in sorted(Path(self.args.output_dir).glob('test_results_epoch*.json')):
            with open(f) as fh:
                r = json.load(fh)
            tag = r.get('tag', '?')
            t = r.get('test', {})
            report_lines.append(
                f'  {tag:12s}: Dice={t.get("dice",0):.4f}  '
                f'mIoU={t.get("miou",0):.4f}  '
                f'Acc={t.get("accuracy",0):.4f}')
        report_lines.append('')
        report_lines.append('=' * 60)
        report_text = '\n'.join(report_lines)
        # Save report
        report_path = os.path.join(self.args.output_dir, 'report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        self.logger.info(f'Report saved to {report_path}')


# ============================================================
# 7. VISUALIZATION
# ============================================================
def plot_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('HamSeg v3 Training', fontsize=16, fontweight='bold')
    for i, (tk, vk, title) in enumerate([
        ('train_loss','val_loss','Loss'), ('train_dice','val_dice','Dice'),
        ('train_miou','val_miou','mIoU'), ('train_precision','val_precision','Precision'),
        ('train_specificity','val_specificity','Specificity'),
        ('train_accuracy','val_accuracy','Accuracy')]):
        ax = axes[i//3, i%3]
        ax.plot(epochs, history[tk], 'b-', label='Train', lw=2)
        ax.plot(epochs, history[vk], 'r-', label='Val', lw=2)
        ax.set_title(title, fontweight='bold'); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_samples(model, loader, device, save_dir, n=8, use_amp=True, num_classes=1):
    model.eval()
    imgs_all, masks_all, preds_all = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            if num_classes > 1:
                preds = logits.float().argmax(dim=1).cpu()  # (B, H, W)
            else:
                preds = (torch.sigmoid(logits.float()) > 0.5).float().cpu()[:, 0]  # (B, H, W)
            # Ensure masks are 2D per sample
            if targets.ndim == 4:
                targets = targets[:, 0]  # (B, 1, H, W) → (B, H, W)
            imgs_all.append(images.cpu()); masks_all.append(targets)
            preds_all.append(preds)
            if sum(i.shape[0] for i in imgs_all) >= n: break
    imgs_all = torch.cat(imgs_all)[:n]
    masks_all = torch.cat(masks_all)[:n]
    preds_all = torch.cat(preds_all)[:n]
    mean = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    std = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    imgs_vis = torch.clamp(imgs_all * std + mean, 0, 1)
    nc = min(n, len(imgs_vis))
    fig, axes = plt.subplots(nc, 4, figsize=(16, 4*nc))
    if nc == 1: axes = axes.reshape(1, -1)
    # Colormap for multi-class
    cmap = 'nipy_spectral' if num_classes > 1 else 'gray'
    for i in range(nc):
        img = imgs_vis[i].permute(1,2,0).numpy()
        axes[i,0].imshow(img); axes[i,0].axis('off')
        if i==0: axes[i,0].set_title('Input')
        axes[i,1].imshow(masks_all[i].numpy(), cmap=cmap, vmin=0, vmax=max(num_classes-1,1))
        axes[i,1].axis('off')
        if i==0: axes[i,1].set_title('GT')
        axes[i,2].imshow(preds_all[i].numpy(), cmap=cmap, vmin=0, vmax=max(num_classes-1,1))
        axes[i,2].axis('off')
        if i==0: axes[i,2].set_title('Pred')
        axes[i,3].imshow(img)
        if num_classes > 1:
            for c in range(1, num_classes):
                colors = ['red', 'lime', 'cyan', 'yellow']
                axes[i,3].contour(preds_all[i].numpy() == c, levels=[.5],
                                  colors=colors[c % len(colors)], linewidths=1.5)
                axes[i,3].contour(masks_all[i].numpy() == c, levels=[.5],
                                  colors=colors[c % len(colors)], linewidths=1, linestyles='--')
        else:
            axes[i,3].contour(preds_all[i].numpy(), levels=[.5], colors='lime', linewidths=2)
            axes[i,3].contour(masks_all[i].numpy(), levels=[.5], colors='red', linewidths=1, linestyles='--')
        axes[i,3].axis('off')
        if i==0: axes[i,3].set_title('Overlay')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'segmentation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 8. MAIN
# ============================================================
def main():
    args = get_args()
    set_seed(args.seed)
    # output_dir is now auto: outputs_hamseg/{dataset}/
    logger = setup_logging(args.output_dir)
    logger.info('='*60)
    logger.info(f'  HamSeg v3: CNN backbone + Hamiltonian innovations')
    logger.info(f'  Dataset: {args.dataset} → {args.output_dir}')
    logger.info('='*60)
    for k in ['device','dataset','img_size','num_classes','embed_dim','depths',
              'epochs','batch_size','lr','weight_decay','drop_rate','use_amp']:
        logger.info(f'  {k:15s}: {getattr(args, k)}')

    logger.info('Loading data...')
    if not Path(args.data_root).exists():
        logger.error(f'{args.data_root} not found!'); sys.exit(1)
    train_ds = MedicalSegDataset(args.data_root, 'train', args.img_size,
                                 args.train_ratio, args.val_ratio, args.num_classes)
    val_ds = MedicalSegDataset(args.data_root, 'val', args.img_size,
                               args.train_ratio, args.val_ratio, args.num_classes)
    test_ds = MedicalSegDataset(args.data_root, 'test', args.img_size,
                                args.train_ratio, args.val_ratio, args.num_classes)
    logger.info(f'  Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    logger.info('Building HamSeg v3...')
    model = HamSeg(args).to(args.device)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f'  Parameters: {total:,} ({total*4/1024**2:.1f} MB)')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    trainer = Trainer(args, model, train_loader, val_loader, test_loader, logger)
    if not args.test_only:
        trainer.train()
    trainer.test()

    if trainer.history['train_loss']:
        plot_curves(trainer.history, args.output_dir)
    base = model.module if isinstance(model, nn.DataParallel) else model
    plot_samples(base, test_loader, args.device, args.output_dir,
                 use_amp=args.use_amp, num_classes=args.num_classes)
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(trainer.history, f, indent=2)
    logger.info('Done!')


if __name__ == '__main__':
    main()
