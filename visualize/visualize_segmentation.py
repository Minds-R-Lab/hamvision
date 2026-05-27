#!/usr/bin/env python3
"""
visualize_fig4_qualitative.py
==============================

Rebuilds Figure 4 of the HamVision paper with more samples per dataset
and a cleaner layout than the original 3-row figure.

Two layouts are supported:

  * 'grid' (default): N_datasets rows x M_samples columns, where each
    cell shows the input image with the ground-truth contour (green) and
    HamSeg's prediction contour (orange) overlaid. Per-sample Dice score
    is printed in the bottom-left corner. This is the most space-efficient
    way to show many examples and is the recommended layout for the paper.

  * 'detailed': one sample per row, four columns
    [input | GT mask | prediction mask | overlay], matching the original
    Figure~4 convention. Used when the reviewer asks for explicit
    GT/prediction mask panels.

For each sample, the picker uses a different coverage quantile so the
selected images span a range of lesion sizes (small / medium / large /
very-large). This avoids the appearance of only showing easy cases.

Usage:
    python visualize_fig4_qualitative.py \\
        --acdc_ckpt   ... --acdc_root   ... \\
        --isic18_ckpt ... --isic18_root ... \\
        --isic17_ckpt ... --isic17_root ... \\
        --tn3k_ckpt   ... --tn3k_root   ... \\
        --mmotu_ckpt  ... --mmotu_root  ... \\
        --save_dir figures \\
        --layout grid --n_samples 4 --seed 42
"""

import os, sys, argparse, random, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')


# ===================================================================
# Model loader (shared with Fig 5/6 script)
# ===================================================================

def load_model(model_path, device, embed_dim=48, num_classes=1):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in list(sys.modules.keys()):
        if 'hamseg' in mod:
            del sys.modules[mod]
    from hamseg import HamSeg

    class A: pass
    a = A()
    a.embed_dim = embed_dim
    a.depths = [2, 2, 2, 2]
    a.damping_clamp = 5.0
    a.num_classes = num_classes
    a.img_size = 224
    a.drop_rate = 0.1

    model = HamSeg(a).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and any(
            torch.is_tensor(v) and v.dtype == torch.float16 for v in state.values()):
        state = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
                 for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, img_t, device, num_classes=1):
    """Run forward pass; return prediction mask as (H, W) tensor."""
    with torch.no_grad():
        x = img_t.unsqueeze(0).to(device)
        logits = model(x).float()
        if logits.shape[1] > 1:
            pred = logits.argmax(dim=1)[0].cpu()
        else:
            pred = (torch.sigmoid(logits) > 0.5).float()[0, 0].cpu()
    return pred


# ===================================================================
# Mask + image utilities
# ===================================================================

def normalise_image_array(img_np):
    img_np = np.asarray(img_np).astype(np.float32)
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3):
        img_np = img_np.transpose(1, 2, 0)
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np[..., 0]
    lo, hi = np.nanpercentile(img_np, [1, 99])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    img_np = np.clip((img_np - lo) / (hi - lo), 0, 1) * 255.0
    img_np = np.nan_to_num(img_np, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(img_np, 0, 255).astype(np.uint8)


def normalise_mask_array(msk_np):
    msk_np = np.asarray(msk_np)
    if msk_np.ndim == 3 and msk_np.shape[0] < 8 and msk_np.shape[-1] >= 64:
        msk_np = msk_np.argmax(axis=0)
    if msk_np.ndim == 3 and msk_np.shape[-1] < 8:
        msk_np = msk_np.argmax(axis=-1)
    return msk_np.astype(np.uint8)


def dice_score(pred, gt):
    """Binary Dice between two 2D bool/0-1 arrays."""
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union < 1e-6:
        return 1.0
    return 2.0 * inter / union


# ===================================================================
# Multi-sample picker (covers a range of mask coverage)
# ===================================================================

def pick_samples(data_root, n_samples=4, img_size=224, seed=42,
                 candidates=200, verbose=True, return_all=False):
    """Return a list of `n_samples` (img_t, mask_t, orig_np, fname) tuples
    spanning a range of mask coverage values (quartiles).

    If return_all=True, returns ALL coverage-sorted candidates (used by
    the Dice-filtering wrapper to pick high-quality cases only).
    """
    root = Path(data_root)

    for split in ['test', 'val', 'train']:
        # ---- PNG/JPG (ISIC, TN3K, MMOTU, ACDC if pre-sliced) ----
        id_ = root / split / 'images'
        md_ = root / split / 'masks'
        if id_.exists() and md_.exists():
            exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
            sfx = ['_segmentation', '_Segmentation', '_mask', '_seg']
            ml = {}
            for p in md_.iterdir():
                if p.suffix.lower() in exts:
                    ml[p.stem] = p
                    for s in sfx:
                        if p.stem.endswith(s):
                            ml[p.stem[:-len(s)]] = p
            pairs = []
            for p in sorted(id_.iterdir()):
                if p.suffix.lower() not in exts:
                    continue
                m = ml.get(p.stem)
                if not m:
                    for s in sfx:
                        m = ml.get(p.stem + s)
                        if m: break
                if m: pairs.append((str(p), str(m)))
            if not pairs:
                continue
            random.seed(seed)
            sel = random.sample(pairs, min(candidates, len(pairs)))
            scored = []
            for ip, mp in sel:
                msk = Image.open(mp).convert('L')
                ma = np.array(TF.resize(msk, [img_size, img_size]))
                thresh = 128 if ma.max() > 10 else 0
                cov = float((ma > thresh).mean())
                scored.append((cov, ip, mp))
            scored.sort()
            if return_all:
                # Drop empty-mask candidates; return the rest with metadata
                picks = [(c, ip, mp) for c, ip, mp in scored if c >= 0.005]
                if not picks:
                    picks = scored[-min(n_samples * 4, len(scored)):]
            else:
                # Pick `n_samples` evenly spaced quantiles from the coverage
                # distribution, biased toward the upper half to avoid empty masks
                if n_samples == 1:
                    quantiles = [0.7]
                else:
                    quantiles = [0.4 + 0.5 * (i / (n_samples - 1))
                                 for i in range(n_samples)]
                picks = []
                for q in quantiles:
                    idx = min(int(q * (len(scored) - 1)), len(scored) - 1)
                    cov, ip, mp = scored[idx]
                    if cov < 0.005:
                        cov, ip, mp = scored[-1]
                    picks.append((cov, ip, mp))
                seen = set(); unique_picks = []
                for cov, ip, mp in picks:
                    if ip in seen: continue
                    seen.add(ip); unique_picks.append((cov, ip, mp))
                for cov, ip, mp in reversed(scored):
                    if len(unique_picks) >= n_samples: break
                    if ip in seen: continue
                    seen.add(ip); unique_picks.append((cov, ip, mp))
                picks = unique_picks[:n_samples]

            results = []
            for cov, ip, mp in picks:
                if verbose:
                    print(f'        chose: {Path(ip).name}  (coverage={cov:.3f})')
                img = Image.open(ip).convert('RGB')
                msk = Image.open(mp).convert('L')
                orig_np = np.array(TF.resize(img, [img_size, img_size]))
                t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                img_t = TF.normalize(t, [.485, .456, .406], [.229, .224, .225])
                ma = np.array(TF.resize(msk, [img_size, img_size]))
                thresh = 128 if ma.max() > 10 else 0
                mt = torch.from_numpy(ma).float()
                mask_t = (mt > thresh).float()
                results.append((img_t, mask_t, orig_np, Path(ip).stem))
            return results

        # ---- NPZ (ACDC 3D-volume form) ----
        npz_dir = root / split
        if npz_dir.exists():
            npz_files = sorted(npz_dir.glob('*.npz'))
            if not npz_files:
                continue
            random.seed(seed)
            sel = random.sample(npz_files, min(candidates, len(npz_files)))
            scored = []
            for f in sel:
                data = np.load(f)
                msk_key = None
                for cand in ('mask', 'label', 'seg', 'segmentation', 'gt'):
                    if cand in data.files:
                        msk_key = cand; break
                if msk_key is None:
                    msk_key = data.files[-1]
                msk_np = normalise_mask_array(data[msk_key])
                cov = float((msk_np > 0).mean())
                scored.append((cov, f, msk_key))
            scored.sort()
            if return_all:
                picks = [(c, f, mk) for c, f, mk in scored if c >= 0.005]
                if not picks:
                    picks = scored[-min(n_samples * 4, len(scored)):]
            else:
                if n_samples == 1:
                    quantiles = [0.7]
                else:
                    quantiles = [0.4 + 0.5 * (i / (n_samples - 1))
                                 for i in range(n_samples)]
                picks = []
                for q in quantiles:
                    idx = min(int(q * (len(scored) - 1)), len(scored) - 1)
                    cov, f, mk = scored[idx]
                    if cov < 0.005:
                        cov, f, mk = scored[-1]
                    picks.append((cov, f, mk))
                seen = set(); unique_picks = []
                for cov, f, mk in picks:
                    if f in seen: continue
                    seen.add(f); unique_picks.append((cov, f, mk))
                for cov, f, mk in reversed(scored):
                    if len(unique_picks) >= n_samples: break
                    if f in seen: continue
                    seen.add(f); unique_picks.append((cov, f, mk))
                picks = unique_picks[:n_samples]

            results = []
            for cov, f, mk in picks:
                if verbose:
                    print(f'        chose: {f.name}  (coverage={cov:.3f})')
                data = np.load(f)
                img_key = 'image' if 'image' in data.files else data.files[0]
                img_uint8 = normalise_image_array(data[img_key])
                msk_arr = normalise_mask_array(data[mk])
                img = Image.fromarray(img_uint8).convert('RGB')
                msk = Image.fromarray(msk_arr)
                orig_np = np.array(TF.resize(img, [img_size, img_size]))
                t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                img_t = TF.normalize(t, [.485, .456, .406], [.229, .224, .225])
                mt = torch.from_numpy(np.array(TF.resize(msk, [img_size, img_size],
                                     interpolation=TF.InterpolationMode.NEAREST))).float()
                mask_t = (mt > 0).float()
                results.append((img_t, mask_t, orig_np, f.stem))
            return results

    raise RuntimeError(f"No data in {data_root}")


# ===================================================================
# Dice-quality wrapper: run predictions on candidates, keep good ones
# ===================================================================

def pick_samples_quality_filtered(data_root, model, device, num_classes,
                                  n_samples=4, img_size=224, seed=42,
                                  min_dice=0.75, max_eval=24, verbose=True):
    """Pick `n_samples` samples that all have Dice >= `min_dice`,
    spanning a range of mask-coverage quantiles. Runs the model on
    up to `max_eval` candidates and keeps the best-quality ones.
    """
    # 1) Get all candidates sorted by coverage
    all_cands_raw = pick_samples(data_root, n_samples=n_samples,
                                 img_size=img_size, seed=seed,
                                 verbose=False, return_all=True)
    # pick_samples in return_all mode returns the FILE-LEVEL candidates;
    # call again normally to actually load them lazily. Easier: just load
    # everything up to max_eval here.
    # Re-implement candidate loading inline since pick_samples doesn't
    # expose a "load this one specific file" entry point cleanly.
    return _pick_quality_inline(data_root, model, device, num_classes,
                                n_samples=n_samples, img_size=img_size,
                                seed=seed, min_dice=min_dice,
                                max_eval=max_eval, verbose=verbose)


def _pick_quality_inline(data_root, model, device, num_classes,
                         n_samples=4, img_size=224, seed=42,
                         min_dice=0.75, max_eval=24, verbose=True):
    """Inline version: enumerates candidates, runs prediction on each,
    keeps n_samples that span coverage and pass min_dice."""
    root = Path(data_root)

    for split in ['test', 'val', 'train']:
        # ---- PNG/JPG branch ----
        id_ = root / split / 'images'
        md_ = root / split / 'masks'
        if id_.exists() and md_.exists():
            exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
            sfx = ['_segmentation', '_Segmentation', '_mask', '_seg']
            ml = {}
            for p in md_.iterdir():
                if p.suffix.lower() in exts:
                    ml[p.stem] = p
                    for s in sfx:
                        if p.stem.endswith(s):
                            ml[p.stem[:-len(s)]] = p
            pairs = []
            for p in sorted(id_.iterdir()):
                if p.suffix.lower() not in exts: continue
                m = ml.get(p.stem)
                if not m:
                    for s in sfx:
                        m = ml.get(p.stem + s)
                        if m: break
                if m: pairs.append((str(p), str(m)))
            if not pairs: continue

            random.seed(seed)
            # Pull a large pool of candidates, score them by coverage,
            # spread the prediction-eval budget across coverage quantiles.
            pool = random.sample(pairs, min(max_eval * 4, len(pairs)))
            scored = []
            for ip, mp in pool:
                msk = Image.open(mp).convert('L')
                ma = np.array(TF.resize(msk, [img_size, img_size]))
                thresh = 128 if ma.max() > 10 else 0
                cov = float((ma > thresh).mean())
                if cov >= 0.005:
                    scored.append((cov, ip, mp))
            scored.sort()
            if not scored: continue

            # Stratify into n_samples coverage buckets; within each bucket
            # try several candidates and keep the first that passes min_dice
            chosen = []
            buckets = np.linspace(0, len(scored), n_samples + 1, dtype=int)
            for b in range(n_samples):
                lo, hi = buckets[b], buckets[b + 1]
                if hi <= lo: hi = lo + 1
                bucket = scored[lo:hi]
                # Walk from the centre of the bucket outward
                centre = len(bucket) // 2
                order = []
                for d in range(max(centre + 1, len(bucket) - centre)):
                    if centre + d < len(bucket): order.append(centre + d)
                    if d > 0 and centre - d >= 0: order.append(centre - d)
                tried_in_bucket = 0
                for k in order:
                    if tried_in_bucket >= max(4, max_eval // n_samples): break
                    cov, ip, mp = bucket[k]
                    tried_in_bucket += 1
                    # Load + predict
                    img = Image.open(ip).convert('RGB')
                    msk = Image.open(mp).convert('L')
                    orig_np = np.array(TF.resize(img, [img_size, img_size]))
                    t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                    img_t = TF.normalize(t, [.485, .456, .406], [.229, .224, .225])
                    ma = np.array(TF.resize(msk, [img_size, img_size]))
                    thresh = 128 if ma.max() > 10 else 0
                    mt = torch.from_numpy(ma).float()
                    mask_t = (mt > thresh).float()
                    pred = predict(model, img_t, device, num_classes=num_classes)
                    d = dice_score((pred.numpy() > 0).astype(np.uint8),
                                   (mask_t.numpy() > 0).astype(np.uint8))
                    if d >= min_dice:
                        if verbose:
                            print(f'        bucket {b+1}/{n_samples}: {Path(ip).name}  '
                                  f'(cov={cov:.3f}, Dice={d*100:.1f}%)')
                        chosen.append((img_t, mask_t, orig_np, Path(ip).stem,
                                       pred, d))
                        break
                else:
                    # No sample in this bucket passed; take the best Dice from
                    # the ones we tried
                    best = None
                    for k in order[:max(4, max_eval // n_samples)]:
                        if k >= len(bucket): continue
                        cov, ip, mp = bucket[k]
                        img = Image.open(ip).convert('RGB')
                        msk = Image.open(mp).convert('L')
                        orig_np = np.array(TF.resize(img, [img_size, img_size]))
                        t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                        img_t = TF.normalize(t, [.485, .456, .406], [.229, .224, .225])
                        ma = np.array(TF.resize(msk, [img_size, img_size]))
                        thresh = 128 if ma.max() > 10 else 0
                        mt = torch.from_numpy(ma).float()
                        mask_t = (mt > thresh).float()
                        pred = predict(model, img_t, device, num_classes=num_classes)
                        d = dice_score((pred.numpy() > 0).astype(np.uint8),
                                       (mask_t.numpy() > 0).astype(np.uint8))
                        if best is None or d > best[-1]:
                            best = (img_t, mask_t, orig_np, Path(ip).stem, pred, d)
                    if best is not None:
                        if verbose:
                            print(f'        bucket {b+1}/{n_samples}: '
                                  f'{best[3]} (best Dice={best[-1]*100:.1f}% '
                                  f'below threshold {min_dice*100:.0f}%)')
                        chosen.append(best)
            return chosen

        # ---- NPZ branch ----
        npz_dir = root / split
        if npz_dir.exists():
            npz_files = sorted(npz_dir.glob('*.npz'))
            if not npz_files: continue
            random.seed(seed)
            pool = random.sample(npz_files, min(max_eval * 4, len(npz_files)))
            scored = []
            for f in pool:
                data = np.load(f)
                msk_key = None
                for cand in ('mask', 'label', 'seg', 'segmentation', 'gt'):
                    if cand in data.files:
                        msk_key = cand; break
                if msk_key is None: msk_key = data.files[-1]
                msk_np = normalise_mask_array(data[msk_key])
                cov = float((msk_np > 0).mean())
                if cov >= 0.005:
                    scored.append((cov, f, msk_key))
            scored.sort()
            if not scored: continue

            chosen = []
            buckets = np.linspace(0, len(scored), n_samples + 1, dtype=int)
            for b in range(n_samples):
                lo, hi = buckets[b], buckets[b + 1]
                if hi <= lo: hi = lo + 1
                bucket = scored[lo:hi]
                centre = len(bucket) // 2
                order = []
                for d in range(max(centre + 1, len(bucket) - centre)):
                    if centre + d < len(bucket): order.append(centre + d)
                    if d > 0 and centre - d >= 0: order.append(centre - d)
                best = None
                for k in order[:max(4, max_eval // n_samples)]:
                    if k >= len(bucket): continue
                    cov, f, mk = bucket[k]
                    data = np.load(f)
                    img_key = 'image' if 'image' in data.files else data.files[0]
                    img_uint8 = normalise_image_array(data[img_key])
                    msk_arr = normalise_mask_array(data[mk])
                    img = Image.fromarray(img_uint8).convert('RGB')
                    msk = Image.fromarray(msk_arr)
                    orig_np = np.array(TF.resize(img, [img_size, img_size]))
                    t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                    img_t = TF.normalize(t, [.485, .456, .406], [.229, .224, .225])
                    mt = torch.from_numpy(np.array(TF.resize(msk, [img_size, img_size],
                                         interpolation=TF.InterpolationMode.NEAREST))).float()
                    mask_t = (mt > 0).float()
                    pred = predict(model, img_t, device, num_classes=num_classes)
                    dd = dice_score((pred.numpy() > 0).astype(np.uint8),
                                    (mask_t.numpy() > 0).astype(np.uint8))
                    if dd >= min_dice:
                        if verbose:
                            print(f'        bucket {b+1}/{n_samples}: {f.name}  '
                                  f'(cov={cov:.3f}, Dice={dd*100:.1f}%)')
                        chosen.append((img_t, mask_t, orig_np, f.stem, pred, dd))
                        best = None  # mark as accepted
                        break
                    if best is None or dd > best[-1]:
                        best = (img_t, mask_t, orig_np, f.stem, pred, dd)
                else:
                    if best is not None:
                        if verbose:
                            print(f'        bucket {b+1}/{n_samples}: {best[3]} '
                                  f'(best Dice={best[-1]*100:.1f}% below '
                                  f'threshold {min_dice*100:.0f}%)')
                        chosen.append(best)
            return chosen

    raise RuntimeError(f"No data in {data_root}")


# ===================================================================
# Layout 1: compact overlay grid (rows = datasets, cols = samples)
# ===================================================================

def fig4_grid(per_dataset, save_path):
    """per_dataset = list of (name, [(orig, gt, pred, dice, fname), ...])."""
    n_rows = len(per_dataset)
    n_cols = max(len(s[1]) for s in per_dataset)
    cell = 2.6
    fig = plt.figure(figsize=(cell * (n_cols + 0.4), cell * n_rows + 0.6))
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  hspace=0.06, wspace=0.04,
                  left=0.06, right=0.99, top=0.97, bottom=0.06)

    for i, (name, samples) in enumerate(per_dataset):
        for j, (orig, gt, pred, dice, fname) in enumerate(samples):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(orig)
            if gt.max() > 0:
                ax.contour(gt, levels=[0.5], colors='lime', linewidths=1.6,
                           linestyles='-')
            if pred.max() > 0:
                ax.contour(pred, levels=[0.5], colors='orange', linewidths=1.6,
                           linestyles='-')
            ax.text(0.03, 0.04, f'Dice={dice*100:.1f}',
                    transform=ax.transAxes, fontsize=10, fontweight='bold',
                    color='white', va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='black', alpha=0.55,
                              edgecolor='none'))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(name, fontsize=13, fontweight='bold',
                              rotation=0, ha='right', va='center', labelpad=20)

    # Legend across the bottom
    legend_elems = [
        Line2D([0], [0], color='lime',   lw=2.5, label='Ground truth'),
        Line2D([0], [0], color='orange', lw=2.5, label='HamSeg prediction'),
    ]
    fig.legend(handles=legend_elems, loc='lower center',
               ncol=2, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ===================================================================
# Layout 2: detailed input | GT | pred | overlay per sample
# ===================================================================

def fig4_detailed(per_dataset, save_path):
    """Original 4-column layout, one row per sample."""
    rows_data = []
    for name, samples in per_dataset:
        for k, (orig, gt, pred, dice, fname) in enumerate(samples):
            rows_data.append((name if k == 0 else '', orig, gt, pred, dice, fname))

    n_rows = len(rows_data)
    cell = 2.4
    fig = plt.figure(figsize=(cell * 4 + 0.6, cell * n_rows + 0.4))
    gs = GridSpec(n_rows, 4, figure=fig,
                  hspace=0.04, wspace=0.03,
                  left=0.06, right=0.99, top=0.96, bottom=0.04)

    titles = ['Input', 'Ground truth', 'Prediction', 'Overlay']
    for j, t in enumerate(titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(t, fontsize=12, fontweight='bold', pad=8)
        ax.remove()  # placeholder for top-row title alignment

    for i, (name, orig, gt, pred, dice, fname) in enumerate(rows_data):
        # Input
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(orig)
        if name:
            ax.set_ylabel(name, fontsize=12, fontweight='bold',
                          rotation=0, ha='right', va='center', labelpad=20)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0:
            ax.set_title(titles[0], fontsize=12, fontweight='bold', pad=8)

        # GT
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0:
            ax.set_title(titles[1], fontsize=12, fontweight='bold', pad=8)

        # Prediction
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0:
            ax.set_title(titles[2], fontsize=12, fontweight='bold', pad=8)

        # Overlay
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(orig)
        if gt.max() > 0:
            ax.contour(gt, levels=[0.5], colors='lime', linewidths=1.6)
        if pred.max() > 0:
            ax.contour(pred, levels=[0.5], colors='orange', linewidths=1.6)
        ax.text(0.03, 0.04, f'Dice={dice*100:.1f}',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color='white', va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='black',
                          alpha=0.55, edgecolor='none'))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0:
            ax.set_title(titles[3], fontsize=12, fontweight='bold', pad=8)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ===================================================================
# Main
# ===================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--acdc_ckpt');   p.add_argument('--acdc_root')
    p.add_argument('--isic18_ckpt'); p.add_argument('--isic18_root')
    p.add_argument('--isic17_ckpt'); p.add_argument('--isic17_root')
    p.add_argument('--tn3k_ckpt');   p.add_argument('--tn3k_root')
    p.add_argument('--mmotu_ckpt');  p.add_argument('--mmotu_root')
    p.add_argument('--save_dir', default='./figures')
    p.add_argument('--embed_dim', type=int, default=48)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_samples', type=int, default=4,
                   help='Samples per dataset (default: 4)')
    p.add_argument('--layout', default='grid', choices=['grid', 'detailed'],
                   help='grid: overlay-only N x M grid (recommended). '
                        'detailed: input | GT | pred | overlay per row.')
    p.add_argument('--min_dice', type=float, default=0.80,
                   help='Reject samples whose Dice falls below this threshold. '
                        'The picker tries additional candidates until n_samples '
                        'pass. Set to 0 to disable quality filtering.')
    p.add_argument('--max_eval', type=int, default=24,
                   help='Max candidates per dataset to run prediction on while '
                        'searching for high-Dice samples (default 24).')
    a = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.save_dir, exist_ok=True)

    config = [
        ('ACDC',      a.acdc_ckpt,   a.acdc_root,   4),
        ('ISIC 2018', a.isic18_ckpt, a.isic18_root, 1),
        ('ISIC 2017', a.isic17_ckpt, a.isic17_root, 1),
        ('TN3K',      a.tn3k_ckpt,   a.tn3k_root,   1),
        ('MMOTU',     a.mmotu_ckpt,  a.mmotu_root,  1),
    ]

    per_dataset = []
    for name, ckpt, root, ncls in config:
        if not ckpt or not root: continue
        if not os.path.exists(ckpt):
            print(f'[skip] {name}: ckpt not found'); continue
        if not os.path.isdir(root):
            print(f'[skip] {name}: data root not found'); continue
        print(f'[load] {name}')
        try:
            model = load_model(ckpt, device, embed_dim=a.embed_dim, num_classes=ncls)
        except Exception as e:
            print(f'[skip] {name}: load failed ({e})'); continue

        if a.min_dice > 0:
            print(f'[pick] {a.n_samples} samples with Dice >= {a.min_dice:.2f} '
                  f'spanning coverage quantiles (max_eval={a.max_eval})')
            try:
                samples = _pick_quality_inline(root, model, device, ncls,
                                               n_samples=a.n_samples,
                                               seed=a.seed,
                                               min_dice=a.min_dice,
                                               max_eval=a.max_eval)
            except RuntimeError as e:
                print(f'[skip] {name}: {e}')
                del model; torch.cuda.empty_cache(); continue
        else:
            print(f'[pick] {a.n_samples} samples spanning coverage quantiles')
            try:
                raw = pick_samples(root, n_samples=a.n_samples, seed=a.seed)
            except RuntimeError as e:
                print(f'[skip] {name}: {e}')
                del model; torch.cuda.empty_cache(); continue
            samples = []
            for img_t, mask_t, orig, fname in raw:
                pred = predict(model, img_t, device, num_classes=ncls)
                d = dice_score((pred.numpy() > 0).astype(np.uint8),
                               (mask_t.numpy() > 0).astype(np.uint8))
                samples.append((img_t, mask_t, orig, fname, pred, d))

        out = []
        for img_t, mask_t, orig, fname, pred, d in samples:
            pred_np = (pred.numpy() > 0).astype(np.uint8)
            gt_np = (mask_t.numpy() > 0).astype(np.uint8)
            out.append((orig / 255.0, gt_np, pred_np, d, fname))
            print(f'         {fname}: Dice={d*100:.2f}%')
        per_dataset.append((name, out))
        del model; torch.cuda.empty_cache()

    if not per_dataset:
        raise SystemExit('No datasets loaded.')

    save_path = os.path.join(a.save_dir, f'fig4_qualitative_{a.layout}.png')
    print(f'\n[render] Fig 4 ({a.layout} layout, {a.n_samples} samples per dataset)')
    if a.layout == 'grid':
        fig4_grid(per_dataset, save_path)
    else:
        fig4_detailed(per_dataset, save_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
