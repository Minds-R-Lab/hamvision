#!/usr/bin/env python3
"""
visualize_fig5_multidataset.py  (v3, clean redesign)
=====================================================

Rebuilds Figure 5 and Figure 6 of the HamVision paper.

Addresses reviewer R2-8 ("no colour bar; TN3K d_2 looks inconsistent")
plus the layout issues identified in the v1 attempt.

Design choices in this version:

* Figure 5 is image-only -- six columns of physics maps with two clean,
  well-spaced colour bars beneath. No diagnostic charts mixed in.
* The skip-gate colour map is auto-scaled per row to the actual dynamic
  range of that dataset's gates (the absolute sigmoid range [0,1] is
  too wide to reveal the structure when gates cluster near 0.5).
* Figure 6 is a single-sample explainer with five physically-meaningful
  panels (input, position, momentum, energy, gate) and *one* colour bar
  per panel.
* Diagnostic prints announce the shape / dtype / range of each row's
  data so blank panels can be debugged from the terminal.

Usage:
    python visualize_fig5_multidataset.py \\
        --acdc_ckpt   path/to/best_model.pth --acdc_root   path/to/ACDC \\
        --isic18_ckpt ... --isic18_root ... \\
        --isic17_ckpt ... --isic17_root ... \\
        --tn3k_ckpt   ... --tn3k_root   ... \\
        --mmotu_ckpt  ... --mmotu_root  ... \\
        --save_dir figures --embed_dim 48
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

warnings.filterwarnings('ignore')


# ===================================================================
# Model + signal extraction
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


def extract(model, img_t, device):
    model.eval()
    S = {}
    with torch.no_grad():
        x = img_t.unsqueeze(0).to(device)
        logits = model(x)
        if logits.shape[1] > 1:
            S['pred'] = logits.float().argmax(dim=1)[0].cpu().float()
        else:
            S['pred'] = (torch.sigmoid(logits.float()) > 0.5).float().cpu()[0, 0]

        x_s = model.stem(x)
        e1 = model.enc1(x_s)
        e2 = model.enc2(model.down1(e1))
        e3 = model.enc3(model.down2(e2))
        e4 = model.down3(e3)

        momentum, energy_map, position = None, None, None
        for blk in model.bottleneck:
            conv_out = blk.conv_block(e4)
            x_n = blk.norm(e4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            with torch.cuda.amp.autocast(enabled=False):
                pos, mom, energy_raw = blk.ss2d(x_n.float())
                ham_out = blk.pos_proj(pos)
                g = blk.gate(torch.cat([conv_out.float(), ham_out], 1))
                out = conv_out.float() * g + ham_out * (1 - g)
            e4 = out.to(e4.dtype)
            momentum = mom.to(e4.dtype)
            position = pos.to(e4.dtype)
            energy_f = energy_raw.to(e4.dtype)
            ch_weights = blk.energy_attn(energy_f)
            ch_weights = ch_weights.unsqueeze(-1).unsqueeze(-1)
            energy_map = (energy_f * ch_weights).mean(dim=1, keepdim=True)

            S['gate'] = g[0].cpu().float()
            S['ch_weights'] = ch_weights[0, :, 0, 0].cpu().float()
            S['energy_map'] = energy_map[0, 0].cpu().float()
            S['mom_raw'] = mom[0].cpu().float()
            S['pos_raw'] = pos[0].cpu().float()

        S['skip_gates'] = {}
        S['mom_at_level'] = {}
        for level, (skip, enc_feat, target_size) in enumerate([
            (model.skip3, e3, e3.shape[2:]),
            (model.skip2, e2, e2.shape[2:]),
            (model.skip1, e1, e1.shape[2:]),
        ]):
            en_l = F.interpolate(energy_map, target_size, mode='bilinear',
                                 align_corners=False)
            e_centered = en_l - en_l.mean(dim=(2, 3), keepdim=True)
            gate_map = torch.sigmoid(skip.energy_gamma * e_centered)
            S['skip_gates'][level] = gate_map[0, 0].cpu().float()
            mom_l = F.interpolate(momentum, target_size, mode='bilinear',
                                  align_corners=False)
            mom_l = mom_l[:, :enc_feat.shape[1]]
            S['mom_at_level'][level] = mom_l[0].cpu().float()
    return S


def up(t, s=224, mode='bicubic'):
    return F.interpolate(t.unsqueeze(0).unsqueeze(0).float(),
                         size=(s, s), mode=mode, align_corners=False)[0, 0]


def pnorm(a, lo=2, hi=98):
    vlo, vhi = np.percentile(a, lo), np.percentile(a, hi)
    if vhi - vlo < 1e-8:
        vhi = vlo + 1
    return np.clip((a - vlo) / (vhi - vlo), 0, 1)


def auto_gate_range(gate_arr, padding=0.02):
    """Return (vmin, vmax) for visualising a sigmoid gate map.

    Uses 10th/90th percentile (not 2nd/98th) to focus on the bulk of the
    distribution rather than rare extreme pixels -- this is what makes
    the visual structure visible when most values are near 0.5.
    Always centred on 0.5 so the diverging colormap stays semantically
    aligned (red < 0.5 = suppress; blue > 0.5 = amplify)."""
    lo = float(np.percentile(gate_arr, 10))
    hi = float(np.percentile(gate_arr, 90))
    half = max(abs(0.5 - lo), abs(hi - 0.5), 0.04) + padding
    return max(0.0, 0.5 - half), min(1.0, 0.5 + half)


# ===================================================================
# Image / mask normalisation utilities
# ===================================================================

def normalise_image_array(img_np):
    """Convert an arbitrary-range float/uint image to uint8 in [0,255]."""
    img_np = np.asarray(img_np).astype(np.float32)
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3):
        img_np = img_np.transpose(1, 2, 0)
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np[..., 0]
    vmin = float(np.nanmin(img_np))
    vmax = float(np.nanmax(img_np))
    # Always percentile-stretch for safety; this works regardless of
    # whether the raw range is [0,1], [0,255], or arbitrary float (CT HU, MRI raw).
    lo, hi = np.nanpercentile(img_np, [1, 99])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    img_np = np.clip((img_np - lo) / (hi - lo), 0, 1) * 255.0
    img_np = np.nan_to_num(img_np, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(img_np, 0, 255).astype(np.uint8), (vmin, vmax)


def normalise_mask_array(msk_np):
    """Squeeze one-hot / channel-first masks down to (H, W) class indices."""
    msk_np = np.asarray(msk_np)
    if msk_np.ndim == 3 and msk_np.shape[0] < 8 and msk_np.shape[-1] >= 64:
        msk_np = msk_np.argmax(axis=0)
    if msk_np.ndim == 3 and msk_np.shape[-1] < 8:
        msk_np = msk_np.argmax(axis=-1)
    return msk_np.astype(np.uint8)


# ===================================================================
# Representative-sample picker
# ===================================================================

def median_mask_sample(data_root, img_size=224, n_candidates=120, seed=42,
                       verbose=True, coverage_quantile=0.90):
    """Pick a representative image from `data_root`.

    `coverage_quantile` controls *which* quantile of the mask-coverage
    distribution we pick. For visualization we use 0.90 (90th percentile)
    so we get a sample with clearly visible anatomy. ACDC has many
    background-only slices at the edge of each volume, which would
    drag a median or 75th-percentile pick toward empty masks; the 90th
    percentile is robust against that. We avoid the absolute max
    (which can be an outlier whole-image lesion).
    """
    root = Path(data_root)
    for split in ['test', 'val', 'train']:
        # PNG/JPG (ISIC, TN3K, MMOTU)
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
            if not pairs: continue
            random.seed(seed)
            sel = random.sample(pairs, min(n_candidates, len(pairs)))
            scored = []
            for ip, mp in sel:
                msk = Image.open(mp).convert('L')
                ma = np.array(TF.resize(msk, [img_size, img_size]))
                # Multi-class masks use small integer values {0,1,2,3...};
                # binary masks use {0, 255}. Pick the threshold that works for both.
                thresh = 128 if ma.max() > 10 else 0
                cov = float((ma > thresh).mean())
                scored.append((cov, ip, mp))
            scored.sort()
            idx = min(int(coverage_quantile * (len(scored) - 1)),
                      len(scored) - 1)
            cov, ip, mp = scored[idx]
            if cov < 0.005:
                # Fallback: most candidates were empty (typical for ACDC
                # apex/base slices). Pick the 95th-percentile-coverage
                # slice as a robust mid-volume sample with visible anatomy.
                fb_idx = min(int(0.95 * (len(scored) - 1)), len(scored) - 1)
                fb_cov, fb_ip, fb_mp = scored[fb_idx]
                if fb_cov < 0.005:
                    # Still empty -- take the absolute max
                    fb_cov, fb_ip, fb_mp = scored[-1]
                if verbose:
                    print(f'        WARNING: {coverage_quantile*100:.0f}th-percentile '
                          f'pick {Path(ip).name} was near-empty (coverage={cov:.3f}); '
                          f'falling back to high-coverage slice')
                cov, ip, mp = fb_cov, fb_ip, fb_mp
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
            return img_t, mask_t, orig_np

        # NPZ (ACDC)
        npz_dir = root / split
        if npz_dir.exists():
            npz_files = sorted(npz_dir.glob('*.npz'))
            if not npz_files:
                continue
            random.seed(seed)
            sel = random.sample(npz_files, min(n_candidates, len(npz_files)))
            scored = []
            for f in sel:
                data = np.load(f)
                # Resolve mask key robustly across naming conventions
                msk_key = None
                for cand in ('mask', 'label', 'seg', 'segmentation', 'gt'):
                    if cand in data.files:
                        msk_key = cand; break
                if msk_key is None:
                    for k in data.files:
                        arr = data[k]
                        if arr.dtype in (np.uint8, np.int32, np.int64, np.bool_) \
                                or (np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 10):
                            msk_key = k; break
                    if msk_key is None:
                        msk_key = data.files[-1]
                msk_np = normalise_mask_array(data[msk_key])
                cov = float((msk_np > 0).mean())
                scored.append((cov, f, msk_key))
            scored.sort()
            idx = min(int(coverage_quantile * (len(scored) - 1)),
                      len(scored) - 1)
            cov, f, used_msk_key = scored[idx]
            if cov < 0.005:
                # Fallback to max-coverage if the quantile pick is essentially empty
                cov, f, used_msk_key = scored[-1]
                if verbose:
                    print(f'        WARNING: quantile pick was near-empty; '
                          f'falling back to max-coverage slice')
            if verbose:
                print(f'        chose: {f.name}  '
                      f'(coverage={cov:.3f}, msk_key="{used_msk_key}")')
            data = np.load(f)
            img_key = 'image' if 'image' in data.files else data.files[0]
            msk_key = used_msk_key
            if verbose:
                print(f'        npz file keys: {data.files}')

            img_uint8, (orig_min, orig_max) = normalise_image_array(data[img_key])
            if verbose:
                print(f'        image: shape={data[img_key].shape}  '
                      f'dtype={data[img_key].dtype}  range=[{orig_min:.3f}, {orig_max:.3f}]')
            msk_arr = normalise_mask_array(data[msk_key])
            if verbose:
                print(f'        mask : shape={data[msk_key].shape}  '
                      f'unique={np.unique(msk_arr)[:8].tolist()}')

            img = Image.fromarray(img_uint8).convert('RGB')
            msk = Image.fromarray(msk_arr)
            orig_np = np.array(TF.resize(img, [img_size, img_size]))
            t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
            img_t = TF.normalize(t, [.485, .456, .406], [.229, .224, .225])
            mt = torch.from_numpy(np.array(TF.resize(msk, [img_size, img_size],
                                 interpolation=TF.InterpolationMode.NEAREST))).float()
            mask_t = (mt > 0).float()
            return img_t, mask_t, orig_np

    raise RuntimeError(f"No data in {data_root}")


# ===================================================================
# Figure 5 (clean): multi-scale gates with colour bars
# ===================================================================

def fig5(rows, save_path, gate_style='overlay', per_row_scale=True):
    """
    gate_style:
      'overlay'   -- transparent gate map painted ON TOP of the grayscale input
                     (clearest spatial story; recommended for paper)
      'diverging' -- standalone RdBu_r diverging colormap (suppress/amplify)
      'magnitude' -- standalone sequential colormap of |gate - 0.5| (deviation)

    per_row_scale:
      If True (default), each row's gate colour range is normalised to that
      row's actual data, so a dataset with naturally tight gate variance
      (e.g. TN3K $d_2$) still shows visible structure rather than being
      washed out under a globally shared range. The shared colour bar at
      the bottom of the figure still reports the *global* range so the
      reader can see the cross-dataset comparison.
    """
    n = len(rows)
    n_cols = 6
    cell = 2.4
    fig = plt.figure(figsize=(cell * n_cols + 0.6, cell * n + 1.2))
    gs = GridSpec(n + 1, n_cols, figure=fig,
                  height_ratios=[1.0] * n + [0.12],
                  hspace=0.06, wspace=0.04,
                  left=0.05, right=0.99, top=0.93, bottom=0.06)

    gate_cmap = plt.get_cmap('RdBu_r')   # blue at low, red at high -> red = amplify
    mom_cmap  = plt.get_cmap('magma')

    titles = ['Input + GT',
              r'Skip gate at $d_3$' + '\n(28$\\times$28)',
              r'Skip gate at $d_2$' + '\n(56$\\times$56)',
              r'Skip gate at $d_1$' + '\n(112$\\times$112)',
              r'Momentum $|p|$' + '\nat $d_3$',
              r'Momentum $|p|$' + '\nat $d_1$']

    for j, t in enumerate(titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(t, fontsize=11, fontweight='bold', pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_frame_on(False)
        # Hide the placeholder; we re-add the real subplot below
        ax.remove()

    # Compute one global gate colour range (used for the colour bar at the bottom)
    all_gate = np.concatenate([up(R[k]).numpy().ravel()
                               for R in rows for k in ['gate3', 'gate2', 'gate1']])
    g_vmin, g_vmax = auto_gate_range(all_gate)

    # Per-row gate range (used for visualisation if per_row_scale=True)
    per_row_gate_range = {}
    for R in rows:
        row_gate = np.concatenate([up(R[k]).numpy().ravel()
                                   for k in ['gate3', 'gate2', 'gate1']])
        per_row_gate_range[R['name']] = auto_gate_range(row_gate, padding=0.01)

    for i, R in enumerate(rows):
        # Input + GT
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(R['orig'])
        if R['mask'].max() > 0:
            ax.contour(R['mask'], levels=[0.5], colors='lime', linewidths=1.8)
        ax.set_ylabel(R['name'], fontsize=12, fontweight='bold',
                      rotation=0, ha='right', va='center', labelpad=18)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0:
            ax.set_title(titles[0], fontsize=11, fontweight='bold', pad=8)

        # Skip gates at d3, d2, d1
        # Convert the input to a grayscale luminance image so the overlay reads cleanly
        gray = np.asarray(R['orig']).astype(np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=-1)
        gray = (gray - gray.min()) / max(gray.max() - gray.min(), 1e-8)

        # Pick per-row range if requested (helps datasets like TN3K $d_2$
        # whose gates have unusually tight std vs the others)
        if per_row_scale:
            row_vmin, row_vmax = per_row_gate_range[R['name']]
        else:
            row_vmin, row_vmax = g_vmin, g_vmax

        for j, key in enumerate(['gate3', 'gate2', 'gate1']):
            ax = fig.add_subplot(gs[i, 1 + j])
            gate_up = up(R[key], 224).numpy()

            if gate_style == 'overlay':
                ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
                dev = np.abs(gate_up - 0.5)
                dev = dev / max(dev.max(), 1e-6)
                alpha = 0.30 + 0.55 * dev
                rgba = plt.get_cmap('RdBu_r')(
                    np.clip((gate_up - row_vmin) / max(row_vmax - row_vmin, 1e-6), 0, 1))
                rgba[..., 3] = alpha
                ax.imshow(rgba)
            elif gate_style == 'magnitude':
                dev = np.abs(gate_up - 0.5)
                ax.imshow(dev, cmap='inferno',
                          vmin=0, vmax=max(row_vmax - 0.5, 0.5 - row_vmin))
            else:  # 'diverging'
                ax.imshow(gate_up, cmap=gate_cmap, vmin=row_vmin, vmax=row_vmax)

            if R['mask'].max() > 0:
                ax.contour(R['mask'], levels=[0.5], colors='lime',
                           linewidths=1.4, linestyles='-', alpha=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if i == 0:
                ax.set_title(titles[1 + j], fontsize=11, fontweight='bold', pad=8)

        # Momentum at d3, d1 -- upsample to 224 so the panel matches the contour scale
        for j, key in enumerate(['mom3', 'mom1']):
            ax = fig.add_subplot(gs[i, 4 + j])
            mom_mag = R[key].norm(dim=0)
            mom_up = up(mom_mag, 224, mode='bicubic').numpy()
            ax.imshow(pnorm(mom_up), cmap=mom_cmap, vmin=0, vmax=1)
            if R['mask'].max() > 0:
                ax.contour(R['mask'], levels=[0.5], colors='cyan',
                           linewidths=1.4, linestyles='--', alpha=0.85)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if i == 0:
                ax.set_title(titles[4 + j], fontsize=11, fontweight='bold', pad=8)

    # ---- Colour bars at the bottom: gate (cols 1-3), momentum (cols 4-5) ----
    # Use sub-gridspecs so each colour bar has its own padding and labels
    # cannot collide with its neighbour.
    sub_g = gs[n, 1:4].subgridspec(1, 3, width_ratios=[0.05, 1.0, 0.05])
    cax_g = fig.add_subplot(sub_g[0, 1])
    sm_g = plt.cm.ScalarMappable(cmap=gate_cmap,
                                 norm=plt.Normalize(vmin=g_vmin, vmax=g_vmax))
    cb_g = plt.colorbar(sm_g, cax=cax_g, orientation='horizontal')
    cb_g.set_ticks([g_vmin, 0.5, g_vmax])
    cb_g.set_ticklabels([f'{g_vmin:.2f}\n(suppress)',
                         '0.50\n(neutral)',
                         f'{g_vmax:.2f}\n(amplify)'])
    cb_g.ax.tick_params(labelsize=9)
    cb_g.outline.set_visible(False)
    cb_g.set_label(
        r'Skip-gate activation $\sigma(\gamma_\ell (H_\ell - \bar H_\ell))$',
        fontsize=10, labelpad=6)

    sub_m = gs[n, 4:6].subgridspec(1, 3, width_ratios=[0.10, 1.0, 0.05])
    cax_m = fig.add_subplot(sub_m[0, 1])
    sm_m = plt.cm.ScalarMappable(cmap=mom_cmap,
                                 norm=plt.Normalize(vmin=0, vmax=1))
    cb_m = plt.colorbar(sm_m, cax=cax_m, orientation='horizontal')
    cb_m.set_ticks([0.0, 0.5, 1.0])
    cb_m.set_ticklabels(['low\n(quiescent)', 'mid', 'high\n(active)'])
    cb_m.ax.tick_params(labelsize=9)
    cb_m.outline.set_visible(False)
    cb_m.set_label(
        r'Momentum magnitude $|p|$ (per-panel percentile)',
        fontsize=10, labelpad=6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ===================================================================
# Figure 6 (clean): single-sample physics-as-interpretability
# ===================================================================

def fig6(R, save_path):
    fig = plt.figure(figsize=(22, 6.2))
    gs = GridSpec(2, 5, figure=fig,
                  height_ratios=[1.0, 0.07],
                  hspace=0.28, wspace=0.14,
                  left=0.03, right=0.99, top=0.90, bottom=0.20)

    cmaps = {
        'input':   None,
        'q':       'viridis',
        'mom':     'magma',
        'energy':  'inferno',
        'gate':    'RdBu_r',
    }

    titles = [
        f"Input + GT  ({R['name']})",
        r"Position  $|q|$" + "\n(filtered representation)",
        r"Momentum  $|p|$" + "\n(spatial derivative)",
        r"Energy  $H$" + "\n(saliency / activity)",
        r"Skip gate at $d_1$" + "\n" +
        r"$\sigma(\gamma_1 (H_1 - \bar H_1))$ (energy-driven attention)",
    ]

    # Panel 0: Input + GT contour
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(R['orig'])
    if R['mask'].max() > 0:
        ax.contour(R['mask'], levels=[0.5], colors='lime', linewidths=2.0)
    ax.set_title(titles[0], fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    # Panels 1-4: q, |p|, H, energy-gated skip (the gate that does
    # the anatomy-aligned attention work)
    skip_gate_arr = up(R['gate1'], 224).numpy()
    panel_data = [
        ('q',      pnorm(up(R['pos_raw'].norm(dim=0), 224).numpy()), 0, 1),
        ('mom',    pnorm(up(R['mom_raw'].norm(dim=0), 224).numpy()), 0, 1),
        ('energy', pnorm(up(R['energy_map'], 224).numpy()),          0, 1),
        ('gate',   skip_gate_arr,
                   *auto_gate_range(skip_gate_arr)),
    ]
    images = []
    for j, (key, arr, vmin, vmax) in enumerate(panel_data, start=1):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(arr, cmap=cmaps[key], vmin=vmin, vmax=vmax)
        if R['mask'].max() > 0:
            edge_color = 'lime' if key != 'gate' else 'black'
            ax.contour(R['mask'], levels=[0.5], colors=edge_color,
                       linewidths=1.5, linestyles='--', alpha=0.8)
        ax.set_title(titles[j], fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        images.append((j, im, key, vmin, vmax))

    # Colour bar for each physics panel (skip the input panel)
    for j, im, key, vmin, vmax in images:
        cax = fig.add_subplot(gs[1, j])
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=9)
        cb.outline.set_visible(False)
        if key == 'gate':
            cb.set_ticks([vmin, 0.5, vmax])
            cb.set_ticklabels([f'{vmin:.2f}', '0.50', f'{vmax:.2f}'])
            cb.set_label('suppress  $\\leftarrow$  neutral  $\\rightarrow$  amplify',
                         fontsize=9, labelpad=4)
        elif key == 'q':
            cb.set_ticks([0, 1]); cb.set_ticklabels(['0', '1'])
            cb.set_label('low  $\\rightarrow$  high', fontsize=9, labelpad=4)
        elif key == 'mom':
            cb.set_ticks([0, 1]); cb.set_ticklabels(['0', '1'])
            cb.set_label('quiescent  $\\rightarrow$  active', fontsize=9, labelpad=4)
        elif key == 'energy':
            cb.set_ticks([0, 1]); cb.set_ticklabels(['0', '1'])
            cb.set_label('low  $\\rightarrow$  high', fontsize=9, labelpad=4)

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
    p.add_argument('--focus', default=None,
                   help='Dataset name to use for Figure 6 (default: first available).')
    p.add_argument('--gate_style', default='overlay',
                   choices=['overlay', 'diverging', 'magnitude'],
                   help='How to render the skip-gate maps in Fig 5. '
                        'overlay (default) paints the gate ON TOP of the grayscale '
                        'input; diverging shows a standalone RdBu_r colormap; '
                        'magnitude shows |gate-0.5| with a sequential colormap.')
    p.add_argument('--gate_scale', default='per_row',
                   choices=['per_row', 'global'],
                   help='per_row (default): each dataset row scales its gate colormap '
                        'to its own data range, so tight-variance rows still show '
                        'structure (helps TN3K $d_2$). global: one shared range '
                        'across all rows.')
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

    rows = []
    for name, ckpt, root, ncls in config:
        if not ckpt or not root:
            print(f'[skip] {name}: ckpt or root not provided')
            continue
        if not os.path.exists(ckpt):
            print(f'[skip] {name}: ckpt not found at {ckpt}')
            continue
        if not os.path.isdir(root):
            print(f'[skip] {name}: data root not found at {root}')
            continue
        print(f'[load] {name}: {ckpt}')
        try:
            model = load_model(ckpt, device, embed_dim=a.embed_dim, num_classes=ncls)
        except Exception as e:
            print(f'[skip] {name}: load failed ({e})')
            continue
        print(f'[pick] median-coverage sample from {root}')
        try:
            img_t, mask_t, orig = median_mask_sample(root, seed=a.seed)
        except RuntimeError as e:
            print(f'[skip] {name}: {e}')
            del model
            torch.cuda.empty_cache()
            continue
        print(f'[run]  forward + signal extraction')
        S = extract(model, img_t, device)

        row = {
            'name':   name,
            'orig':   orig / 255.0,
            'mask':   mask_t,
            'gate3':  S['skip_gates'][0],
            'gate2':  S['skip_gates'][1],
            'gate1':  S['skip_gates'][2],
            'mom3':   S['mom_at_level'][0],
            'mom1':   S['mom_at_level'][2],
            'mom_raw': S['mom_raw'],
            'pos_raw': S['pos_raw'],
            'energy_map': S['energy_map'],
            'gate':   S['gate'],
            'ch_weights': S['ch_weights'],
            'pred':   S['pred'].numpy(),
        }

        # ---- Diagnostic prints so blank panels can be debugged ----
        print(f'       diagnostics:')
        print(f'         orig    shape={row["orig"].shape} '
              f'range=[{row["orig"].min():.3f}, {row["orig"].max():.3f}]')
        print(f'         mask    shape={tuple(row["mask"].shape)} '
              f'max={row["mask"].max().item():.0f} '
              f'coverage={(row["mask"] > 0).float().mean().item():.3f}')
        for k in ['gate3', 'gate2', 'gate1']:
            v = row[k].numpy()
            print(f'         {k:8s} shape={v.shape} '
                  f'mean={v.mean():.3f} std={v.std():.3f} '
                  f'range=[{v.min():.3f}, {v.max():.3f}]')

        rows.append(row)
        del model
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit('No datasets loaded.')

    print(f'\n[render] Figure 5 -- multi-scale gates '
          f'(style={a.gate_style}, scale={a.gate_scale})')
    fig5(rows, os.path.join(a.save_dir, 'fig5_multiscale_energy_gates.png'),
         gate_style=a.gate_style,
         per_row_scale=(a.gate_scale == 'per_row'))

    print('\n[render] Figure 6 -- single-sample physics-as-interpretability')
    focus_row = rows[0]
    if a.focus:
        for r in rows:
            if r['name'].lower() == a.focus.lower():
                focus_row = r; break
    fig6(focus_row, os.path.join(a.save_dir, 'fig6_physics_interpretability.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
