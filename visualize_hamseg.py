#!/usr/bin/env python3
"""
HamSeg v3 Interpretability Visualization
==========================================

Generates 5 publication-quality figures:

1. Physics Signals: Energy H (ch-attn) + Momentum |p| + overlay + prediction
2. Channel Specialization: ConvNeXt vs Hamiltonian channels + gate histogram
3. Scan Directions: Per-direction (row→←, col↓↑) momentum maps
4. Boundary Analysis: Bar + box plot of momentum by region (interior/boundary/exterior)
5. Multi-Scale Energy Gates: Skip gate maps at all 3 decoder levels + momentum injection

Usage:
    python visualize_hamseg.py --model_path ./outputs_hamseg/isic2018/best_model.pth \
                               --data_root ./data/ISIC2018 --n_samples 6
"""

import os, sys, argparse, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_model(model_path, device, embed_dim=48, num_classes=1):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in list(sys.modules.keys()):
        if 'hamseg' in mod:
            del sys.modules[mod]
    from hamseg import HamSeg
    class A: pass
    a = A()
    a.embed_dim = embed_dim; a.depths = [2,2,2,2]
    a.damping_clamp = 5.0; a.num_classes = num_classes
    a.img_size = 224; a.drop_rate = 0.1
    model = HamSeg(a).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def extract(model, img_t, device):
    """Run model forward, capture all v3 physics signals."""
    model.eval()
    S = {}
    with torch.no_grad():
        x = img_t.unsqueeze(0).to(device)

        # Prediction
        logits = model(x)
        if logits.shape[1] > 1:
            S['pred'] = logits.float().argmax(dim=1)[0].cpu().float()
        else:
            S['pred'] = (torch.sigmoid(logits.float()) > 0.5).float().cpu()[0, 0]

        # Re-run encoder + bottleneck to capture signals
        x_s = model.stem(x)
        e1 = model.enc1(x_s)
        e2 = model.enc2(model.down1(e1))
        e3 = model.enc3(model.down2(e2))
        e4 = model.down3(e3)

        momentum, energy_map = None, None
        for blk in model.bottleneck:
            conv_out = blk.conv_block(e4)
            x_n = blk.norm(e4.permute(0,2,3,1)).permute(0,3,1,2)
            with torch.cuda.amp.autocast(enabled=False):
                pos, mom, energy_raw = blk.ss2d(x_n.float())
                ham_out = blk.pos_proj(pos)
                g = blk.gate(torch.cat([conv_out.float(), ham_out], 1))
                out = conv_out.float() * g + ham_out * (1 - g)
            e4 = out.to(e4.dtype)
            momentum = mom.to(e4.dtype)

            # Replicate learned energy channel attention
            energy_f = energy_raw.to(e4.dtype)
            ch_weights = blk.energy_attn(energy_f)
            ch_weights = ch_weights.unsqueeze(-1).unsqueeze(-1)
            energy_map = (energy_f * ch_weights).mean(dim=1, keepdim=True)

            # Store signals from last block
            S['pos_raw'] = pos[0].cpu().float()
            S['mom_raw'] = mom[0].cpu().float()
            S['eng_raw'] = energy_raw[0].cpu().float()
            S['conv_out'] = conv_out[0].cpu().float()
            S['ham_out'] = ham_out[0].cpu().float()
            S['gate'] = g[0].cpu().float()
            S['energy_map'] = energy_map[0, 0].cpu().float()
            S['ch_weights'] = ch_weights[0, :, 0, 0].cpu().float()

        # Compute skip gate maps at all 3 decoder levels
        S['skip_gates'] = {}
        S['mom_at_level'] = {}
        for level, (skip, enc_feat, target_size) in enumerate([
            (model.skip3, e3, e3.shape[2:]),
            (model.skip2, e2, e2.shape[2:]),
            (model.skip1, e1, e1.shape[2:]),
        ]):
            en_l = F.interpolate(energy_map, target_size, mode='bilinear', align_corners=False)
            e_centered = en_l - en_l.mean(dim=(2, 3), keepdim=True)
            gate_map = torch.sigmoid(skip.energy_gamma * e_centered)
            S['skip_gates'][level] = gate_map[0, 0].cpu().float()

            mom_l = F.interpolate(momentum, target_size, mode='bilinear', align_corners=False)
            mom_l = mom_l[:, :enc_feat.shape[1]]
            S['mom_at_level'][level] = mom_l[0].cpu().float()

    return S


def up(t, s=224):
    return F.interpolate(t.unsqueeze(0).unsqueeze(0).float(),
                         size=(s,s), mode='bilinear', align_corners=False)[0,0]


def pnorm(a, lo=2, hi=98):
    vlo, vhi = np.percentile(a, lo), np.percentile(a, hi)
    if vhi - vlo < 1e-8: vhi = vlo + 1
    return np.clip((a - vlo) / (vhi - vlo), 0, 1)


def load_samples(data_root, n=6, img_size=224):
    root = Path(data_root)
    for split in ['test','val','train']:
        # PNG/JPG (ISIC, TN3K)
        id_, md_ = root/split/'images', root/split/'masks'
        if id_.exists() and md_.exists():
            exts = {'.png','.jpg','.jpeg','.bmp','.tif'}
            sfx = ['_segmentation','_Segmentation','_mask','_seg']
            ml = {}
            for p in md_.iterdir():
                if p.suffix.lower() in exts:
                    ml[p.stem] = p
                    for s in sfx:
                        if p.stem.endswith(s): ml[p.stem[:-len(s)]] = p
            pairs = []
            for p in sorted(id_.iterdir()):
                if p.suffix.lower() not in exts: continue
                m = ml.get(p.stem)
                if not m:
                    for s in sfx:
                        m = ml.get(p.stem+s)
                        if m: break
                if m: pairs.append((str(p), str(m)))
            if pairs:
                random.seed(42)
                sel = random.sample(pairs, min(n, len(pairs)))
                images, masks, originals = [], [], []
                for ip, mp in sel:
                    img = Image.open(ip).convert('RGB')
                    msk = Image.open(mp).convert('L')
                    originals.append(np.array(TF.resize(img, [img_size, img_size])))
                    t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                    images.append(TF.normalize(t, [.485,.456,.406], [.229,.224,.225]))
                    mt = torch.from_numpy(np.array(TF.resize(msk, [img_size, img_size]))).float()
                    masks.append((mt > 128).float())
                return images, masks, originals

        # NPZ (ACDC)
        npz_dir = root / split
        if npz_dir.exists():
            npz_files = sorted(npz_dir.glob('*.npz'))
            if npz_files:
                random.seed(42)
                sel = random.sample(npz_files, min(n, len(npz_files)))
                images, masks, originals = [], [], []
                for f in sel:
                    data = np.load(f)
                    img_np = data['image']
                    msk_np = data['mask']
                    img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(img_uint8).convert('RGB')
                    msk = Image.fromarray(msk_np)
                    originals.append(np.array(TF.resize(img, [img_size, img_size])))
                    t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                    images.append(TF.normalize(t, [.485,.456,.406], [.229,.224,.225]))
                    mt = torch.from_numpy(np.array(TF.resize(msk, [img_size, img_size],
                                         interpolation=TF.InterpolationMode.NEAREST))).float()
                    masks.append((mt > 0).float())
                return images, masks, originals
    raise RuntimeError(f"No data in {data_root}")


# ============================================================
# FIGURE 1: Physics Signals
# ============================================================
def fig1_physics_signals(model, images, masks, originals, save_dir, device):
    n = len(images)
    fig, ax = plt.subplots(n, 6, figsize=(22, 3.2*n))
    if n == 1: ax = ax.reshape(1, -1)

    for i in range(n):
        S = extract(model, images[i], device)
        orig = originals[i] / 255.0
        energy_up = up(S['energy_map']).numpy()
        mom = S['mom_raw']
        mom_mag = mom.norm(dim=0)
        mom_up = up(mom_mag).numpy()
        mom_overlay = pnorm(mom_up)
        pred = S['pred'].numpy()

        ax[i,0].imshow(orig); ax[i,0].axis('off')
        ax[i,1].imshow(masks[i].numpy(), cmap='gray', vmin=0, vmax=1); ax[i,1].axis('off')
        ax[i,2].imshow(pnorm(energy_up), cmap='inferno'); ax[i,2].axis('off')
        ax[i,3].imshow(pnorm(mom_up), cmap='magma'); ax[i,3].axis('off')

        ax[i,4].imshow(orig)
        ax[i,4].imshow(mom_overlay, cmap='hot', alpha=0.55, vmin=0.3, vmax=1.0)
        if masks[i].max() > 0:
            ax[i,4].contour(masks[i].numpy(), levels=[.5], colors='cyan',
                           linewidths=1.5, linestyles='-')
        ax[i,4].axis('off')

        ax[i,5].imshow(pred, cmap='gray', vmin=0, vmax=1)
        if masks[i].max() > 0:
            ax[i,5].contour(masks[i].numpy(), levels=[.5], colors='r',
                           linewidths=1.2, linestyles='--')
        ax[i,5].axis('off')

    titles = ['Input', 'Ground truth',
              'Energy $H$ (ch-attn)\n(boundary activity)',
              'Momentum $|p|$\n(spatial gradients)',
              'Momentum overlay\n(+ GT boundary)',
              'Prediction']
    for j, t in enumerate(titles):
        ax[0, j].set_title(t, fontsize=10, fontweight='bold')

    plt.tight_layout(pad=0.3)
    p = os.path.join(save_dir, 'fig_physics_signals.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'  Saved: {p}')


# ============================================================
# FIGURE 2: Channel Specialization
# ============================================================
def fig2_channel_specialization(model, images, masks, originals, save_dir, device):
    n = min(4, len(images))
    fig, ax = plt.subplots(n, 5, figsize=(20, 3.5*n))
    if n == 1: ax = ax.reshape(1, -1)

    for i in range(n):
        S = extract(model, images[i], device)
        orig = originals[i] / 255.0
        g = S['gate']
        conv = S['conv_out']
        ham = S['ham_out']
        g_ch = g.mean(dim=(1,2))
        k = 48
        _, conv_idx = g_ch.topk(k, largest=True)
        conv_feat = conv[conv_idx].mean(dim=0)
        _, ham_idx = g_ch.topk(k, largest=False)
        ham_feat = ham[ham_idx].mean(dim=0)
        conv_up = up(conv_feat.abs()).numpy()
        ham_up = up(ham_feat.abs()).numpy()
        diff = np.abs(pnorm(ham_up) - pnorm(conv_up))

        ax[i,0].imshow(orig)
        if masks[i].max() > 0:
            ax[i,0].contour(masks[i].numpy(), levels=[.5], colors='lime', linewidths=1.5)
        ax[i,0].axis('off')
        ax[i,1].imshow(pnorm(conv_up), cmap='viridis'); ax[i,1].axis('off')
        ax[i,2].imshow(pnorm(ham_up), cmap='viridis'); ax[i,2].axis('off')
        ax[i,3].imshow(diff, cmap='hot'); ax[i,3].axis('off')

        ax[i,4].hist(g_ch.numpy(), bins=50, color='#534AB7', alpha=0.8, edgecolor='none')
        ax[i,4].axvline(0.5, color='red', linestyle='--', linewidth=1)
        ax[i,4].set_xlim(0, 1)
        n_conv = (g_ch > 0.5).sum().item()
        n_ham = (g_ch <= 0.5).sum().item()
        ax[i,4].text(0.75, 0.85, f'ConvNeXt\n{n_conv} ch',
                    transform=ax[i,4].transAxes, fontsize=8,
                    ha='center', color='#0F6E56', fontweight='bold')
        ax[i,4].text(0.25, 0.85, f'Hamiltonian\n{n_ham} ch',
                    transform=ax[i,4].transAxes, fontsize=8,
                    ha='center', color='#993C1D', fontweight='bold')
        ax[i,4].set_xlabel('Gate $g$', fontsize=8)
        ax[i,4].set_ylabel('Channels', fontsize=8)
        ax[i,4].tick_params(labelsize=7)

    titles = ['Input + GT', 'ConvNeXt channels\n(top-48 by gate)',
              'Hamiltonian channels\n(top-48 by gate)',
              'Absolute difference\n(unique Ham. info)',
              'Gate distribution\nper channel']
    for j, t in enumerate(titles):
        ax[0, j].set_title(t, fontsize=10, fontweight='bold')

    plt.tight_layout(pad=0.5)
    p = os.path.join(save_dir, 'fig_channel_specialization.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'  Saved: {p}')


# ============================================================
# FIGURE 3: Scan Direction Momentum Maps
# ============================================================
def fig3_scan_directions(model, images, masks, originals, save_dir, device):
    n = min(4, len(images))
    fig, ax = plt.subplots(n, 6, figsize=(22, 3.5*n))
    if n == 1: ax = ax.reshape(1, -1)

    for i in range(n):
        model.eval()
        with torch.no_grad():
            x = images[i].unsqueeze(0).to(device)
            x_s = model.stem(x)
            e1 = model.enc1(x_s)
            e2 = model.enc2(model.down1(e1))
            e3 = model.enc3(model.down2(e2))
            e4 = model.down3(e3)

            for blk in model.bottleneck:
                conv_out = blk.conv_block(e4)
                x_n = blk.norm(e4.permute(0,2,3,1)).permute(0,3,1,2)
                with torch.cuda.amp.autocast(enabled=False):
                    pos, mom, eng = blk.ss2d(x_n.float())
                    ham_out = blk.pos_proj(pos)
                    g = blk.gate(torch.cat([conv_out.float(), ham_out], 1))
                    e4 = (conv_out.float() * g + ham_out * (1 - g)).to(e4.dtype)

            blk = list(model.bottleneck)[-1]
            x_n = blk.norm(e4.permute(0,2,3,1)).permute(0,3,1,2)
            x_f = x_n.float()

            dir_moms = []
            for d in range(4):
                lines, h, w = blk.ss2d._to_lines(x_f, d)
                q, p, e = blk.ss2d.scans[d](lines)
                p_2d = blk.ss2d._to_2d(p, 1, h, w, d)
                mom_d = p_2d[0].norm(dim=0).cpu()
                dir_moms.append(mom_d)

        orig = originals[i] / 255.0
        ax[i,0].imshow(orig)
        if masks[i].max() > 0:
            ax[i,0].contour(masks[i].numpy(), levels=[.5], colors='lime', linewidths=1.5)
        ax[i,0].axis('off')

        combined = torch.stack(dir_moms).mean(dim=0)
        for d in range(4):
            m = up(dir_moms[d]).numpy()
            ax[i, d+1].imshow(pnorm(m), cmap='magma'); ax[i, d+1].axis('off')
        m_c = up(combined).numpy()
        ax[i, 5].imshow(pnorm(m_c), cmap='magma'); ax[i, 5].axis('off')

    titles = ['Input + GT', 'Scan: Row $\\rightarrow$', 'Scan: Row $\\leftarrow$',
              'Scan: Col $\\downarrow$', 'Scan: Col $\\uparrow$', 'Combined $|p|$']
    for j, t in enumerate(titles):
        ax[0, j].set_title(t, fontsize=10, fontweight='bold')

    plt.tight_layout(pad=0.3)
    p = os.path.join(save_dir, 'fig_scan_directions.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'  Saved: {p}')


# ============================================================
# FIGURE 4: Boundary Analysis
# ============================================================
def fig4_boundary_analysis(model, images, masks, originals, save_dir, device):
    from torch.nn.functional import max_pool2d
    n = len(images)
    boundary_moms, interior_moms, exterior_moms = [], [], []

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(n):
        S = extract(model, images[i], device)
        mom = S['mom_raw']
        mom_mag = mom.norm(dim=0)

        msk_s = F.interpolate(masks[i].unsqueeze(0).unsqueeze(0), size=(28,28),
                              mode='nearest')[0,0]
        msk_4d = msk_s.unsqueeze(0).unsqueeze(0)
        dilated = max_pool2d(msk_4d, 3, stride=1, padding=1)[0,0]
        eroded = 1 - max_pool2d(1 - msk_4d, 3, stride=1, padding=1)[0,0]
        boundary = ((dilated - eroded) > 0.5).float()
        interior = (eroded > 0.5).float()
        exterior = ((1 - dilated) > 0.5).float()

        if boundary.sum() > 0:
            boundary_moms.append(mom_mag[boundary > 0.5].mean().item())
        if interior.sum() > 0:
            interior_moms.append(mom_mag[interior > 0.5].mean().item())
        if exterior.sum() > 0:
            exterior_moms.append(mom_mag[exterior > 0.5].mean().item())

    means = [np.mean(interior_moms), np.mean(boundary_moms), np.mean(exterior_moms)]
    stds = [np.std(interior_moms), np.std(boundary_moms), np.std(exterior_moms)]
    colors = ['#534AB7', '#D85A30', '#0F6E56']
    labels = ['Interior\n(lesion)', 'Boundary\n(edge)', 'Exterior\n(skin)']
    bars = ax[0].bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                     edgecolor='white', linewidth=1.5, capsize=5)
    ax[0].set_ylabel('Mean momentum $|p|$', fontsize=12)
    ax[0].set_title('Momentum magnitude by region', fontsize=12, fontweight='bold')
    ax[0].grid(axis='y', alpha=0.3)
    for bar, m in zip(bars, means):
        ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                  f'{m:.1f}', ha='center', fontsize=10, fontweight='bold')

    data = [interior_moms, boundary_moms, exterior_moms]
    bp = ax[1].boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax[1].set_ylabel('Momentum $|p|$ per sample', fontsize=12)
    ax[1].set_title('Distribution across samples', fontsize=12, fontweight='bold')
    ax[1].grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=1.0)
    p = os.path.join(save_dir, 'fig_boundary_analysis.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'  Saved: {p}')


# ============================================================
# FIGURE 5: Multi-Scale Energy Gates + Momentum (NEW for v3)
# ============================================================
def fig5_multiscale_gates(model, images, masks, originals, save_dir, device):
    """
    v3 novelty: energy gates and momentum injection at ALL 3 decoder levels.
    Shows skip gate maps at 3 scales, momentum at coarse and fine levels,
    and the learned energy channel attention weights.
    """
    n = min(4, len(images))
    fig, ax = plt.subplots(n, 7, figsize=(26, 3.5*n))
    if n == 1: ax = ax.reshape(1, -1)

    for i in range(n):
        S = extract(model, images[i], device)
        orig = originals[i] / 255.0

        ax[i,0].imshow(orig)
        if masks[i].max() > 0:
            ax[i,0].contour(masks[i].numpy(), levels=[.5], colors='lime', linewidths=1.5)
        ax[i,0].axis('off')

        # Skip gate maps at 3 levels
        for l in range(3):
            gate_map = S['skip_gates'][l]
            gate_up = up(gate_map).numpy()
            ax[i, l+1].imshow(gate_up, cmap='RdYlGn', vmin=0.3, vmax=0.7)
            if masks[i].max() > 0:
                ax[i, l+1].contour(masks[i].numpy(), levels=[.5],
                                   colors='black', linewidths=1, linestyles='--')
            ax[i, l+1].axis('off')

        # Momentum at d3 (coarsest Hamiltonian signal)
        mom_d3 = S['mom_at_level'][0]
        mom_d3_mag = mom_d3.norm(dim=0)
        mom_d3_up = up(mom_d3_mag).numpy()
        ax[i, 4].imshow(pnorm(mom_d3_up), cmap='magma'); ax[i, 4].axis('off')

        # Momentum at d1 (finest resolution)
        mom_d1 = S['mom_at_level'][2]
        mom_d1_mag = mom_d1.norm(dim=0)
        mom_d1_np = mom_d1_mag.numpy()
        ax[i, 5].imshow(pnorm(mom_d1_np), cmap='magma'); ax[i, 5].axis('off')

        # Energy channel attention weights
        cw = S['ch_weights'].numpy()
        ax[i, 6].bar(range(len(cw)), np.sort(cw)[::-1], color='#BA7517',
                     alpha=0.7, edgecolor='none', width=1.0)
        ax[i, 6].set_xlim(0, len(cw))
        ax[i, 6].set_ylabel('Weight', fontsize=7)
        ax[i, 6].set_xlabel('Channel (sorted)', fontsize=7)
        ax[i, 6].tick_params(labelsize=6)
        top_k = int((cw > np.median(cw) * 1.5).sum())
        ax[i, 6].set_title(f'{top_k} high-weight ch', fontsize=8)

    titles = ['Input + GT',
              'Skip gate $d_3$\n(56×56)',
              'Skip gate $d_2$\n(112×112)',
              'Skip gate $d_1$\n(224×224)',
              'Momentum at $d_3$',
              'Momentum at $d_1$',
              'Energy ch-attn\nweights']
    for j, t in enumerate(titles):
        ax[0, j].set_title(t, fontsize=10, fontweight='bold')

    plt.tight_layout(pad=0.4)
    p = os.path.join(save_dir, 'fig_multiscale_gates.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'  Saved: {p}')


def main():
    parser = argparse.ArgumentParser(description='HamSeg v3 Visualization')
    parser.add_argument('--model_path', default='./outputs_hamseg/isic2018/best_model.pth')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--save_dir', default=None,
                       help='Output dir (default: same as model_path dir)')
    parser.add_argument('--n_samples', type=int, default=6)
    parser.add_argument('--embed_dim', type=int, default=48)
    parser.add_argument('--num_classes', type=int, default=1)
    a = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if a.save_dir is None:
        a.save_dir = os.path.dirname(a.model_path)
    os.makedirs(a.save_dir, exist_ok=True)

    print(f'HamSeg v3 Visualization')
    print(f'  Model: {a.model_path}')
    print(f'  Data:  {a.data_root}')
    print(f'  Output: {a.save_dir}')
    print()

    print('Loading model...')
    model = load_model(a.model_path, device, a.embed_dim, a.num_classes)
    print('Loading samples...')
    images, masks, originals = load_samples(a.data_root, a.n_samples)
    print(f'Generating 5 figures ({len(images)} samples)...\n')

    fig1_physics_signals(model, images, masks, originals, a.save_dir, device)
    fig2_channel_specialization(model, images, masks, originals, a.save_dir, device)
    fig3_scan_directions(model, images, masks, originals, a.save_dir, device)
    fig4_boundary_analysis(model, images, masks, originals, a.save_dir, device)
    fig5_multiscale_gates(model, images, masks, originals, a.save_dir, device)

    print('\nDone! Generated figures:')
    print('  fig_physics_signals.png        - Energy H (ch-attn) + momentum maps')
    print('  fig_channel_specialization.png - ConvNeXt vs Hamiltonian channels')
    print('  fig_scan_directions.png        - Per-direction momentum maps')
    print('  fig_boundary_analysis.png      - Quantitative boundary correlation')
    print('  fig_multiscale_gates.png       - Multi-scale skip gates + momentum (v3 NEW)')


if __name__ == '__main__':
    main()
