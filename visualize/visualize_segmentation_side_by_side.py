#!/usr/bin/env python3
"""Side-by-side qualitative comparison of HamSeg vs its §4.3 ablations.

Columns: GT | HamSeg(full) | ConvNeXt-only (abl_A) | Oscillator-only (abl_B).
Strategies: full_wins (default), disagreement, quantile.
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
for p in [HERE, HERE.parent / 'hamvision', HERE.parent / 'src',
          Path('/mnt/c/Users/Z/Dropbox/claude_work_to_share/HamVision_V2/hamvision')]:
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

visfig4 = None
for m in ('visualize_fig4_qualitative', 'visualize_segmentation'):
    try:
        visfig4 = __import__(m); break
    except ImportError: continue
if visfig4 is None:
    raise ImportError('Need visualize_fig4_qualitative.py or visualize_segmentation.py on path')

predict, pick_samples = visfig4.predict, visfig4.pick_samples
dice_score, normalise_image_array = visfig4.dice_score, visfig4.normalise_image_array


def load_model(ckpt_path, device, embed_dim=48, num_classes=1,
               ablation='none', img_size=224, drop_rate=0.1, damping_clamp=5.0):
    for mod in list(sys.modules.keys()):
        if mod == 'hamseg' or mod.startswith('hamseg.'):
            del sys.modules[mod]
    from hamseg import HamSeg
    class _A: pass
    a = _A()
    a.embed_dim = embed_dim; a.depths = [2, 2, 2, 2]
    a.damping_clamp = damping_clamp; a.num_classes = num_classes
    a.img_size = img_size; a.drop_rate = drop_rate; a.ablation = ablation
    model = HamSeg(a).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(torch.is_tensor(v) and v.dtype == torch.float16 for v in state.values()):
        state = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
                 for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class DatasetBundle:
    def __init__(self, name, label, data_root, num_classes, img_size,
                 full_ckpt, abl_a_ckpt=None, abl_b_ckpt=None,
                 embed_dim=48, n_samples=3, seed=42, device='cuda',
                 pick_strategy='full_wins', candidate_pool=80):
        self.name, self.label = name, label
        self.num_classes, self.n_samples = num_classes, n_samples
        self.device = device

        amap = {'full': 'none', 'abl_a': 'A', 'abl_b': 'B'}
        self.variants = {}
        for tag, ckpt in [('full', full_ckpt), ('abl_a', abl_a_ckpt), ('abl_b', abl_b_ckpt)]:
            if ckpt and Path(ckpt).exists():
                self.variants[tag] = load_model(
                    ckpt, device, embed_dim=embed_dim, num_classes=num_classes,
                    ablation=amap[tag], img_size=img_size)
                print(f'  [{name}] loaded {tag} (ablation={amap[tag]}): {ckpt}')
            else:
                self.variants[tag] = None
                if ckpt:
                    print(f'  [{name}] WARNING: {tag} ckpt missing: {ckpt}')

        if pick_strategy in ('disagreement', 'full_wins'):
            self.samples = self._pick_scored(data_root, img_size, seed,
                                             candidate_pool, pick_strategy)
        else:
            self.samples = pick_samples(data_root, n_samples=n_samples,
                                        img_size=img_size, seed=seed, verbose=False)
        self._pred_cache = {}

    def _pick_scored(self, data_root, img_size, seed, candidate_pool, strategy):
        cands = pick_samples(data_root, n_samples=candidate_pool,
                             img_size=img_size, seed=seed, verbose=False)
        active = [t for t, m in self.variants.items() if m is not None]
        if len(active) < 2 or 'full' not in active:
            return cands[:self.n_samples]
        others = [t for t in active if t != 'full']
        scored = []
        for img_t, mask_t, orig_np, fname in cands:
            gt_np = mask_t.numpy() if isinstance(mask_t, torch.Tensor) else np.array(mask_t)
            if gt_np.ndim == 3:
                gt_np = gt_np.squeeze(0) if gt_np.shape[0] == 1 else gt_np.argmax(0)
            gt_bin = (gt_np > 0).astype(np.uint8) if self.num_classes > 1 else gt_np.astype(np.uint8)
            dices = {}
            for tag in active:
                pred = predict(self.variants[tag], img_t, self.device,
                               num_classes=self.num_classes).numpy()
                pred_bin = (pred > 0).astype(np.uint8) if self.num_classes > 1 else pred.astype(np.uint8)
                dices[tag] = dice_score(pred_bin, gt_bin)
            if dices['full'] < 0.50:
                continue
            if strategy == 'full_wins':
                score = dices['full'] - max(dices[t] for t in others)
            else:
                score = max(dices.values()) - min(dices.values())
            scored.append((score, dices, (img_t, mask_t, orig_np, fname)))
        if not scored:
            return cands[:self.n_samples]
        scored.sort(key=lambda x: -x[0])
        chosen = [s[2] for s in scored[:self.n_samples]]
        for sc, dices, (_, _, _, fname) in scored[:self.n_samples]:
            print(f'    [{self.name}] picked {fname}: score={sc:+.3f}, '
                  f'dices={ {k: round(v, 3) for k, v in dices.items()} }')
        return chosen

    def predict_all(self):
        if self._pred_cache:
            return self._pred_cache
        preds = {}
        for tag, model in self.variants.items():
            if model is None:
                preds[tag] = [None] * len(self.samples); continue
            row = []
            for img_t, mask_t, orig_np, fname in self.samples:
                pred = predict(model, img_t, self.device,
                               num_classes=self.num_classes).numpy()
                gt_np = mask_t.numpy() if isinstance(mask_t, torch.Tensor) else np.array(mask_t)
                if gt_np.ndim == 3:
                    gt_np = gt_np.squeeze(0) if gt_np.shape[0] == 1 else gt_np.argmax(0)
                pred_bin = (pred > 0).astype(np.uint8) if self.num_classes > 1 else pred.astype(np.uint8)
                gt_bin = (gt_np > 0).astype(np.uint8) if self.num_classes > 1 else gt_np.astype(np.uint8)
                row.append((pred, gt_np, dice_score(pred_bin, gt_bin)))
            preds[tag] = row
        self._pred_cache = preds
        return preds


VARIANT_ORDER = ['full', 'abl_a', 'abl_b']
COL_TITLES = ['Input + GT', 'HamSeg (full)', 'ConvNeXt-only', 'Oscillator-only']


def _overlay(ax, img, mask, color, lw=1.8):
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
    if mask is not None and np.any(mask > 0):
        ax.contour(mask > 0, levels=[0.5], colors=[color], linewidths=[lw])
    ax.set_xticks([]); ax.set_yticks([])


def build_panel(bundles, save_path, n_cols=4):
    n_rows = sum(b.n_samples for b in bundles)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    GT, PRED, NA = '#00b04f', '#ff8c1a', (0.92, 0.92, 0.92)
    row = 0
    for bundle in bundles:
        preds = bundle.predict_all()
        for i in range(len(bundle.samples)):
            img_t, mask_t, orig_np, fname = bundle.samples[i]
            img_u = normalise_image_array(orig_np)
            gt_np = mask_t.numpy() if isinstance(mask_t, torch.Tensor) else np.array(mask_t)
            if gt_np.ndim == 3:
                gt_np = gt_np.squeeze(0) if gt_np.shape[0] == 1 else gt_np.argmax(0)
            ax = axes[row, 0]
            _overlay(ax, img_u, gt_np, GT)
            if i == 0: ax.set_title('Input + GT', fontsize=11)
            ax.set_ylabel(bundle.label, fontsize=11, rotation=0,
                          ha='right', va='center', labelpad=24)
            for c, tag in enumerate(VARIANT_ORDER, start=1):
                ax = axes[row, c]
                entry = preds[tag][i]
                if entry is None:
                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                               facecolor=NA, edgecolor='none'))
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                            ha='center', va='center', fontsize=14, color='gray')
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    pred_np, _, d = entry
                    _overlay(ax, img_u, pred_np, PRED)
                    ax.text(0.02, 0.96, f'Dice = {d:.3f}',
                            transform=ax.transAxes, ha='left', va='top',
                            fontsize=9, color='white',
                            bbox=dict(facecolor='black', alpha=0.55,
                                      pad=2, edgecolor='none'))
                if i == 0: ax.set_title(COL_TITLES[c], fontsize=11)
            row += 1
    handles = [Line2D([0], [0], color=GT,   lw=2, label='Ground truth'),
               Line2D([0], [0], color=PRED, lw=2, label='Prediction')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.04)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(save_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {save_path.with_suffix(".pdf")} and {save_path.with_suffix(".png")}')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--acdc_full_ckpt');  p.add_argument('--acdc_abl_a_ckpt')
    p.add_argument('--acdc_abl_b_ckpt'); p.add_argument('--acdc_root')
    p.add_argument('--acdc_num_classes', type=int, default=4)
    p.add_argument('--isic18_full_ckpt');  p.add_argument('--isic18_abl_a_ckpt')
    p.add_argument('--isic18_abl_b_ckpt'); p.add_argument('--isic18_root')
    p.add_argument('--isic18_num_classes', type=int, default=1)
    p.add_argument('--n_samples', type=int, default=3)
    p.add_argument('--pick_strategy', choices=['full_wins', 'disagreement', 'quantile'],
                   default='full_wins')
    p.add_argument('--candidate_pool', type=int, default=80)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--embed_dim', type=int, default=48)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_dir', default='./outputs/fig4_side_by_side')
    p.add_argument('--save_name', default='fig4_side_by_side')
    return p.parse_args()


def main():
    a = get_args()
    bundles = []
    if a.acdc_root and a.acdc_full_ckpt:
        print('Building ACDC bundle ...')
        bundles.append(DatasetBundle(
            'acdc', 'ACDC', a.acdc_root, a.acdc_num_classes, a.img_size,
            a.acdc_full_ckpt, a.acdc_abl_a_ckpt, a.acdc_abl_b_ckpt,
            a.embed_dim, a.n_samples, a.seed, a.device,
            a.pick_strategy, a.candidate_pool))
    if a.isic18_root and a.isic18_full_ckpt:
        print('Building ISIC2018 bundle ...')
        bundles.append(DatasetBundle(
            'isic2018', 'ISIC 2018', a.isic18_root, a.isic18_num_classes, a.img_size,
            a.isic18_full_ckpt, a.isic18_abl_a_ckpt, a.isic18_abl_b_ckpt,
            a.embed_dim, a.n_samples, a.seed, a.device,
            a.pick_strategy, a.candidate_pool))
    if not bundles:
        raise SystemExit('Need at least one of --acdc_full_ckpt / --isic18_full_ckpt.')
    build_panel(bundles, Path(a.save_dir) / a.save_name)
    print('done.')


if __name__ == '__main__':
    main()
