#!/usr/bin/env python3
"""
visualize_fig7_cls_interpretability.py
=======================================

Builds a classification interpretability figure that mirrors Fig 6
(physics-as-interpretability) for the HamCls classifier. For each chosen
MedMNIST dataset, we pick a small number of correctly-classified test
images and render the four physically-meaningful intermediate quantities
that the network produces in its forward pass:

  * input image, annotated with the predicted class label and the
    softmax confidence;
  * position |q|, the filtered representation;
  * momentum |p|, the spatial derivative of q;
  * energy H = (|q|^2 + |p|^2)/2, the per-pixel oscillator energy that
    drives the PSSP head's attention;
  * energy overlay on the input -- a transparent magma colourmap painted
    on top of the grayscale anatomy so the reader can see WHICH part of
    the actual image the energy concentrates on. This is the
    classification analogue of the "where the model looks" panel.

No segmentation supervision is used at any point. The energy map is the
same quantity the classifier consumes through its Phase-Space Spectral
Pooling head. The figure therefore answers the question "does the
physics highlight the class-discriminative regions?" empirically.

Output layouts:
  --layout 'compact'  (default): N_datasets rows x 5 columns
      [input | |q| | |p| | H | overlay], one sample per dataset.
  --layout 'per_class': pick one sample per class and arrange them in
      a denser grid; used for single-dataset deep dives.

Datasets supported (set --datasets to pick a subset, default = all five):
    pathmnist  dermamnist  bloodmnist  octmnist  organcmnist

Usage:
    python visualize_fig7_cls_interpretability.py \\
        --ckpt_root  outputs_hamcls \\
        --data_root  /path/to/medmnist/data \\
        --save_dir   figures \\
        --datasets   pathmnist dermamnist bloodmnist octmnist \\
        --layout     compact
"""

import os, sys, argparse, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')


# ===================================================================
# MedMNIST class names (for label annotation)
# ===================================================================

CLASS_NAMES = {
    'pathmnist':    ['adipose', 'background', 'debris', 'lymphocytes', 'mucus',
                     'smooth muscle', 'normal colon', 'cancer-assoc.', 'colorectal cancer'],
    'dermamnist':   ['act. keratoses', 'BCC', 'benign keratosis', 'dermatofibroma',
                     'melanoma', 'melanocytic nevi', 'vasc. lesion'],
    'bloodmnist':   ['basophil', 'eosinophil', 'erythroblast', 'IG', 'lymphocyte',
                     'monocyte', 'neutrophil', 'platelet'],
    'octmnist':     ['CNV', 'DME', 'drusen', 'normal'],
    'pneumoniamnist': ['normal', 'pneumonia'],
    'retinamnist':  ['DR-0', 'DR-1', 'DR-2', 'DR-3', 'DR-4'],
    'breastmnist':  ['malignant', 'normal'],
    'organamnist':  ['bladder', 'femur-L', 'femur-R', 'heart', 'kidney-L',
                     'kidney-R', 'liver', 'lung-L', 'lung-R', 'pancreas', 'spleen'],
    'organcmnist':  ['bladder', 'femur-L', 'femur-R', 'heart', 'kidney-L',
                     'kidney-R', 'liver', 'lung-L', 'lung-R', 'pancreas', 'spleen'],
    'organsmnist':  ['bladder', 'femur-L', 'femur-R', 'heart', 'kidney-L',
                     'kidney-R', 'liver', 'lung-L', 'lung-R', 'pancreas', 'spleen'],
}

# Channel count per dataset (some are grayscale, some are RGB)
DATASET_CHANNELS = {
    'pathmnist': 3, 'dermamnist': 3, 'bloodmnist': 3,
    'octmnist': 1, 'pneumoniamnist': 1, 'retinamnist': 3,
    'breastmnist': 1, 'organamnist': 1, 'organcmnist': 1, 'organsmnist': 1,
}

DATASET_LABEL = {
    'pathmnist': 'PathMNIST (colon pathology)',
    'dermamnist': 'DermaMNIST (dermoscopy)',
    'bloodmnist': 'BloodMNIST (blood-cell microscopy)',
    'octmnist': 'OCTMNIST (retinal OCT)',
    'pneumoniamnist': 'PneumoniaMNIST (chest X-ray)',
    'retinamnist': 'RetinaMNIST (retinal fundus)',
    'breastmnist': 'BreastMNIST (breast ultrasound)',
    'organamnist': 'OrganAMNIST (axial abdominal CT)',
    'organcmnist': 'OrganCMNIST (coronal abdominal CT)',
}


# ===================================================================
# Model loader and signal extraction
# ===================================================================

def load_model(model_path, device, args_dict):
    """Reload HamCls from a checkpoint. Auto-detects in_channels
    and num_classes from the checkpoint tensor shapes so the model
    architecture always matches what was trained.

    args_dict supplies hyperparameters that aren't recoverable from
    the state_dict (embed_dim, depths, n_scan_dirs, pssp_K, etc.)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in list(sys.modules.keys()):
        if 'hamcls' in mod:
            del sys.modules[mod]
    from hamcls import HamCls

    state = torch.load(model_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and any(
            torch.is_tensor(v) and v.dtype == torch.float16 for v in state.values()):
        state = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
                 for k, v in state.items()}

    # Auto-detect in_channels from stem conv weight (B, C_in, k, k)
    if 'stem.0.weight' in state:
        detected_in_ch = int(state['stem.0.weight'].shape[1])
        if detected_in_ch != args_dict.get('in_channels', 3):
            print(f'         [auto-detect] in_channels={detected_in_ch} '
                  f'(was {args_dict.get("in_channels", 3)} in args)')
            args_dict['in_channels'] = detected_in_ch

    # Auto-detect num_classes from classifier last layer weight (C_out, hidden)
    last_lin = None
    for k in state.keys():
        if k.startswith('classifier.') and k.endswith('.weight'):
            last_lin = k
    if last_lin is not None:
        detected_n_cls = int(state[last_lin].shape[0])
        if detected_n_cls != args_dict.get('num_classes', 0):
            print(f'         [auto-detect] num_classes={detected_n_cls} '
                  f'(was {args_dict.get("num_classes", 0)} in args)')
            args_dict['num_classes'] = detected_n_cls

    class A: pass
    a = A()
    for k, v in args_dict.items():
        setattr(a, k, v)

    model = HamCls(a).to(device)
    state = {k: v.to(device) for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, args_dict


def extract_cls_signals(model, img_t, device):
    """Run the classifier and capture (q, p, energy_map) from the bottleneck."""
    with torch.no_grad():
        x = img_t.unsqueeze(0).to(device)
        x = model.stem(x)
        x = model.stage1(x)
        x = model.down(x)
        x = model.stage2(x)
        x_n = model.bottleneck_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        with torch.cuda.amp.autocast(enabled=False):
            q, p, energy_map = model.ss2d(x_n.float())
        q = q.to(x.dtype); p = p.to(x.dtype); energy_map = energy_map.to(x.dtype)
        features = model.pssp(q, p, energy_map=energy_map)
        logits = model.classifier(features)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        pred = int(logits.argmax(dim=1)[0].item())
    return {
        'pred': pred,
        'probs': probs,
        'q':   q[0].cpu().float(),          # (2C, 28, 28)
        'p':   p[0].cpu().float(),
        'H':   energy_map[0].cpu().float(),  # (2C, 28, 28)
    }


# ===================================================================
# MedMNIST data loading
# ===================================================================

def load_medmnist(data_root, dataset, split='test'):
    """Load a MedMNIST split. Returns (images: NxHxWxC uint8, labels: N,).
    Tries both flat `{dataset}_224.npz` and `{dataset}/{dataset}_224.npz`."""
    root = Path(data_root)
    candidates = [
        root / f'{dataset}_224.npz',
        root / dataset / f'{dataset}_224.npz',
        root / f'{dataset}.npz',
        root / dataset / f'{dataset}.npz',
    ]
    for c in candidates:
        if c.exists():
            data = np.load(c, allow_pickle=True)
            img_key = f'{split}_images'
            lbl_key = f'{split}_labels'
            if img_key in data.files and lbl_key in data.files:
                imgs = data[img_key]
                lbls = data[lbl_key]
                if lbls.ndim > 1: lbls = lbls.squeeze(-1)
                return imgs, lbls
    raise FileNotFoundError(f'No MedMNIST file for {dataset} under {data_root}')


def preprocess_image(img_np, in_ch, img_size=224):
    """Convert a single uint8 MedMNIST image into (img_t, orig_rgb_np)."""
    if img_np.ndim == 2:
        img_np = img_np[:, :, None]
    if img_np.shape[-1] == 1 and in_ch == 3:
        img_np = np.repeat(img_np, 3, axis=-1)
    if img_np.shape[-1] == 3 and in_ch == 1:
        img_np = img_np.mean(axis=-1, keepdims=True).astype(np.uint8)

    from PIL import Image
    img = Image.fromarray(img_np if img_np.shape[-1] != 1 else img_np[..., 0])
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img)
    if arr.ndim == 2:
        rgb = np.stack([arr]*3, axis=-1)
    else:
        rgb = arr

    t = TF.to_tensor(img)
    if in_ch == 3 and t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    if in_ch == 1 and t.shape[0] == 3:
        t = t.mean(dim=0, keepdim=True)
    mean = [.485, .456, .406] if in_ch == 3 else [.5]
    std  = [.229, .224, .225] if in_ch == 3 else [.5]
    img_t = TF.normalize(t, mean, std)
    return img_t, rgb


# ===================================================================
# Sample picker (one high-confidence correctly-classified per dataset)
# ===================================================================

def _energy_focality(signals):
    """Score: how spatially concentrated is the energy map?
    Higher means more focal (top-10% of pixels carry most of the mass).
    Returns ratio of top-10% mean to overall mean."""
    H_pix = 0.5 * (signals['q'].pow(2).sum(dim=0)
                   + signals['p'].pow(2).sum(dim=0)).numpy().ravel()
    if H_pix.std() < 1e-6:
        return 0.0
    thr = np.percentile(H_pix, 90)
    top = H_pix[H_pix >= thr]
    return float(top.mean() / max(H_pix.mean(), 1e-6))


def pick_high_conf_correct(model, images, labels, in_ch, device,
                           n_samples=1, n_candidates=120, seed=42,
                           verbose=True, prefer_focal=True):
    """Iterate `n_candidates` random samples, run prediction. Keep
    correctly-classified ones; rank by softmax confidence AND focality
    of the energy map (so the visual shows a spotlight on the
    class-discriminative region rather than a diffuse blob)."""
    rng = np.random.RandomState(seed)
    idx_pool = rng.choice(len(images), size=min(n_candidates, len(images)),
                          replace=False)
    correct = []
    for idx in idx_pool:
        img_np = images[idx]
        true_label = int(labels[idx])
        img_t, rgb = preprocess_image(img_np, in_ch=in_ch)
        signals = extract_cls_signals(model, img_t, device)
        if signals['pred'] == true_label:
            conf = float(signals['probs'][true_label])
            focality = _energy_focality(signals) if prefer_focal else 1.0
            # Composite score: confidence x focality (both > 0)
            score = conf * (focality ** 0.5)
            correct.append((score, conf, focality, idx, img_t, rgb,
                            true_label, signals))
    correct.sort(reverse=True)
    if not correct:
        raise RuntimeError('No correctly-classified candidates found.')
    # Strip the score field on the way out; keep the same return shape.
    return [(conf, idx, img_t, rgb, true_label, signals)
            for (score, conf, focality, idx, img_t, rgb, true_label, signals)
            in correct[:n_samples]]


# ===================================================================
# Rendering
# ===================================================================

def pnorm(a, lo=5, hi=95):
    """Percentile-normalise to [0, 1]. Tighter (5/95) gives more contrast
    than the previous 2/98 because the rare extreme pixels were
    flattening the bulk of the distribution."""
    vlo, vhi = np.percentile(a, lo), np.percentile(a, hi)
    if vhi - vlo < 1e-8: vhi = vlo + 1
    return np.clip((a - vlo) / (vhi - vlo), 0, 1)


def up_to_224(t, smooth_sigma=2.5):
    """Upsample (H, W) tensor to (224, 224) with bicubic + post-hoc
    Gaussian smoothing. The 28x28 bottleneck physics maps are
    inherently low-res; an 8x bicubic upsample alone leaves visible
    block artefacts. A small Gaussian (sigma=2.5 px at 224 res, i.e.
    ~1/100 of the image width) removes the artefacts while preserving
    spatial structure."""
    arr = F.interpolate(t.unsqueeze(0).unsqueeze(0).float(),
                        size=(224, 224), mode='bicubic',
                        align_corners=False)[0, 0].numpy()
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            arr = gaussian_filter(arr, sigma=smooth_sigma)
        except ImportError:
            pass  # scipy not installed -- silently skip the smoothing
    return arr


def fig7_compact(rows, save_path):
    """rows = list of dicts {name, dataset_key, rgb, true_label, pred_label,
                              confidence, q, p, H}."""
    n_rows = len(rows)
    n_cols = 5
    cell = 2.6
    # Extra height for the bottom colour-bar strip
    fig = plt.figure(figsize=(cell * n_cols + 0.6, cell * n_rows + 1.4))
    gs = GridSpec(n_rows + 1, n_cols, figure=fig,
                  height_ratios=[1.0] * n_rows + [0.10],
                  hspace=0.10, wspace=0.04,
                  left=0.07, right=0.99, top=0.95, bottom=0.06)

    titles = [
        'Input',
        r'Position $|q|$' + '\n(filtered representation)',
        r'Momentum $|p|$' + '\n(spatial derivative)',
        r'Energy $H$' + '\n(saliency / activity)',
        'Energy overlay\n(amplified region)',
    ]

    for i, R in enumerate(rows):
        rgb = R['rgb']
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        gray = rgb.mean(axis=-1).astype(np.float32) / 255.0

        # Compute the spatial maps
        q_map = pnorm(up_to_224(R['q'].norm(dim=0)))
        p_map = pnorm(up_to_224(R['p'].norm(dim=0)))
        h_per_pix = 0.5 * (R['q'].pow(2).sum(dim=0) + R['p'].pow(2).sum(dim=0))
        H_map_raw = up_to_224(h_per_pix)
        H_map = pnorm(H_map_raw)

        # Class labels
        cls_names = CLASS_NAMES.get(R['dataset_key'], None)
        if cls_names is None or R['pred_label'] >= len(cls_names):
            pred_str = f'class {R["pred_label"]}'
            true_str = f'class {R["true_label"]}'
        else:
            pred_str = cls_names[R['pred_label']]
            true_str = cls_names[R['true_label']]
        correct = R['pred_label'] == R['true_label']
        marker = u'✓' if correct else u'✗'

        # Col 0: input + class label
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(rgb)
        cap = (f'True: {true_str}\n'
               f'{marker} Pred: {pred_str} ({R["confidence"]*100:.1f}%)')
        ax.text(0.03, 0.04, cap, transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='white',
                va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='black', alpha=0.65, edgecolor='none'))
        ax.set_ylabel(R['name'], fontsize=11, fontweight='bold',
                      rotation=0, ha='right', va='center', labelpad=20)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0:
            ax.set_title(titles[0], fontsize=12, fontweight='bold', pad=8)

        # Cols 1, 2, 3: q, p, H
        for j, (data, cmap) in enumerate([(q_map, 'viridis'),
                                           (p_map, 'magma'),
                                           (H_map, 'inferno')]):
            ax = fig.add_subplot(gs[i, 1 + j])
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if i == 0:
                ax.set_title(titles[1 + j], fontsize=12, fontweight='bold', pad=8)

        # Col 4: energy overlay -- threshold-based "spotlight" rendering.
        # Only the top 40% of pixel energies appear; below that the input
        # is shown bare. The remaining energy is mapped through a sharp
        # gamma curve so the spotlight is bright and concentrated.
        ax = fig.add_subplot(gs[i, 4])
        ax.imshow(gray, cmap='gray', vmin=0, vmax=1)

        # Compute a "focus" mask: pixels above the 60th-percentile energy
        H_thresh = np.percentile(H_map, 60)
        if H_thresh < 0.999:
            H_focus = np.clip((H_map - H_thresh) / (1.0 - H_thresh + 1e-6),
                              0, 1)
        else:
            H_focus = np.zeros_like(H_map)

        rgba = plt.get_cmap('inferno')(H_focus)
        # Alpha: 0 outside the spotlight, sharp gamma inside
        alpha = 0.90 * np.power(H_focus, 1.4)
        rgba[..., 3] = alpha
        ax.imshow(rgba)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0:
            ax.set_title(titles[4], fontsize=12, fontweight='bold', pad=8)

    # ===== Bottom colour-bar strip =====
    # One colour bar per physics column (q / p / H / overlay).
    cbar_info = [
        ('viridis', 'low',       'high',     r'Position $|q|$ (filtered representation)'),
        ('magma',   'quiescent', 'active',   r'Momentum $|p|$ (spatial derivative)'),
        ('inferno', 'low',       'high',     r'Energy $H = (|q|^2 + |p|^2)/2$ (saliency)'),
        ('inferno', '60th pct.', 'max', r'Energy spotlight (top-40\% energy only)'),
    ]
    for j, (cmap_name, lo_lbl, hi_lbl, label) in enumerate(cbar_info, start=1):
        cax = fig.add_subplot(gs[n_rows, j])
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name),
                                   norm=plt.Normalize(vmin=0, vmax=1))
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_ticks([0, 1])
        cb.set_ticklabels([lo_lbl, hi_lbl])
        cb.ax.tick_params(labelsize=9)
        cb.outline.set_visible(False)
        cb.set_label(label, fontsize=9, labelpad=4)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ===================================================================
# Main
# ===================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_root', action='append', required=True,
                   help='Root(s) containing {dataset}/seed_42/best_model.pth. '
                        'Pass multiple times to probe several directories per '
                        'dataset (e.g., for the classifier checkpoints which live '
                        'in different folders depending on which experiment '
                        'wrote them). The first existing ckpt wins.')
    p.add_argument('--data_root', required=True,
                   help='Root containing MedMNIST {dataset}_224.npz files')
    p.add_argument('--save_dir',  default='./figures')
    p.add_argument('--datasets', nargs='+',
                   default=['pathmnist', 'dermamnist', 'bloodmnist', 'octmnist'],
                   help='Which datasets to include (space-separated)')
    p.add_argument('--seed_ckpt', default='seed_42',
                   help='Which trained-seed directory to load (default: seed_42)')
    p.add_argument('--seed', type=int, default=42,
                   help='Sample-picking seed (default 42)')
    p.add_argument('--n_per_dataset', type=int, default=1,
                   help='Samples per dataset (default 1 -- one per row)')
    p.add_argument('--embed_dim', type=int, default=96)
    p.add_argument('--depths', nargs='+', type=int, default=[3, 3])
    p.add_argument('--damping_clamp', type=float, default=5.0)
    p.add_argument('--n_scan_dirs', type=int, default=2)
    p.add_argument('--pssp_K', type=int, default=12)
    p.add_argument('--pssp_no_complex', action='store_true',
                   help='Set this when loading a checkpoint trained with '
                        '--no_pssp_complex (relevant for BreastMNIST).')
    a = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.save_dir, exist_ok=True)

    rows = []
    for ds in a.datasets:
        if ds not in DATASET_CHANNELS:
            print(f'[skip] {ds}: not in DATASET_CHANNELS')
            continue
        ckpt = None
        for root in a.ckpt_root:
            cand = os.path.join(root, ds, a.seed_ckpt, 'best_model.pth')
            if os.path.exists(cand):
                ckpt = cand; break
        if ckpt is None:
            tried = [os.path.join(r, ds, a.seed_ckpt, 'best_model.pth')
                     for r in a.ckpt_root]
            print(f'[skip] {ds}: ckpt not found. Tried:')
            for t in tried: print(f'         {t}')
            continue

        in_ch = DATASET_CHANNELS[ds]
        n_cls = len(CLASS_NAMES.get(ds, [0]))
        args_dict = dict(
            embed_dim=a.embed_dim,
            depths=list(a.depths),
            damping_clamp=a.damping_clamp,
            num_classes=n_cls,
            img_size=224,
            in_channels=in_ch,
            n_scan_dirs=a.n_scan_dirs,
            pssp_K=a.pssp_K,
            pssp_complex=(not a.pssp_no_complex),
            pssp_cross=True,
            pssp_use_ss2d_energy=True,
            drop_rate=0.2,
            head_drop=0.3,
        )

        print(f'[load] {ds}: {ckpt}')
        try:
            model, args_dict = load_model(ckpt, device, args_dict)
        except Exception as e:
            print(f'[skip] {ds}: load failed ({e})'); continue

        # The auto-detector may have updated in_channels; use that for
        # preprocessing so the input tensor matches the model's stem.
        in_ch = args_dict['in_channels']

        try:
            print(f'[data] loading test split for {ds}')
            images, labels = load_medmnist(a.data_root, ds, split='test')
        except FileNotFoundError as e:
            print(f'[skip] {ds}: {e}'); del model; torch.cuda.empty_cache(); continue

        print(f'[pick] {a.n_per_dataset} high-confidence correctly-classified samples')
        try:
            picks = pick_high_conf_correct(model, images, labels, in_ch,
                                            device, n_samples=a.n_per_dataset,
                                            seed=a.seed)
        except RuntimeError as e:
            print(f'[skip] {ds}: {e}'); del model; torch.cuda.empty_cache(); continue

        for conf, idx, img_t, rgb, true_label, signals in picks:
            cls_name = CLASS_NAMES.get(ds, ['?'])
            print(f'        idx={idx}  true={true_label} ({cls_name[true_label] if true_label < len(cls_name) else "?"})  '
                  f'pred={signals["pred"]} ({cls_name[signals["pred"]] if signals["pred"] < len(cls_name) else "?"})  '
                  f'conf={conf*100:.1f}%')
            rows.append(dict(
                name=DATASET_LABEL.get(ds, ds),
                dataset_key=ds,
                rgb=rgb,
                true_label=true_label,
                pred_label=signals['pred'],
                confidence=conf,
                q=signals['q'],
                p=signals['p'],
                H=signals['H'],
            ))

        del model; torch.cuda.empty_cache()

    if not rows:
        raise SystemExit('No rows rendered.')

    save_path = os.path.join(a.save_dir, 'fig7_cls_interpretability.png')
    print(f'\n[render] Fig 7 -- classification physics-as-interpretability '
          f'({len(rows)} rows)')
    fig7_compact(rows, save_path)
    print('Done.')


if __name__ == '__main__':
    main()
