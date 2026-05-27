#!/usr/bin/env python3
"""Grad-CAM vs intrinsic energy map. Reports pointing-game + top-20% IoU."""
from __future__ import annotations
import argparse, csv, sys, warnings
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch
import torch.nn.functional as F
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
for p in [HERE, HERE.parent/'hamvision', HERE.parent/'src',
          Path('/mnt/c/Users/Z/Dropbox/claude_work_to_share/HamVision_V2/hamvision')]:
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))
vf = None
for m in ('visualize_fig4_qualitative', 'visualize_segmentation'):
    try: vf = __import__(m); break
    except ImportError: continue
pick_samples = vf.pick_samples
normimg = vf.normalise_image_array
dice_score = vf.dice_score


def load_model(ckpt, device, ed=48, nc=1, ims=224):
    for mod in list(sys.modules.keys()):
        if mod == 'hamseg' or mod.startswith('hamseg.'): del sys.modules[mod]
    from hamseg import HamSeg, HamiltonianBottleneck
    class _A: pass
    a = _A(); a.embed_dim = ed; a.depths = [2, 2, 2, 2]
    a.damping_clamp = 5.0; a.num_classes = nc; a.img_size = ims
    a.drop_rate = 0.1; a.ablation = 'none'
    model = HamSeg(a).to(device)
    st = torch.load(ckpt, map_location=device, weights_only=True)
    if any(torch.is_tensor(v) and v.dtype == torch.float16 for v in st.values()):
        st = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
              for k, v in st.items()}
    model.load_state_dict(st, strict=True); model.eval()
    return model, HamiltonianBottleneck


def last_bn(model, BC):
    last = None
    for n, m in model.named_modules():
        if isinstance(m, BC): last = (n, m)
    return last


def n01(x):
    x = np.asarray(x, np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn) if mx - mn > 1e-8 else x * 0.0


def run_one(model, tgt, img_t, dev, nc=1):
    acts, grads, ens = [], [], []
    def fh(m, i, o):
        if isinstance(o, (tuple, list)):
            acts.append(o[0])
            if len(o) >= 3 and o[2] is not None: ens.append(o[2].detach())
        else: acts.append(o)
    def bh(m, gi, go):
        for g in go:
            if g is not None: grads.append(g); return
    h1 = tgt.register_forward_hook(fh)
    h2 = tgt.register_full_backward_hook(bh)
    try:
        model.zero_grad()
        x = img_t.unsqueeze(0).to(dev)
        lo = model(x).float()
        if nc > 1:
            mk = (lo.argmax(dim=1) > 0).float()
            t = (lo[:, 1:].sum(dim=1) * mk).sum()
        else:
            mk = (torch.sigmoid(lo) > 0.5).float()
            t = (lo * mk).sum()
        t.backward()
    finally:
        h1.remove(); h2.remove()
    w = grads[0].detach().mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts[0].detach()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=img_t.shape[-2:], mode='bilinear', align_corners=False)
    cam = n01(cam[0, 0].cpu().numpy())
    en = None
    if ens:
        e = F.interpolate(ens[0].float(), size=img_t.shape[-2:],
                          mode='bilinear', align_corners=False)
        en = n01(e[0, 0].cpu().numpy())
    with torch.no_grad():
        lo = model(img_t.unsqueeze(0).to(dev)).float()
        pred = (lo.argmax(dim=1)[0].cpu().numpy() if nc > 1
                else (torch.sigmoid(lo) > 0.5)[0, 0].cpu().numpy())
    return cam, en, pred


def pg(h, gt):
    if gt.sum() == 0: return None
    y, x = np.unravel_index(int(h.argmax()), h.shape)
    return int(bool(gt[y, x] > 0))


def tiou(h, gt, frac=0.20):
    if gt.sum() == 0: return None
    t = (h >= np.quantile(h, 1.0 - frac)).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    i, u = (t & g).sum(), (t | g).sum()
    return float(i) / float(u) if u > 0 else None


def render(rows, sp):
    n = len(rows); nc = 4 if any(r['en'] is not None for r in rows) else 3
    fig, ax = plt.subplots(n, nc, figsize=(3.5*nc, 3.3*n), squeeze=False)
    GT, PR = '#00b04f', '#ff8c1a'
    for i, r in enumerate(rows):
        a = ax[i, 0]; a.imshow(r['img'])
        a.contour(r['gt'] > 0, levels=[0.5], colors=[GT], linewidths=[1.8])
        a.set_title('Input + GT', fontsize=10); a.set_xticks([]); a.set_yticks([])
        a = ax[i, 1]; a.imshow(r['img'])
        a.contour(r['pred'] > 0, levels=[0.5], colors=[PR], linewidths=[1.8])
        a.set_title(f"Pred (D={r['dice']:.3f})", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        a = ax[i, 2]; a.imshow(r['img']); a.imshow(r['cam'], cmap='jet', alpha=0.5)
        a.contour(r['gt'] > 0, levels=[0.5], colors=[GT], linewidths=[1.6])
        a.set_title(f"Grad-CAM pg={r['cpg']} IoU={r['ciou']:.3f}", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        if nc == 4:
            a = ax[i, 3]; a.imshow(r['img']); a.imshow(r['en'], cmap='magma', alpha=0.55)
            a.contour(r['gt'] > 0, levels=[0.5], colors=[GT], linewidths=[1.6])
            a.set_title(f"Energy pg={r['epg']} IoU={r['eiou']:.3f}", fontsize=10)
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    sp = Path(sp); sp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sp.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(sp.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {sp}.pdf and {sp}.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--data_root', required=True)
    p.add_argument('--num_classes', type=int, default=1)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--embed_dim', type=int, default=48)
    p.add_argument('--n_samples', type=int, default=6); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', default='./outputs/gradcam')
    p.add_argument('--save_name', default='gradcam_panel')
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, BC = load_model(a.ckpt, a.device, a.embed_dim, a.num_classes, a.img_size)
    name, tgt = last_bn(model, BC); print(f'target: {name}')
    samples = pick_samples(a.data_root, n_samples=a.n_samples,
                           img_size=a.img_size, seed=a.seed, verbose=False)
    rows, csv_rows = [], []
    for i, (img_t, mask_t, orig, fn) in enumerate(samples):
        cam, en, pred = run_one(model, tgt, img_t, a.device, a.num_classes)
        gt = mask_t.numpy() if isinstance(mask_t, torch.Tensor) else np.array(mask_t)
        if gt.ndim == 3: gt = gt.squeeze(0) if gt.shape[0] == 1 else gt.argmax(0)
        gtb = (gt > 0).astype(np.uint8) if a.num_classes > 1 else gt.astype(np.uint8)
        prb = (pred > 0).astype(np.uint8) if a.num_classes > 1 else pred.astype(np.uint8)
        d = dice_score(prb, gtb)
        cpg, ciou = pg(cam, gtb), tiou(cam, gtb)
        epg, eiou = (pg(en, gtb), tiou(en, gtb)) if en is not None else (None, None)
        rows.append({'img': normimg(orig), 'gt': gtb, 'pred': prb, 'cam': cam,
                     'en': en, 'dice': d, 'cpg': cpg,
                     'ciou': ciou if ciou is not None else 0.0,
                     'epg': epg, 'eiou': eiou if eiou is not None else 0.0})
        csv_rows.append({'sample': fn, 'dice': round(d, 4),
                         'cam_pg': cpg, 'cam_iou': None if ciou is None else round(ciou, 4),
                         'energy_pg': epg, 'energy_iou': None if eiou is None else round(eiou, 4)})
        et = f"en(pg={epg},IoU={eiou:.3f})" if en is not None else "en=N/A"
        print(f'  [{i+1}/{len(samples)}] {fn}: D={d:.3f}, cam(pg={cpg},IoU={ciou:.3f}), {et}')
    render(rows, out / a.save_name)
    with (out / 'gradcam_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample', 'dice', 'cam_pg', 'cam_iou',
                                          'energy_pg', 'energy_iou'])
        w.writeheader(); w.writerows(csv_rows)
    for k in ['cam_pg', 'cam_iou', 'energy_pg', 'energy_iou']:
        xs = [r[k] for r in csv_rows if r[k] is not None]
        if xs: print(f'  agg {k:>11s} = {sum(xs)/len(xs):.3f} (n={len(xs)})')


if __name__ == '__main__':
    main()
