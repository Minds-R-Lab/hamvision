#!/usr/bin/env python3
"""
Dataset Preparation for HamVision

Downloads and organises every dataset used in Tables 2 and 3 of the manuscript,
plus MMOTU (the FreqConvMamba comparison benchmark).

Segmentation:
  1. ISIC2017  - Dermoscopic (2,150 samples)  - S3 auto-download
  2. ISIC2018  - Dermoscopic (2,694 samples)  - S3 auto-download
  3. TN3K      - Thyroid ultrasound (3,493)   - Google Drive auto-download
  4. MMOTU     - Ovarian ultrasound (1,469)   - GitHub clone
  5. ACDC      - Cardiac MRI (~1,526 slices)  - Manual or in-place NIfTI conversion

Classification (MedMNIST v2):
  6. PathMNIST, BloodMNIST, DermaMNIST, BreastMNIST, OrganSMNIST, RetinaMNIST
     (and other MedMNIST datasets) — auto-download via the medmnist package.

Usage
-----

    pip install gdown medmnist                            # for TN3K + MedMNIST

    # Per-dataset:
    python prepare_data.py --dataset isic2018 --data_root ./data/ISIC2018
    python prepare_data.py --dataset tn3k     --data_root ./data/TN3K
    python prepare_data.py --dataset mmotu    --data_root ./data/MMOTU
    python prepare_data.py --dataset bloodmnist --data_root ./data
    python prepare_data.py --dataset pathmnist --data_root ./data --size 224

    # Everything at once (segmentation only):
    python prepare_data.py --dataset all_seg --data_root ./data

    # All MedMNIST classification datasets:
    python prepare_data.py --dataset all_cls --data_root ./data

    # All 11 datasets:
    python prepare_data.py --dataset all --data_root ./data

This script is also imported by hamvision.data_setup (the lightweight wrapper
called by hamseg.py and hamcls.py at training startup).
"""
import os, sys, argparse, zipfile, shutil, random, subprocess
from pathlib import Path
from glob import glob


# ============================================================
# Generic helpers (kept identical to the user's original prepare_data.py)
# ============================================================

def download_url(url, dest, desc=None):
    import urllib.request
    if os.path.exists(dest): return True
    print(f"  Downloading {desc or Path(dest).name} ...")
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        def _p(bn, bs, ts):
            if ts > 0:
                print(f"\r    [{min(100, bn*bs*100//ts):3d}%] "
                      f"{bn*bs/1048576:.1f}/{ts/1048576:.1f}MB",
                      end='', flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=_p)
        print()
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        if os.path.exists(dest): os.remove(dest)
        return False


def download_gdrive(fid, dest, desc=None):
    if os.path.exists(dest): return True
    print(f"  Downloading from Google Drive: {desc or fid} ...")
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown', '-q'])
        import gdown
    try:
        gdown.download(f"https://drive.google.com/uc?id={fid}", dest, quiet=False)
        return os.path.exists(dest)
    except Exception as e:
        print(f"  GDrive failed: {e}")
        return False


def extract(path, dest):
    print(f"  Extracting {Path(path).name} ...")
    if path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(dest)
    print(f"  -> {dest}")


def find_imgs(d, rec=False):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    f = []
    for e in exts:
        f.extend(glob(os.path.join(d, '**' if rec else '', e), recursive=rec))
    return sorted(f)


def pair(img_dir, msk_dir):
    sfx = ['_segmentation', '_Segmentation', '_mask', '_seg', '_label', '_gt']
    ml = {}
    for f in find_imgs(msk_dir):
        s = Path(f).stem
        ml[s] = f
        for sx in sfx:
            if s.endswith(sx): ml[s[:-len(sx)]] = f
    pairs = []
    for f in sorted(find_imgs(img_dir)):
        s = Path(f).stem
        m = ml.get(s)
        if not m:
            for sx in sfx:
                m = ml.get(s + sx)
                if m: break
        if m: pairs.append((f, m))
    return pairs


def split_copy(pairs, root, tr=0.7, vr=0.15):
    rng = random.Random(42); rng.shuffle(pairs); n = len(pairs)
    nt = int(n * tr); nv = int(n * vr)
    sp = {'train': pairs[:nt], 'val': pairs[nt:nt + nv], 'test': pairs[nt + nv:]}
    for name, items in sp.items():
        id_ = os.path.join(root, name, 'images')
        md_ = os.path.join(root, name, 'masks')
        os.makedirs(id_, exist_ok=True); os.makedirs(md_, exist_ok=True)
        for ip, mp in items:
            shutil.copy2(ip, os.path.join(id_, Path(ip).name))
            shutil.copy2(mp, os.path.join(md_, Path(mp).name))
        print(f"    {name}: {len(items)}")
    return True


def copy_split(si, sm, root, name):
    ps = pair(si, sm)
    if not ps:
        print(f"    WARN: no pairs for {name}"); return 0
    id_ = os.path.join(root, name, 'images'); md_ = os.path.join(root, name, 'masks')
    os.makedirs(id_, exist_ok=True); os.makedirs(md_, exist_ok=True)
    for ip, mp in ps:
        shutil.copy2(ip, os.path.join(id_, Path(ip).name))
        shutil.copy2(mp, os.path.join(md_, Path(mp).name))
    print(f"    {name}: {len(ps)}"); return len(ps)


def ready(root):
    for s in ['train', 'test']:
        d = os.path.join(root, s, 'images')
        if not os.path.isdir(d) or not find_imgs(d): return False
    return True


def find_subdir(root, keywords):
    for d in Path(root).rglob('*'):
        if d.is_dir() and any(k in d.name for k in keywords): return str(d)
    return None


# ============================================================
# Segmentation: ISIC 2018
# ============================================================

def prep_isic2018(R):
    print("\n" + "=" * 60 + "\n  ISIC2018 - Skin Lesion Segmentation (2,694)\n" + "=" * 60)
    if ready(R): print("  Ready!"); return True
    raw = os.path.join(R, '_raw'); os.makedirs(raw, exist_ok=True)
    u = {
        'img': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip',
        'msk': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip',
    }
    ok = all(download_url(v, os.path.join(raw, v.split('/')[-1]), k) for k, v in u.items())
    if not ok:
        print("  AUTO-DOWNLOAD FAILED. Manual options:")
        print("    https://challenge.isic-archive.com/data/#2018")
        print("    https://www.kaggle.com/datasets/salviohexia/isic-2018-task-1")
        print(f"    Place in {R}/images/ and {R}/masks/, then re-run.")
        return False
    for v in u.values():
        zp = os.path.join(raw, v.split('/')[-1])
        if os.path.exists(zp): extract(zp, raw)
    id_ = find_subdir(raw, ['Training_Input']); md_ = find_subdir(raw, ['GroundTruth'])
    if id_ and md_:
        ps = pair(id_, md_); print(f"  {len(ps)} pairs")
        return split_copy(ps, R)
    return False


# ============================================================
# Segmentation: ISIC 2017
# ============================================================

def prep_isic2017(R):
    print("\n" + "=" * 60 + "\n  ISIC2017 - Skin Lesion Segmentation (2,150)\n" + "=" * 60)
    if ready(R): print("  Ready!"); return True
    raw = os.path.join(R, '_raw'); os.makedirs(raw, exist_ok=True)
    S = {
        'train': ('https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip',
                  'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip'),
        'val':   ('https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip',
                  'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip'),
        'test':  ('https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip',
                  'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip'),
    }
    ok = True
    for sp, (iu, mu) in S.items():
        for u in [iu, mu]:
            if not download_url(u, os.path.join(raw, u.split('/')[-1]), sp):
                ok = False
    if not ok:
        print("  FAILED. Manual: https://challenge.isic-archive.com/data/#2017")
        return False
    for sp, (iu, mu) in S.items():
        for u in [iu, mu]:
            zp = os.path.join(raw, u.split('/')[-1])
            if os.path.exists(zp): extract(zp, raw)
    M = {
        'train': ('Training_Data', 'Training_Part1_GroundTruth'),
        'val':   ('Validation_Data', 'Validation_Part1_GroundTruth'),
        'test':  ('Test_v2_Data', 'Test_v2_Part1_GroundTruth'),
    }
    tot = 0
    for sp, (ik, mk) in M.items():
        id_ = find_subdir(raw, [ik]); md_ = find_subdir(raw, [mk])
        if id_ and md_:
            tot += copy_split(id_, md_, R, sp)
    return tot > 0


# ============================================================
# Segmentation: TN3K
# ============================================================

def prep_tn3k(R):
    print("\n" + "=" * 60 + "\n  TN3K - Thyroid Nodule Ultrasound (3,493)\n" + "=" * 60)
    if ready(R): print("  Ready!"); return True
    raw = os.path.join(R, '_raw'); os.makedirs(raw, exist_ok=True)
    zp = os.path.join(raw, 'tn3k.zip')
    ok = download_gdrive('1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F', zp, 'TN3K')
    if not ok:
        print("  FAILED. Manual options:")
        print("    https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F")
        print("    https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation")
        print("    Baidu: https://pan.baidu.com/s/1byqO5sBlt6OQdOxC4-SYng code:trfe")
        return False
    extract(zp, raw)
    ti = find_subdir(raw, ['trainval-image', 'trainval_image'])
    tm = find_subdir(raw, ['trainval-mask', 'trainval_mask'])
    ei = find_subdir(raw, ['test-image', 'test_image'])
    em = find_subdir(raw, ['test-mask', 'test_mask'])
    if ti and tm:
        ps = pair(ti, tm); rng = random.Random(42); rng.shuffle(ps)
        nt = int(len(ps) * 0.85)
        for nm, items in [('train', ps[:nt]), ('val', ps[nt:])]:
            id_ = os.path.join(R, nm, 'images'); md_ = os.path.join(R, nm, 'masks')
            os.makedirs(id_, exist_ok=True); os.makedirs(md_, exist_ok=True)
            for ip, mp in items:
                shutil.copy2(ip, os.path.join(id_, Path(ip).name))
                shutil.copy2(mp, os.path.join(md_, Path(mp).name))
            print(f"    {nm}: {len(items)}")
        if ei and em: copy_split(ei, em, R, 'test')
        return True
    # Fallback: flat search
    for sr in [raw] + [str(d) for d in Path(raw).iterdir() if d.is_dir()]:
        for iname in ['trainval-image', 'images']:
            for mname in ['trainval-mask', 'masks']:
                ip = os.path.join(sr, iname); mp = os.path.join(sr, mname)
                if os.path.isdir(ip) and os.path.isdir(mp):
                    ps = pair(ip, mp)
                    if ps: return split_copy(ps, R)
    print("  Could not find extracted TN3K dirs"); return False


# ============================================================
# Segmentation: MMOTU
# ============================================================

def prep_mmotu(R):
    print("\n" + "=" * 60 + "\n  MMOTU - Ovarian Tumor Ultrasound (1,469)\n" + "=" * 60)
    if ready(R): print("  Ready!"); return True
    raw = os.path.join(R, '_raw'); os.makedirs(raw, exist_ok=True)
    for repo, url in [('MMOTU_DS2Net', 'https://github.com/cv516Buaa/MMOTU_DS2Net.git'),
                      ('OTU-2D-Dataset', 'https://github.com/SonBH0410/OTU-2D-Dataset.git')]:
        rd = os.path.join(raw, repo)
        if not os.path.isdir(rd):
            print(f"  Cloning {repo}...")
            try:
                subprocess.run(['git', 'clone', '--depth=1', url, rd],
                               check=True, capture_output=True, timeout=180)
                print('    OK')
            except Exception as e:
                print(f"    Failed: {e}")
    fi = fm = None
    for sr in [raw, R] + [str(d) for d in Path(raw).rglob('*') if d.is_dir()]:
        if not os.path.isdir(sr): continue
        nm = Path(sr).name.lower()
        if nm in ['otu_2d', 'otu2d', 'images', 'image', 'img'] and not fi:
            if find_imgs(sr): fi = sr
        if nm in ['otu_2d_seg', 'otu2d_seg', 'masks', 'mask', 'gt', 'seg',
                   'annotations', 'label'] and not fm:
            if find_imgs(sr): fm = sr
    if fi and fm:
        ps = pair(fi, fm)
        if ps: print(f"  {len(ps)} pairs"); return split_copy(ps, R)
    print("  AUTO-DOWNLOAD INCOMPLETE. Manual options:")
    print("    1. https://github.com/cv516Buaa/MMOTU_DS2Net (check README for data links)")
    print("    2. https://figshare.com/articles/dataset/_zip/25058690")
    print(f"    Place OTU_2d images in {R}/images/ and masks in {R}/masks/")
    for iname in ['images', 'OTU_2d']:
        for mname in ['masks', 'OTU_2d_seg']:
            ip = os.path.join(R, iname); mp = os.path.join(R, mname)
            if os.path.isdir(ip) and os.path.isdir(mp):
                ps = pair(ip, mp)
                if ps: return split_copy(ps, R)
    return False


# ============================================================
# Segmentation: ACDC
# ============================================================

def prep_acdc(R):
    print("\n" + "=" * 60 + "\n  ACDC - Cardiac MRI Segmentation (~1,526 slices)\n" + "=" * 60)
    if ready(R): print("  Ready!"); return True
    raw = os.path.join(R, '_raw'); os.makedirs(raw, exist_ok=True)
    pdirs = [str(d) for d in Path(raw).rglob('patient*')
             if d.is_dir() and list(d.glob('*.nii.gz'))]
    if not pdirs:
        pdirs = [str(d) for d in Path(R).rglob('patient*')
                 if d.is_dir() and list(d.glob('*.nii.gz'))]
    for iname in ['images', 'slices']:
        ip = os.path.join(R, iname)
        if os.path.isdir(ip) and find_imgs(ip):
            for mname in ['masks', 'labels', 'gt']:
                mp = os.path.join(R, mname)
                if os.path.isdir(mp) and find_imgs(mp):
                    ps = pair(ip, mp)
                    if ps:
                        print(f"  {len(ps)} pre-converted slices")
                        return split_copy(ps, R)
    if pdirs:
        print(f"  Found {len(pdirs)} patient NIfTI directories. Converting...")
        return convert_acdc(pdirs, R)
    print("  ACDC requires registration for download.")
    print("  Options:")
    print("    A) Official: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html")
    print("    B) Kaggle:   https://www.kaggle.com/datasets/anhoangvo/acdc-dataset")
    print("    C) HuggingFace: https://huggingface.co/datasets/msepulvedagodoy/acdc")
    print(f"  After download, extract patient folders into {raw}/ and re-run.")
    print(f"  Or place pre-converted PNGs in {R}/images/ and {R}/masks/")
    return False


def convert_acdc(pdirs, R):
    try:
        import nibabel as nib
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nibabel', '-q'])
        import nibabel as nib
    from PIL import Image
    import numpy as np
    all_p = {}
    for pd in sorted(pdirs):
        pn = Path(pd).name
        for nf in sorted(Path(pd).glob('*.nii.gz')):
            fn = nf.stem.replace('.nii', '')
            if '_gt' in fn or 'frame' not in fn: continue
            gt = nf.parent / (fn + '_gt.nii.gz')
            if not gt.exists(): continue
            iv = nib.load(str(nf)).get_fdata()
            gv = nib.load(str(gt)).get_fdata()
            for s in range(iv.shape[2]):
                gs = gv[:, :, s]
                if gs.max() == 0: continue
                isl = iv[:, :, s].astype(np.float32)
                if isl.max() > isl.min(): isl = (isl - isl.min()) / (isl.max() - isl.min())
                sn = f"{fn}_s{s:02d}"
                io = os.path.join(R, '_slices', 'images', f"{sn}.png")
                mo = os.path.join(R, '_slices', 'masks', f"{sn}.png")
                os.makedirs(os.path.dirname(io), exist_ok=True)
                os.makedirs(os.path.dirname(mo), exist_ok=True)
                Image.fromarray((isl * 255).astype(np.uint8), 'L').save(io)
                # ACDC has 4 classes (0=bg, 1=RV, 2=MYO, 3=LV).  Save raw class indices
                # as pixel values 0/1/2/3 — the loader reads them with .long() to recover
                # the correct multi-class targets.  Do NOT binarise.
                gs_u8 = np.clip(gs, 0, 255).astype(np.uint8)
                Image.fromarray(gs_u8, 'L').save(mo)
                if pn not in all_p: all_p[pn] = []
                all_p[pn].append((io, mo))
    if not all_p: print('  No valid slices'); return False
    total = sum(len(v) for v in all_p.values())
    print(f"  Converted {total} slices from {len(all_p)} patients")
    pts = sorted(all_p.keys()); rng = random.Random(42); rng.shuffle(pts)
    n = len(pts); nt = int(n * 0.7); nv = int(n * 0.15)
    sp = {'train': pts[:nt], 'val': pts[nt:nt + nv], 'test': pts[nt + nv:]}
    for nm, ptlist in sp.items():
        id_ = os.path.join(R, nm, 'images'); md_ = os.path.join(R, nm, 'masks')
        os.makedirs(id_, exist_ok=True); os.makedirs(md_, exist_ok=True)
        c = 0
        for p in ptlist:
            for ip, mp in all_p[p]:
                shutil.copy2(ip, os.path.join(id_, Path(ip).name))
                shutil.copy2(mp, os.path.join(md_, Path(mp).name))
                c += 1
        print(f"    {nm}: {c} slices ({len(ptlist)} patients)")
    return True


# ============================================================
# Classification: MedMNIST (any of the 12 datasets in v2)
# ============================================================

MEDMNIST_NAMES = {
    'pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
    'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
    'organamnist', 'organcmnist', 'organsmnist',
}


def prep_medmnist(name, R, size=224):
    """Use the medmnist package to download. Returns True iff the .npz files end up in R."""
    print("\n" + "=" * 60 +
          f"\n  {name.upper()} - MedMNIST (size={size})\n" + "=" * 60)
    R = str(R)
    os.makedirs(R, exist_ok=True)

    candidates = [
        os.path.join(R, f'{name}_{size}.npz'),
        os.path.join(R, f'{name}.npz'),
    ]
    if any(os.path.exists(p) for p in candidates):
        print(f"  Ready! ({[p for p in candidates if os.path.exists(p)][0]})")
        return True

    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        print("  medmnist not installed; installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'medmnist', '-q'])
        import medmnist
        from medmnist import INFO

    if name not in INFO:
        print(f"  ERROR: {name!r} is not a known MedMNIST dataset.  "
              f"Known: {sorted(INFO.keys())}")
        return False

    DataClass = getattr(medmnist, INFO[name]['python_class'])
    try:
        for split in ('train', 'val', 'test'):
            DataClass(split=split, download=True, root=R, size=size)
        ok = any(os.path.exists(p) for p in candidates)
        if ok: print(f"  {name} ready.")
        return ok
    except Exception as e:
        print(f"  medmnist download failed: {e}")
        print(f"  Manual: download {name}_{size}.npz from https://medmnist.com/ to {R}/")
        return False


# ============================================================
# Dispatch
# ============================================================

# Per-dataset prep functions, indexed by lowercase name.
PREP = {
    'isic2017': prep_isic2017,
    'isic2018': prep_isic2018,
    'tn3k':     prep_tn3k,
    'mmotu':    prep_mmotu,
    'acdc':     prep_acdc,
}

SEG_DATASETS = set(PREP.keys())
CLS_DATASETS = MEDMNIST_NAMES


def prep(name, R, size=224):
    """Dispatch entry point used by data_setup.prepare()."""
    name = name.lower()
    if name in PREP:
        return PREP[name](R)
    if name in CLS_DATASETS:
        return prep_medmnist(name, R, size)
    print(f"  ERROR: unknown dataset name {name!r}")
    return False


# ============================================================
# CLI
# ============================================================

ALL_SEG = ['isic2017', 'isic2018', 'tn3k', 'mmotu', 'acdc']
ALL_CLS_TAB3 = ['pathmnist', 'bloodmnist', 'dermamnist',
                'breastmnist', 'organsmnist', 'retinamnist']


def main():
    p = argparse.ArgumentParser(description='Prepare datasets for HamVision')
    p.add_argument('--dataset', required=True,
                   choices=ALL_SEG + sorted(MEDMNIST_NAMES) + ['all_seg', 'all_cls', 'all'])
    p.add_argument('--data_root', required=True,
                   help='For segmentation: dataset folder.  For MedMNIST: the parent '
                        'data folder (.npz files land directly here).')
    p.add_argument('--size', type=int, default=224,
                   help='Resolution for MedMNIST (28, 64, 128, 224).  Default 224.')
    a = p.parse_args()

    if a.dataset in ('all_seg', 'all'):
        seg_results = {}
        for nm in ALL_SEG:
            dr = os.path.join(a.data_root, nm.upper())
            try:
                seg_results[nm] = PREP[nm](dr)
            except Exception as e:
                print(f"  ERROR {nm}: {e}"); seg_results[nm] = False

    if a.dataset in ('all_cls', 'all'):
        cls_results = {}
        for nm in ALL_CLS_TAB3:
            try:
                cls_results[nm] = prep_medmnist(nm, a.data_root, a.size)
            except Exception as e:
                print(f"  ERROR {nm}: {e}"); cls_results[nm] = False

    if a.dataset in ('all_seg', 'all_cls', 'all'):
        print("\n" + "=" * 60 + "\n  SUMMARY\n" + "=" * 60)
        if a.dataset in ('all_seg', 'all'):
            for nm, ok in seg_results.items():
                print(f"    {nm.upper():10s}: {'READY' if ok else 'NEEDS MANUAL DOWNLOAD'}")
        if a.dataset in ('all_cls', 'all'):
            for nm, ok in cls_results.items():
                print(f"    {nm.upper():10s}: {'READY' if ok else 'NEEDS MANUAL DOWNLOAD'}")
        if a.dataset in ('all_cls', 'all'):
            for nm, ok in cls_results.items():
                print(f"    {nm.upper():12s}: {'READY' if ok else 'NEEDS MANUAL DOWNLOAD'}")
    else:
        ok = prep(a.dataset, a.data_root, a.size)
        if ok:
            print(f"\n  {a.dataset.upper()} READY!")
        else:
            print(f"\n  {a.dataset.upper()} needs manual download.")
            sys.exit(1)


if __name__ == '__main__':
    main()
