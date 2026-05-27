#!/usr/bin/env python3
"""
Build the train/val/test layout for MMOTU using the dataset's OFFICIAL splits.

OTU_2d/train.txt  -> our train (90%) + held-out val (10%) for early stopping
OTU_2d/val.txt    -> our test set (this is what FreqConvMamba reports on)

This matches the standard MMOTU benchmark protocol used by FreqConvMamba and others.
"""
import argparse, os, random, shutil
from pathlib import Path

EXTS_IMG = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp')
EXTS_MSK = ('.png', '.PNG', '.bmp', '.BMP', '.jpg', '.JPG')


def read_split(p):
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def find_for(stem, d, exts):
    for ext in exts:
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def pair_ids(ids, img_dir, ann_dir):
    pairs, missing = [], []
    for sid in ids:
        ip = find_for(sid, img_dir, EXTS_IMG)
        mp = find_for(sid, ann_dir, EXTS_MSK)
        if ip and mp:
            pairs.append((ip, mp))
        else:
            missing.append(sid)
    return pairs, missing


def copy_pairs(pairs, dest_root, split_name):
    id_ = os.path.join(dest_root, split_name, 'images')
    md_ = os.path.join(dest_root, split_name, 'masks')
    os.makedirs(id_, exist_ok=True)
    os.makedirs(md_, exist_ok=True)
    for ip, mp in pairs:
        shutil.copy2(ip, os.path.join(id_, Path(ip).name))
        shutil.copy2(mp, os.path.join(md_, Path(mp).name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='/mnt/c/Users/Z/Desktop/research/hamvision_data/seg/data/MMOTU')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_frac', type=float, default=0.10)
    a = p.parse_args()

    src = os.path.join(a.data_root, 'OTU_2d')
    img_dir = os.path.join(src, 'images')
    ann_dir = os.path.join(src, 'annotations')
    train_txt = os.path.join(src, 'train.txt')
    val_txt = os.path.join(src, 'val.txt')

    for p_ in (img_dir, ann_dir, train_txt, val_txt):
        if not os.path.exists(p_):
            raise SystemExit(f'Missing: {p_}')

    train_ids = read_split(train_txt)
    test_ids = read_split(val_txt)

    print(f'  IDs read:  train.txt={len(train_ids)}   val.txt={len(test_ids)}')

    train_pairs, train_missing = pair_ids(train_ids, img_dir, ann_dir)
    test_pairs, test_missing = pair_ids(test_ids, img_dir, ann_dir)

    print(f'  Resolved:  train={len(train_pairs)}/{len(train_ids)} (missing {len(train_missing)})   '
          f'test={len(test_pairs)}/{len(test_ids)} (missing {len(test_missing)})')

    if train_missing[:3]:
        print(f'  Sample missing train IDs: {train_missing[:3]}')

    # Hold out val_frac of train.txt for our internal validation set
    rng = random.Random(a.seed)
    rng.shuffle(train_pairs)
    nval = max(1, int(len(train_pairs) * a.val_frac))
    val_pairs = train_pairs[:nval]
    train_pairs = train_pairs[nval:]

    print(f'  Final:     train={len(train_pairs)}   val={len(val_pairs)}   test={len(test_pairs)}')

    # Wipe any prior train/val/test dirs to avoid stale files
    for name in ('train', 'val', 'test'):
        d = os.path.join(a.data_root, name)
        if os.path.exists(d):
            shutil.rmtree(d)

    copy_pairs(train_pairs, a.data_root, 'train')
    copy_pairs(val_pairs, a.data_root, 'val')
    copy_pairs(test_pairs, a.data_root, 'test')
    print(f'  ✓ MMOTU ready at {a.data_root}')


if __name__ == '__main__':
    main()
