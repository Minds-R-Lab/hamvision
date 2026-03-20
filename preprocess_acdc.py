#!/usr/bin/env python3
"""
Preprocess ACDC dataset for HamSeg.
Converts 3D NIfTI cardiac MRI volumes → 2D .npz slices.

Each .npz contains:
  'image': float32 (H, W) — normalized to [0, 1]
  'mask':  uint8 (H, W)   — class labels (0=bg, 1=RV, 2=Myo, 3=LV)

Only slices containing foreground are saved.

Usage:
    python preprocess_acdc.py --data_root ./data/ACDC/database --output_dir ./data/ACDC_2d
"""

import os, argparse
import numpy as np
from pathlib import Path


def process_acdc(data_root, output_dir, test_patients=20):
    try:
        import nibabel as nib
    except ImportError:
        print("Installing nibabel...")
        os.system("pip install nibabel")
        import nibabel as nib

    data_root = Path(data_root)
    output_dir = Path(output_dir)

    # Find training folder
    train_dir = None
    for name in ['training', 'train']:
        d = data_root / name
        if d.exists():
            train_dir = d; break
    if train_dir is None:
        if list(data_root.glob('patient*')):
            train_dir = data_root
        else:
            raise RuntimeError(f"Cannot find training data in {data_root}")

    patients = sorted([d for d in train_dir.iterdir()
                      if d.is_dir() and 'patient' in d.name.lower()])
    print(f"Found {len(patients)} patients in {train_dir}")
    if len(patients) == 0:
        raise RuntimeError(f"No patient folders found in {train_dir}")

    # ACDC has 5 pathology groups of 20 patients each (sorted by ID):
    # 001-020: Normal, 021-040: MINF, 041-060: DCM, 061-080: HCM, 081-100: RV
    # Stratified split: equal proportion from each group
    n_total = len(patients)
    group_size = 20
    n_groups = n_total // group_size if n_total >= 100 else 1

    if n_groups >= 5 and n_total >= 100:
        test_per_group = test_patients // n_groups  # 30/5 = 6
        train_patients = []
        test_patients_list = []
        for g in range(n_groups):
            group = patients[g * group_size:(g + 1) * group_size]
            train_patients.extend(group[:group_size - test_per_group])
            test_patients_list.extend(group[group_size - test_per_group:])
    else:
        n_train = len(patients) - test_patients
        train_patients = patients[:n_train]
        test_patients_list = patients[n_train:]
    print(f"Train: {len(train_patients)} patients, Test: {len(test_patients_list)} patients")

    total_slices = {'train': 0, 'test': 0}

    for split, patient_list in [('train', train_patients), ('test', test_patients_list)]:
        out_dir = output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for pat_dir in patient_list:
            gt_files = sorted(pat_dir.glob('*_gt.nii.gz'))
            if not gt_files:
                gt_files = sorted(pat_dir.glob('*_gt.nii'))

            for gt_path in gt_files:
                img_name = gt_path.name.replace('_gt.nii', '.nii')
                img_path = pat_dir / img_name
                if not img_path.exists():
                    print(f"  Warning: {img_path} not found, skipping"); continue

                img_vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
                gt_vol = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

                if img_vol.ndim == 4: img_vol = img_vol[:, :, :, 0]
                if gt_vol.ndim == 4: gt_vol = gt_vol[:, :, :, 0]

                pat_name = pat_dir.name
                frame_name = gt_path.stem.replace('_gt', '')

                for s in range(img_vol.shape[2]):
                    img_slice = img_vol[:, :, s]
                    gt_slice = gt_vol[:, :, s]

                    # Only keep slices where ALL 3 cardiac structures are present
                    # (0=bg, 1=RV, 2=Myo, 3=LV) — matches standard ACDC protocol
                    classes_present = set(np.unique(gt_slice).astype(int))
                    if not {1, 2, 3}.issubset(classes_present):
                        continue

                    # Normalize to [0, 1] float32
                    vmin = np.percentile(img_slice, 1)
                    vmax = np.percentile(img_slice, 99)
                    if vmax - vmin < 1e-8: vmax = vmin + 1
                    img_norm = np.clip((img_slice - vmin) / (vmax - vmin), 0, 1).astype(np.float32)

                    fname = f"{pat_name}_{frame_name}_s{s:03d}.npz"
                    np.savez_compressed(str(out_dir / fname),
                                       image=img_norm, mask=gt_slice)
                    total_slices[split] += 1

        print(f"  {split}: {total_slices[split]} slices")

    print(f"\nDone! Output: {output_dir}")
    print(f"  Train: {total_slices['train']} slices")
    print(f"  Test:  {total_slices['test']} slices")
    print(f"\nTo train:")
    print(f"  python hamseg.py --dataset acdc --data_root {output_dir} "
          f"--num_classes 4 --val_ratio 0.0 --patience 80 --batch_size 8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./data/ACDC_2d')
    parser.add_argument('--test_patients', type=int, default=20)
    args = parser.parse_args()
    process_acdc(args.data_root, args.output_dir, args.test_patients)


if __name__ == '__main__':
    main()
