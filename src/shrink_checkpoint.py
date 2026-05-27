#!/usr/bin/env python3
"""
shrink_ckpt.py
==============

Take a HamSeg / HamCls checkpoint and emit a minimal-size copy that
contains only the model weights (no optimizer / scheduler / amp state)
and that is cast to fp16. Useful for transferring trained models off
a remote machine when bandwidth is constrained or downloads crash on
larger files.

Typical size reduction: a 30 MB fp32 best_model.pth -> ~12-15 MB at fp16.

Usage:
    python shrink_ckpt.py path/to/best_model.pth path/to/best_model.fp16.pth

The output file can be loaded back at inference time with:

    state = torch.load('best_model.fp16.pth', map_location='cpu', weights_only=True)
    # Cast each tensor back to the target dtype before loading
    state = {k: v.float() for k, v in state.items()}
    model.load_state_dict(state)

The visualize_fig5_multidataset.py script already does this cast
automatically if you pass --fp16_ckpt.
"""

import argparse, os, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src',  help='Source checkpoint (.pth)')
    p.add_argument('dst',  help='Destination (.pth, will be overwritten)')
    p.add_argument('--keep_buffers', action='store_true',
                   help='Keep buffer tensors (running stats etc.) at fp32. '
                        'Default: cast everything to fp16 except integer buffers.')
    a = p.parse_args()

    print(f'[load]   {a.src}')
    state = torch.load(a.src, map_location='cpu', weights_only=True)

    # Some checkpoints wrap state_dict under a key like 'model' or 'state_dict'
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']

    if not isinstance(state, dict):
        raise SystemExit('Source file does not look like a state_dict')

    shrunk = {}
    n_params = 0
    for k, v in state.items():
        if not torch.is_tensor(v):
            shrunk[k] = v
            continue
        # Don't cast integer / bool tensors
        if v.dtype in (torch.int32, torch.int64, torch.bool, torch.uint8, torch.long):
            shrunk[k] = v
            continue
        if a.keep_buffers and v.numel() < 64:
            # Tiny buffers like running_mean / running_var: keep at original dtype
            shrunk[k] = v.cpu()
        else:
            shrunk[k] = v.to(torch.float16).cpu().contiguous()
            n_params += v.numel()

    print(f'[save]   {a.dst}')
    print(f'[stats]  {n_params/1e6:.2f}M params cast to fp16')

    torch.save(shrunk, a.dst, _use_new_zipfile_serialization=True)

    src_sz = os.path.getsize(a.src) / 1e6
    dst_sz = os.path.getsize(a.dst) / 1e6
    print(f'[size]   {src_sz:7.2f} MB -> {dst_sz:7.2f} MB   (x{src_sz/dst_sz:.2f} compression)')


if __name__ == '__main__':
    main()
