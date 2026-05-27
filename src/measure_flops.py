#!/usr/bin/env python3
"""
Measure FLOPs and parameter counts for HamCls and HamSeg.

Why the wrapper: HamiltonianBottleneck uses `torch.cuda.amp.autocast(enabled=False)`
to keep the oscillator scan in fp32. fvcore's symbolic tracer cannot follow tensors
across that context manager, so any submodule whose only path to the output goes
through it (here: ss2d.mom_merge, energy_attn) is silently dropped from the count.

Trick: monkey-patch `torch.cuda.amp.autocast` to `contextlib.nullcontext` for the
duration of the trace. Since we are running on CPU and the inner section is already
`enabled=False`, this changes nothing computationally - but fvcore can now see
through it.

Reports both the raw and "patched" counts so the gap between them is visible.
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))


class _Args:
    """Stand-in for argparse Namespace expected by HamCls/HamSeg constructors."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def measure(model: nn.Module, input_shape, name: str, patch_autocast: bool = True):
    model.eval()
    x = torch.zeros(*input_shape)

    if patch_autocast:
        original = torch.cuda.amp.autocast
        torch.cuda.amp.autocast = lambda *a, **kw: contextlib.nullcontext()
    try:
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.tracer_warnings('none')
        gflops = flops.total() / 1e9
    finally:
        if patch_autocast:
            torch.cuda.amp.autocast = original

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'model': name,
        'input_shape': list(input_shape),
        'gflops': round(gflops, 3),
        'params_M': round(n_params / 1e6, 3),
        'trainable_M': round(n_trainable / 1e6, 3),
        'patched_autocast': patch_autocast,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='outputs/flops_summary.json')
    # HamCls config (paper-reported: embed=96, depths=[3,3], pssp_K=12, n_dirs=2)
    ap.add_argument('--embed_dim_cls', type=int, default=96)
    ap.add_argument('--depths_cls', type=int, nargs=2, default=[3, 3])
    ap.add_argument('--pssp_K', type=int, default=12)
    ap.add_argument('--n_scan_dirs', type=int, default=2)
    # HamSeg config
    ap.add_argument('--embed_dim_seg', type=int, default=48)
    ap.add_argument('--depths_seg', type=int, nargs=4, default=[2, 2, 2, 2])
    # Shared
    ap.add_argument('--damping_clamp', type=float, default=5.0)
    ap.add_argument('--drop_rate', type=float, default=0.2)
    args = ap.parse_args()

    rows = []

    # ===========================================================
    # HamCls - main paper architecture (Tab. 3 row entries)
    # ===========================================================
    from hamcls import HamCls
    cls_args = _Args(
        in_channels=3, num_classes=9, size=224, head_drop=0.3,
        embed_dim=args.embed_dim_cls, depths=args.depths_cls,
        damping_clamp=args.damping_clamp, drop_rate=args.drop_rate,
        pssp_K=args.pssp_K, n_scan_dirs=args.n_scan_dirs,
        pssp_complex=True, pssp_cross=True, pssp_use_ss2d_energy=True,
    )
    m_cls = HamCls(cls_args)
    rows.append(measure(m_cls, (1, 3, 224, 224), 'HamCls @ 224',         patch_autocast=False))
    rows.append(measure(m_cls, (1, 3, 224, 224), 'HamCls @ 224 (fixed)', patch_autocast=True))

    # 11-class variant (OrganA/C/SMNIST)
    cls_args_11 = _Args(
        in_channels=3, num_classes=11, size=224, head_drop=0.3,
        embed_dim=args.embed_dim_cls, depths=args.depths_cls,
        damping_clamp=args.damping_clamp, drop_rate=args.drop_rate,
        pssp_K=args.pssp_K, n_scan_dirs=args.n_scan_dirs,
        pssp_complex=True, pssp_cross=True, pssp_use_ss2d_energy=True,
    )
    m_cls_11 = HamCls(cls_args_11)
    rows.append(measure(m_cls_11, (1, 3, 224, 224), 'HamCls @ 224 11-cls (fixed)', patch_autocast=True))

    # ===========================================================
    # PSSP component ablations (informative for the cost claims)
    # ===========================================================
    cls_args_no_complex = _Args(
        in_channels=3, num_classes=9, size=224, head_drop=0.3,
        embed_dim=args.embed_dim_cls, depths=args.depths_cls,
        damping_clamp=args.damping_clamp, drop_rate=args.drop_rate,
        pssp_K=args.pssp_K, n_scan_dirs=args.n_scan_dirs,
        pssp_complex=False, pssp_cross=True, pssp_use_ss2d_energy=True,
    )
    m_no_complex = HamCls(cls_args_no_complex)
    rows.append(measure(m_no_complex, (1, 3, 224, 224), 'HamCls (-complex FFT)', patch_autocast=True))

    cls_args_no_cross = _Args(
        in_channels=3, num_classes=9, size=224, head_drop=0.3,
        embed_dim=args.embed_dim_cls, depths=args.depths_cls,
        damping_clamp=args.damping_clamp, drop_rate=args.drop_rate,
        pssp_K=args.pssp_K, n_scan_dirs=args.n_scan_dirs,
        pssp_complex=True, pssp_cross=False, pssp_use_ss2d_energy=True,
    )
    m_no_cross = HamCls(cls_args_no_cross)
    rows.append(measure(m_no_cross, (1, 3, 224, 224), 'HamCls (-cross)', patch_autocast=True))

    cls_args_n1 = _Args(
        in_channels=3, num_classes=9, size=224, head_drop=0.3,
        embed_dim=args.embed_dim_cls, depths=args.depths_cls,
        damping_clamp=args.damping_clamp, drop_rate=args.drop_rate,
        pssp_K=args.pssp_K, n_scan_dirs=1,
        pssp_complex=True, pssp_cross=True, pssp_use_ss2d_energy=True,
    )
    m_n1 = HamCls(cls_args_n1)
    rows.append(measure(m_n1, (1, 3, 224, 224), 'HamCls (n_dirs=1)', patch_autocast=True))

    # ===========================================================
    # HamSeg
    # ===========================================================
    from hamseg import HamSeg
    seg_args = _Args(in_channels=3, num_classes=1, size=256, head_drop=0.3,
                     embed_dim=args.embed_dim_seg, depths=args.depths_seg,
                     damping_clamp=args.damping_clamp, drop_rate=args.drop_rate)
    m_seg = HamSeg(seg_args)
    rows.append(measure(m_seg, (1, 3, 256, 256), 'HamSeg @ 256 binary (fixed)', patch_autocast=True))

    seg_args4 = _Args(in_channels=3, num_classes=4, size=256, head_drop=0.3,
                      embed_dim=args.embed_dim_seg, depths=args.depths_seg,
                      damping_clamp=args.damping_clamp, drop_rate=args.drop_rate)
    m_seg4 = HamSeg(seg_args4)
    rows.append(measure(m_seg4, (1, 3, 256, 256), 'HamSeg @ 256 4-cls (fixed)', patch_autocast=True))

    # ===========================================================
    # Pretty print
    # ===========================================================
    print(f"{'model':<40} {'input':<18} {'GFLOPs':>10} {'Params (M)':>12}")
    print('-' * 84)
    for r in rows:
        shape = 'x'.join(str(s) for s in r['input_shape'])
        print(f"{r['model']:<40} {shape:<18} {r['gflops']:>10.2f} {r['params_M']:>12.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print()
    print(f"-> saved to {out}")


if __name__ == '__main__':
    main()
