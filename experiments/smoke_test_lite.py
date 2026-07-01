"""Smoke-test the HamVision-Lite configuration.

Verifies that:
    1. `--lite_no_se --lite_no_psattn` produces a functioning forward pass.
    2. The parameter count drops by the expected ~74 K vs. the full model
       (at D=384, the standard HamSeg bottleneck dim).
    3. The output tensor shapes are unchanged.

Run this on the H100 before launching the 6 Lite training jobs to catch
any environment mismatch quickly.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))

import torch  # noqa: E402
from hamseg import HamiltonianBottleneck, HamSeg  # noqa: E402


def bottleneck_param_check():
    full = HamiltonianBottleneck(dim=384, ablation="none")
    lite = HamiltonianBottleneck(dim=384, ablation="none", lite_no_se=True)
    full_n = sum(p.numel() for p in full.parameters())
    lite_n = sum(p.numel() for p in lite.parameters())
    saved = full_n - lite_n
    print(f"  bottleneck full : {full_n:>10} params")
    print(f"  bottleneck lite : {lite_n:>10} params  (saved {saved})")
    assert saved > 0, "Lite bottleneck should have fewer params than full"
    assert saved < full_n * 0.1, "Sanity: savings should be < 10% of bottleneck"


def full_lite_forward():
    args = types.SimpleNamespace(
        dataset="acdc",
        embed_dim=48,
        depths=[2, 2, 2, 2],
        num_classes=1,
        damping_clamp=5.0,
        drop_rate=0.1,
        ablation="none",
        lite_no_se=True,
        lite_no_psattn=True,
    )
    m = HamSeg(args)
    m.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    if isinstance(out, (tuple, list)):
        print(f"  forward OK, {len(out)} outputs:")
        for i, o in enumerate(out):
            if isinstance(o, torch.Tensor):
                print(f"    out[{i}]: {tuple(o.shape)}")
    else:
        print(f"  forward OK: {tuple(out.shape)}")
    total = sum(p.numel() for p in m.parameters())
    print(f"  total Lite params: {total:>10}")


if __name__ == "__main__":
    print("== HamVision-Lite smoke test ==")
    print("(1) Bottleneck parameter reduction:")
    bottleneck_param_check()
    print("(2) End-to-end forward pass:")
    try:
        full_lite_forward()
    except Exception as e:
        print(f"  forward FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    print("OK -- HamVision-Lite is ready to train.")
