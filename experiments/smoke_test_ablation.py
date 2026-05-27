#!/usr/bin/env python3
"""Smoke test for the round-2 --ablation flag in HamSeg and HamCls.

Run on a machine that has torch installed (your WSL box). It builds all 6
model variants, runs a single forward pass on dummy CPU tensors, and prints
output shapes + parameter counts. ~30 seconds end-to-end.

If anything errors out, fix it BEFORE launching a 25h ablation training run.
"""
import sys, warnings
import torch
warnings.filterwarnings('ignore')

# Allow running either from repo root or from inside hamvision/
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

class A:
    pass

def cfg(num_classes=1, img_size=224, **kw):
    a = A()
    a.embed_dim = 48
    a.depths = [2, 2, 2, 2]
    a.damping_clamp = 5.0
    a.drop_rate = 0.1
    a.head_drop = 0.3
    a.num_classes = num_classes
    a.img_size = img_size
    a.size = img_size
    a.in_channels = 3
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def header(s):
    print(f"\n{'='*60}\n{s}\n{'='*60}")


def test_seg():
    from hamseg import HamSeg
    header('HamSeg @ 224, binary')
    for abl in ['none', 'A', 'B']:
        m = HamSeg(cfg(num_classes=1, img_size=224, ablation=abl)).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(torch.zeros(2, 3, 224, 224))
        assert y.shape == (2, 1, 224, 224), y.shape
        print(f'  ablation={abl}: out={tuple(y.shape)}, params={n:,}')

    header('HamSeg @ 256, ACDC 4-class')
    for abl in ['none', 'A', 'B']:
        m = HamSeg(cfg(num_classes=4, img_size=256, ablation=abl)).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(torch.zeros(2, 3, 256, 256))
        assert y.shape == (2, 4, 256, 256), y.shape
        print(f'  ablation={abl}: out={tuple(y.shape)}, params={n:,}')


def test_cls():
    from hamcls import HamCls

    header('HamCls @ 224, 9-class — PSSA head (round-2 default)')
    for abl in ['none', 'A', 'B']:
        m = HamCls(cfg(num_classes=9, img_size=224, ablation=abl,
                       head_variant='pssa')).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(torch.zeros(2, 3, 224, 224))
        assert y.shape == (2, 9), y.shape
        print(f'  ablation={abl}: out={tuple(y.shape)}, params={n:,}')

    header('HamCls @ 224, 9-class — GAP head (legacy, back-compat sanity)')
    m = HamCls(cfg(num_classes=9, img_size=224, ablation='none',
                   head_variant='gap')).eval()
    n = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        y = m(torch.zeros(2, 3, 224, 224))
    assert y.shape == (2, 9), y.shape
    print(f'  head_variant=gap, ablation=none: out={tuple(y.shape)}, params={n:,}')


if __name__ == '__main__':
    test_seg()
    test_cls()
    print('\nALL OK')
