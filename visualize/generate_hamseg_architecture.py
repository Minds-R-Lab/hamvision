#!/usr/bin/env python3
"""HamSeg architecture figure, styled to match the Figure 1 bottleneck panel."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


CONV_F, CONV_E = '#d8ecd2', '#3f7d3f'
OSC_F,  OSC_E  = '#ffe1b3', '#c47100'
FUS_F,  FUS_E  = '#e0d4ec', '#7b3fa0'
FEAT_F, FEAT_E = '#d4e6f7', '#2e6fa8'
MOM_F,  MOM_E  = '#f7d4d4', '#b13a3a'
ENG_F,  ENG_E  = '#fbeec1', '#a98a2a'
GREY_F         = '#e8eaed'
WHITE          = '#ffffff'
INK            = '#0d0d0c'
ARROW          = '#3a3a36'
LIGHT_BG       = '#fafafa'
LW = 1.3


def box(ax, x, y, w, h, t, s=None, f=WHITE, e=INK, tc=INK, lw=LW, r=0.05):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0.02,rounding_size={r}',
                       linewidth=lw, edgecolor=e, facecolor=f, zorder=2)
    ax.add_patch(p)
    if s:
        ax.text(x+w/2, y+h*0.66, t, ha='center', va='center',
                fontsize=10.5, color=tc, fontweight='bold', zorder=3)
        ax.text(x+w/2, y+h*0.30, s, ha='center', va='center',
                fontsize=8.5, color=tc, zorder=3)
    else:
        ax.text(x+w/2, y+h/2, t, ha='center', va='center',
                fontsize=10.5, color=tc, fontweight='bold', zorder=3)


def arrow(ax, x1, y1, x2, y2, c=ARROW, lw=1.5, rad=0.0, mut=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=mut,
                                 color=c, linewidth=lw,
                                 connectionstyle=f'arc3,rad={rad}',
                                 zorder=4, shrinkA=0, shrinkB=2))


def tee(ax, x, y, c):
    ax.scatter([x], [y], s=42, color=c, zorder=6,
               edgecolors=INK, linewidths=0.6)


def shape(ax, x, y, t, c=INK):
    ax.text(x, y, t, ha='center', va='top', fontsize=8.0,
            color=c, style='italic')


def make_figure(save_path):
    fig, ax = plt.subplots(figsize=(15.0, 9.5))
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ====== Title ======
    ax.text(7.5, 9.10, 'HamSeg: Segmentation Architecture',
            ha='center', va='center', fontsize=14, fontweight='bold', color=INK)
    ax.text(7.5, 8.78,
            'shared Hamiltonian bottleneck $\\to$ U-Net-style decoder '
            'with energy-gated skips and momentum injection',
            ha='center', va='center', fontsize=9.5, color='#555555')

    # ====== Encoder column (left) ======
    EX, EW, EH = 0.40, 1.90, 0.85
    rows = [
        ('Input',       r'$224\!\times\!224\!\times\!3$',   GREY_F, INK,   None),
        ('Stem',        r'$2\!\times$\,Conv 3$\times$3',    CONV_F, CONV_E, r'$224\!\times\!224\!\times\!48$'),
        ('Enc Stage 1', r'ConvNeXt$\,\times\,2$',           CONV_F, CONV_E, r'$224\!\times\!224\!\times\!48$'),
        ('Enc Stage 2', r'ConvNeXt$\,\times\,2$',           CONV_F, CONV_E, r'$112\!\times\!112\!\times\!96$'),
        ('Enc Stage 3', r'ConvNeXt$\,\times\,2$',           CONV_F, CONV_E, r'$56\!\times\!56\!\times\!192$'),
    ]
    enc_ys = [7.50, 6.30, 5.10, 3.90, 2.70]
    for (t, s, f, e, sh), y in zip(rows, enc_ys):
        box(ax, EX, y, EW, EH, t, s, f=f, e=e)
        if sh is not None:
            shape(ax, EX + EW/2, y - 0.05, sh, c=e)
    # down arrows + tiny PatchMerging labels between encoder boxes (except input->stem)
    for i, (y_top, y_bot) in enumerate(zip(enc_ys[:-1], enc_ys[1:])):
        arrow(ax, EX + EW/2, y_top, EX + EW/2, y_bot + EH, rad=0.0)
        if i >= 1:  # between Enc1->Enc2, Enc2->Enc3 we down-sample
            ax.text(EX + EW + 0.05, (y_top + y_bot + EH)/2,
                    'PatchMerging\n$\\downarrow2\\times$',
                    ha='left', va='center', fontsize=7.5,
                    color='#555555', style='italic')

    # ====== Decoder column (right) ======
    DX = 12.70
    dec_rows = [
        ('Output',      r'$224\!\times\!224\!\times\!\mathrm{C_{out}}$', GREY_F, INK,   None),
        ('Seg head',    r'Conv 3$\times$3 + Conv 1$\times$1',           CONV_F, CONV_E, None),
        ('Dec Stage 1', r'EGS $\to$ ConvNeXt$\,\times\,2$',             CONV_F, CONV_E, r'$224\!\times\!224\!\times\!48$'),
        ('Dec Stage 2', r'EGS $\to$ ConvNeXt$\,\times\,2$',             CONV_F, CONV_E, r'$112\!\times\!112\!\times\!96$'),
        ('Dec Stage 3', r'EGS $\to$ PSA $\to$ ConvNeXt$\,\times\,2$',    CONV_F, CONV_E, r'$56\!\times\!56\!\times\!192$'),
    ]
    dec_ys = enc_ys  # mirror
    for (t, s, f, e, sh), y in zip(dec_rows, dec_ys):
        box(ax, DX, y, EW, EH, t, s, f=f, e=e)
        if sh is not None:
            shape(ax, DX + EW/2, y - 0.05, sh, c=e)
    # up arrows + PatchExpanding labels between decoder boxes
    for i, (y_bot, y_top) in enumerate(zip(dec_ys[1:][::-1], dec_ys[:-1][::-1])):
        # going from lower y (Dec3) up to higher y
        arrow(ax, DX + EW/2, y_bot + EH, DX + EW/2, y_top, rad=0.0)
        # Between Dec3->Dec2, Dec2->Dec1 (i=0,1) we up-sample
        if i <= 1:
            ax.text(DX - 0.05, (y_bot + EH + y_top)/2,
                    'PatchExpand\n$\\uparrow2\\times$',
                    ha='right', va='center', fontsize=7.5,
                    color='#555555', style='italic')

    # ====== Bottleneck (compact box at bottom center) ======
    BX, BY, BW, BH = 4.45, 1.10, 5.40, 0.95
    box(ax, BX, BY, BW, BH,
        r'Hamiltonian Bottleneck  ($\times 2$)',
        s=r'ConvNeXt $\|$ 4-dir selective oscillator $\to$ gated fusion (see Fig.~1)',
        f=OSC_F, e=OSC_E)
    shape(ax, BX + BW/2, BY - 0.05, r'$28\!\times\!28\!\times\!384$', c=OSC_E)

    # Encoder Stage 3 -> Bottleneck (down + into the left side)
    arrow(ax, EX + EW/2, enc_ys[-1], BX + 0.5, BY + BH, rad=-0.20)
    ax.text(EX + EW + 0.05, (enc_ys[-1] + BY + BH)/2,
            'PatchMerging\n$\\downarrow2\\times$',
            ha='left', va='center', fontsize=7.5,
            color='#555555', style='italic')

    # ====== Three output cards (right of bottleneck) ======
    CX, CW, CH = BX + BW + 0.20, 1.20, 0.50
    box(ax, CX, BY + 0.95 - CH/2 + 0.30, CW, CH, r'$f$', s='fused', f=FEAT_F, e=FEAT_E)
    box(ax, CX, BY + 0.95 - CH/2 - 0.30, CW, CH, r'$p$', s='momentum', f=MOM_F, e=MOM_E)
    box(ax, CX, BY + 0.95 - CH/2 - 0.90, CW, CH, r'$H_{\mathrm{map}}$', s='energy',
        f=ENG_F, e=ENG_E)
    arrow(ax, BX + BW, BY + BH * 0.75, CX, BY + 0.95 - CH/2 + 0.30 + CH/2, c=FEAT_E)
    arrow(ax, BX + BW, BY + BH * 0.50, CX, BY + 0.95 - CH/2 - 0.30 + CH/2, c=MOM_E)
    arrow(ax, BX + BW, BY + BH * 0.25, CX, BY + 0.95 - CH/2 - 0.90 + CH/2, c=ENG_E)

    # ====== Three buses running UP between cards and decoder ======
    BUS_F_X = CX + CW + 0.30
    BUS_P_X = CX + CW + 0.65
    BUS_H_X = CX + CW + 1.00

    # f goes only to Dec Stage 3 (the coarsest decoder)
    dec3_y = dec_ys[-1]  # Dec Stage 3 y position (lowest)
    arrow(ax, CX + CW, BY + 0.95 - CH/2 + 0.30 + CH/2,
          DX, dec3_y + EH/2, c=FEAT_E, lw=1.6, rad=-0.05)
    ax.text(BUS_F_X + 0.20, BY - 0.25, r'$f$ bus',
            ha='left', va='center', fontsize=8.5, color=FEAT_E, style='italic')

    # p and H_map buses go vertical up the BUS column, then horizontal into each decoder stage
    bus_top = enc_ys[2] + EH/2  # Dec Stage 1 level
    # p bus
    ax.plot([BUS_P_X, BUS_P_X], [BY + 0.95 - CH/2 - 0.30 + CH/2, bus_top + 0.08],
            color=MOM_E, lw=1.6, zorder=3)
    for y in [dec_ys[2] + EH/2, dec_ys[3] + EH/2, dec_ys[4] + EH/2]:
        tee(ax, BUS_P_X, y, MOM_E)
        arrow(ax, BUS_P_X, y, DX, y - 0.10, c=MOM_E, lw=1.2)
    ax.text(BUS_P_X, bus_top + 0.18, r'$p$ bus',
            ha='center', va='bottom', fontsize=8.5, color=MOM_E, style='italic')

    # H_map bus (slightly right)
    ax.plot([BUS_H_X, BUS_H_X], [BY + 0.95 - CH/2 - 0.90 + CH/2, bus_top + 0.08],
            color=ENG_E, lw=1.6, zorder=3)
    for y in [dec_ys[2] + EH/2, dec_ys[3] + EH/2, dec_ys[4] + EH/2]:
        tee(ax, BUS_H_X, y, ENG_E)
        arrow(ax, BUS_H_X, y + 0.10, DX, y + 0.20, c=ENG_E, lw=1.2)
    ax.text(BUS_H_X, bus_top + 0.18, r'$H_{\mathrm{map}}$ bus',
            ha='center', va='bottom', fontsize=8.5, color=ENG_E, style='italic')

    # ====== Skip connections (encoder -> decoder), arched HIGH over the bottleneck ======
    # Three skips, each at a different arc height to avoid mutual overlap
    skip_levels = [
        (enc_ys[2] + EH/2, dec_ys[2] + EH/2),  # Stage 1 skip (top)
        (enc_ys[3] + EH/2, dec_ys[3] + EH/2),  # Stage 2 skip (mid)
        (enc_ys[4] + EH/2, dec_ys[4] + EH/2),  # Stage 3 skip (bot)
    ]
    for (y_enc, y_dec) in skip_levels:
        arrow(ax, EX + EW, y_enc, DX, y_dec, c=CONV_E, lw=1.4, rad=-0.05)
    # one centred annotation
    ax.text(7.5, enc_ys[2] + EH/2 + 0.20,
            'skip connections $\\to$ EGS (energy-gated, momentum-modulated)',
            ha='center', va='bottom', fontsize=8.5,
            color=CONV_E, style='italic')

    # ====== Annotation panel for EGS (bottom left, under the encoder) ======
    box(ax, 0.30, 0.20, 4.00, 0.75,
        'Energy-gated skip (EGS)',
        s=(r'$g_\ell = \sigma(\gamma_\ell (H_\ell - \bar H_\ell)),\;\;'
           r'\mathrm{out} = g_\ell\!\cdot\!\mathrm{skip} + \beta\,p$'),
        f=LIGHT_BG, e=ENG_E)

    # ====== Bottom legend ======
    items = [
        (CONV_F, CONV_E, 'ConvNeXt'),
        (OSC_F,  OSC_E,  'Oscillator / bottleneck'),
        (FUS_F,  FUS_E,  'PSA fusion'),
        (FEAT_F, FEAT_E, r'Feature $f$'),
        (MOM_F,  MOM_E,  r'Momentum $p$'),
        (ENG_F,  ENG_E,  r'Energy $H$'),
        (GREY_F, INK,    'Input / output'),
    ]
    lx = 4.60
    sw = 0.22
    for f, e, lbl in items:
        ax.add_patch(Rectangle((lx, 0.45), sw, sw,
                               linewidth=0.8, edgecolor=e,
                               facecolor=f, zorder=2))
        ax.text(lx + sw + 0.08, 0.56, lbl, ha='left', va='center',
                fontsize=8.5, color=INK)
        lx += sw + 0.08 + 0.075 * len(lbl) + 0.20

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    sp = Path(save_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sp.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.06)
    fig.savefig(sp.with_suffix('.png'), dpi=220, bbox_inches='tight',
                pad_inches=0.06, facecolor='white')
    plt.close(fig)
    print(f'Saved:\n  {sp}.pdf\n  {sp}.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='figures/fig2_hamseg_architecture_v2')
    a = p.parse_args()
    make_figure(a.out)
