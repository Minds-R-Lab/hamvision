#!/usr/bin/env python3
"""Redesigned Figure 1: Hamiltonian-bottleneck block diagram.

Same colour palette as Figure 3 (HamCls architecture): navy / charcoal.
Cleaner arrows: single style, consistent thickness, no crossings.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# Colour palette aligned with Figure 3 (hamcls architecture)
NAVY_DARK   = '#1d4866'   # primary navy
NAVY_MED    = '#23577a'   # lighter navy for ConvNeXt path
CHARCOAL    = '#1a1a18'   # gate / dark accent
INK         = '#0d0d0c'   # main text / strong borders
GREY_STROKE = '#444444'   # arrow stroke
GREY_LIGHT  = '#e8eaed'   # input box background
WHITE       = '#ffffff'

# Accent colours for the three structured outputs (q, p, H)
# Sit in the same hue family but are visually distinct.
C_POS  = '#3a8fbf'   # sky blue   - position q
C_MOM  = '#bf7a3a'   # warm tan   - momentum p
C_ENG  = '#bfa83a'   # muted gold - energy H

ARROW_COLOR = '#3a3a36'
ARROW_LW    = 1.6
ARROW_MUT   = 14
FONT_TITLE  = 11
FONT_BODY   = 9.5
FONT_SMALL  = 8.5


def box(ax, x, y, w, h, label, sublabel=None,
        face=WHITE, edge=INK, text_color=INK, lw=1.4, radius=0.06):
    """A rounded rectangle with a centred (multi-line) label."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(patch)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label,
                ha='center', va='center', fontsize=FONT_TITLE,
                color=text_color, fontweight='bold', zorder=3)
        ax.text(x + w/2, y + h*0.30, sublabel,
                ha='center', va='center', fontsize=FONT_SMALL,
                color=text_color, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, label,
                ha='center', va='center', fontsize=FONT_TITLE,
                color=text_color, fontweight='bold', zorder=3)


def arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=ARROW_LW, mut=ARROW_MUT,
          connectionstyle='arc3,rad=0.0'):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=mut,
        color=color, linewidth=lw,
        connectionstyle=connectionstyle, zorder=4,
        shrinkA=0, shrinkB=2)
    ax.add_patch(a)


def make_figure(save_path):
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 5.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # ====================================================
    # Stage 1: Input feature map (left edge)
    # ====================================================
    box(ax, 0.30, 2.45, 1.30, 0.95,
        'Input', sublabel=r'$x_\ell$ feature map',
        face=GREY_LIGHT, edge=INK)

    # Splitter dot
    ax.scatter([2.05], [2.93], s=44, color=INK, zorder=5)

    # ====================================================
    # Stage 2: Two parallel paths
    # ====================================================
    # ConvNeXt path (top)
    box(ax, 2.85, 3.95, 2.20, 1.05,
        'ConvNeXt block', sublabel=r'depth-wise $7\!\times\!7$ + MLP',
        face=NAVY_MED, edge=NAVY_DARK, text_color=WHITE)

    # Oscillator path (bottom)
    box(ax, 2.85, 0.85, 2.20, 1.05,
        'Hamiltonian SS2D', sublabel=r'damped oscillator, 4-dir scan',
        face=NAVY_DARK, edge=INK, text_color=WHITE)

    # Branching from splitter
    arrow(ax, 1.60, 2.93, 2.05, 2.93)                 # input -> splitter
    arrow(ax, 2.05, 2.93, 2.85, 4.47,                 # up to ConvNeXt
          connectionstyle='arc3,rad=-0.18')
    arrow(ax, 2.05, 2.93, 2.85, 1.37,                 # down to Oscillator
          connectionstyle='arc3,rad=0.18')

    # ====================================================
    # Stage 3: Three structured outputs from the oscillator
    # ====================================================
    # Output cards (small, on the right of the oscillator), stacked vertically.
    # These are the canonical (q, p, H) maps the rest of the network consumes.
    card_x = 5.85
    box(ax, card_x, 1.85, 1.40, 0.70,
        r'$q$', sublabel='position',
        face=C_POS, edge=NAVY_DARK, text_color=WHITE)
    box(ax, card_x, 0.95, 1.40, 0.70,
        r'$p$', sublabel='momentum',
        face=C_MOM, edge=INK, text_color=WHITE)
    box(ax, card_x, 0.05, 1.40, 0.70,
        r'$H$', sublabel='energy',
        face=C_ENG, edge=INK, text_color=WHITE)

    # Oscillator -> (q, p, H) cards
    arrow(ax, 5.05, 1.55, card_x, 2.20)               # q
    arrow(ax, 5.05, 1.37, card_x, 1.30)               # p
    arrow(ax, 5.05, 1.10, card_x, 0.40)               # H

    # ====================================================
    # Stage 4: Gate fusion (top right)
    # ====================================================
    box(ax, 7.85, 3.95, 1.85, 1.05,
        'Gate fusion', sublabel=r'$g\!\cdot\! c + (1{-}g)\!\cdot\! q$',
        face=CHARCOAL, edge=INK, text_color=WHITE)

    # ConvNeXt -> Gate (straight right)
    arrow(ax, 5.05, 4.47, 7.85, 4.47)

    # q -> Gate (up to top right)
    arrow(ax, card_x + 1.40, 2.20, 8.78, 3.95,
          connectionstyle='arc3,rad=-0.25', color=C_POS, lw=ARROW_LW)

    # ====================================================
    # Stage 5: Three outputs to the decoder/head (right edge)
    # ====================================================
    # Fused features (out)
    box(ax, 10.00, 3.95, 1.40, 1.05,
        r'$y_\ell$', sublabel='fused output',
        face=NAVY_DARK, edge=INK, text_color=WHITE)
    arrow(ax, 9.70, 4.47, 10.00, 4.47)

    # p_out card
    box(ax, 10.00, 2.30, 1.40, 0.70,
        r'$p$', sublabel='to decoder',
        face=C_MOM, edge=INK, text_color=WHITE)
    arrow(ax, card_x + 1.40, 1.30, 10.00, 2.65,
          connectionstyle='arc3,rad=-0.18', color=C_MOM, lw=ARROW_LW)

    # H_out card
    box(ax, 10.00, 1.10, 1.40, 0.70,
        r'$H$', sublabel='to skip gates',
        face=C_ENG, edge=INK, text_color=WHITE)
    arrow(ax, card_x + 1.40, 0.40, 10.00, 1.45,
          connectionstyle='arc3,rad=-0.20', color=C_ENG, lw=ARROW_LW)

    # ====================================================
    # Stage labels and title
    # ====================================================
    ax.text(5.85, 5.45, 'Hamiltonian bottleneck',
            ha='center', va='center', fontsize=13,
            fontweight='bold', color=INK)
    ax.text(5.85, 5.10,
            'shared by HamSeg (decoder) and HamCls (PSSP head)',
            ha='center', va='center', fontsize=9, color=GREY_STROKE)

    # Subtle path labels
    ax.text(3.95, 3.65, 'feature path',
            ha='center', va='center', fontsize=FONT_SMALL,
            color=NAVY_DARK, style='italic')
    ax.text(3.95, 2.15, 'physics path',
            ha='center', va='center', fontsize=FONT_SMALL,
            color=NAVY_DARK, style='italic')

    # ====================================================
    # Legend strip at the bottom
    # ====================================================
    legend_y = 0.0 - 0.45  # just below the axes; will be cropped by tight_layout
    items = [
        (GREY_LIGHT, INK,       'Input / output feature map'),
        (NAVY_MED,   NAVY_DARK, 'ConvNeXt feature path'),
        (NAVY_DARK,  INK,       'Hamiltonian oscillator path'),
        (CHARCOAL,   INK,       'Fusion gate'),
        (C_POS,      NAVY_DARK, r'Position $q$'),
        (C_MOM,      INK,       r'Momentum $p$'),
        (C_ENG,      INK,       r'Energy $H$'),
    ]
    # Render the legend inside the figure bottom strip
    legend_y = 0.20
    lx = 0.30
    swatch_w, swatch_h = 0.22, 0.22
    for face, edge, label in items:
        sq = Rectangle((lx, legend_y - swatch_h/2), swatch_w, swatch_h,
                       linewidth=0.9, edgecolor=edge, facecolor=face, zorder=2)
        ax.add_patch(sq)
        ax.text(lx + swatch_w + 0.10, legend_y, label,
                ha='left', va='center', fontsize=FONT_SMALL, color=INK)
        # Crude width estimate per legend item
        lx += swatch_w + 0.12 + 0.085 * len(label) + 0.12

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix('.pdf'),
                bbox_inches='tight', pad_inches=0.06)
    fig.savefig(save_path.with_suffix('.png'),
                dpi=220, bbox_inches='tight', pad_inches=0.06,
                facecolor='white')
    plt.close(fig)
    print(f'Saved:\n  {save_path}.pdf\n  {save_path}.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='figures/fig1_block_diagram_v2',
                   help='Output basename (no extension)')
    args = p.parse_args()
    make_figure(args.out)
