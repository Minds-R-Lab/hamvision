#!/usr/bin/env python3
"""
generate_block_diagram.py
==========================

Generates a clean, professional replacement for the Figure 1 block diagram
of the Hamiltonian Bottleneck. Outputs PDF (for LaTeX inclusion) and PNG
(for quick preview).

Design principles:
  - Left-to-right horizontal flow, single visual row.
  - Two parallel paths (ConvNeXt + Oscillator) clearly grouped inside a
    labelled "Hamiltonian Bottleneck" container.
  - Three structured outputs (features f, momentum p, energy H_map) as
    distinct cards on the right, each with shape + downstream arrow.
  - Compact equation labels beside arrows, not inside boxes.
  - One colour per data type. No marketing language inside boxes.

Usage:
    python generate_block_diagram.py --out figures/fig1_block_diagram
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


# ------------------------------------------------------------------
# Colour palette (chosen to be print-friendly and colour-blind safe)
# ------------------------------------------------------------------
C_INPUT      = '#E6E8EB'   # neutral gray
C_INPUT_EDGE = '#8A8E94'

C_CONVNEXT   = '#DDE7F2'   # light blue
C_CONVNEXT_E = '#3D6EA8'

C_OSC        = '#F4E4C9'   # warm amber
C_OSC_E      = '#B5862E'

C_GATE       = '#E8DDF2'   # light purple
C_GATE_E     = '#6C4FA1'

C_FEAT       = '#374A5A'   # dark slate (fused features)
C_FEAT_FILL  = '#D9DFE6'

C_POS        = '#C6E2DC'   # teal (position q)
C_POS_E      = '#2F8377'

C_MOM        = '#F2D6DE'   # rose (momentum p)
C_MOM_E      = '#B23C5E'

C_ENG        = '#F8DDB8'   # gold (energy H)
C_ENG_E      = '#C77F1A'

C_TEXT       = '#1F1F1F'
C_LIGHT_TEXT = '#5A5A5A'
C_ARROW      = '#3A3A3A'


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------

def box(ax, x, y, w, h, label, sub=None, fill='#FFFFFF', edge='#1F1F1F',
        label_fs=11, sub_fs=8.5, label_color=C_TEXT, sub_color=C_LIGHT_TEXT,
        radius=0.04, lw=1.2, label_weight='bold'):
    """Rounded rectangle with bold label and optional small subtitle."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0.005,rounding_size={radius}',
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       zorder=2)
    ax.add_patch(p)
    cx = x + w / 2
    if sub is not None:
        ax.text(cx, y + h * 0.62, label, ha='center', va='center',
                fontsize=label_fs, color=label_color, weight=label_weight,
                zorder=3)
        ax.text(cx, y + h * 0.28, sub, ha='center', va='center',
                fontsize=sub_fs, color=sub_color, style='italic',
                zorder=3)
    else:
        ax.text(cx, y + h / 2, label, ha='center', va='center',
                fontsize=label_fs, color=label_color, weight=label_weight,
                zorder=3)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, mut=14,
          label=None, label_off=(0, 0.10), label_fs=9, style='-',
          curve=0.0):
    """Straight or slightly curved arrow with optional centred label."""
    connstyle = f'arc3,rad={curve}' if curve else 'arc3,rad=0'
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle='-|>', mutation_scale=mut,
                        linewidth=lw, color=color,
                        connectionstyle=connstyle,
                        linestyle=style,
                        zorder=1)
    ax.add_patch(a)
    if label is not None:
        mx = (x0 + x1) / 2 + label_off[0]
        my = (y0 + y1) / 2 + label_off[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=label_fs, color=C_LIGHT_TEXT, zorder=3,
                bbox=dict(facecolor='white', edgecolor='none',
                          pad=1.2, alpha=0.92))


def output_card(ax, x, y, w, h, symbol, name, shape, dest,
                fill, edge, sym_color=None):
    """A small card with a big math symbol, a name, a shape, and a
    downstream destination label below."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.005,rounding_size=0.06',
                       linewidth=1.4, edgecolor=edge, facecolor=fill,
                       zorder=2)
    ax.add_patch(p)
    sc = sym_color if sym_color is not None else edge
    ax.text(x + w * 0.22, y + h * 0.62, symbol, ha='center', va='center',
            fontsize=18, color=sc, weight='bold', zorder=3)
    ax.text(x + w * 0.62, y + h * 0.70, name, ha='left', va='center',
            fontsize=10.5, color=C_TEXT, weight='bold', zorder=3)
    ax.text(x + w * 0.62, y + h * 0.43, shape, ha='left', va='center',
            fontsize=8.5, color=C_LIGHT_TEXT, style='italic', zorder=3)
    ax.text(x + w * 0.62, y + h * 0.20, dest, ha='left', va='center',
            fontsize=8.5, color=C_LIGHT_TEXT, zorder=3)


# ------------------------------------------------------------------
# Main figure
# ------------------------------------------------------------------

def make_figure(save_path):
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.4)
    ax.axis('off')

    # ----- Container for the Hamiltonian Bottleneck -----
    container = Rectangle((1.55, 0.50), 7.10, 4.55,
                          linewidth=1.0, edgecolor='#BFC2C7',
                          facecolor='#FAFAFB', linestyle='--',
                          zorder=0)
    ax.add_patch(container)
    ax.text(1.65, 4.78, 'Hamiltonian Bottleneck',
            ha='left', va='center',
            fontsize=10.5, color=C_LIGHT_TEXT, weight='bold', style='italic')

    # ----- Input -----
    box(ax, 0.10, 2.20, 1.35, 1.00,
        label=r'Input $\mathbf{x}$',
        sub=r'$8C{=}384,\ 28{\times}28$',
        fill=C_INPUT, edge=C_INPUT_EDGE)

    # ----- ConvNeXt path (upper) -----
    box(ax, 1.85, 3.45, 2.45, 1.10,
        label='ConvNeXt block',
        sub=r'DW $7\!\times\!7\,\to\,$PW$\,\to\,$GELU',
        fill=C_CONVNEXT, edge=C_CONVNEXT_E)

    # ----- Oscillator path (lower) -----
    box(ax, 1.85, 1.05, 2.45, 1.10,
        label='Oscillator bank',
        sub=r'$4$-dir scan, complex $z = q + ip$',
        fill=C_OSC, edge=C_OSC_E)

    # Equation label below oscillator
    ax.text(3.075, 0.78,
            r'$z_t = e^{(-\nu+i\omega)\Delta}\, z_{t-1} + u_t$',
            ha='center', va='center', fontsize=9.5,
            color=C_LIGHT_TEXT)

    # ----- Arrows from input into the two paths -----
    arrow(ax, 1.45, 2.85, 1.85, 4.00, lw=1.4)   # to ConvNeXt
    arrow(ax, 1.45, 2.55, 1.85, 1.60, lw=1.4)   # to Oscillator

    # ----- Re / Im / |z|^2 extraction from oscillator -----
    # Three small chip-style markers attached to the right edge of the
    # oscillator box, each clearly labelled.
    chips = [
        (r'$q = \mathrm{Re}(z)$',     C_POS, C_POS_E, 2.05),
        (r'$p = \mathrm{Im}(z)$',     C_MOM, C_MOM_E, 1.60),
        (r'$H = \tfrac{1}{2}|z|^2$',  C_ENG, C_ENG_E, 1.15),
    ]
    for label, fill, edge, cy in chips:
        c = FancyBboxPatch((4.32, cy - 0.16), 0.95, 0.32,
                           boxstyle='round,pad=0.005,rounding_size=0.05',
                           linewidth=1.0, edgecolor=edge, facecolor=fill,
                           zorder=3)
        ax.add_patch(c)
        ax.text(4.32 + 0.475, cy, label,
                ha='center', va='center', fontsize=8.5,
                color=edge, weight='bold', zorder=4)

    # ----- Gate fusion -----
    box(ax, 5.20, 2.65, 1.55, 0.65,
        label='Gate',
        sub=r'$g = \sigma(W[f_{\mathrm{conv}};f_{\mathrm{osc}}])$',
        fill=C_GATE, edge=C_GATE_E,
        label_fs=10.5, sub_fs=8)

    # ConvNeXt output flows down into gate
    arrow(ax, 4.30, 4.00, 5.20, 3.10, lw=1.4)
    # Position chip flows up into gate
    arrow(ax, 5.27, 2.05, 5.20, 2.80, lw=1.4)

    # Bias annotation, placed just under the gate so it doesn't collide
    # with the momentum arrow
    ax.text(5.975, 2.55,
            r'bias $b_g = +2.0$',
            ha='center', va='top', fontsize=8.5, color=C_LIGHT_TEXT,
            style='italic')

    # ----- SE channel attention block (for energy) -----
    box(ax, 5.20, 0.85, 1.55, 0.85,
        label='SE channel\nattention',
        sub=r'$w_c = \sigma(\mathrm{MLP}(\mathrm{GAP}\,H_c))$',
        fill=C_OSC, edge=C_OSC_E,
        label_fs=10, sub_fs=8)

    # Energy chip flows down into SE
    arrow(ax, 5.27, 1.00, 5.30, 1.65, lw=1.4)

    # ----- Three output cards on the right -----
    out_x = 8.95
    out_w = 3.85
    out_h = 1.10

    # Features (top)
    output_card(ax, out_x, 3.55, out_w, out_h,
                symbol=r'$f$', name='Fused features',
                shape=r'$\,384 \times 28 \times 28$',
                dest=r'$\to$ task-specific head',
                fill='#EEF1F5', edge=C_FEAT,
                sym_color=C_FEAT)

    # Momentum (middle)
    output_card(ax, out_x, 2.20, out_w, out_h,
                symbol=r'$p$', name='Momentum',
                shape=r'$\,384 \times 28 \times 28$',
                dest=r'$\to$ injected at every decoder level',
                fill=C_MOM, edge=C_MOM_E)

    # Energy (bottom)
    output_card(ax, out_x, 0.85, out_w, out_h,
                symbol=r'$H_{\mathrm{map}}$', name='Energy map',
                shape=r'$\,1 \times 28 \times 28$',
                dest=r'$\to$ gates encoder skip connections',
                fill=C_ENG, edge=C_ENG_E)

    # ----- Arrows from gate / chips / SE into the output cards -----
    # Features f: from the gate
    arrow(ax, 6.75, 2.97, 8.95, 4.10, lw=1.6, mut=14)
    # Momentum p: routed straight from the Im(z) chip; we draw it as
    # a two-segment polyline that goes right, then down to the card,
    # so it never crosses the SE block.
    ax.plot([5.27, 7.95], [1.60, 1.60], color=C_MOM_E, lw=1.6, zorder=1)
    arrow(ax, 7.95, 1.60, 8.95, 2.75, lw=1.6, mut=14, color=C_MOM_E)
    # Energy H_map: from the SE block to its card
    arrow(ax, 6.75, 1.27, 8.95, 1.40, lw=1.6, mut=14, color=C_ENG_E)

    # ----- Subtle legend strip at the bottom -----
    legend_y = 0.18
    legend_items = [
        (C_CONVNEXT, C_CONVNEXT_E, 'ConvNeXt path'),
        (C_OSC,      C_OSC_E,      'Oscillator path'),
        (C_GATE,     C_GATE_E,     'Fusion gate'),
        (C_POS,      C_POS_E,      r'Position $q$'),
        (C_MOM,      C_MOM_E,      r'Momentum $p$'),
        (C_ENG,      C_ENG_E,      r'Energy $H$'),
    ]
    lx = 1.50
    for f, e, name in legend_items:
        sq = Rectangle((lx, legend_y - 0.10), 0.20, 0.20,
                       linewidth=0.9, edgecolor=e, facecolor=f, zorder=2)
        ax.add_patch(sq)
        ax.text(lx + 0.28, legend_y, name, ha='left', va='center',
                fontsize=8.5, color=C_TEXT)
        lx += 1.85

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(save_path + '.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(save_path + '.png', dpi=200, bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig1_block_diagram',
                    help='Output path without extension (.pdf and .png are added).')
    args = ap.parse_args()
    draw(args.out)
    print(f'wrote {args.out}.pdf and {args.out}.png')
