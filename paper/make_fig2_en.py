"""Render the report's four figures from the summary tables in report/figdata/.

The main figure (make_main_figure.py) already carries the external-cohort
quality-vs-bias scatter, the LAA-950 agreement interval, the displacement bars and
the site gate. This file draws only what the main figure does not: the public
cohort, LAA-910, and the percentile biases. Splitting it that way keeps every
number in the report in exactly one place.

Colour is used to separate the arms, matching the main figure so one hue means one
arm across both. The palette follows the results-to-figure asset
(assets/color-palette.txt), extended with two hues for the arms that file does not
name; every hue is paired with a distinct marker or hatch so the panels still read
when printed in greyscale.

Panel ids and filenames follow that skill's naming policy (figure1_*.pdf, lowercase,
deterministic), and every panel states its source table in the caption, per
references/caption-minimum.md.

  /home/prinlab/miniconda3/envs/scte/bin/python report/make_figures.py
"""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'report', 'figdata')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,          # keep the minus sign from becoming a box
    'font.size': 8,
    'axes.linewidth': 0.7,
    'axes.edgecolor': '0',
    'axes.labelcolor': '0',
    'text.color': '0',
    'xtick.color': '0', 'ytick.color': '0',
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False,
})

# results-to-figure/assets/color-palette.txt supplies the first two; the rest keep
# the same saturation so no arm dominates by colour weight alone.
C_BASE1 = '#4C78A8'      # control  -> Lanczos
C_BASE2 = '#F58518'      # treated  -> CTHNet
C_BASE3 = '#54A24B'      # cluster_1 -> TVSRN
C_BASE4 = '#E45756'      # cluster_2 -> I3Net
C_OURS = '#6B4C9A'       # SCTE-R, zero-shot
C_OURS2 = '#B07AA1'      # SCTE-R, site-adapted
C_GRID = '#9A9A9A'

C_THICK = '#7F7F7F'          # doing nothing: the origin every other arm moves from
COLOR = {'Thick5mm': C_THICK, 'Lanczos': C_BASE1, 'CTHNet': C_BASE2,
         'TVSRN': C_BASE3, 'I3Net': C_BASE4,
         'SCTE-R': C_OURS, 'SCTE-R-zeroshot': C_OURS, 'SCTE-R-adapted': C_OURS2}

OURS = 'SCTE-R-zeroshot'
LBL = {'Thick5mm': '5 mm direct', 'Lanczos': 'Lanczos', 'CTHNet': 'CTHNet',
       'TVSRN': 'TVSRN', 'I3Net': 'I3Net',
       'SCTE-R-zeroshot': 'SCTE-R\n(zero-shot)', 'SCTE-R-adapted': 'SCTE-R\n(site-adapted)',
       'SCTE-R': 'SCTE-R'}


def load(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(fh))


def style(m):
    """ours: solid fill; baselines: lighter fill with a hatch, so the distinction
    survives a greyscale print as well as colour."""
    c = COLOR.get(m, C_GRID)
    if m.startswith('SCTE-R'):
        return dict(facecolor=c, edgecolor='0.15', hatch='')
    return dict(facecolor=c, edgecolor='0.15', hatch='///', alpha=0.55)


from matplotlib.patches import Patch

def tag(ax, letter, dx=-0.10, dy=1.10):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='left')


def panel_public(ax):
    """The public benchmark, which the main figure leaves out."""
    rows = [r for r in load('arms_quality_vs_bias.csv') if r['cohort'] == 'public']
    tag(ax, 'a')
    ax.set_title('Public cohort ($n$ = 50)', fontsize=8.2, pad=8)
    x = [float(r['psnr_db']) for r in rows]
    y = [abs(float(r['laa950_bias_pp'])) for r in rows]
    cluster = {'CTHNet', 'TVSRN', 'I3Net'}
    for xi, yi, r in zip(x, y, rows):
        ours = r['method'].startswith('SCTE-R')
        ax.scatter(xi, yi, s=62 if ours else 44, facecolor=COLOR.get(r['method'], C_GRID),
                   edgecolor='0.15', linewidth=0.9, marker='o' if ours else 's', zorder=3)
        if r['method'] not in cluster:
            ax.annotate(r['method'], (xi, yi), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=6.6)
    cx = np.mean([a for a, r in zip(x, rows) if r['method'] in cluster])
    cy = np.mean([b for b, r in zip(y, rows) if r['method'] in cluster])
    ax.annotate('CTHNet, TVSRN, I3Net', (cx, cy), textcoords='offset points',
                xytext=(-6, 26), ha='center', fontsize=6.6,
                arrowprops=dict(arrowstyle='-', lw=0.6, color=C_GRID, shrinkA=1, shrinkB=6))
    base = [(a, b) for a, b, r in zip(x, y, rows) if not r['method'].startswith('SCTE-R')]
    bx, by = [q[0] for q in base], [q[1] for q in base]
    ax.plot([min(bx), max(bx)], [np.mean(by)] * 2, color=C_GRID, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel('PSNR (dB)'); ax.set_ylabel('|LAA-950 bias| (pp)')
    ax.set_ylim(-0.25, max(y) * 1.55); ax.margins(x=0.22)
    ax.spines[['top', 'right']].set_visible(False)


def panel_laa910(ax):
    rows = [r for r in load('agreement_loa.csv') if r['endpoint'] == 'LAA-910']
    tag(ax, 'b')
    ax.set_title('LAA-910 agreement ($n$ = 444)', fontsize=8.2, pad=8)
    order = ['Thick5mm', 'Lanczos', 'CTHNet', OURS, 'SCTE-R-adapted']
    sub = {r['method']: r for r in rows}
    ys = np.arange(len(order))[::-1]
    for yi, m in zip(ys, order):
        r = sub[m]
        b, lo, hi = float(r['bias']), float(r['loa_lo']), float(r['loa_hi'])
        ours = m.startswith('SCTE-R'); c = COLOR[m]
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.6 if ours else 1.6, solid_capstyle='butt')
        for e in (lo, hi):
            ax.plot([e, e], [yi - .15, yi + .15], color=c, lw=1.1)
        ax.scatter([b], [yi], s=38, facecolor=c, edgecolor='0.15', lw=0.9,
                   marker='o' if ours else 's', zorder=3)
        ax.annotate(f'{b:+.2f}', (b, yi), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=6.5)
    ax.axvline(0, color='0.35', lw=0.8, ls=(0, (2, 2)))
    ax.set_yticks(ys, [LBL[m] for m in order], fontsize=6.9)
    ax.set_xlabel('bias and 95% limits of agreement (pp)')
    ax.set_ylim(-0.7, len(order) - 0.25)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


def panel_percentile(ax):
    pr = load('percentile_bias.csv')
    tag(ax, 'c')
    ax.set_title('Percentile bias ($n$ = 444)', fontsize=8.2, pad=8)
    ms = [r['method'] for r in pr]
    x = np.arange(len(ms)); w = 0.36
    for i, (col, sh) in enumerate((('perc15_bias_hu', ''), ('perc10_bias_hu', '///'))):
        v = [float(r[col]) for r in pr]
        ax.bar(x + (i - .5) * w, v, w, color=[COLOR.get(m, C_GRID) for m in ms],
               edgecolor='0.15', linewidth=0.7, hatch=sh, alpha=1.0 if i == 0 else 0.55)
        for xi, vi in zip(x + (i - .5) * w, v):
            ax.annotate(f'{vi:+.1f}', (xi, vi), textcoords='offset points',
                        xytext=(0, 2.5), ha='center', fontsize=6.3)
    ax.axhline(0, color='0.35', lw=0.8)
    # four categories in a narrow panel: the full labels collide, so this panel
    # uses short forms and the caption carries the full names
    short = {'Thick5mm': '5 mm\ndirect', 'Lanczos': 'Lanczos', 'CTHNet': 'CTHNet',
             'SCTE-R-zeroshot': 'SCTE-R\nzero-shot'}
    ax.set_xticks(x, [short.get(m, m) for m in ms], fontsize=6.6)
    ax.set_ylabel('bias (HU)')
    ax.set_ylim(0, 35)
    ax.legend(handles=[Patch(facecolor='0.6', edgecolor='0.15', label='Perc15'),
                       Patch(facecolor='0.6', edgecolor='0.15', hatch='///', alpha=0.55,
                             label='Perc10')], fontsize=7, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)


fig = plt.figure(figsize=(7.2, 2.95))
gs = fig.add_gridspec(1, 3, wspace=0.50, left=0.085, right=0.985, top=0.84, bottom=0.27)
panel_public(fig.add_subplot(gs[0, 0]))
panel_laa910(fig.add_subplot(gs[0, 1]))
panel_percentile(fig.add_subplot(gs[0, 2]))
out = os.path.join(OUT, 'fig2.pdf')
fig.savefig(out)
print('Output', out)
