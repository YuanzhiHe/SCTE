"""The paper's main figure, laid out the way this literature lays one out.

The closest precedent is CTHNet (npj Digital Medicine 2024), our own backbone and
baseline: a schematic, then representative through-plane views with one column per
method, then the quantitative panels. The earlier version of this figure had no
images at all, which for a slice-thickness paper leaves out the thing a radiologist
looks at first.

Panels a-c come from the public aligned cohort, where the reconstructions are held
locally; panels d-f report the 444-case external cohort from the summary tables in
figdata/. The caption states which cohort each panel belongs to, because they are
not the same patients.

  /home/prinlab/miniconda3/envs/scte/bin/python report/make_main_figure.py
"""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root: DATA/ and RECON/ live beside report/
DATA = os.path.join(os.path.dirname(HERE), 'report', 'figdata')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 7.2,
    'axes.linewidth': 0.7, 'axes.edgecolor': '0.15', 'axes.labelcolor': '0',
    'text.color': '0', 'xtick.color': '0.15', 'ytick.color': '0.15',
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False,
})

C_BASE1, C_BASE2, C_BASE3, C_BASE4 = '#4C78A8', '#F58518', '#54A24B', '#E45756'
C_OURS, C_OURS2, C_GRID = '#6B4C9A', '#B07AA1', '#9A9A9A'
C_THICK = '#7F7F7F'          # doing nothing: the origin every other arm moves from
COLOR = {'Thick5mm': C_THICK, 'Lanczos': C_BASE1, 'CTHNet': C_BASE2,
         'TVSRN': C_BASE3, 'I3Net': C_BASE4,
         'SCTE-R': C_OURS, 'SCTE-R-zeroshot': C_OURS, 'SCTE-R-adapted': C_OURS2}
OURS = 'SCTE-R-zeroshot'
LBL = {'Thick5mm': '5 mm direct', 'Lanczos': 'Lanczos', 'CTHNet': 'CTHNet',
       'SCTE-R-zeroshot': 'SCTE-R (zero-shot)', 'SCTE-R-adapted': 'SCTE-R (site-adapted)'}

CASE = 'CT00000102'          # the local case with the largest low-density burden
CORONAL_Y = 147
WIN = (-1000, -200)          # lung window for display
R = 5


def load(n):
    with open(os.path.join(DATA, n)) as fh:
        return list(csv.DictReader(fh))


def tag(ax, letter, dx=-0.09, dy=1.10, fs=10):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fs,
            fontweight='bold', va='top', ha='left')


# =============================================================== panel a
def sub(ax, x, y, w, h, lines, fc, ec, lw=1.0, title=None, tfs=6.6, fs=5.9,
        hatch=None):
    """A module box with a title strip and its own sub-cells.

    One box per stage was too coarse: the stages differ in what they consume and
    what they hand on, and a single block of text inside one rectangle hides that.
    Each stage is drawn as a titled module whose sub-cells carry one item each.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.010',
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
                                clip_on=False))
    if hatch:
        # a band at the left edge rather than a fill: a hatch across the whole box
        # runs through every line of text inside it
        ax.add_patch(Rectangle((x + 0.004, y + 0.012), 0.009, h - 0.024,
                               facecolor=ec, alpha=0.65, edgecolor='none',
                               zorder=3, clip_on=False))
    ty = y + h
    if title:
        ax.add_patch(Rectangle((x, y + h - 0.085), w, 0.085, facecolor=ec, alpha=0.90,
                               edgecolor='none', zorder=3, clip_on=False))
        ax.text(x + w / 2, y + h - 0.0425, title, ha='center', va='center',
                fontsize=tfs, color='white', fontweight='bold', zorder=4, clip_on=False)
        ty = y + h - 0.085
    n = len(lines)
    ch = (ty - y) / max(n, 1)
    for i, t in enumerate(lines):
        cy = ty - (i + 0.5) * ch
        if i:
            ax.plot([x + 0.006, x + w - 0.006], [ty - i * ch] * 2, color=ec, lw=0.5,
                    alpha=0.55, zorder=3, clip_on=False)
        ax.text(x + w / 2, cy, t, ha='center', va='center', fontsize=fs, zorder=4,
                clip_on=False, linespacing=1.25)


def arrow(ax, p, q, ls='-', color='0.3', rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7.5,
                                 linewidth=lw, color=color, linestyle=ls, zorder=5,
                                 clip_on=False, connectionstyle=f'arc3,rad={rad}'))


def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    tag(ax, 'a', dx=-0.012, dy=1.10)
    ax.text(0.035, 1.09, 'Method: from the 5 mm series to endpoints and a validity indicator',
            transform=ax.transAxes, fontsize=8.4, fontweight='bold', va='top')

    yT, hT = 0.545, 0.40         # top row: the pipeline
    yB, hB = 0.035, 0.375        # bottom row: the physics and what it enables

    sub(ax, 0.000, yT, 0.152, hT, ['5 mm series $y$', 'the only one\navailable'],
        '#E8EFF7', C_BASE1, title='Input', hatch='.')
    sub(ax, 0.178, yT, 0.150, hT, ['upsampling', 'or frozen backbone', 'gives $x_0$'],
        '#F4F4F4', '#6E6E6E', title='Base predictor')
    sub(ax, 0.354, yT, 0.220, hT,
        ['$u=(x-x_0)/s_r$', 'Euler, $t\\!:\\!0\\!\\to\\!1$', 'data-consistency step'],
        '#EFE9F6', C_OURS, lw=1.3, title='Flow matching')
    sub(ax, 0.600, yT, 0.130, hT, ['1 mm estimate', '$\\hat{x}$'],
        '#EFE9F6', C_OURS, lw=1.3, title='Output')
    sub(ax, 0.756, yT, 0.244, hT, ['LAA-950 / LAA-910', 'Perc15 / Perc10', 'five lobes'],
        'white', '#3A3A3A', title='Endpoints')

    sub(ax, 0.354, yB, 0.220, hB,
        ['$z$ Gaussian (FWHM $w$)', 'slab averaging ($r=5$)', 'mean-preserving'],
        '#FDF0E1', C_BASE2, lw=1.1, title='Forward operator $A_w$', hatch='.', fs=5.7)
    sub(ax, 0.600, yB, 0.130, hB, ['$\\hat{\\delta}$', '$\\rho_{\\rm struct}$', '$s$'],
        '#FDF0E1', C_BASE2, lw=1.1, title='Reference-free', fs=6.4)
    sub(ax, 0.756, yB, 0.244, hB,
        ['pass / flag', 'site-level competence', 'no per-scan claim'],
        'white', C_BASE2, lw=1.1, title='Validity indicator')

    for a, b in ((0.152, 0.178), (0.328, 0.354), (0.574, 0.600), (0.730, 0.756)):
        arrow(ax, (a, yT + hT / 2), (b, yT + hT / 2))
    for a, b in ((0.574, 0.600), (0.730, 0.756)):
        arrow(ax, (a, yB + hB / 2), (b, yB + hB / 2), color=C_BASE2)
    arrow(ax, (0.464, yT), (0.464, yB + hB), ls=(0, (3, 2)), color=C_BASE2)
    arrow(ax, (0.665, yT), (0.665, yB + hB), ls=(0, (3, 2)), color=C_OURS)
    ax.text(0.479, (yT + yB + hB) / 2, 'constrains', fontsize=5.8, color=C_BASE2,
            va='center', ha='left')
    ax.text(0.680, (yT + yB + hB) / 2, '$\\hat{x}$', fontsize=6.2, color=C_OURS,
            va='center', ha='left')
    ax.text(0.878, yB - 0.075, 'no 1 mm reference', fontsize=6.0,
            color=C_BASE2, ha='center', va='top')


# =============================================================== panels b, c
def coronal(vol, y):
    return vol[:, y, :]


def upsample_z(v, r=R):
    return np.repeat(v, r, axis=0)


def load_case():
    d = os.path.join(ROOT, 'DATA', 'aligned_test')
    thin = np.load(os.path.join(d, f'{CASE}_thin.npy')).astype(np.float32)
    thick = np.load(os.path.join(d, f'{CASE}_thick.npy')).astype(np.float32)
    base = np.load(os.path.join(d, f'{CASE}_base.npy')).astype(np.float32)
    rec = np.load(os.path.join(ROOT, 'RECON', 's1', f'{CASE}_rec.npy')).astype(np.float32)
    lung = np.load(os.path.join(d, f'{CASE}_lung.npy')) > 0
    n = min(thin.shape[0], base.shape[0], rec.shape[0], thick.shape[0] * R)
    return thin[:n], upsample_z(thick)[:n], base[:n], rec[:n], lung[:n]


def panel_b(axes, thin, thickup, base, rec):
    """One column per method, through-plane (coronal) view, plus a zoom row.

    Through-plane is the only view where 5 mm acquisition differs from 1 mm, so it
    is the view this literature shows.
    """
    cols = [(thickup, '5 mm (upsampled)'), (base, 'CTHNet'), (rec, 'SCTE-R'), (thin, '1 mm reference')]
    # zoom box chosen as the parenchyma-richest window of this slice, so the inset
    # shows lung texture rather than mediastinum
    zx, zw, zz, zh = 80, 95, 50, 70
    for j, (vol, name) in enumerate(cols):
        im = coronal(vol, CORONAL_Y)
        ax = axes[0][j]
        ax.imshow(im, cmap='gray', vmin=WIN[0], vmax=WIN[1], aspect='equal',
                  interpolation='nearest')
        ax.add_patch(Rectangle((zx, zz), zw, zh, fill=False, edgecolor='#F5D000',
                               linewidth=0.8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_anchor('S')            # sit at the bottom of the cell
        for s in ax.spines.values():
            s.set_color(COLOR.get(name.split('（')[0], '0.3') if name in ('CTHNet', 'SCTE-R')
                        else '0.3')
            s.set_linewidth(1.4 if name == 'SCTE-R' else 0.8)
        ax.set_title(name, fontsize=7.0, pad=3)
        if j == 0:
            ax.set_ylabel('coronal', fontsize=6.8)
            tag(ax, 'b', dx=-0.16, dy=1.34)
        axz = axes[1][j]
        axz.imshow(im[zz:zz + zh, zx:zx + zw], cmap='gray', vmin=WIN[0], vmax=WIN[1],
                   aspect='equal', interpolation='nearest')
        axz.set_xticks([]); axz.set_yticks([])
        axz.set_anchor('N')           # and the zoom row at the top of its own
        for s in axz.spines.values():
            s.set_color('#F5D000'); s.set_linewidth(1.0)
        if j == 0:
            axz.set_ylabel('zoom', fontsize=6.8)


def panel_c(ax, thin, thickup, base, rec, lung):
    """LAA as a function of its own threshold.

    A histogram buries the informative region under the parenchymal peak. Sweeping
    the threshold shows the whole family of threshold statistics at once: where the
    curves separate is where a method will misreport, and the standard cut-offs are
    read straight off the x-axis.
    """
    tag(ax, 'c')
    ax.set_title('LAA vs. threshold', fontsize=8, pad=6)
    thr = np.arange(-1000, -860, 2.0)
    vals = {}
    for vol, lab, c, lw, ls in ((thin, '1 mm reference', '0.15', 1.6, '-'),
                                (thickup, '5 mm', C_BASE1, 1.1, (0, (4, 2))),
                                (base, 'CTHNet', C_BASE2, 1.2, '-'),
                                (rec, 'SCTE-R', C_OURS, 1.6, '-')):
        v = vol[lung]
        y = np.array([100.0 * (v < t).mean() for t in thr])
        vals[lab] = y
        ax.plot(thr, y, color=c, lw=lw, ls=ls, label=lab, zorder=3 if lab == 'SCTE-R' else 2)
    for t, name in ((-950, 'LAA-950'), (-910, 'LAA-910')):
        ax.axvline(t, color='0.45', lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.text(t, 0.035, f' {name}', fontsize=6.0, color='0.3',
                rotation=90, va='bottom', ha='left')
    ax.set_xlabel('threshold (HU)'); ax.set_ylabel('below threshold (%)')
    ax.set_yscale('log'); ax.set_ylim(0.03, 60)
    ax.set_xlim(thr[0], thr[-1])
    ax.legend(fontsize=6.0, loc='upper left', frameon=True, framealpha=1.0,
              facecolor='white', edgecolor='none', borderpad=0.25,
              handlelength=1.5, labelspacing=0.35)
    ax.spines[['top', 'right']].set_visible(False)


# =============================================================== panels d-f
def panel_d(ax):
    rows = [r for r in load('arms_quality_vs_bias.csv') if r['cohort'] == 'external']
    tag(ax, 'd')
    ax.set_title('Image quality gains do not reach the measurement', fontsize=8, pad=6)
    x = [float(r['psnr_db']) for r in rows]
    y = [abs(float(r['laa950_bias_pp'])) for r in rows]
    for xi, yi, r in zip(x, y, rows):
        ours = r['method'].startswith('SCTE-R')
        ax.scatter(xi, yi, s=58 if ours else 42, facecolor=COLOR[r['method']],
                   edgecolor='0.15', lw=0.9, marker='o' if ours else 's', zorder=3)
        below = r['method'] == OURS          # its label would sit under the 95% arrow
        ax.annotate(LBL[r['method']].replace('（', '\n（'), (xi, yi),
                    textcoords='offset points', xytext=(0, -9 if below else 8),
                    ha='center', va='top' if below else 'baseline', fontsize=6.0)
    base = [(a, b) for a, b, r in zip(x, y, rows) if not r['method'].startswith('SCTE-R')]
    bx, by = [p[0] for p in base], [p[1] for p in base]
    ax.plot([min(bx), max(bx)], [np.mean(by)] * 2, color=C_GRID, lw=1.0, ls=(0, (4, 3)))
    ax.annotate('', (min(bx), np.mean(by) - 0.42), (max(bx), np.mean(by) - 0.42),
                arrowprops=dict(arrowstyle='<->', lw=0.8, color='0.35'))
    ax.text(np.mean(bx), np.mean(by) - 0.95,
            '3.2 dB apart; 0.5 pp recovered (10%)',
            ha='center', va='top', fontsize=6.1, color='0.2')
    # SCTE-R sits at lower PSNR than doing nothing, with the bias nearly gone
    thick = next(q for q, r in zip(y, rows) if r['method'] == 'Thick5mm')
    ours = next((a, b) for a, b, r in zip(x, y, rows) if r['method'] == OURS)
    # a diagonal arrow from doing nothing to the sampler shows the move itself,
    # and keeps clear of the two labels stacked near x = 27-28.5
    tx = next(a for a, r in zip(x, rows) if r['method'] == 'Thick5mm')
    ax.annotate('', xy=(ours[0] - 0.06, ours[1] + 0.32), xytext=(tx - 0.10, thick - 0.18),
                arrowprops=dict(arrowstyle='-|>', lw=1.2, color=C_OURS,
                                connectionstyle='arc3,rad=0.28'))
    ax.text(tx - 1.30, thick - 0.55, '95%\nrecovered', fontsize=6.4, color=C_OURS,
            ha='left', va='top', linespacing=1.25)
    ax.set_xlabel('PSNR (dB)'); ax.set_ylabel('|LAA-950 bias| (pp)')
    ax.set_ylim(-2.3, max(y) * 1.30); ax.margins(x=0.26)
    ax.spines[['top', 'right']].set_visible(False)


def panel_e(ax):
    rows = load('agreement_loa.csv')
    tag(ax, 'e')
    ax.set_title('LAA-950 bias and 95% limits of agreement ($n$ = 444)', fontsize=8, pad=6)
    order = ['Thick5mm', 'Lanczos', 'CTHNet', OURS, 'SCTE-R-adapted']
    sub = {r['method']: r for r in rows if r['endpoint'] == 'LAA-950'}
    ys = np.arange(len(order))[::-1]
    for yi, m in zip(ys, order):
        r = sub[m]
        b, lo, hi = float(r['bias']), float(r['loa_lo']), float(r['loa_hi'])
        ours = m.startswith('SCTE-R'); c = COLOR[m]
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.5 if ours else 1.5, solid_capstyle='butt')
        for e in (lo, hi):
            ax.plot([e, e], [yi - .15, yi + .15], color=c, lw=1.1)
        ax.scatter([b], [yi], s=36, facecolor=c, edgecolor='0.15', lw=0.9,
                   marker='o' if ours else 's', zorder=3)
        ax.annotate(f'{b:+.2f}', (b, yi), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=6.2)
    ax.axvline(0, color='0.35', lw=0.8, ls=(0, (2, 2)))
    ax.set_yticks(ys, [LBL[m] for m in order], fontsize=6.6)
    ax.set_xlabel('bias and 95% limits of agreement (pp)')
    ax.set_ylim(-0.7, len(order) - 0.2)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


def panel_f(ax):
    rows = load('displacement.csv')
    tag(ax, 'f')
    ax.set_title('A constant attenuation displacement', fontsize=8, pad=6)
    lab = {'deterministic-regression': 'Regression', 'bias-only-control': 'Scalar offset',
           'SCTE-R': 'SCTE-R'}
    v = [float(r['delta_hu']) for r in rows]
    e = [float(r['delta_sd']) for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, v, 0.5, xerr=e, color=[C_BASE2, C_BASE4, C_OURS], edgecolor='0.15',
            lw=0.7, alpha=0.85, error_kw=dict(ecolor='0.2', lw=0.9, capsize=2.5))
    for yi, vi, ei in zip(y, v, e):
        ax.annotate(f'{vi:.2f} ± {ei:.2f}', (vi - ei, yi), textcoords='offset points',
                    xytext=(-6, -2.3), fontsize=6.1, ha='right')
    ax.axvline(0, color='0.35', lw=0.8)
    ax.set_yticks(y, [lab[r['arm']] for r in rows], fontsize=6.6)
    ax.set_xlabel('global attenuation displacement $\\delta$ (HU)')
    ax.set_xlim(min(v) - 19, 3.5)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


def panel_g(ax):
    rows = load('site_gate.csv')
    tag(ax, 'g')
    ax.set_title('Pass rate by model-domain competence', fontsize=8, pad=6)
    lab = {'public-test': 'public test\n(in domain)', 'external-soft-kernel': 'external, soft\n(same family)',
           'external-sharp-kernel': 'external, sharp\n(unseen)',
           'external-sharp-adapted': 'external, sharp\n(adapted)'}
    x = np.arange(len(rows))
    v = [float(r['certified_pct']) for r in rows]
    failed = [r['endpoint_status'] == 'failed' for r in rows]
    ax.bar(x, v, 0.56, color=[C_BASE4 if f else C_OURS for f in failed],
           edgecolor='0.15', lw=0.7, alpha=0.85,
           hatch=['///' if f else '' for f in failed])
    for xi, vi in zip(x, v):
        ax.annotate(f'{vi:.1f}%', (xi, vi), textcoords='offset points', xytext=(0, 3),
                    ha='center', fontsize=6.3)
    # the endpoint status goes in the tick label: a legend collides either with the
    # title or with the 99% bar, whichever corner it is placed in
    ax.set_xticks(x, ['%s\n%s' % (lab[r['domain']], 'endpoints failed' if f else 'endpoints intact')
                      for r, f in zip(rows, failed)], fontsize=6.0)
    ax.set_ylabel('pass rate (%)'); ax.set_ylim(0, 114)
    ax.spines[['top', 'right']].set_visible(False)


# =============================================================== layout
thin, thickup, base, rec, lung = load_case()

fig = plt.figure(figsize=(7.2, 9.0))
L, Rt = 0.105, 0.978

# Panel a sits in its own grid: it is a schematic and needs far less vertical
# breathing room than the data panels, and one shared hspace cannot serve both.
gsA = fig.add_gridspec(1, 1, left=L, right=Rt, top=0.972, bottom=0.788)
panel_a(fig.add_subplot(gsA[0, 0]))

gsM = fig.add_gridspec(3, 4, left=L, right=Rt, top=0.735, bottom=0.050,
                       height_ratios=[1.28, 0.92, 0.86], hspace=0.62, wspace=0.42)

gsb = gsM[0, 0:3].subgridspec(2, 4, hspace=0.06, wspace=0.04, height_ratios=[1.0, 1.0])
axes_b = [[fig.add_subplot(gsb[i, j]) for j in range(4)] for i in range(2)]
panel_b(axes_b, thin, thickup, base, rec)
panel_c(fig.add_subplot(gsM[0, 3]), thin, thickup, base, rec, lung)

panel_d(fig.add_subplot(gsM[1, 0:2]))
panel_e(fig.add_subplot(gsM[1, 2:4]))
panel_f(fig.add_subplot(gsM[2, 0:2]))
panel_g(fig.add_subplot(gsM[2, 2:4]))

fig.savefig(os.path.join(OUT, 'fig3_results.pdf'))
fig.savefig(os.path.join(OUT, 'fig3_results.png'), dpi=300)
print('Output', os.path.join(OUT, 'figure0_main.pdf'))
