"""Fig. 1, study overview, and Fig. 2, method.

Both target papers carry these as separate display items: the 2026 benchmark
opens with a study overview giving cohort composition and the evaluation
framework, and puts the architecture in its own figure; the 2024 paper spends
its first figure entirely on architecture. A single block diagram covering both
is thinner than either convention allows.

  /home/prinlab/miniconda3/envs/scte/bin/python paper/make_fig_overview.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 7.2,
    'text.color': '0', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})

C_TRAIN, C_EXT, C_PUB = '#4C78A8', '#6B4C9A', '#54A24B'
C_OP, C_OURS, C_GREY = '#F58518', '#6B4C9A', '#8A8A8A'


def box(ax, x, y, w, h, lines, fc='white', ec='0.3', lw=0.9, title=None,
        tfs=6.5, fs=6.0, band=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.003,rounding_size=0.008',
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
                                clip_on=False))
    if band:
        ax.add_patch(Rectangle((x + 0.003, y + 0.010), 0.007, h - 0.020, facecolor=band,
                               alpha=0.75, edgecolor='none', zorder=3, clip_on=False))
    top = y + h
    if title:
        ax.add_patch(Rectangle((x, y + h - 0.062), w, 0.062, facecolor=ec, alpha=0.92,
                               edgecolor='none', zorder=3, clip_on=False))
        ax.text(x + w / 2, y + h - 0.031, title, ha='center', va='center', color='white',
                fontsize=tfs if len(title) < 24 else tfs - 0.8, fontweight='bold',
                zorder=4, clip_on=False)
        top = y + h - 0.062
    n = max(len(lines), 1)
    ch = (top - y) / n
    for i, t in enumerate(lines):
        if i:
            ax.plot([x + 0.005, x + w - 0.005], [top - i * ch] * 2, color=ec, lw=0.45,
                    alpha=0.5, zorder=3, clip_on=False)
        ax.text(x + w / 2, top - (i + 0.5) * ch, t, ha='center', va='center',
                fontsize=fs, zorder=4, clip_on=False, linespacing=1.2)


def arr(ax, p, q, c='0.35', ls='-', rad=0.0, lw=0.9, ms=7.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, linewidth=lw,
                                 color=c, linestyle=ls, zorder=5, clip_on=False,
                                 connectionstyle=f'arc3,rad={rad}'))


def tag(ax, t, dx=-0.01, dy=1.05):
    ax.text(dx, dy, t, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', ha='left')


# ═══════════════════════════════════════════════════════ Fig. 1 study overview
fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(2, 1, left=0.045, right=0.985, top=0.945, bottom=0.035,
                      height_ratios=[1.0, 1.12], hspace=0.30)

# ---- a: cohorts
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
tag(ax, 'a', dy=1.14)
ax.text(0.030, 1.13, 'Cohorts', transform=ax.transAxes, fontsize=8.4,
        fontweight='bold', va='top')

box(ax, 0.000, 0.52, 0.205, 0.40,
    ['203 paired examinations', 'thick and thin from the', 'same raw acquisition'],
    '#EDF2F8', C_TRAIN, title='Hospital A  (training)', band=C_TRAIN)
box(ax, 0.000, 0.03, 0.205, 0.40,
    ['50 whole volumes', 'released real-paired', 'benchmark'],
    '#EEF6EC', C_PUB, title='Public  (RPLHR-CT)', band=C_PUB)
box(ax, 0.280, 0.03, 0.235, 0.89,
    ['494 paired examinations', 'two manufacturers,', 'two reconstruction kernels',
     'contributed no training data'],
    '#F0EBF6', C_EXT, title='Hospital B  (external)', band=C_EXT)

box(ax, 0.590, 0.52, 0.185, 0.40, ['50 examinations', 'protocol constants', 'indicator thresholds'],
    'white', C_EXT, title='Calibration split')
box(ax, 0.590, 0.03, 0.185, 0.40, ['444 examinations', 'soft kernel  311', 'sharp kernel  126'],
    'white', C_EXT, title='Reported split')

box(ax, 0.840, 0.27, 0.160, 0.42,
    ['444  descriptive', '438  paired tests', '6 excluded, non-finite'],
    'white', '0.35', title='Analysis set')

arr(ax, (0.205, 0.72), (0.280, 0.62), rad=-0.12, c=C_TRAIN)
arr(ax, (0.205, 0.23), (0.280, 0.33), rad=0.12, c=C_PUB)
arr(ax, (0.515, 0.72), (0.590, 0.72), c=C_EXT)
arr(ax, (0.515, 0.23), (0.590, 0.23), c=C_EXT)
arr(ax, (0.775, 0.23), (0.840, 0.42), rad=-0.18, c='0.4')
ax.text(0.243, 0.86, 'model\ntraining', fontsize=5.6, color=C_TRAIN, ha='center', va='top')
ax.text(0.243, 0.14, 'pretraining\nweights', fontsize=5.6, color=C_PUB, ha='center', va='top')

# ---- b: evaluation framework
ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
tag(ax, 'b', dy=1.10)
ax.text(0.030, 1.09, 'Evaluation framework', transform=ax.transAxes, fontsize=8.4,
        fontweight='bold', va='top')

box(ax, 0.000, 0.30, 0.150, 0.46, ['5 mm series', 'as acquired'], '#F2F2F2', C_GREY,
    title='Input')
box(ax, 0.205, 0.045, 0.215, 0.90,
    ['read directly, no reconstruction', 'Lanczos interpolation', 'CTHNet',
     'TVSRN   (public cohort)', 'I3Net   (public cohort)',
     'SCTE-R, zero-shot', 'SCTE-R, site-adapted'],
    'white', '0.3', title='Arms compared', fs=5.7)
box(ax, 0.475, 0.30, 0.185, 0.46, ['matched 1 mm series', 'same raw acquisition'],
    '#EFE9F6', C_OURS, title='Reference', band=C_OURS)
box(ax, 0.745, 0.045, 0.255, 0.90,
    ['LAA-950, LAA-910', 'Perc15, Perc10', 'agreement by lobe',
     'PSNR, SSIM', 'displacement  $\\hat{\\delta}$', 'indicator pass rate'],
    'white', '0.3', title='Endpoints', fs=6.0)

arr(ax, (0.150, 0.53), (0.205, 0.53), c=C_GREY)
arr(ax, (0.420, 0.53), (0.475, 0.53), c='0.35')
arr(ax, (0.660, 0.53), (0.745, 0.53), c=C_OURS)
ax.text(0.447, 0.24, 'compared\nagainst', fontsize=5.6, color='0.3', ha='center', va='top')

fig.savefig(os.path.join(OUT, 'fig1_overview.pdf'))
fig.savefig(os.path.join(OUT, 'fig1_overview.png'), dpi=300)
plt.close(fig)
print('wrote', os.path.join(OUT, 'fig1_overview.pdf'))


# ═══════════════════════════════════════════════════════════ Fig. 2 method
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(3, 1, left=0.055, right=0.985, top=0.945, bottom=0.030,
                      height_ratios=[0.86, 1.10, 0.92], hspace=0.42)

# ---- a: forward model
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
tag(ax, 'a', dy=1.16)
ax.text(0.030, 1.15, 'Forward model of thick-slice reconstruction',
        transform=ax.transAxes, fontsize=8.2, fontweight='bold', va='top')

# 1 mm stack drawn as thin slabs, 5 mm as thick ones
for i in range(15):
    ax.add_patch(Rectangle((0.030, 0.10 + i * 0.052), 0.085, 0.044,
                           facecolor='#EFE9F6', edgecolor=C_OURS, lw=0.5, clip_on=False))
ax.text(0.0725, 0.02, '1 mm  $x$', fontsize=6.4, ha='center')
for i in range(3):
    ax.add_patch(Rectangle((0.700, 0.13 + i * 0.255), 0.085, 0.225,
                           facecolor='#EDF2F8', edgecolor=C_TRAIN, lw=0.8, clip_on=False))
ax.text(0.7425, 0.02, '5 mm  $y$', fontsize=6.4, ha='center')

# the kernel: a Gaussian profile drawn along z
zz = np.linspace(-1, 1, 120)
g = np.exp(-4.0 * zz ** 2)
ax.plot(0.235 + 0.075 * g, 0.50 + 0.36 * zz, color=C_OP, lw=1.5, clip_on=False)
ax.plot([0.235, 0.235], [0.14, 0.86], color='0.6', lw=0.6, ls=(0, (2, 2)), clip_on=False)
ax.annotate('', (0.235 + 0.075 * np.exp(-1.0), 0.50 + 0.36 * 0.5),
            (0.235 + 0.075 * np.exp(-1.0), 0.50 - 0.36 * 0.5),
            arrowprops=dict(arrowstyle='<->', lw=0.8, color=C_OP))
ax.text(0.330, 0.50, 'FWHM $w$', fontsize=6.2, color=C_OP, va='center')
ax.text(0.150, 0.97, 'through-plane sensitivity', fontsize=6.0, color=C_OP,
        ha='left', va='top')

box(ax, 0.455, 0.30, 0.195, 0.44,
    ['$\\downarrow r$,  $r = 5$', 'slab mean preserved'],
    '#FDF0E1', C_OP, title='slab averaging', band=C_OP)
arr(ax, (0.120, 0.50), (0.225, 0.50), c=C_OP)
arr(ax, (0.420, 0.50), (0.455, 0.50), c=C_OP)
arr(ax, (0.650, 0.50), (0.695, 0.50), c=C_OP)
ax.text(0.7425, 0.97, '$y = A_w x + \\varepsilon$', fontsize=7.4, ha='center', va='top')
ax.text(0.845, 0.44, 'non-trivial null space:\ndetail that averages to\nzero in a slab leaves\nno trace in $y$',
        fontsize=5.8, ha='left', va='center', linespacing=1.35)

# ---- b: sampling loop, unrolled
ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
tag(ax, 'b', dy=1.12)
ax.text(0.030, 1.11, 'Residual-space sampling, unrolled', transform=ax.transAxes,
        fontsize=8.2, fontweight='bold', va='top')

box(ax, 0.000, 0.44, 0.140, 0.42, ['$x_0$ = upsample($y$)', 'or frozen backbone'],
    '#F4F4F4', '0.45', title='Base predictor', fs=5.8)
ax.text(0.070, 0.33, '$u_0 \\sim \\mathcal{N}(0, I)$', fontsize=6.4, ha='center')

xs = [0.215, 0.375, 0.535, 0.695]
for k, x0 in enumerate(xs):
    lab = ['$t = 0$', '$t = 1/T$', '$\\cdots$', '$t = 1$'][k]
    box(ax, x0, 0.44, 0.120, 0.42,
        ['$u \\leftarrow u + \\frac{1}{T} v_\\theta$', 'project to $A_w\\hat{x} \\approx y$'],
        '#EFE9F6', C_OURS, lw=1.1, title=lab, fs=5.5, tfs=6.2)
    if k:
        arr(ax, (xs[k - 1] + 0.120, 0.65), (x0, 0.65), c=C_OURS)
arr(ax, (0.140, 0.65), (0.215, 0.65), c='0.45')

box(ax, 0.855, 0.44, 0.145, 0.42, ['$\\hat{x} = x_0 + s_r u$', 'texture retained'],
    '#EFE9F6', C_OURS, lw=1.2, title='Estimate', fs=5.8)
arr(ax, (0.815, 0.65), (0.855, 0.65), c=C_OURS)

box(ax, 0.215, 0.02, 0.600, 0.30,
    ['velocity field  $v_\\theta(u, t \\mid y, w)$  conditioned on the observation and the fitted width'],
    'white', C_OURS, lw=0.9, fs=6.2)
for x0 in xs:
    arr(ax, (x0 + 0.060, 0.32), (x0 + 0.060, 0.44), c=C_OURS, ls=(0, (2, 2)), lw=0.7, ms=5.5)

# ---- c: the three reference-free terms
ax = fig.add_subplot(gs[2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
tag(ax, 'c', dy=1.14)
ax.text(0.030, 1.13, 'Reference-free terms and the site-level verdict',
        transform=ax.transAxes, fontsize=8.2, fontweight='bold', va='top')

box(ax, 0.000, 0.30, 0.150, 0.50, ['$R = A_w\\hat{x} - y$', 'on eroded lung $\\Omega_e$'],
    '#FDF0E1', C_OP, title='Residual', fs=5.9, band=C_OP)
terms = [(0.200, '$\\hat{\\delta} = \\overline{R}$', 'mean direction of $R$;\nslab-mean preservation\nmakes it identifiable'),
         (0.420, r'$\rho_{\rm struct} = \|R - \hat{\delta}\| \,/\, \|y - \bar{y}\|$',
          'what remains once the\ndisplacement is removed'),
         (0.640, '$s = (\\hat{v} - {\\rm Var}_z\\hat{x}) / \\sigma_v$',
          'through-plane variance\nagainst its calibrated\nprediction')]
for x0, eq, note in terms:
    ax.add_patch(FancyBboxPatch((x0, 0.30), 0.175, 0.50,
                                boxstyle='round,pad=0.003,rounding_size=0.008',
                                facecolor='white', edgecolor=C_OP, lw=1.0, zorder=2,
                                clip_on=False))
    ax.text(x0 + 0.0875, 0.70, eq, fontsize=7.0, ha='center', va='center')
    ax.plot([x0 + 0.008, x0 + 0.167], [0.60] * 2, color=C_OP, lw=0.45, alpha=0.5)
    ax.text(x0 + 0.0875, 0.45, note, fontsize=5.5, ha='center', va='center', linespacing=1.3)
# the three terms are parallel readouts of the same residual. A direct fan-out
# would cross the intervening boxes, so the feed runs as a bus underneath them
BUS = 0.19
ax.plot([0.075, 0.075], [0.30, BUS], color=C_OP, lw=0.8, clip_on=False)
ax.plot([0.075, 0.728 + 0.0875], [BUS] * 2, color=C_OP, lw=0.8, clip_on=False)
for x0, _, _ in terms:
    arr(ax, (x0 + 0.0875, BUS), (x0 + 0.0875, 0.295), c=C_OP, lw=0.8, ms=6.0)
for x0, _, _ in terms:
    ax.plot([x0 + 0.0875, x0 + 0.0875], [0.80, 0.88], color=C_OP, lw=0.7, clip_on=False)
ax.plot([0.2875, 0.7275], [0.88] * 2, color=C_OP, lw=0.7, clip_on=False)
arr(ax, (0.7275, 0.88), (0.9225, 0.88), c=C_OP, lw=0.8)
ax.plot([0.9225, 0.9225], [0.88, 0.80], color=C_OP, lw=0.7, clip_on=False)

box(ax, 0.845, 0.30, 0.155, 0.50,
    ['$|\\hat{\\delta}| < \\tau_\\delta$', '$\\rho_{\\rm struct} < \\tau_\\rho$', '$|s| < \\tau_s$'],
    'white', C_OP, lw=1.2, title='Verdict', fs=6.2)

ax.text(0.9225, 0.26, 'thresholds from the\nnull distribution on\ncalibration cases',
        fontsize=5.5, ha='center', va='top', color='0.35', linespacing=1.3)

fig.savefig(os.path.join(OUT, 'fig2_method.pdf'))
fig.savefig(os.path.join(OUT, 'fig2_method.png'), dpi=300)
plt.close(fig)
print('wrote', os.path.join(OUT, 'fig2_method.pdf'))
