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
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root: DATA/ and RECON/ live beside report/
DATA, OUT = os.path.join(HERE, 'figdata'), os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False, 'font.size': 7.2,
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
LBL = {'Thick5mm': '5 mm 直测', 'Lanczos': 'Lanczos 插值', 'CTHNet': 'CTHNet',
       'SCTE-R-zeroshot': 'SCTE-R（零样本）', 'SCTE-R-adapted': 'SCTE-R（站点适配）'}

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
def box(ax, x, y, w, h, text, fc='white', ec='0.25', lw=0.9, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.005,rounding_size=0.012',
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
                                clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            zorder=3, linespacing=1.35, clip_on=False)


def arrow(ax, p, q, ls='-', color='0.3', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7.5,
                                 linewidth=0.9, color=color, linestyle=ls, zorder=2,
                                 clip_on=False, connectionstyle=f'arc3,rad={rad}'))


def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    tag(ax, 'a', dx=-0.012, dy=1.26)
    ax.text(0.035, 1.25, '方法与效度判据', transform=ax.transAxes, fontsize=8.4,
            fontweight='bold', va='top')

    y1, h1 = 0.58, 0.34
    y2, h2 = 0.02, 0.32
    box(ax, 0.000, y1, 0.112, h1, '5 mm\n厚层观测 y', fc='#EDF2F8')
    box(ax, 0.142, y1, 0.138, h1, '基预测 x0\n上采样／冻结骨干', fc='#F7F7F7', fs=6.3)
    box(ax, 0.310, y1, 0.218, h1,
        '残差空间流匹配采样\nu = (x − x0) / s_r\nEuler 积分 + 数据一致性投影',
        fc='#F0EBF6', ec=C_OURS, lw=1.2, fs=6.1)
    box(ax, 0.558, y1, 0.108, h1, '1 mm\n重建 x', fc='#F0EBF6', ec=C_OURS, lw=1.2)
    box(ax, 0.700, y1, 0.300, h1, '密度学终点\nLAA-950／LAA-910／Perc15')
    box(ax, 0.700, y2, 0.300, h2, '无参考效度判据\nδ、ρ_struct、s → 通过／标记',
        ec=C_BASE2, lw=1.2)
    box(ax, 0.310, y2, 0.310, h2,
        '前向算子 A_w\n(z 向高斯 + 跨层平均, r = 5)\n均值保持', fc='#FDF3E8',
        ec=C_BASE2, lw=1.0, fs=6.1)

    for a, b in ((0.112, 0.142), (0.280, 0.310), (0.528, 0.558), (0.666, 0.700)):
        arrow(ax, (a, y1 + h1 / 2), (b, y1 + h1 / 2))
    arrow(ax, (0.612, y1), (0.700, y2 + h2 * 0.74), rad=-0.28)
    arrow(ax, (0.419, y1), (0.419, y2 + h2), ls=(0, (3, 2)), color=C_BASE2)
    arrow(ax, (0.620, y2 + h2 * 0.36), (0.700, y2 + h2 * 0.36), ls=(0, (3, 2)), color=C_BASE2)
    ax.text(0.660, y2 + h2 * 0.36 - 0.04, '均值保持\n无需 1 mm 参考', fontsize=5.9,
            color=C_BASE2, ha='center', va='top', linespacing=1.3)
    ax.text(0.0, y1 - 0.10, '（无薄层设备的医院仅有此项）', fontsize=6.0, color='0.35')


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
    cols = [(thickup, '5 mm（上采样）'), (base, 'CTHNet'), (rec, 'SCTE-R'), (thin, '1 mm 参考')]
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
            ax.set_ylabel('冠状位', fontsize=6.8)
            tag(ax, 'b', dx=-0.16, dy=1.34)
        axz = axes[1][j]
        axz.imshow(im[zz:zz + zh, zx:zx + zw], cmap='gray', vmin=WIN[0], vmax=WIN[1],
                   aspect='equal', interpolation='nearest')
        axz.set_xticks([]); axz.set_yticks([])
        axz.set_anchor('N')           # and the zoom row at the top of its own
        for s in axz.spines.values():
            s.set_color('#F5D000'); s.set_linewidth(1.0)
        if j == 0:
            axz.set_ylabel('放大', fontsize=6.8)


def panel_c(ax, thin, thickup, base, rec, lung):
    """LAA as a function of its own threshold.

    A histogram buries the informative region under the parenchymal peak. Sweeping
    the threshold shows the whole family of threshold statistics at once: where the
    curves separate is where a method will misreport, and the standard cut-offs are
    read straight off the x-axis.
    """
    tag(ax, 'c')
    ax.set_title('LAA 随阈值的变化', fontsize=8, pad=6)
    thr = np.arange(-1000, -860, 2.0)
    vals = {}
    for vol, lab, c, lw, ls in ((thin, '1 mm 参考', '0.15', 1.6, '-'),
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
    ax.set_xlabel('阈值 (HU)'); ax.set_ylabel('低于阈值的肺体素 (%)')
    ax.set_yscale('log'); ax.set_ylim(0.03, 60)
    ax.set_xlim(thr[0], thr[-1])
    ax.legend(fontsize=6.2, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)


# =============================================================== panels d-f
def panel_d(ax):
    rows = [r for r in load('arms_quality_vs_bias.csv') if r['cohort'] == 'external']
    tag(ax, 'd')
    ax.set_title('图像质量的提升不转化为定量准确性', fontsize=8, pad=6)
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
            'PSNR 相差 3.2 dB，偏差仅挽回 0.5 pp（10%）',
            ha='center', va='top', fontsize=6.1, color='0.2')
    # SCTE-R sits at lower PSNR than doing nothing, with the bias nearly gone
    thick = next(q for q, r in zip(y, rows) if r['method'] == 'Thick5mm')
    ours = next((a, b) for a, b, r in zip(x, y, rows) if r['method'] == OURS)
    ax.annotate('', (ours[0], ours[1] + 0.3), (ours[0], thick - 0.1),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color=C_OURS))
    ax.text(ours[0] + 0.10, (thick + ours[1]) / 2, '挽回 95%', fontsize=6.4,
            color=C_OURS, ha='left', va='center')
    ax.set_xlabel('PSNR (dB)'); ax.set_ylabel('|LAA-950 偏差| (pp)')
    ax.set_ylim(-1.6, max(y) * 1.38); ax.margins(x=0.24)
    ax.spines[['top', 'right']].set_visible(False)


def panel_e(ax):
    rows = load('agreement_loa.csv')
    tag(ax, 'e')
    ax.set_title('LAA-950 的偏差与 95% 一致性界（n = 444）', fontsize=8, pad=6)
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
    ax.set_xlabel('偏差与 95% 一致性界 (pp)')
    ax.set_ylim(-0.7, len(order) - 0.2)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


def panel_f(ax):
    rows = load('displacement.csv')
    tag(ax, 'f')
    ax.set_title('确定性回归收敛到恒定的密度平移', fontsize=8, pad=6)
    lab = {'deterministic-regression': '确定性回归', 'bias-only-control': '纯标量偏置',
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
    ax.set_xlabel('整体密度位移 δ (HU)')
    ax.set_xlim(min(v) - 19, 3.5)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


def panel_g(ax):
    rows = load('site_gate.csv')
    tag(ax, 'g')
    ax.set_title('判据通过率随模型对本域的胜任度变化', fontsize=8, pad=6)
    lab = {'public-test': '公开测试集\n(学过)', 'external-soft-kernel': '外部软核\n(同核族)',
           'external-sharp-kernel': '外部锐核\n(未学过)',
           'external-sharp-adapted': '外部锐核\n(适配后)'}
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
    ax.set_xticks(x, ['%s\n%s' % (lab[r['domain']], '终点失效' if f else '终点正常')
                      for r, f in zip(rows, failed)], fontsize=6.0)
    ax.set_ylabel('判据通过率 (%)'); ax.set_ylim(0, 112)
    ax.spines[['top', 'right']].set_visible(False)


# =============================================================== layout
thin, thickup, base, rec, lung = load_case()

fig = plt.figure(figsize=(7.2, 8.9))
gs = GridSpec(5, 4, figure=fig,
              height_ratios=[0.80, 0.78, 0.78, 1.02, 0.96],
              hspace=0.52, wspace=0.40,
              left=0.085, right=0.975, top=0.955, bottom=0.045)

panel_a(fig.add_subplot(gs[0, :]))

# b occupies the left three columns over two rows (full view + zoom); c sits right
gsb = gs[1:3, 0:3].subgridspec(2, 4, hspace=0.06, wspace=0.04,
                               height_ratios=[1.0, 1.0])
axes_b = [[fig.add_subplot(gsb[i, j]) for j in range(4)] for i in range(2)]
panel_b(axes_b, thin, thickup, base, rec)
panel_c(fig.add_subplot(gs[1:3, 3]), thin, thickup, base, rec, lung)

panel_d(fig.add_subplot(gs[3, 0:2]))
panel_e(fig.add_subplot(gs[3, 2:4]))
panel_f(fig.add_subplot(gs[4, 0:2]))
panel_g(fig.add_subplot(gs[4, 2:4]))

fig.savefig(os.path.join(OUT, 'figure0_main.pdf'))
fig.savefig(os.path.join(OUT, 'figure0_main.png'), dpi=300)
print('输出', os.path.join(OUT, 'figure0_main.pdf'))
