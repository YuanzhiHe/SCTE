"""The paper's main figure: workflow schematic plus the panels that carry the claim.

Panel A is a schematic and has no source table; the data panels reuse the same
tables and the same palette as the standalone figures in make_figures.py, so a
reader comparing them sees one colour per arm throughout.

The biomedical-codex-skills results-to-figure planner covers volcano, heatmap,
UMAP and bar panels only, so it returns no plan for a schematic. Its naming
policy still applies to the export.

  /home/prinlab/miniconda3/envs/scte/bin/python report/make_main_figure.py
"""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, 'figdata'), os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False, 'font.size': 7.5,
    'axes.linewidth': 0.7, 'axes.edgecolor': '0.15', 'axes.labelcolor': '0',
    'text.color': '0', 'xtick.color': '0.15', 'ytick.color': '0.15',
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False,
})

C_BASE1, C_BASE2, C_BASE3, C_BASE4 = '#4C78A8', '#F58518', '#54A24B', '#E45756'
C_OURS, C_OURS2, C_GRID = '#6B4C9A', '#B07AA1', '#9A9A9A'
COLOR = {'Lanczos': C_BASE1, 'CTHNet': C_BASE2, 'TVSRN': C_BASE3, 'I3Net': C_BASE4,
         'SCTE-R': C_OURS, 'SCTE-R-zeroshot': C_OURS, 'SCTE-R-adapted': C_OURS2}
OURS = 'SCTE-R-zeroshot'
LBL = {'Lanczos': 'Lanczos 插值', 'CTHNet': 'CTHNet',
       'SCTE-R-zeroshot': 'SCTE-R（零样本）', 'SCTE-R-adapted': 'SCTE-R（站点适配）'}


def load(n):
    with open(os.path.join(DATA, n)) as fh:
        return list(csv.DictReader(fh))


def tag(ax, letter, dx=-0.085, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10.5,
            fontweight='bold', va='top', ha='left')


# ------------------------------------------------------------------ panel A
def box(ax, x, y, w, h, text, fc='white', ec='0.25', lw=0.9, fs=7.2, bold=False):
    # clip_on=False: a box whose edge lands on the axes boundary loses that edge,
    # which is how the two output boxes came out with no right-hand border.
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
                                clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            zorder=3, fontweight='bold' if bold else 'normal', linespacing=1.35,
            clip_on=False)


def arrow(ax, p, q, text=None, rad=0.0, ls='-', color='0.3', tdy=0.032, fs=6.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8,
                                 linewidth=0.9, color=color, zorder=2,
                                 linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + tdy, text, ha='center',
                va='bottom', fontsize=fs, color='0.2', zorder=3)


def panelA(ax):
    """Left-to-right pipeline on the top row, the physics and the two outputs below.

    Combining marks (x-hat, delta-hat) are absent from Noto Sans CJK SC and render as
    boxes, so the schematic writes the plain letters and the caption carries the hats.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(-0.012, 1.14, 'A', transform=ax.transAxes, fontsize=10.5,
            fontweight='bold', va='top', ha='left')
    ax.text(0.038, 1.13, '工作流：厚层观测 → 残差空间采样 → 定量终点与效度判据',
            transform=ax.transAxes, fontsize=8.2, fontweight='bold', va='top')

    y1, h1 = 0.60, 0.30          # pipeline row
    y2, h2 = 0.05, 0.28          # physics and certificate row
    box(ax, 0.000, y1, 0.110, h1, '5 mm\n厚层观测 y', fc='#EDF2F8', fs=6.9)
    box(ax, 0.140, y1, 0.135, h1, '基预测 x0\n上采样／冻结骨干', fc='#F7F7F7', fs=6.6)
    box(ax, 0.305, y1, 0.215, h1,
        '残差空间流匹配采样\nu = (x − x0) / s_r\nEuler 积分 + 数据一致性投影',
        fc='#F0EBF6', ec=C_OURS, lw=1.2, fs=6.4)
    box(ax, 0.550, y1, 0.105, h1, '1 mm\n重建 x', fc='#F0EBF6', ec=C_OURS, lw=1.2, fs=6.9)
    box(ax, 0.690, y1, 0.300, h1, '密度学终点\nLAA-950／LAA-910／Perc15', fs=6.6)
    box(ax, 0.690, y2, 0.300, h2,
        '无参考效度判据\nδ、ρ_struct、s → 通过／标记', ec=C_BASE2, lw=1.2, fs=6.6)
    box(ax, 0.305, y2, 0.305, h2,
        '前向算子 A_w\n(z 向高斯 + 跨层平均, r = 5)\n均值保持', fc='#FDF3E8',
        ec=C_BASE2, lw=1.0, fs=6.4)

    for a, b in ((0.110, 0.140), (0.275, 0.305), (0.520, 0.550), (0.655, 0.690)):
        arrow(ax, (a, y1 + h1 / 2), (b, y1 + h1 / 2))
    ax.add_patch(FancyArrowPatch((0.603, y1), (0.690, y2 + h2 * 0.72), arrowstyle='-|>',
                                 mutation_scale=8, linewidth=0.9, color='0.3',
                                 connectionstyle='arc3,rad=-0.28', zorder=2,
                                 clip_on=False))
    # the operator constrains the sampler and supplies the certificate's estimator
    arrow(ax, (0.410, y1), (0.410, y2 + h2), ls=(0, (3, 2)), color=C_BASE2)
    ax.add_patch(FancyArrowPatch((0.610, y2 + h2 * 0.35), (0.690, y2 + h2 * 0.35),
                                 arrowstyle='-|>', mutation_scale=8, linewidth=0.9,
                                 color=C_BASE2, linestyle=(0, (3, 2)), zorder=2,
                                 clip_on=False))
    ax.text(0.650, y2 + h2 * 0.35 - 0.035, '均值保持\n无需 1 mm 参考', fontsize=6.0,
            color=C_BASE2, ha='center', va='top', linespacing=1.3)
    ax.text(0.0, y1 - 0.075, '（无薄层设备的医院仅有此项）', fontsize=6.2, color='0.35')


# ------------------------------------------------------------------ panel B
def panelB(ax):
    rows = [r for r in load('arms_quality_vs_bias.csv') if r['cohort'] == 'external']
    tag(ax, 'B')
    ax.set_title('图像质量的提升不转化为定量准确性', fontsize=8, pad=7)
    x = [float(r['psnr_db']) for r in rows]
    y = [abs(float(r['laa950_bias_pp'])) for r in rows]
    for xi, yi, r in zip(x, y, rows):
        ours = r['method'].startswith('SCTE-R')
        ax.scatter(xi, yi, s=64 if ours else 46, facecolor=COLOR[r['method']],
                   edgecolor='0.15', lw=0.9, marker='o' if ours else 's', zorder=3)
        ax.annotate(LBL[r['method']].replace('（', '\n（'), (xi, yi),
                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=6.2)
    base = [(a, b) for a, b, r in zip(x, y, rows) if not r['method'].startswith('SCTE-R')]
    bx, by = [p[0] for p in base], [p[1] for p in base]
    ax.plot([min(bx), max(bx)], [np.mean(by)] * 2, color=C_GRID, lw=1.0, ls=(0, (4, 3)))
    ax.annotate('', (min(bx), np.mean(by) - 0.42), (max(bx), np.mean(by) - 0.42),
                arrowprops=dict(arrowstyle='<->', lw=0.8, color='0.35'))
    ax.text(np.mean(bx), np.mean(by) - 1.55, 'PSNR 相差 2.0 dB，\n偏差相差 0.07 pp',
            ha='center', fontsize=6.4, color='0.2', linespacing=1.3)
    ax.set_xlabel('PSNR (dB)'); ax.set_ylabel('|LAA-950 偏差| (pp)')
    ax.set_ylim(-0.4, max(y) * 1.5); ax.margins(x=0.22)
    ax.spines[['top', 'right']].set_visible(False)


# ------------------------------------------------------------------ panel C
def panelC(ax):
    rows = load('agreement_loa.csv')
    tag(ax, 'C')
    ax.set_title('LAA-950 的偏差与 95% 一致性界（n = 444）', fontsize=8, pad=7)
    order = ['Lanczos', 'CTHNet', OURS, 'SCTE-R-adapted']
    sub = {r['method']: r for r in rows if r['endpoint'] == 'LAA-950'}
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
                    ha='center', fontsize=6.4)
    ax.axvline(0, color='0.35', lw=0.8, ls=(0, (2, 2)))
    ax.set_yticks(ys, [LBL[m] for m in order], fontsize=6.8)
    ax.set_xlabel('偏差与 95% 一致性界 (pp)')
    ax.set_ylim(-0.7, len(order) - 0.25)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


# ------------------------------------------------------------------ panel D
def panelD(ax):
    rows = load('displacement.csv')
    tag(ax, 'D')
    ax.set_title('确定性回归收敛到一个恒定的密度平移', fontsize=8, pad=7)
    lab = {'deterministic-regression': '确定性回归', 'bias-only-control': '纯标量偏置',
           'SCTE-R': 'SCTE-R'}
    v = [float(r['delta_hu']) for r in rows]
    e = [float(r['delta_sd']) for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, v, 0.52, xerr=e, color=[C_BASE2, C_BASE4, C_OURS], edgecolor='0.15',
            lw=0.7, alpha=0.85, error_kw=dict(ecolor='0.2', lw=0.9, capsize=2.5))
    for yi, vi, ei in zip(y, v, e):
        ax.annotate(f'{vi:.2f} ± {ei:.2f}', (vi - ei, yi), textcoords='offset points',
                    xytext=(-5, -2.4), fontsize=6.4, ha='right')
    ax.axvline(0, color='0.35', lw=0.8)
    ax.set_yticks(y, [lab[r['arm']] for r in rows], fontsize=6.8)
    ax.set_xlabel('整体密度位移 δ (HU)')
    ax.set_xlim(min(v) - 18, 3.5)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


# ------------------------------------------------------------------ panel E
def panelE(ax):
    rows = load('site_gate.csv')
    tag(ax, 'E')
    ax.set_title('判据通过率随模型对本域的胜任度变化', fontsize=8, pad=7)
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
                    ha='center', fontsize=6.6)
    ax.set_xticks(x, [lab[r['domain']] for r in rows], fontsize=6.4)
    ax.set_ylabel('判据通过率 (%)'); ax.set_ylim(0, 112)
    ax.legend(handles=[Patch(facecolor=C_OURS, edgecolor='0.15', alpha=0.85, label='终点正常'),
                       Patch(facecolor=C_BASE4, edgecolor='0.15', alpha=0.85, hatch='///',
                             label='终点失效')],
              fontsize=6.4, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.spines[['top', 'right']].set_visible(False)


fig = plt.figure(figsize=(7.2, 7.3))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.86, 1.0, 0.95],
              hspace=0.50, wspace=0.32,
              left=0.085, right=0.975, top=0.905, bottom=0.055)
panelA(fig.add_subplot(gs[0, :]))
panelB(fig.add_subplot(gs[1, 0]))
panelC(fig.add_subplot(gs[1, 1]))
panelD(fig.add_subplot(gs[2, 0]))
panelE(fig.add_subplot(gs[2, 1]))
fig.savefig(os.path.join(OUT, 'figure0_main.pdf'))
fig.savefig(os.path.join(OUT, 'figure0_main.png'), dpi=300)
print('输出', os.path.join(OUT, 'figure0_main.pdf'))
