"""Render the report's four figures from the summary tables in report/figdata/.

Greyscale only: the report is printed and circulated in black and white, so the
arms are separated by fill value and hatch rather than hue. That departs from the
results-to-figure palette (assets/color-palette.txt), which assumes colour.

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
DATA = os.path.join(HERE, 'figdata')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Noto Sans CJK SC',
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

OURS = 'SCTE-R-zeroshot'
LBL = {'Lanczos': 'Lanczos 插值', 'CTHNet': 'CTHNet', 'TVSRN': 'TVSRN', 'I3Net': 'I3Net',
       'SCTE-R-zeroshot': 'SCTE-R\n（零样本）', 'SCTE-R-adapted': 'SCTE-R\n（站点适配）',
       'SCTE-R': 'SCTE-R'}


def load(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(fh))


def style(m):
    """ours: solid dark; baselines: light with hatch."""
    if m.startswith('SCTE-R'):
        return dict(facecolor='0.25', edgecolor='0', hatch='')
    return dict(facecolor='0.88', edgecolor='0', hatch='///')


# ---------------------------------------------------------------- figure 1
def figure1():
    rows = load('arms_quality_vs_bias.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    for ax, cohort, title in ((axes[0], 'external', '外部临床队列（n = 444）'),
                              (axes[1], 'public', '公开数据（n = 50）')):
        sub = [r for r in rows if r['cohort'] == cohort]
        x = [float(r['psnr_db']) for r in sub]
        y = [abs(float(r['laa950_bias_pp'])) for r in sub]
        # On the public cohort the three published methods sit on top of one another;
        # that coincidence IS the finding, so they get one shared label instead of
        # three overlapping ones.
        cluster = {'CTHNet', 'TVSRN', 'I3Net'} if cohort == 'public' else set()
        for xi, yi, r in zip(x, y, sub):
            ours = r['method'].startswith('SCTE-R')
            ax.scatter(xi, yi, s=58 if ours else 40,
                       facecolor='0.25' if ours else 'white',
                       edgecolor='0', linewidth=0.9, marker='o' if ours else 's', zorder=3)
            if r['method'] in cluster:
                continue
            ax.annotate(r['method'].replace('SCTE-R-zeroshot', 'SCTE-R')
                        .replace('SCTE-R-adapted', 'SCTE-R (适配)'),
                        (xi, yi), textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=6.6)
        if cluster:
            cx = np.mean([a for a, r in zip(x, sub) if r['method'] in cluster])
            cy = np.mean([b for b, r in zip(y, sub) if r['method'] in cluster])
            ax.annotate('CTHNet、TVSRN、I3Net', (cx, cy), textcoords='offset points',
                        xytext=(-4, 26), ha='center', fontsize=6.6,
                        arrowprops=dict(arrowstyle='-', lw=0.6, color='0.4',
                                        shrinkA=1, shrinkB=6))
        # the baselines' horizontal run: same bias across a wide PSNR range
        base = [(a, b) for a, b, r in zip(x, y, sub) if not r['method'].startswith('SCTE-R')]
        if len(base) >= 2:
            bx = [p[0] for p in base]; by = [p[1] for p in base]
            ax.plot([min(bx), max(bx)], [np.mean(by)] * 2, color='0.45', lw=0.8,
                    ls=(0, (4, 3)), zorder=1)
        ax.set_title(title, fontsize=8.5, pad=9)
        ax.set_xlabel('PSNR (dB)')
        ax.set_ylim(-0.25, max(y) * 1.55)
        ax.margins(x=0.22)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('|LAA-950 偏差| (pp)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure1_quality_vs_bias.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def figure2():
    rows = load('agreement_loa.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    order = ['Lanczos', 'CTHNet', OURS, 'SCTE-R-adapted']
    for ax, ep in zip(axes, ('LAA-950', 'LAA-910')):
        sub = {r['method']: r for r in rows if r['endpoint'] == ep}
        ys = np.arange(len(order))[::-1]
        for yi, m in zip(ys, order):
            r = sub[m]
            b, lo, hi = float(r['bias']), float(r['loa_lo']), float(r['loa_hi'])
            ours = m.startswith('SCTE-R')
            ax.plot([lo, hi], [yi, yi], color='0' if ours else '0.5',
                    lw=2.4 if ours else 1.4, solid_capstyle='butt', zorder=2)
            ax.plot([lo, lo], [yi - .16, yi + .16], color='0' if ours else '0.5', lw=1)
            ax.plot([hi, hi], [yi - .16, yi + .16], color='0' if ours else '0.5', lw=1)
            ax.scatter([b], [yi], s=34, facecolor='0.15' if ours else 'white',
                       edgecolor='0', zorder=3, linewidth=0.9)
            ax.annotate(f'{b:+.2f}', (b, yi), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=6.6)
        ax.axvline(0, color='0', lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.set_yticks(ys, [LBL[m].replace('\n', '') for m in order], fontsize=7.2)
        ax.set_xlabel('偏差与 95% 一致性界 (pp)')
        ax.set_title(ep, fontsize=8.5, pad=9)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure2_agreement_loa.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def figure3():
    pr = load('percentile_bias.csv')
    dp = load('displacement.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))

    ax = axes[0]
    ms = [r['method'] for r in pr]
    x = np.arange(len(ms)); w = 0.36
    for i, (col, lab, sh) in enumerate((('perc15_bias_hu', 'Perc15', ''),
                                        ('perc10_bias_hu', 'Perc10', '///'))):
        v = [float(r[col]) for r in pr]
        ax.bar(x + (i - .5) * w, v, w, label=lab, facecolor='0.35' if i == 0 else '0.8',
               edgecolor='0', linewidth=0.7, hatch=sh)
    ax.axhline(0, color='0', lw=0.7)
    ax.set_xticks(x, [LBL[m].replace('\n', '') for m in ms], fontsize=7.2)
    ax.set_ylabel('偏差 (HU)')
    ax.set_title('低密度百分位的偏差', fontsize=8.5, pad=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    arms = [r['arm'] for r in dp]
    lab = {'deterministic-regression': '确定性回归', 'bias-only-control': '纯标量偏置',
           'SCTE-R': 'SCTE-R'}
    v = [float(r['delta_hu']) for r in dp]
    e = [float(r['delta_sd']) for r in dp]
    y = np.arange(len(arms))[::-1]
    ax.barh(y, v, 0.5, xerr=e, facecolor=['0.8', '0.8', '0.35'],
            edgecolor='0', linewidth=0.7, error_kw=dict(ecolor='0', lw=0.9, capsize=2.5))
    for yi, vi, ei in zip(y, v, e):
        ax.annotate(f'{vi:.2f}', (vi - ei, yi), textcoords='offset points',
                    xytext=(-4, -2.5), fontsize=6.8, ha='right')
    ax.axvline(0, color='0', lw=0.7)
    ax.set_xlim(min(v) - 9, 2.5)
    ax.set_yticks(y, [lab[a] for a in arms], fontsize=7.2)
    ax.set_xlabel('整体密度位移 δ (HU)')
    ax.set_title('位移的直接测量（n = 444）', fontsize=8.5, pad=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure3_percentile_and_displacement.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def figure4():
    rows = load('site_gate.csv')
    lab = {'public-test': '公开测试集\n（模型学过）',
           'external-soft-kernel': '外部软核\n（同核族）',
           'external-sharp-kernel': '外部锐核\n（未学过）',
           'external-sharp-adapted': '外部锐核\n（站点适配后）'}
    status = {True: '终点失效', False: '终点正常'}
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    x = np.arange(len(rows))
    v = [float(r['certified_pct']) for r in rows]
    failed = [r['endpoint_status'] == 'failed' for r in rows]
    ax.bar(x, v, 0.56, facecolor=['0.8' if f else '0.35' for f in failed],
           edgecolor='0', linewidth=0.7,
           hatch=['///' if f else '' for f in failed])
    for xi, vi in zip(x, v):
        ax.annotate(f'{vi:.1f}%', (xi, vi), textcoords='offset points', xytext=(0, 3),
                    ha='center', fontsize=7)
    ax.set_xticks(x, ['%s\n%s' % (lab[r['domain']], status[f])
                      for r, f in zip(rows, failed)], fontsize=6.9)
    ax.set_ylabel('判据通过率 (%)')
    ax.set_ylim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure4_site_gate.pdf'))
    plt.close(fig)


for fn in (figure1, figure2, figure3, figure4):
    fn(); print('绘制', fn.__name__)
print('输出目录', OUT)
