"""Compare arms ONLY on the cases every arm has finished.

The arms finish at very different speeds - Lanczos is one matmul, CTHNet one
feedforward pass, ours is a few hundred network passes per volume - so at any
moment during a run the CSVs have different numbers of rows. Reading the
per-arm summaries side by side then compares different patients, and for the
densitometry endpoints that is not a small effect: LAA-950 bias is bounded
below by -LAA_ref, so on a handful of low-emphysema cases EVERY method looks
unbiased. A method can appear to remove a -4.5 pp bias purely by having been
evaluated on cases where a -4.5 pp bias was arithmetically impossible.

So: intersect on `case` first, then report. The reference column is printed
too, because the first thing to check is whether the completed subset has the
same emphysema burden as the full cohort - if it does not, the subset is early,
not representative, and the numbers are provisional for that reason and not
because n is small.

  python scripts/common_subset.py --arms PRIVATE/hebei/results/volume_lanczos.csv:Lanczos \
      PRIVATE/hebei/results/volume_zs_cthnet.csv:CTHNet \
      PRIVATE/hebei/results/volume_zs_ours.csv:Ours
"""
import argparse, csv, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--arms', nargs='+', required=True, metavar='CSV:LABEL')
ap.add_argument('--metrics', nargs='+',
                default=['psnr', 'lung_psnr', 'ssim', 'laa950', 'laa910', 'p15', 'p10'])
ap.add_argument('--boot', type=int, default=5000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--csv', default=None, help='write the common-subset table here')
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

arms = []
for spec in a.arms:
    fn, label = spec.rsplit(':', 1)
    if not os.path.exists(fn):
        sys.exit(f'missing {fn}')
    rows = {r['case']: r for r in csv.DictReader(open(fn))}
    if not rows:
        sys.exit(f'{fn} has no rows yet')
    arms.append((label, fn, rows))

common = set.intersection(*[set(r) for _, _, r in arms])
if not common:
    sys.exit('the arms share no cases')
common = sorted(common)
print(f'\n共同完成的病例 {len(common)} 例  ' +
      '  '.join(f'{lb}:{len(r)}' for lb, _, r in arms))
if len(common) < min(len(r) for _, _, r in arms):
    print('（已丢弃各臂中其他臂尚未完成的病例）')


def ccc(x, y):
    mx, my = x.mean(), y.mean()
    return 2 * ((x - mx) * (y - my)).mean() / (x.var() + y.var() + (mx - my) ** 2 + 1e-12)


def col(rows, k):
    return np.array([float(rows[c][k]) for c in common])


for m in a.metrics:
    paired = (m + '_ref') in next(iter(arms[0][2].values()))
    print()
    if not paired:
        if m not in next(iter(arms[0][2].values())):
            continue
        print(f'{m:<12s}' + ''.join(f'{lb:>16s}' for lb, _, _ in arms))
        print(f'{"":<12s}' + ''.join(f'{col(r, m).mean():>16.4f}' for _, _, r in arms))
        continue
    ref = col(arms[0][2], m + '_ref')
    print(f'{m}   参考值（1mm 真值）{ref.mean():.2f} +- {ref.std():.2f}')
    print(f"  {'臂':<14s}{'重建值':>10s}{'bias':>9s}{'MAE':>8s}{'95% LoA':>20s}{'CCC':>8s}")
    for lb, _, rows in arms:
        r2 = col(rows, m + '_ref')
        if not np.allclose(r2, ref, atol=1e-6):
            print(f'  !! {lb}: 参考值与首臂不一致，说明 CSV 不是同一批数据')
        rec = col(rows, m + '_rec')
        d = rec - r2
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        print(f'  {lb:<14s}{rec.mean():>10.2f}{d.mean():>9.2f}{np.abs(d).mean():>8.2f}'
              f'{d.mean() - 1.96 * sd:>10.2f} 到{d.mean() + 1.96 * sd:6.2f}'
              f'{ccc(rec, r2):>8.3f}')
    # paired |error| difference against the LAST arm, on these same cases
    lb_last, _, rows_last = arms[-1]
    e_last = np.abs(col(rows_last, m + '_rec') - col(rows_last, m + '_ref'))
    for lb, _, rows in arms[:-1]:
        d = e_last - np.abs(col(rows, m + '_rec') - col(rows, m + '_ref'))
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'    {lb_last} vs {lb:<12s} |误差| 差 {d.mean():+7.2f} '
              f'[{lo:+.2f}, {hi:+.2f}]{"" if lo <= 0 <= hi else "  *"}')

print('\n* = 自助法 95% 置信区间不含 0（负值有利于最后一臂）。')
print('配对比较只在共同病例上做，因此不受各臂进度不同的影响；但若共同病例的参考')
print('LAA 与全队列（444 例）差别明显，结论仍是暂时的 —— 原因是子集不代表总体，')
print('不是 n 小。')

if a.csv and common:
    out = []
    for lb, fn, rows in arms:
        rec_row = dict(arm=lb, n=len(common))
        for m in a.metrics:
            if (m + '_ref') in next(iter(rows.values())):
                r2, rc = col(rows, m + '_ref'), col(rows, m + '_rec')
                rec_row[m + '_bias'] = round(float((rc - r2).mean()), 4)
                rec_row[m + '_mae'] = round(float(np.abs(rc - r2).mean()), 4)
                rec_row[m + '_ccc'] = round(float(ccc(rc, r2)), 4)
            elif m in next(iter(rows.values())):
                rec_row[m] = round(float(col(rows, m).mean()), 4)
        out.append(rec_row)
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print('wrote', a.csv)
