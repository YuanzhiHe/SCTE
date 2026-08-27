"""Publication figures for the thick-slice densitometry study."""
import csv, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figstyle import (PALETTE, FigureStyle, apply_publication_style, create_subplots,
                      finalize_figure, make_grouped_bar, annotate_bars, make_trend, make_scatter)
from scte_r import metrics

R = "results"
load = lambda p: list(csv.DictReader(open(os.path.join(R, p))))
col = lambda rows, k: np.array([float(r[k]) for r in rows], dtype=float)

# ---------------------------------------------------------------- Figure 1
# Whole-lung bias per metric. Two panels because the two metric families carry the
# SAME clinical direction with OPPOSITE arithmetic signs: for LAA a negative bias
# means "reports less emphysema than the 1 mm truth", while for the percentiles a
# POSITIVE HU bias means the same thing (denser = less emphysema). Plotting them on
# one axis would put identical clinical errors on opposite sides of zero.
apply_publication_style(FigureStyle(font_size=17, axes_linewidth=2.2))
arms = [("volume_trilinear.csv", "Thick input (trilinear)", PALETTE["neutral"], None),
        ("volume_realft.csv", "Real-pair fine-tuned\n(displacement-driven)", PALETTE["red_strong"], "//"),
        ("volume_flow.csv", "Flow decoder (ours)", PALETTE["blue_main"], None)]
panels = [(["laa950", "laa910"], ["LAA-950", "LAA-910"], "Whole-lung bias  (pp)",
           "reports LESS emphysema\nthan the 1 mm truth", "reports MORE emphysema"),
          (["p15", "p10"], ["Perc15", "Perc10"], "Whole-lung bias  (HU)",
           "reports MORE emphysema", "reports LESS emphysema\nthan the 1 mm truth")]
data = {}
for f, lab, _, _ in arms:
    rows = load(f)
    data[lab] = {k: float(np.mean(col(rows, k + "_rec") - col(rows, k + "_ref")))
                 for k in ["laa950", "laa910", "p15", "p10"]}

fig, axes = create_subplots(1, 2, figsize=(15, 6.4))
for pi, (keys, names, ylab, top_note, bot_note) in enumerate(panels):
    ax = axes[pi]
    series = [[data[a[1]][k] for k in keys] for a in arms]
    make_grouped_bar(ax, names, series, [a[1] for a in arms], ylabel=ylab,
                     colors=[a[2] for a in arms], hatches=[a[3] for a in arms], lw=2.0)
    lo = min(min(s) for s in series); hi = max(max(s) for s in series)
    pad = 0.30 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    for i, a in enumerate(arms):
        for j, v in enumerate(series[i]):
            ax.annotate(f"{v:+.2f}", (j - 0.4 + (0.8 / 3) * (i + 0.5), v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=12,
                        xytext=(0, 5 if v >= 0 else -7), textcoords="offset points")
    ax.axhline(0, color="black", lw=2.2, zorder=1)
    ax.text(0.015, 0.975, top_note, transform=ax.transAxes, fontsize=11.5,
            color="#666666", va="top")
    ax.text(0.015, 0.025, bot_note, transform=ax.transAxes, fontsize=11.5,
            color="#666666", va="bottom")
    ax.get_legend().remove() if ax.get_legend() else None
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=13.5,
           bbox_to_anchor=(0.5, -0.09), handlelength=1.8, columnspacing=2.2)
print(finalize_figure(fig, "figures/fig1_wholelung_bias", formats=["pdf", "png"]))

# ---------------------------------------------------------------- Figure 2
# Dose-response: the larger the displacement a model carries, the "better" its LAA
# concordance looks - and none of those models passes the certificate.
apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2.2))
# (file, label, dx, dy, ha, va) - offsets hand-placed; the four displacement-driven
# arms cluster tightly at both ends of the x range and collide under any auto rule.
pts = [("final_zeroshot.csv", "public.pt\nzero-shot", 10, 2, "left", "top"),
       ("final_realft.csv", "real-pair\nfine-tuned", 8, 12, "left", "bottom"),
       ("final_blocked.csv", "displacement\nblocked", 2, -12, "left", "top"),
       ("final_landweber.csv", "Landweber\ndeconvolution", 2, 12, "left", "bottom"),
       ("steps_64.csv", "flow decoder\n(ours)", 26, -8, "left", "top"),
       ("final_oracle.csv", "perfect\nreconstruction", 26, 6, "left", "bottom")]
fig, axes = create_subplots(1, 2, figsize=(15, 6))
ax = axes[0]
for f, lab, dx, dy, ha, va in pts:
    rows = load(f)
    d = np.abs(col(rows, "delta_HU")).mean()
    ref, rec = col(rows, "laa_ref"), col(rows, "laa_rec")
    ccc = metrics.ccc(torch.tensor(rec), torch.tensor(ref)).item()
    cert = sum(r["verdict"] == "certified" for r in rows) / len(rows)
    c = PALETTE["blue_main"] if "ours" in lab else (
        PALETTE["green_3"] if "perfect" in lab else PALETTE["red_strong"])
    make_scatter(ax, [d], [ccc], color=c, size=90 + 520 * cert, alpha=0.9)
    ax.annotate(lab, (d, ccc), textcoords="offset points", xytext=(dx, dy),
                fontsize=12.5, ha=ha, va=va)
ax.set_xlabel("displacement carried by the model,  |δ̂|  (HU)")
ax.set_ylabel("LAA-950 concordance (CCC)")
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlim(-0.15, 90); ax.set_ylim(-0.10, 1.16)
ax.text(0.97, 0.03, "marker area ∝ fraction certified", transform=ax.transAxes,
        fontsize=12, color="#666666", ha="right", va="bottom")

ax2 = axes[1]
steps = [8, 16, 32, 64, 128]
laa, ccc_s, cert_s = [], [], []
for s in steps:
    rows = load(f"steps_{s}.csv")
    ref, rec = col(rows, "laa_ref"), col(rows, "laa_rec")
    laa.append(np.abs(rec - ref).mean())
    ccc_s.append(metrics.ccc(torch.tensor(rec), torch.tensor(ref)).item())
    cert_s.append(100 * sum(r["verdict"] == "certified" for r in rows) / len(rows))
make_trend(ax2, steps, [ccc_s], ["LAA-950 CCC"], colors=[PALETTE["blue_main"]],
           xlabel="sampling steps", ylabel="LAA-950 CCC")
ax2.set_xscale("log", base=2); ax2.set_xticks(steps); ax2.set_xticklabels(steps)
ax2.set_ylim(0.5, 0.95)
ax3 = ax2.twinx()
ax3.spines["right"].set_visible(True); ax3.spines["top"].set_visible(False)
ax3.plot(steps, cert_s, marker="s", lw=3, ms=9, color=PALETTE["green_3"],
         markeredgecolor="white", markeredgewidth=1.5, label="certified (%)")
ax3.set_ylabel("scans certified (%)", color=PALETTE["green_3"])
ax3.tick_params(axis="y", colors=PALETTE["green_3"]); ax3.set_ylim(-3, 100)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=13)
ax2.set_title("accuracy comes from integration, not capacity", fontsize=13, pad=10)
print(finalize_figure(fig, "figures/fig2_displacement_and_steps", formats=["pdf", "png"]))

# ---------------------------------------------------------------- Figure 3
# The fitted slice-sensitivity profile: a narrow core plus a heavy tail, which is why a
# single Gaussian has to inflate its width and still leaves a third of the signal unexplained.
apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2.2))
d = torch.load("runs/forward_op_rplhr.pt", map_location="cpu", weights_only=False)
from scte_r.forward_learned import LearnedOperator
op = LearnedOperator(d["downsample"], d["widths"], correction=d["correction"])
op.load_state_dict(d["state_dict"])
w = np.array(d["widths"], dtype=float); a = op.coeffs.detach().numpy()
fig, axes = create_subplots(1, 2, figsize=(14, 5.6))
ax = axes[0]
ax.bar(np.arange(len(w)), a, color=[PALETTE["blue_main"] if x == 5.0 else PALETTE["blue_secondary"]
                                    for x in w], edgecolor="black", linewidth=1.8, width=0.72)
for i, (x_, y_) in enumerate(zip(w, a)):
    if y_ > 0.015:
        ax.annotate(f"{y_:.3f}", (i, y_), ha="center", va="bottom", fontsize=12,
                    xytext=(0, 4), textcoords="offset points")
ax.set_xticks(np.arange(len(w))); ax.set_xticklabels([f"{x:g}" for x in w])
ax.set_xlabel("Gaussian basis width (mm FWHM)"); ax.set_ylabel("fitted weight")
ax.axvline(4.5, color=PALETTE["red_strong"], ls="--", lw=2)
ax.text(4.65, 0.55, "heavy tail\n13.5% of\nthe weight", fontsize=12.5,
        color=PALETTE["red_strong"], va="center")
ax.set_title("real slice profile is not Gaussian", fontsize=13, pad=10)

ax = axes[1]
vals = [d["unexplained_before"], d["unexplained_after"]]
b = ax.bar([0, 1], vals, color=[PALETTE["red_strong"], PALETTE["blue_main"]],
           edgecolor="black", linewidth=1.8, width=0.55)
annotate_bars(ax, b, fmt="{:.3f}", fontsize=13)
ax.set_xticks([0, 1])
ax.set_xticklabels(["single Gaussian\n(w = 6.25 mm)", "fitted profile\n+ zero-mean correction"])
ax.set_ylabel("unexplained residual\n‖A(x) − y‖ / ‖y − ȳ‖")
ax.set_ylim(0, 0.44)
ax.annotate("", xy=(1, vals[1] + 0.02), xytext=(1, vals[0] - 0.01),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="black"))
ax.text(1.08, (vals[0] + vals[1]) / 2, "−55%", fontsize=15, va="center")
ax.set_title("half the model misfit was the tail", fontsize=13, pad=10)
print(finalize_figure(fig, "figures/fig3_forward_operator", formats=["pdf", "png"]))

# ---------------------------------------------------------------- Figure 4
# The aggregate LAA number is diluted by near-normal lungs. Stratifying by disease
# burden is the honest view - and it is the one that matters for a target cohort
# whose mean LAA-950 is 13x higher than this test set's.
apply_publication_style(FigureStyle(font_size=17, axes_linewidth=2.2))
flow, zs, ft = load("steps_64.csv"), load("final_zeroshot.csv"), load("final_realft.csv")
ref = col(flow, "laa_ref"); q1, q2 = np.percentile(ref, [33.3, 66.7])
bands = [(-1, q1, "mild"), (q1, q2, "moderate"), (q2, ref.max(), "severe")]
arms = [("thick input", PALETTE["neutral"], None, lambda m: col(flow, "laa_thick")[m]),
        ("displacement-driven", PALETTE["red_strong"], "//", lambda m: col(ft, "laa_rec")[m]),
        ("flow decoder (ours)", PALETTE["blue_main"], None, lambda m: col(flow, "laa_rec")[m])]
masks = [(ref > lo) & (ref <= hi) for lo, hi, _ in bands]
names = [f"{n}\nref LAA {ref[m].mean():.2f}%  (n={m.sum()})"
         for (_, _, n), m in zip(bands, masks)]
series = [[float(np.mean(np.abs(get(m) - ref[m]))) for m in masks] for _, _, _, get in arms]

fig, axes = create_subplots(1, 1, figsize=(10.5, 6.4))
ax = axes[0]
make_grouped_bar(ax, names, series, [a[0] for a in arms],
                 ylabel="LAA-950 MAE  (pp)", colors=[a[1] for a in arms],
                 hatches=[a[2] for a in arms], lw=2.0)
for i in range(len(arms)):
    for j, v in enumerate(series[i]):
        ax.annotate(f"{v:.2f}", (j - 0.4 + (0.8 / 3) * (i + 0.5), v), ha="center",
                    va="bottom", fontsize=12, xytext=(0, 4), textcoords="offset points")
ax.set_ylim(0, max(max(x) for x in series) * 1.22)
ax.legend(loc="upper left", fontsize=13)
ax.annotate("", xy=(2.42, 0.63), xytext=(2.42, 1.58),
            arrowprops=dict(arrowstyle="<->", lw=2, color="#444444"))
ax.text(2.50, 1.10, "2.5×", fontsize=13.5, color="#444444", va="center")
ax.text(0.33, 0.985, "the aggregate 0.31 pp is diluted by the mild\nthird, where nothing can be recovered",
        transform=ax.transAxes, fontsize=11.5, color="#666666", va="top")
print(finalize_figure(fig, "figures/fig4_severity", formats=["pdf", "png"]))
