"""House publication style for this project's figures (scientific-figure-making skill)."""
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
                  PALETTE["teal"], PALETTE["violet"], PALETTE["neutral"]]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle = None):
    s = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(s.font_family),
        "font.size": s.font_size, "axes.linewidth": s.axes_linewidth,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": s.font_size, "axes.titlesize": s.font_size,
        "xtick.labelsize": s.font_size - 2, "ytick.labelsize": s.font_size - 2,
        "xtick.major.width": s.axes_linewidth, "ytick.major.width": s.axes_linewidth,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "legend.frameon": False, "legend.fontsize": s.font_size - 2,
        "figure.dpi": 120, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.usetex": s.use_tex,
    })


def create_subplots(nrows=1, ncols=1, figsize=None, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    return fig, np.atleast_1d(np.asarray(axes)).ravel()


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.05, **kw):
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    formats = formats or (["pdf", "svg", "eps"] if not out.suffix else [out.suffix[1:]])
    saved = []
    for f in formats:
        p = out.with_suffix("." + f)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad, **kw)
        saved.append(p)
    if close:
        plt.close(fig)
    return saved


def make_grouped_bar(ax, categories, series, labels, ylabel="Value", colors=None,
                     annotate=False, hatches=None, edgecolor="black", lw=1.5):
    series = [np.asarray(s, dtype=float) for s in series]
    n_c, n_s = len(categories), len(series)
    for s in series:
        if s.shape[0] != n_c:
            raise ValueError("each series must match len(categories)")
    colors = colors or DEFAULT_COLORS
    x = np.arange(n_c, dtype=float)
    w = 0.8 / n_s
    bars = None
    for i, (s, lab) in enumerate(zip(series, labels)):
        bars = ax.bar(x - 0.4 + w * (i + 0.5), s, width=w * 0.92, label=lab,
                      color=colors[i % len(colors)], edgecolor=edgecolor, linewidth=lw,
                      hatch=None if hatches is None else hatches[i % len(hatches)])
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if annotate:
        annotate_bars(ax, bars)
    return bars


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=10, padding=3):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, padding if h >= 0 else -padding - 8),
                    ha="center", fontsize=fontsize)


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None,
               markers=None, lw=3, ms=9):
    colors = colors or DEFAULT_COLORS
    markers = markers or ["o", "s", "^", "D", "v"]
    for i, (y, lab) in enumerate(zip(y_series, labels)):
        ax.plot(x, y, marker=markers[i % len(markers)], label=lab, lw=lw, ms=ms,
                color=colors[i % len(colors)], markeredgecolor="white", markeredgewidth=1.5)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)


def make_scatter(ax, x, y, label=None, color=None, size=50, alpha=0.75, marker="o"):
    ax.scatter(x, y, s=size, alpha=alpha, label=label, marker=marker,
               color=color or PALETTE["blue_main"], edgecolors="black", linewidths=1.2)
