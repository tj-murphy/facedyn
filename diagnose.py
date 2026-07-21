"""
smoothing_diagnostic.py  (v2)
==============================
Visual diagnostic for AU rolling-mean smoothing parameters.
Replicates the R ggplot / patchwork aesthetic and extends it with
alignment × window comparisons and two intuitive metrics.

Metrics
-------
fidelity           — Pearson r(original, smoothed). How much of the
                     original signal is preserved. Higher = better.
roughness_reduction — Fraction of frame-to-frame jitter removed.
                     (mean|Δraw| − mean|Δsmooth|) / mean|Δraw|
                     Higher = smoother. Expressed as a percentage.

The scatter plot (bottom) shows both metrics together. Top-right
corner is best: high fidelity AND high jitter removal.

Layout
------
Row 0–1 left:   Raw / Smoothed stacked  (replicates R patchwork)
Row 0–1 right:  Overlay of both
Row 2:          Alignment comparison    (left / center / right at K_BASE)
Row 3 left:     Window-size comparison  (all windows at ALN_BASE)
Row 3 right:    Jitter-removed heatmap  (window × alignment, higher = greener)
Row 4:          Scatter — fidelity vs jitter removed across all combos

Usage
-----
Edit CONFIG then run:
    python smoothing_diagnostic.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = "Data/full_dataset_preprocessed.csv"
VIDEO     = "20__exit_phone_room"
AU_COL    = "AU07_r"
WINDOWS   = [2, 3, 4, 5, 7, 10]
ALIGNS    = ["left", "center", "right"]
K_BASE    = 4          # window shown in the patchwork panels and alignment row
ALN_BASE  = "left"     # alignment shown in the patchwork panels and window row
OUT_PATH  = "smoothing_diagnostic.png"
DPI       = 150

# Colours — matched to the R script palette
RAW_COLOR    = "#009E73"
SMTH_COLOR   = "#E69F00"
ALIGN_COLORS = {"left": "#0072B2", "center": "#CC79A7", "right": "#D55E00"}
WINDOW_CMAP  = plt.cm.plasma
# ─────────────────────────────────────────────────────────────────────────────


# ── Smoothing ─────────────────────────────────────────────────────────────────
def smooth(series: pd.Series, window: int, align: str) -> pd.Series:
    """
    Rolling mean replicating zoo::rollmean(fill='extend').
    Left-aligned uses the reverse-roll-reverse trick.
    """
    if align == "center":
        rolled = series.rolling(window=window, min_periods=window, center=True).mean()
    elif align == "right":
        rolled = series.rolling(window=window, min_periods=window).mean()
    else:  # left
        rev    = series[::-1]
        rolled = rev.rolling(window=window, min_periods=window).mean()[::-1]
    return rolled.ffill().bfill()


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(original: pd.Series, smoothed: pd.Series) -> tuple[float, float]:
    """
    Returns (fidelity, roughness_reduction).

    fidelity            — Pearson r. How much of the original signal shape
                          is preserved. Close to 1 = faithful.

    roughness_reduction — What fraction of frame-to-frame jitter was removed.
                          0% = did nothing. 100% = perfectly flat line.
                          Alignment-agnostic: based only on the smoothed
                          signal's own step sizes, not its phase vs original.
    """
    fidelity  = original.corr(smoothed)
    raw_rough  = original.diff().abs().mean()
    smth_rough = smoothed.diff().abs().mean()
    roughness_reduction = (raw_rough - smth_rough) / raw_rough
    return fidelity, roughness_reduction


# ── Load & pre-compute ────────────────────────────────────────────────────────
df  = pd.read_csv(DATA_PATH)
vid = df[df["video_filename"] == VIDEO].reset_index(drop=True)
raw = vid[AU_COL]
frames = vid["frame"] if "frame" in vid.columns else pd.Series(vid.index)

smoothed_versions = {(w, a): smooth(raw, w, a) for w in WINDOWS for a in ALIGNS}
current           = smoothed_versions[(K_BASE, ALN_BASE)]

metrics_records = []
for w in WINDOWS:
    for a in ALIGNS:
        f, rr = evaluate(raw, smoothed_versions[(w, a)])
        metrics_records.append(
            {"window": w, "align": a, "fidelity": f, "roughness_reduction": rr}
        )
metrics_df = pd.DataFrame(metrics_records)


# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
})

def style_ax(ax, title, xlabel="Frame", ylabel="AU Intensity", hide_xlabel=False):
    ax.set_title(title, fontsize=12, fontweight="bold", color="black", pad=5)
    ax.set_xlabel("" if hide_xlabel else xlabel, fontsize=11, color="black")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", color="black")
    ax.tick_params(labelsize=9, colors="black")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.margins(y=0.08)
    if hide_xlabel:
        plt.setp(ax.get_xticklabels(), visible=False)


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 22), facecolor="white")
gs  = gridspec.GridSpec(
    5, 3, figure=fig,
    height_ratios=[1, 1, 1, 1.2, 1.3],
    hspace=0.65,
    wspace=0.33,
)

# ── Rows 0–1 left: Raw / Smoothed stacked ─────────────────────────────────────
ax_raw = fig.add_subplot(gs[0, :2])
ax_raw.plot(frames, raw, color=RAW_COLOR, alpha=0.6, linewidth=0.8)
style_ax(ax_raw, "AU07 (Raw)", hide_xlabel=True)

ax_smth = fig.add_subplot(gs[1, :2], sharex=ax_raw)
ax_smth.plot(frames, current, color=SMTH_COLOR, linewidth=0.9)
style_ax(ax_smth, f"AU07 (Smoothed)  —  k={K_BASE}, align='{ALN_BASE}'")

# ── Rows 0–1 right: Overlay ────────────────────────────────────────────────────
ax_ov = fig.add_subplot(gs[0:2, 2])
ax_ov.plot(frames, raw,     color=RAW_COLOR,  alpha=0.45, linewidth=0.75, label="Raw")
ax_ov.plot(frames, current, color=SMTH_COLOR, linewidth=0.95,             label="Smoothed")
ax_ov.legend(fontsize=10, frameon=False, loc="upper right")
style_ax(ax_ov, "Overlay")

# ── Row 2: Alignment comparison at K_BASE ─────────────────────────────────────
for i, align in enumerate(ALIGNS):
    ax = fig.add_subplot(gs[2, i])
    sm = smoothed_versions[(K_BASE, align)]
    ax.plot(frames, raw, color=RAW_COLOR, alpha=0.22, linewidth=0.7)
    ax.plot(frames, sm,  color=ALIGN_COLORS[align], linewidth=0.9)
    f, rr = evaluate(raw, sm)
    ax.set_title(
        f"align='{align}'  (k={K_BASE})\n"
        f"fidelity={f:.4f}   jitter removed={rr:.1%}",
        fontsize=10, fontweight="bold", color="black", pad=4,
    )
    ax.set_xlabel("Frame", fontsize=10, color="black")
    ax.set_ylabel("AU Intensity" if i == 0 else "", fontsize=10,
                  fontweight="bold", color="black")
    ax.tick_params(labelsize=8, colors="black")
    ax.margins(y=0.08)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

# ── Row 3 left: Window-size comparison at ALN_BASE ────────────────────────────
ax_win = fig.add_subplot(gs[3, :2])
ax_win.plot(frames, raw, color=RAW_COLOR, alpha=0.22, linewidth=0.7, label="Raw")
for k_idx, w in enumerate(WINDOWS):
    col       = WINDOW_CMAP(k_idx / (len(WINDOWS) - 1))
    _, rr     = evaluate(raw, smoothed_versions[(w, ALN_BASE)])
    ax_win.plot(
        frames, smoothed_versions[(w, ALN_BASE)],
        color=col, linewidth=0.85,
        label=f"k={w}  ({rr:.0%} jitter removed)",
    )
ax_win.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
style_ax(ax_win, f"Window-size comparison  —  align='{ALN_BASE}'")

# ── Row 3 right: Jitter-removed heatmap  (higher = greener = better) ──────────
ax_hm = fig.add_subplot(gs[3, 2])
heat  = metrics_df.pivot(index="window", columns="align",
                          values="roughness_reduction")[ALIGNS]
vmin, vmax = heat.values.min(), heat.values.max()

# RdYlGn (NOT reversed): green = high = good
im = ax_hm.imshow(heat.values, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
ax_hm.set_xticks(range(len(ALIGNS)))
ax_hm.set_xticklabels(ALIGNS, fontsize=10, color="black")
ax_hm.set_yticks(range(len(WINDOWS)))
ax_hm.set_yticklabels([f"k={w}" for w in WINDOWS], fontsize=10, color="black")
ax_hm.set_title("Jitter removed\n(higher = better)",
                 fontsize=11, fontweight="bold", color="black", pad=5)
plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04,
             format=ticker.PercentFormatter(xmax=1))
mid = (vmin + vmax) / 2
for i, w in enumerate(WINDOWS):
    for j, a in enumerate(ALIGNS):
        val     = heat.loc[w, a]
        txt_col = "black" if val < mid else "white"
        ax_hm.text(j, i, f"{val:.0%}",
                   ha="center", va="center", fontsize=8,
                   color=txt_col, fontweight="bold")

# ── Row 4: Scatter — fidelity vs roughness reduction ──────────────────────────
ax_sc = fig.add_subplot(gs[4, :])

for a in ALIGNS:
    sub = metrics_df[metrics_df["align"] == a]
    ax_sc.scatter(
        sub["fidelity"], sub["roughness_reduction"],
        color=ALIGN_COLORS[a], s=90, zorder=3, label=f"align='{a}'",
    )
    for _, row in sub.iterrows():
        ax_sc.annotate(
            f"k={int(row['window'])}",
            (row["fidelity"], row["roughness_reduction"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8.5, color=ALIGN_COLORS[a],
        )

# Highlight current selection
cur_f, cur_rr = evaluate(raw, current)
ax_sc.scatter(
    [cur_f], [cur_rr], s=220,
    facecolors="none", edgecolors="black", linewidths=2.2,
    zorder=4, label=f"current  (k={K_BASE}, align='{ALN_BASE}')",
)

# "Better" diagonal arrow
x_min = metrics_df["fidelity"].min()
x_max = metrics_df["fidelity"].max()
y_min = metrics_df["roughness_reduction"].min()
y_max = metrics_df["roughness_reduction"].max()
x_pad = (x_max - x_min) * 0.12
y_pad = (y_max - y_min) * 0.12
ax_sc.set_xlim(x_min - x_pad, x_max + x_pad)
ax_sc.set_ylim(y_min - y_pad, y_max + y_pad)

ax_sc.annotate(
    "", xy=(x_max + x_pad * 0.7, y_max + y_pad * 0.7),
    xytext=(x_min - x_pad * 0.3, y_min - y_pad * 0.3),
    arrowprops=dict(arrowstyle="->", color="lightgray", lw=1.8),
)
ax_sc.text(
    x_max + x_pad * 0.75, y_max + y_pad * 0.75,
    "better", fontsize=9, color="gray", style="italic", va="bottom",
)

ax_sc.set_xlabel(
    "Fidelity (correlation with original)  —  higher = more faithful to original signal",
    fontsize=11, color="black",
)
ax_sc.set_ylabel(
    "Jitter removed  —  higher = smoother",
    fontsize=11, fontweight="bold", color="black",
)
ax_sc.set_title(
    "All parameter combinations: fidelity vs jitter removed  —  top-right is best",
    fontsize=12, fontweight="bold", color="black", pad=5,
)
ax_sc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_sc.tick_params(labelsize=9, colors="black")
ax_sc.legend(fontsize=9, frameon=False, loc="lower right")
for sp in ["top", "right"]:
    ax_sc.spines[sp].set_visible(False)

# ── Super-title ────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Smoothing Diagnostic  —  {AU_COL}  |  {VIDEO}",
    fontsize=15, fontweight="bold", color="black", y=0.999,
)

plt.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved → {OUT_PATH}")