#!/usr/bin/env python3
r"""What the depth factor tracks: the observation's share of the action representation, layer by layer.

Panel (a) is the observation footprint decay across network depth across all four suites.
Panel (b) is the 56-element Action Entropy heatmap across layers and chunk elements, showing the decision wavefront
as an academic dashed isocontour (Scheme B).

Fonts: DejaVu Serif, bold.
Colormap: viridis.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
D = HERE.parent.parent / "experiments/results/measurements"
OUT = HERE.parent

sys.path.insert(0, str(HERE))
import _style as S
S.apply()

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
prop = fm.FontProperties(fname=FONT, weight="bold")
fm.fontManager.addfont(FONT)

plt.rcParams.update({
    "font.family": prop.get_name(),
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "mathtext.fontset": "dejavuserif",
    "mathtext.default": "bf",
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42,
})

SUITES = [("libero_spatial", "Spatial", "#9467bd", "o"),
          ("libero_object",  "Object",  "#4aa3df", "s"),
          ("libero_goal",    "Goal",    "#f4776d", "^"),
          ("libero_10",      "Long",    "#5fbb6a", "D")]
CHANCE = (513 / 4096) ** 0.5
TICK, LAB, LEG = 6.0, 7.0, 5.8
ANISO_FROM = 26

def smooth_2d(a, k=3, sigma_l=1.2, sigma_s=2.5):
    pad = np.pad(a, ((k, k), (k, k)), mode='edge')
    res = np.zeros_like(a, dtype=float)
    y_idx, x_idx = np.ogrid[-k:k+1, -k:k+1]
    W = np.exp(-0.5 * ((y_idx**2 / sigma_l**2) + (x_idx**2 / sigma_s**2)))
    W /= W.sum()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            res[i, j] = np.sum(pad[i:i+2*k+1, j:j+2*k+1] * W)
    return res

def main():
    r_foot = {k: json.loads((D / f"footprint2_{k}.json").read_text()) for k, _, _, _ in SUITES}
    L = len(r_foot["libero_10"]["same_layer_centered"])
    x = np.arange(L)

    r_ent = json.loads((D / "entropy_spread_libero_10.json").read_text())
    per = np.array(r_ent["per_element"]) # [32, 56]

    # Calculate empirical wavefront line across steps
    smooth_per = smooth_2d(per)
    settle_layers = []
    x_centers = [3.5 + 7 * s for s in range(8)]
    for s in range(8):
        col = smooth_per[:, int(x_centers[s])]
        l_cross = np.interp(0.45, col[::-1], np.arange(32)[::-1])
        settle_layers.append(l_cross)
    z = np.polyfit(x_centers, settle_layers, 2)
    p = np.poly1d(z)
    x_fine = np.linspace(3.5, 52.5, 60)

    # Highly compact top margin (zero wasted dead white space)
    # Bottom margin provides ample vertical breathing room between xlabels and subcaptions
    W = 3.45
    H = 1.35
    pb_in = 0.46
    ph_in = 0.80
    PB = pb_in / H
    PH = ph_in / H

    wa_in = 1.30
    wb_in = 1.12
    gap_in = 0.38
    cb_gap_in = 0.025
    cb_w_in = 0.035
    left_margin_in = 0.34

    PW_a = wa_in / W
    PW_b = wb_in / W
    PW_cb = cb_w_in / W

    left_a = left_margin_in / W
    left_b = left_a + PW_a + (gap_in / W)
    left_cb = left_b + PW_b + (cb_gap_in / W)

    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([left_a, PB, PW_a, PH])
    bx = fig.add_axes([left_b, PB, PW_b, PH])
    cax = fig.add_axes([left_cb, PB, PW_cb, PH])

    # -----------------------------------------------------------
    # Panel (a): Footprint decay
    # -----------------------------------------------------------
    ax.axvspan(ANISO_FROM, L - 1, color="#f2f2f2", zorder=0, lw=0)
    for k, label, col, mk in SUITES:
        y = (np.array(r_foot[k]["same_layer_centered"]) - CHANCE) / (1 - CHANCE)
        ax.plot(x, y, "-", color=col, lw=1.4, zorder=4)
        ax.plot(x[::6], y[::6], mk, color=col, ms=2.8, zorder=5, label=label)

    ax.set_xlim(-0.7, L - 0.3); ax.set_ylim(0.42, 1.045)
    ax.set_xticks([0, 8, 16, 24, 31]); ax.set_yticks([0.5, 0.7, 0.9])
    ax.tick_params(labelsize=TICK, pad=1.5)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(prop); t.set_fontsize(TICK)

    ax.set_xlabel(r"layer $\ell$", fontproperties=prop, fontsize=LAB, labelpad=2.0)
    ax.set_ylabel("observation share", fontproperties=prop, fontsize=LAB, labelpad=1.8)
    ax.grid(True, color="#dddddd", lw=0.5, ls=(0, (2.5, 2.5)), zorder=0)
    ax.set_axisbelow(True)

    leg = ax.legend(frameon=False, fontsize=LEG, loc="lower left", bbox_to_anchor=(0.02, 0.04), ncol=2,
              handletextpad=0.15, borderpad=0.04, labelspacing=0.08, columnspacing=0.35,
              handlelength=0.7, borderaxespad=0.0, markerscale=0.7)
    for t in leg.get_texts():
        t.set_fontproperties(prop); t.set_fontsize(LEG)

    for sp in ax.spines.values():
        sp.set_linewidth(1.0)

    # -----------------------------------------------------------
    # Panel (b): 56-element Action Entropy with Decision Wavefront (Scheme B)
    # -----------------------------------------------------------
    im = bx.imshow(per, origin="lower", cmap="viridis", vmin=0.15, vmax=0.82, aspect="auto",
                   interpolation="nearest")

    # Scheme B: Academic White Dashed Isocontour (no beads/dots, pure and clean)
    bx.plot(x_fine, p(x_fine), color="#ffffff", lw=1.5, ls=(0, (4.0, 2.5)), zorder=6, alpha=0.95)

    bx.set_xticks([3.5 + 7 * i for i in range(8)])
    bx.set_xticklabels([f"{i+1}" for i in range(8)])
    bx.set_yticks([0, 8, 16, 24, 31])
    bx.tick_params(labelsize=TICK, pad=1.5)
    for t in bx.get_xticklabels() + bx.get_yticklabels():
        t.set_fontproperties(prop); t.set_fontsize(TICK)

    bx.set_xlabel("chunk elements", fontproperties=prop, fontsize=LAB, labelpad=2.0)
    bx.set_ylabel(r"layer $\ell$", fontproperties=prop, fontsize=LAB, labelpad=1.8)

    cb = fig.colorbar(im, cax=cax, ticks=[0.2, 0.5, 0.8])
    cb.ax.set_yticklabels(["0.2", "0.5", "0.8"])
    cb.ax.tick_params(labelsize=TICK, pad=1.2)
    for lab in cb.ax.get_yticklabels():
        lab.set_fontproperties(prop); lab.set_fontsize(TICK)

    for sp in bx.spines.values():
        sp.set_linewidth(1.0)

    # -----------------------------------------------------------
    # Subcaptions placed cleanly at the bottom without any collision
    # -----------------------------------------------------------
    pos_a = ax.get_position()
    fig.text((pos_a.x0 + pos_a.x1) / 2, 0.025, "(a) Observation share",
             ha="center", va="bottom", fontproperties=prop, fontsize=LAB)

    pos_b = bx.get_position()
    fig.text((pos_b.x0 + pos_b.x1) / 2, 0.025, "(b) Action entropy",
             ha="center", va="bottom", fontproperties=prop, fontsize=LAB)

    fig.savefig(OUT / "footprint.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / "footprint.png", dpi=300, bbox_inches="tight", pad_inches=0.025)
    print("wrote", OUT / "footprint.pdf and .png")

if __name__ == "__main__":
    main()
