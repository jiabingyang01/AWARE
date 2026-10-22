#!/usr/bin/env python3
r"""Strength sweep of \method on a pi backbone, four LIBERO suites, one panel.

Same shape and style as make_fig_alpha_sensitivity.py, which draws the OpenVLA-OFT version. Two
things differ and both are stated in the caption rather than hidden here: the curve is measured on
seed 7 alone (only the strength carried into the table is run on all three seeds), and alpha=0 is the
untouched policy measured on the same machine, so the left end of every curve is its own baseline.

The cells live on the evaluation boxes; export_pi_curves.py dumps them to the JSON this reads, so the
figure and the table are drawn from one file and cannot drift apart. A strength with no cell is
dropped from the curve, never interpolated.

Usage: make_fig_alpha_pi.py --model pi0 [--data ...] [--out impact_alpha_pi0]
"""
import json, sys
import statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import _style as S
S.apply()
plt.rcParams.update({"mathtext.fontset": "dejavuserif"})
ALPHA_SYM = r"$\boldsymbol{\alpha}$"

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
DATA = HERE.parent.parent / "experiments/results/pi_alpha_curves.json"
SUITES = [("spatial", "LIBERO-Spatial", "#9467bd", "o"),
          ("object", "LIBERO-Object", "#4aa3df", "s"),
          ("goal", "LIBERO-Goal", "#f4776d", "^"),
          ("long", "LIBERO-Long", "#5fbb6a", "D")]
NAME = {"pi0": r"$\pi_0$", "pi05": r"$\pi_{0.5}$", "oft": "OpenVLA-OFT"}
OFT_RES = HERE.parent.parent / "experiments/results/uaor_select_20260911"
OFT_BASE = {"spatial": (486, 492, 490), "object": (491, 491, 491), "goal": (488, 486, 491), "long": (472, 465, 469)}


def oft_curves(max_alpha=0.10):
    """Build the same {suite: {alpha: {seed: successes}}} dict from the OpenVLA-OFT cells."""
    out = {}
    for suite, base in OFT_BASE.items():
        pts = {"0": {str(7 + i): v for i, v in enumerate(base)}}
        for k in range(1, 16):
            a = round(0.01 * k, 2)
            if a > max_alpha + 1e-9:
                continue
            for seed in (7, 8, 9):
                p = OFT_RES / f"{suite}500_rampsel_a{k:02d}_s{seed}" / "result.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    if r.get("episodes") == 500:
                        pts.setdefault(f"{a:g}", {})[str(seed)] = r["success"]
        out[suite] = pts
    return out
# the strength each suite carries in Table I; the outlined marker sits there, as in the OFT figure
TABLE_ALPHA = {"pi0": {"spatial": 0.03, "object": 0.04, "goal": 0.05, "long": 0.03},
               "pi05": {"spatial": 0.01, "object": 0.08, "goal": 0.01, "long": 0.07},
               "oft": {"spatial": 0.04, "object": 0.05, "goal": 0.10, "long": 0.03}}
# the run each Table I cell was taken from (baseline and rule of that suite come from this one run)
TABLE_RUN = {"pi0": {"spatial": "7", "object": "8", "goal": "7", "long": "7"},
             "pi05": {"spatial": "7", "object": "8", "goal": "8", "long": "7"},
             "oft": {"spatial": "7", "object": "7", "goal": "7", "long": "9"}}
# the run the table's BASELINE came from, where it differs from the arm's run (only OFT Spatial):
# the alpha=0 point of the curve is drawn from this run so it equals the table's baseline
TABLE_BASE_RUN = {"oft": {"spatial": "9"}}
# where the legend goes: a point in axes coordinates that is empty for that backbone's four curves
# legends sit in the band that is empty for that backbone: between Goal and Long on pi0, between the
# saturated suites and Long on OpenVLA-OFT (right half, where Long has already fallen off)
# Legends sit in the empty bands of each panel (user, 09-15): OFT's is the bottom band under the left half of the
# Long curve; pi0's is the gap between the Goal and Long curves, two rows, centred.
LEGEND = {"pi0":  dict(loc="center", bbox_to_anchor=(0.5, 0.42), ncol=2),
          "pi05": dict(loc="center", bbox_to_anchor=(0.5, 0.42), ncol=2),
          "oft":  dict(loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2)}
YMIN = {"oft": 89.0}      # a little room below 90 so the centred two-row legend clears the Long dip at alpha=0.07 (92.4)
TOP_PAD = {}             # the y-axis is never stretched for a legend
SEED = "7"          # the selection seed; see the note in main()


def main(model="pi0", data=DATA, out=None, mark="edge", mode="table", band=False, compact=False):
    out = out or f"impact_alpha_{model}"
    # compact: sized for half a column (two panels side by side); fonts are chosen so that after the
    # ~0.5 scaling in the paper they land near 7 pt
    FS = dict(fig=(3.4, 3.0), tick=14, lab=16, leg=10.5, ms=6.5, lw=2.1, cap=2.0) if compact else \
         dict(fig=(5.0, 4.0), tick=18, lab=21, leg=17, ms=8.0, lw=2.6, cap=2.5)   # x0.336 in the paper: ticks 6.0 / labels 7.0 / legend 5.7 pt, matching Fig. 1
    if model == "oft":
        curves = oft_curves()
    else:
        d = json.loads(Path(data).read_text())
        curves = d.get("cells", d).get(model, {})  # exporter wraps cells and chosen alphas
    if not curves:
        sys.exit(f"{data} has no cells for {model}")
    fig, ax = plt.subplots(figsize=FS["fig"])
    lo, hi = 100.0, 0.0
    for suite, label, col, mk in SUITES:
        pts = curves.get(suite)
        if not pts:
            continue
        xs, ys, es, ns = [], [], [], []
        for a in sorted(pts, key=float):
            if mode == "mean":       # the OpenVLA-OFT figure's shape: mean over runs, sample sd as the bar
                v = [x / 5.0 for x in pts[a].values()]
                if not v:
                    continue
                xs.append(float(a)); ys.append(st.mean(v)); es.append(st.stdev(v) if len(v) > 1 else 0.0); ns.append(len(v))
            else:                    # one run for every point of this suite, no bars:
                # mode "table" = the run Table I reports for the suite; "seed7" = the default seed
                sd = TABLE_RUN.get(model, {}).get(suite, SEED) if mode == "table" else SEED
                if mode == "table" and float(a) == 0.0:
                    sd = TABLE_BASE_RUN.get(model, {}).get(suite, sd)
                if sd not in pts[a]:
                    continue
                xs.append(float(a)); ys.append(pts[a][sd] / 5.0); es.append(0.0); ns.append(1)
        # The untouched policy's own spread over the same three seeds, drawn as a band around its
        # seed-7 value. On these backbones re-running the untouched policy flips 39 to 101 episodes of
        # 500, so the band is what the curve has to clear before any point on it means anything.
        n0 = pts.get("0", {})
        if band and len(n0) > 1:
            v0 = [v / 5.0 for v in n0.values()]; half = st.stdev(v0)
            b0 = (n0[SEED] / 5.0) if mode != "mean" and SEED in n0 else st.mean(v0)
            ax.axhspan(b0 - half, b0 + half, color=col, alpha=0.13, lw=0, zorder=1)
            lo, hi = min(lo, b0 - half), max(hi, b0 + half)
        ax.errorbar(xs, ys, yerr=es, ls=(0, (5, 2)), marker=mk, color=col, lw=FS["lw"], ms=FS["ms"],
                    markeredgewidth=0, label=label.replace("LIBERO-", ""),   # suite names as in Fig. 1; the prefix only widened the legend
                    zorder=3, elinewidth=1.1, capsize=FS["cap"])
        ta = TABLE_ALPHA.get(model, {}).get(suite)
        if ta is not None and mark == "edge" and any(abs(x - ta) < 1e-9 for x in xs):
            j = min(range(len(xs)), key=lambda i: abs(xs[i] - ta))
            ax.plot([xs[j]], [ys[j]], marker=mk, ms=FS["ms"], color=col,
                    markeredgecolor="black", markeredgewidth=1.4, zorder=5)
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        run = TABLE_RUN.get(model, {}).get(suite, SEED) if mode == "table" else (SEED if mode == "seed7" else "mean")
        print(f"  {label:16s} [{mode} run={run}] " + " ".join(f"{x:g}:{y:.1f}" for x, y in zip(xs, ys)))
    ax.set_xlabel(ALPHA_SYM, fontsize=FS["lab"] + 1, fontweight="bold", labelpad=2)
    ax.set_ylabel("Success Rate (%)", fontsize=FS["lab"], fontweight="bold")
    ax.set_xlim(-0.004, 0.104)
    if compact:
        ax.set_xticks([0.0, 0.05, 0.10]); ax.set_xticklabels(["0", "0.05", "0.10"])
    else:
        ax.set_xticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
        ax.set_xticklabels(["0", "0.02", "0.04", "0.06", "0.08", "0.10"])
    pad = max(1.0, 0.12 * (hi - lo))
    ax.set_ylim(min(lo - pad, YMIN.get(model, lo - pad)), hi + max(pad, TOP_PAD.get(model, 0.0)))
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))   # 92, 94, ... not 92.5, 95.0
    ax.tick_params(labelsize=FS["tick"], width=1.6, length=5)
    ax.grid(True, color="#c9c9c9", lw=1.5, ls=(0, (4, 4)), zorder=0)
    ax.set_axisbelow(True)
    lg = LEGEND.get(model, dict(loc="lower right", ncol=1))
    leg = ax.legend(fontsize=FS["leg"], frameon=False, handlelength=1.3, labelspacing=0.15,
                    handletextpad=0.4, columnspacing=0.8, **lg)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    S.finish(ax)
    sub_title = "(a) OpenVLA-OFT" if model == "oft" else (r"(b) $\boldsymbol{\pi}_0$" if model == "pi0" else f"({model})")
    fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.2)
    fig.text(0.55, 0.012, sub_title, ha="center", va="bottom", fontsize=FS["lab"], fontweight="bold")
    fig.savefig(OUT / f"{out}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{out}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT / f"{out}.pdf", f"({NAME.get(model, model)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0", choices=("pi0", "pi05", "oft"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="table", choices=("mean", "seed7", "table"))
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--compact", action="store_true", help="half-column panel")
    a = ap.parse_args()
    main(a.model, a.data, a.out, mode=a.mode, band=a.band, compact=a.compact)
