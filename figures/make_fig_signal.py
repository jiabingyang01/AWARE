#!/usr/bin/env python3
r"""Why the per-element dose is read from the Action Entropy, and what is written.

Panel (a): the same rule with the per-element weight read from a different signal, each swept over its
own strength on LIBERO-Long. The alternatives are rescaled to the mean of the entropy they replace, so
every point on every curve injects the same total mass -- only the ranking of the elements changes.

Panel (b): the strength held at the rule's own value while the retrieved content or its routing is
destroyed, so the dose is identical by construction across the bars.

All numbers are read from the evaluation result files; nothing is typed in here.
Fonts follow figures/scripts/make_fig_footprint.py so the three figures match in the paper.
"""
import json, glob, os, re, sys, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = ROOT / "experiments/results/uaor_select_20260911"
OUT = HERE.parent

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
prop = fm.FontProperties(fname=FONT, weight="bold")
fm.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.family": prop.get_name(), "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "mathtext.fontset": "dejavuserif", "mathtext.default": "bf",
    "axes.linewidth": 1.1, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42,
})
TICK, LAB, LEG = 6.1, 7.1, 5.9
N_EP = 500.0
BASE = np.array([472, 469, 466], dtype=float)          # LIBERO-Long, seeds 7/8/9 (ledger 1.1)


def load(tag):
    p = RES / tag / "result.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    return j.get("success"), (j.get("extra_env") or {})


# The canonical run directories for each weighting signal. Selecting by tag rather than by switches
# matters: many exploratory runs share the same switch set at the same alpha (selrows, gl8/gl16, hs,
# ml16, fast ...), and matching on switches alone silently averages them into the curve.
TAGS = {
    "entropy":  [r"^long500_rampsel_a(\d+)_s(\d)$"],
    "step":     [r"^long500_sig_step_s(\d)$", r"^long500_tune_sigstep_a(\d+)_s(\d)$"],
    "inverted": [r"^long500_sig_inv_s(\d)$", r"^long500_tune_siginv_a(\d+)_s(\d)$"],
    "margin":   [r"^long500_sig_margin_s(\d)$"],
}


def sweep(sig):
    """{alpha: [s7, s8, s9]} for one weighting signal, keeping only alphas with all three seeds.

    alpha is read from the run's own recorded environment, never parsed out of the directory name.
    """
    pats = [re.compile(r) for r in TAGS[sig]]
    out = collections.defaultdict(dict)
    for p in glob.glob(str(RES / "long500_*/result.json")):
        tag = os.path.basename(os.path.dirname(p))
        if not any(r.match(tag) for r in pats):
            continue
        m = re.search(r"_s(\d)$", tag)
        j = json.loads(Path(p).read_text())
        e = j.get("extra_env") or {}
        assert (e.get("UAOR_SOFT_SIGNAL") or "entropy") == sig, (tag, e.get("UAOR_SOFT_SIGNAL"))
        assert e.get("UAOR_TRIGGER") == "soft" and e.get("UAOR_SOFT_DEPTH") == "ramp" \
            and e.get("UAOR_SELECT") == "instruction", (tag, e)
        a = float(e["UAOR_ALPHA"])
        assert m.group(1) not in out[a], f"duplicate {sig} alpha={a} seed={m.group(1)}: {tag}"
        out[a][m.group(1)] = float(j["success"])
    full = {a: [v["7"], v["8"], v["9"]] for a, v in out.items() if set(v) >= {"7", "8", "9"}}
    return dict(sorted(full.items()))


SIGNALS = [("entropy", "Action Entropy (ours)", "#d62728", "o", "-"),
           ("step",    "chunk position",        "#4aa3df", "s", "--"),
           ("inverted","entropy inverted",      "#9467bd", "^", ":"),
           ("margin",  "top-1 margin",          "#7f7f7f", "D", "-.")]

CONTENT = [("long500_rampsel_a03", "the rule"),
           ("long500_abl_selinv",  "tilt inverted"),
           ("long500_abl_other",   "other episode"),
           ("long500_abl_shuffle", "weights shuffled"),
           ("long500_abl_gauss",   "random content"),
           ("long500_abl_damp",    "no content")]


def main():
    data = {s: sweep(s) for s, *_ in SIGNALS}
    for s, d in data.items():
        print(f"{s:9s}: {len(d)} alphas with 3 seeds -> " +
              ", ".join(f"{a:g}:{np.mean(v)/5:.1f}" for a, v in d.items()))

    W, H = 3.45, 1.66
    PB, PH = 0.425 / H, 1.13 / H
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0.125, PB, 1.20 / W, PH])
    bx = fig.add_axes([0.700, PB, 1.00 / W, PH])

    # ---------------- (a) which signal sets the per-element dose ----------------
    b_m, b_s = BASE.mean() / 5, BASE.std(ddof=1) / 5
    ax.axhspan(b_m - b_s, b_m + b_s, color="#e8e8e8", zorder=0, lw=0)
    ax.axhline(b_m, color="#888888", lw=1.0, ls=(0, (3, 2)), zorder=1)
    for sig, label, col, mk, ls in SIGNALS:
        d = data[sig]
        if not d:
            continue
        a = np.array(list(d)); y = np.array([np.mean(v) for v in d.values()]) / 5
        e = np.array([np.std(v, ddof=1) for v in d.values()]) / 5
        ax.errorbar(a, y, yerr=e, fmt="none", ecolor=col, elinewidth=0.6, capsize=0.9,
                    capthick=0.6, alpha=0.85, zorder=3)
        if len(a) >= 3:
            ax.plot(a, y, ls, color=col, lw=1.4, zorder=4)
            ax.plot(a, y, mk, color=col, ms=2.6, zorder=5, label=label)
        else:
            ax.plot(a, y, mk, color=col, ms=3.4, zorder=5, label=label)
        j = int(np.argmax(y))
        ax.plot(a[j], y[j], mk, color=col, ms=5.0, mfc="none", mew=1.2, zorder=6)

    ax.set_xlim(0.003, 0.107); ax.set_ylim(92.15, 96.4)
    ax.set_xticks([0.02, 0.04, 0.06, 0.08, 0.10]); ax.set_yticks([93, 94, 95, 96])
    ax.tick_params(labelsize=TICK, pad=1.5)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(prop); t.set_fontsize(TICK)
    ax.set_xlabel(r"strength $\alpha$", fontproperties=prop, fontsize=LAB, labelpad=1.5)
    ax.set_ylabel("success rate (%)", fontproperties=prop, fontsize=LAB, labelpad=1.5)
    ax.grid(True, color="#dddddd", lw=0.6, ls=(0, (2.5, 2.5)), zorder=0); ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=LEG - 0.7, loc="lower left", bbox_to_anchor=(-0.018, 0.005),
                    ncol=1, handletextpad=0.22, borderpad=0.0, labelspacing=0.10,
                    handlelength=0.9, borderaxespad=0.0, markerscale=0.8)
    for t in leg.get_texts():
        t.set_fontproperties(prop); t.set_fontsize(LEG - 0.7)

    # ---------------- (b) what is written, at the rule's own strength ----------------
    names, mus, sds = [], [], []
    for stem, label in CONTENT:
        v = [load(f"{stem}_s{s}") for s in (7, 8, 9)]
        v = [x[0] for x in v if x]
        if len(v) != 3:
            print(f"  missing: {stem}"); continue
        names.append(label); mus.append(np.mean(v) / 5); sds.append(np.std(v, ddof=1) / 5)
    y = np.arange(len(names))[::-1]
    mus = np.array(mus); sds = np.array(sds)
    cols = ["#d62728"] + ["#9aa0a6"] * (len(names) - 1)
    bx.barh(y, mus - b_m, left=b_m, xerr=sds, height=0.62, color=cols, zorder=3,
            error_kw=dict(elinewidth=0.7, capsize=1.2, capthick=0.7, ecolor="#444444"))
    bx.axvline(b_m, color="#888888", lw=1.0, ls=(0, (3, 2)), zorder=4)
    bx.set_yticks(y); bx.set_yticklabels(names)
    for t in bx.get_yticklabels():
        t.set_fontproperties(prop); t.set_fontsize(LEG - 0.5)
    bx.set_xlim(93.35, 96.25); bx.set_xticks([94, 95, 96])
    bx.tick_params(labelsize=TICK, pad=1.5)
    for t in bx.get_xticklabels():
        t.set_fontproperties(prop); t.set_fontsize(TICK)
    bx.set_xlabel("success rate (%)", fontproperties=prop, fontsize=LAB, labelpad=1.5)
    bx.grid(True, axis="x", color="#dddddd", lw=0.6, ls=(0, (2.5, 2.5)), zorder=0); bx.set_axisbelow(True)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)

    pa = ax.get_position(); pb = bx.get_position()
    fig.text((pa.x0 + pa.x1) / 2, 0.020, "(a) which signal sets the dose",
             ha="center", va="bottom", fontproperties=prop, fontsize=LAB)
    fig.text((pb.x0 + pb.x1) / 2, 0.020, "(b) what is written",
             ha="center", va="bottom", fontproperties=prop, fontsize=LAB)

    fig.savefig(OUT / "signal.pdf"); fig.savefig(OUT / "signal.png", dpi=300)
    print("wrote", OUT / "signal.pdf")
    print(f"baseline {b_m:.2f} +- {b_s:.2f}")
    for n, m, s in zip(names, mus, sds):
        print(f"  {n.replace(chr(10),' '):34s} {m:5.2f} +- {s:.2f}   ({m-b_m:+.2f} pp)")


if __name__ == "__main__":
    main()
