#!/usr/bin/env python3
"""One AWARE cell on CogACT x SIMPLER Visual Matching, through the repo's own 8-GPU launcher.

The paired baseline is not re-run: three repeats of the released policy under this exact launcher
are already on disk (cogact_released_vm_primary_optix_s1_*), and they reproduce both CoRL's
reproduction and the CogACT paper (per task 92.0 / 85.0 / 73.4 / 46.9, mean 74.3).

The arm differs from those runs in one environment variable. The launcher passes os.environ through
to its workers, so `--mode baseline --replan-stride 1` with AWARE_ALPHA set is the same condition
with the rule switched on, episode for episode.

As with the pi0 cells, the arm is only accepted if every worker log shows the executor installed.
"""
# --- AWARE path bootstrap ------------------------------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# -------------------------------------------------------------------------------------------
import argparse, collections, glob, json, os, subprocess, sys, time

SIMPLER = _paths.SIMPLER
# The repo's own condition runner, not a private call. It holds the OptiX/Vulkan environment, the
# PYTHONPATH, the per-condition lock and the host-wide GPU lock, and it invokes the launcher as a
# MODULE -- running the launcher by path puts simpler_env/evaluation on sys.path, where its own
# argparse.py shadows the standard library and the process dies on import.
RUNNER = f"{SIMPLER}/scripts/cogact_run_condition_v2.sh"
# Visual Matching has its OWN launcher. The variant-aggregation one is a different benchmark with a
# different grid (1992 episodes against 864); the two manifests differ only in this module name, so
# using the wrong one produces a full, plausible, incomparable result.
LAUNCHER = "simpler_env.evaluation.launch_cogact_visual_matching"
PROTOCOL = "released_code_horizon2_total"      # the label the three baseline repeats were run under
RESULTS = os.environ.get("AWARE_COGACT_RESULTS",
                         _os.path.join(_paths.SIMPLER, "results/icra27_aware"))


def score(root):
    agg = collections.defaultdict(lambda: [0, 0])
    for f in glob.glob(os.path.join(root, "workers", "*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("record_type") not in (None, "episode"):
                continue
            g = r["task_group"]
            agg[g][1] += 1
            agg[g][0] += int(bool(r["success"]))
    return {g: {"success": v[0], "episodes": v[1], "rate": v[0] / v[1]} for g, v in agg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--alpha", default="")
    p.add_argument("--base-seed", type=int, default=20260724)
    p.add_argument("--extra", nargs="*", default=[])
    a = p.parse_args()

    out = os.path.join(RESULTS, a.tag)
    os.makedirs(out, exist_ok=True)
    env = os.environ.copy()
    if a.alpha:
        env.update(AWARE_ALPHA=str(a.alpha), AWARE_TRIGGER="soft",
                   AWARE_SOFT_DEPTH="ramp", AWARE_SELECT="instruction")
    for kv in a.extra:
        k, _, v = kv.partition("=")
        env[k] = v

    settings = {"tag": a.tag, "alpha": a.alpha, "base_seed": a.base_seed,
                "extra": a.extra, "started": time.strftime("%F %T")}
    json.dump(settings, open(os.path.join(out, "settings.json"), "w"), indent=2)

    cmd = ["bash", RUNNER, "visual_matching", "baseline", out, "1", PROTOCOL,
           LAUNCHER, str(a.base_seed)]
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=SIMPLER, env=env)
    wall = time.time() - t0

    mode_root = os.path.join(out, "baseline")
    logs = glob.glob(os.path.join(mode_root, "logs", "worker_*.log"))
    proof = sum(1 for f in logs if "[AWARE-cogact] installed" in open(f, errors="ignore").read())
    per = score(mode_root)
    tot = sum(v["episodes"] for v in per.values())
    suc = sum(v["success"] for v in per.values())

    res = {**settings, "finished": time.strftime("%F %T"), "wall_s": round(wall, 1), "rc": rc,
           "per_task": per, "success": suc, "episodes": tot,
           "rate": (suc / tot if tot else 0.0),
           "task_mean": (sum(v["rate"] for v in per.values()) / len(per) if per else 0.0),
           "n_workers": len(logs), "aware_proof": proof}
    ok = (proof == len(logs) == 8) if a.alpha else (proof == 0)
    if not ok or tot != 864 or suc == 0:
        res["error"] = ("arm proof failed" if not ok else
                        f"expected 864 episodes, got {tot}" if tot != 864 else "zero successes")
        json.dump(res, open(os.path.join(out, "result_partial.json"), "w"), indent=2)
        print("FAIL:", res["error"], file=sys.stderr)
        return 3
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=2)
    print(f"{a.tag}: {suc}/{tot} episodes, task mean {100*res['task_mean']:.1f}%  "
          f"({wall/60:.1f} min, proof {proof}/{len(logs)})")
    for g in sorted(per):
        print(f"    {g:26s} {per[g]['success']:3d}/{per[g]['episodes']:3d} = {100*per[g]['rate']:5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
