# --- AWARE path bootstrap: every host path comes from aware/paths.py, overridable by env ---------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# ------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
"""One AWARE cell on pi0 / pi0.5 x LIBERO, through the repo's own parallel harness.

The cell is the unit the queue claims: 10 tasks x 50 trials = 500 episodes on the box's 8 GPUs,
replan_steps 5 and 10 denoise steps, which is the protocol the paired baseline was measured under.

The one thing this wrapper adds to run_parallel_eval.sh is the proof. An environment variable
reaching a process is not evidence that the executor ran: the arm is only accepted if every policy
server logged that AWARE installed. A cell whose proof fails writes result_partial.json and exits
non-zero, so the queue releases the claim instead of recording a plausible-looking baseline score.
"""
import argparse, glob, json, os, re, subprocess, sys, time

ROOT = _paths.OPENPI
RESULTS = os.environ.get("AWARE_PI0_RESULTS", _os.path.join(_paths.OPENPI, "eval_runs/icra27_aware"))
CKPT_ROOT = _os.environ.get("AWARE_PI_CKPTS", _os.path.join(_os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi"), "openpi-assets/pytorch_checkpoints"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default="pi0", choices=["pi0", "pi05"])
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--alpha", default="")               # empty -> released policy, no hooks
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--replan", type=int, default=5)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--port", type=int, default=8400)
    p.add_argument("--extra", nargs="*", default=[])     # KEY=VAL passed straight into the env
    a = p.parse_args()

    out = os.path.join(RESULTS, a.tag)
    os.makedirs(out, exist_ok=True)
    env = os.environ.copy()
    env.update(CKPT_ROOT=CKPT_ROOT, MODEL=a.model, TASK_SUITE=a.suite,
               NUM_GPUS=str(a.gpus), NUM_TRIALS=str(a.trials), REPLAN_STEPS=str(a.replan),
               SEED=str(a.seed), BASE_PORT=str(a.port), RUN_ID=a.tag, LOG_DIR=out)
    if a.alpha:
        env.update(AWARE_ALPHA=str(a.alpha), AWARE_TRIGGER="soft",
                   AWARE_SOFT_DEPTH="ramp", AWARE_SELECT="instruction")
    for kv in a.extra:
        k, _, v = kv.partition("=")
        env[k] = v

    settings = {"tag": a.tag, "model": a.model, "suite": a.suite, "seed": a.seed,
                "alpha": a.alpha, "trials": a.trials, "replan": a.replan,
                "extra": a.extra, "started": time.strftime("%F %T")}
    json.dump(settings, open(os.path.join(out, "settings.json"), "w"), indent=2)

    t0 = time.time()
    rc = subprocess.call(["bash", "examples/libero/run_parallel_eval.sh"], cwd=ROOT, env=env)
    wall = time.time() - t0

    combined = sorted(glob.glob(os.path.join(out, "*-combined.json")), key=os.path.getmtime)
    if not combined:
        json.dump({**settings, "error": "no combined json", "rc": rc, "wall_s": wall},
                  open(os.path.join(out, "result_partial.json"), "w"), indent=2)
        print("FAIL: no combined json", file=sys.stderr)
        return 2
    d = json.load(open(combined[-1]))

    # proof: every server must have logged that the executor installed
    logs = glob.glob(os.path.join(out, "server-gpu*.log"))
    proof = sum(1 for f in logs if "[AWARE-pi0] installed" in open(f, errors="ignore").read())
    alphas = set(re.findall(r"\[AWARE-pi0\] installed alpha=([0-9.]+)",
                            "".join(open(f, errors="ignore").read() for f in logs)))
    ok = (proof == len(logs) == a.gpus) if a.alpha else (proof == 0)

    res = {**settings, "finished": time.strftime("%F %T"), "wall_s": round(wall, 1), "rc": rc,
           "success": d["total_successes"], "episodes": d["total_episodes"],
           "rate": d["overall_success_rate"], "n_servers": len(logs),
           "aware_proof": proof, "alpha_in_logs": sorted(alphas),
           "latency_ms": d.get("overall_vla_latency_ms"), "control_hz": d.get("overall_control_hz")}
    if not ok or d["total_episodes"] != 10 * a.trials:
        res["error"] = ("arm proof failed" if not ok else
                        f"expected {10 * a.trials} episodes, got {d['total_episodes']}")
        json.dump(res, open(os.path.join(out, "result_partial.json"), "w"), indent=2)
        print("FAIL:", res["error"], file=sys.stderr)
        return 3
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=2)
    print(f"{a.tag}: {res['success']}/{res['episodes']} = {100*res['rate']:.1f}%  "
          f"({wall/60:.1f} min, proof {proof}/{len(logs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
