"""Where does the AWARE arm's latency go on OpenVLA-7B?

The arm measured 496 ms per control step against a released arm of roughly 150. OpenVLA decodes
seven action tokens per step, so every hook runs on eight forwards, and the logit lens is a
[4096 x 32000] product each time: 32 layers x 8 forwards = 256 of them per control step. Two
switches already exist for exactly this and are default-off pending a no-loss check:

  AWARE_HEAD_SLICE=action   read only the last 256 vocabulary ids, which on this backbone ARE the
                            action bins, so the lens becomes a [4096 x 256] product (125x less)
  AWARE_MIN_LAYER=k         skip layers below k entirely; with the depth ramp their weight is
                            proportional to l/(L-1) and the shallow ones are nearly zero anyway

This measures each, and also checks that head_slice does not change the action the policy returns
by more than the dose it is supposed to carry.
"""
# --- AWARE path bootstrap: every host path comes from aware/paths.py, overridable by env ---------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# ------------------------------------------------------------------------------------------------
import importlib.util, os, sys, time
import numpy as np
import torch

SRV = _os.environ.get("AWARE_OPENVLA_SERVER", _os.path.join(_os.path.dirname(_paths.OPENVLA_OFT), "openvla/serve/openvla_server.py"))
os.environ.setdefault("OVLA_CKPT", "openvla/openvla-7b-finetuned-libero-10")
os.environ.setdefault("OVLA_UNNORM_KEY", "libero_10")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, _os.path.join(_paths.OPENPI, "src"))

CONFIGS = [
    ("released (no hooks)", {}),
    ("AWARE default", {"AWARE_ALPHA": "0.03", "AWARE_TRIGGER": "soft",
                       "AWARE_SOFT_DEPTH": "ramp", "AWARE_SELECT": "instruction"}),
    ("AWARE + head_slice", {"AWARE_ALPHA": "0.03", "AWARE_TRIGGER": "soft",
                            "AWARE_SOFT_DEPTH": "ramp", "AWARE_SELECT": "instruction",
                            "AWARE_HEAD_SLICE": "action"}),
    ("AWARE + head_slice + min_layer 16",
     {"AWARE_ALPHA": "0.03", "AWARE_TRIGGER": "soft", "AWARE_SOFT_DEPTH": "ramp",
      "AWARE_SELECT": "instruction", "AWARE_HEAD_SLICE": "action", "AWARE_MIN_LAYER": "16"}),
]

rng = np.random.default_rng(0)
obs = {"observation/image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
       "observation/state": np.zeros(8, dtype=np.float32),
       "prompt": "pick up the black bowl", "episode": "0"}

results = []
for name, extra in CONFIGS:
    for k in [k for k in os.environ if k.startswith("AWARE_")]:
        os.environ.pop(k)
    os.environ.update(extra)
    for m in [m for m in list(sys.modules) if m in ("openvla_server", "aware_hooks", "uaor_hooks")]:
        del sys.modules[m]
    spec = importlib.util.spec_from_file_location("openvla_server", SRV)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["openvla_server"] = mod
    spec.loader.exec_module(mod)
    pol = mod.OpenVLAPolicy()
    for _ in range(3):
        out = pol.infer(dict(obs))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 12
    for _ in range(N):
        out = pol.infer(dict(obs))
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / N
    a = np.asarray(out["actions"] if "actions" in out else out.get("action", []))
    results.append((name, ms, a.copy()))
    print(f"  {name:36s} {ms:7.1f} ms/step")
    del pol, mod
    torch.cuda.empty_cache()

base = results[0][1]
print()
for name, ms, _ in results:
    print(f"  {name:36s} {ms:7.1f} ms  = {ms/base:5.2f}x released")
print()
d = np.abs(results[2][2] - results[1][2]).max() if len(results) > 2 else float("nan")
print(f"head_slice vs full head, max |Delta action| = {d:.3e}")
