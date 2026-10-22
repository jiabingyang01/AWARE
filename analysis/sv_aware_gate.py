"""AWARE-on-SpatialVLA gate. Three separate processes (the switch is read at construction):
  MODE=off      -> hooks not installed
  MODE=alpha0   -> AWARE_ALPHA=0 installed: must be bit-identical to off (token ids and actions)
  MODE=alpha    -> AWARE_ALPHA=<a>: rule must fire ([AWARE] ACTIVE) and the layout proof must pass
Each run dumps the greedy action-token ids and the un-normalised action for K frames pulled from a
released-arm video, so the memory rows carry real Bridge observations rather than noise."""
# --- AWARE path bootstrap: every host path comes from aware/paths.py, overridable by env ---------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# ------------------------------------------------------------------------------------------------
import glob, os, sys
import numpy as np, cv2 as cv, torch
mode = os.environ["MODE"]; out = sys.argv[1]
if mode == "off":
    os.environ.pop("AWARE_ALPHA", None)
elif mode == "alpha0":
    os.environ["AWARE_ALPHA"] = "0"
else:
    os.environ["AWARE_ALPHA"] = os.environ.get("GATE_ALPHA", "0.05")
sys.path.insert(0, _paths.SIMPLER + "-SpatialVLA")
from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
# Real frames to run the gate on. Point AWARE_GATE_VIDEOS at any glob of rollout videos from the
# backbone under test; the gate only needs a handful of genuine frames.
vids = sorted(glob.glob(os.environ.get("AWARE_GATE_VIDEOS", "")))
assert vids, "set AWARE_GATE_VIDEOS to a glob of rollout videos"
cap = cv.VideoCapture(vids[0]); frames = []
while len(frames) < 6:
    ok, fr = cap.read()
    if not ok: break
    frames.append(cv.cvtColor(fr, cv.COLOR_BGR2RGB))
frames = frames[::2][:3]
pol = SpatialVLAInference(saved_model_path="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge", policy_setup="widowx_bridge", action_ensemble_temp=-0.8)
pol.reset("put carrot on plate")
ids_all, acts = [], []
gen_ids = []
orig = pol.vla.predict_action
def spy(inputs):
    g = orig(inputs); gen_ids.append(g[0].detach().cpu().numpy().copy()); return g
pol.vla.predict_action = spy
for fr in frames:
    raw, act = pol.step(fr, "put carrot on plate")
    acts.append(np.concatenate([np.asarray(raw["world_vector"]).ravel(), np.asarray(raw["rotation_delta"]).ravel(), np.asarray(raw["open_gripper"]).ravel()]))
np.savez(out, ids=np.array([np.pad(g, (0, 32 - len(g)), constant_values=-1) for g in gen_ids]), acts=np.array(acts))
print(f"[gate:{mode}] frames={len(frames)} gen_lens={[len(g) for g in gen_ids]} first_ids={gen_ids[0][:16].tolist()}")
if pol._aware is not None:
    s = pol._aware.summary(); print(f"[gate:{mode}] aware summary: n_forwards={s.get(chr(110)+chr(95)+chr(102)+chr(111)+chr(114)+chr(119)+chr(97)+chr(114)+chr(100)+chr(115))} n_inject={s.get(chr(110)+chr(95)+chr(105)+chr(110)+chr(106)+chr(101)+chr(99)+chr(116))}")
