"""!! CONTAMINATED MEASUREMENT -- DO NOT USE THE PAIRED DIFFERENCES !!

This probe compares a rule-on forward against a rule-off forward element by element. That design is
invalid on this policy: running the SAME observation through it twice with nothing injected already
produces a relative deviation of 0.22 at the action rows, because a bf16-level difference of 0.0019 at
layer 0 is amplified roughly two hundredfold across depth. The noise floor is about three quarters of
the signal, and the deviation is already 0.13 eleven layers BELOW the layer the write happens at,
which is impossible for a real effect.

Verified 2026-09-16 with an identity control (alpha = 0 in both forwards): mean delta 0.224, versus
0.25 at the write layer with the real write.

Comparisons of DISTRIBUTIONS over many steps are unaffected and remain valid. Per-sample differences
are not. A clean isolation needs one forward that branches at the write layer and runs the upper stack
twice from the same tensor; that probe is not written yet.

Kept for the record and for whoever writes the branching version.

ORIGINAL HEADER FOLLOWS
"""

"""Does the rule act on the elements it says it acts on?  Measured element by element.

For every control step the policy is run twice on the SAME observation: once with the rule at its
strength, once with the strength forced to zero (hooks installed, blend inert, so the two forwards are
identical except for the write).  Per element n we keep

  H_n   its Action Entropy at layer --layer (the quantity the weight is read from), and
  d_n   how far the rule moved that element's decoded action, |a_n(rule) - a_n(off)|, in units of the
        element's own scale over the run.

If the mechanism is what the paper says, d_n grows with H_n and the settled elements (low H) are left
essentially untouched.  Thousands of paired samples per suite; no success rate enters.

  CUDA_VISIBLE_DEVICES=k python probe_element_response.py --suite libero_10 --layer 16 --out resp.json
"""
# --- AWARE path bootstrap: every host path comes from aware/paths.py, overridable by env ---------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# ------------------------------------------------------------------------------------------------
import argparse, json, os, sys

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LIBERO_CONFIG_PATH", _os.path.expanduser(_os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")))
sys.path.insert(0, _paths.OPENVLA_OFT); sys.path.insert(0, _paths.LIBERO)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--suite", default="libero_10")
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--layer", type=int, default=16)
ap.add_argument("--alpha", default="0.03")
ap.add_argument("--out", default="")
a = ap.parse_args()
os.environ.update(AWARE_ALPHA=a.alpha, AWARE_TRIGGER="soft", AWARE_SOFT_DEPTH="ramp", AWARE_SELECT="instruction")

import numpy as np, torch, glob
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image, quat2axisangle
from experiments.robot.robot_utils import get_action
from libero.libero import benchmark

key = {"libero_10": "libero-10", "libero_spatial": "libero-spatial",
       "libero_object": "libero-object", "libero_goal": "libero-goal"}[a.suite]
ckpt = sorted(glob.glob(_paths.HF_HUB + f"/models--moojink--openvla-7b-oft-finetuned-{key}/snapshots/*/"))[0]
cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name=a.suite, center_crop=True,
                     num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False)
model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

import importlib.util as ilu
spec = ilu.spec_from_file_location("aware_hooks", _paths.HOOKS_FILE)
uh = ilu.module_from_spec(spec); sys.modules["aware_hooks"] = uh; spec.loader.exec_module(uh)
st = uh.install(model, use_proprio=cfg.use_proprio)
ALPHA = float(a.alpha)
ell = min(a.layer, st.L - 1)

cap = {"H": None, "hlast": None}
def ent_hook(module, args, output):
    h = output[0] if isinstance(output, (tuple, list)) else output
    if h.shape[1] > st.obs_offset + st.n_obs:                # prefill only
        with torch.no_grad():
            cap["H"] = st.action_entropy_vec(h)[0].float().cpu().numpy()
    return None
st.layers[ell].register_forward_hook(ent_hook)

def last_hook(module, args, output):
    h = output[0] if isinstance(output, (tuple, list)) else output
    if h.shape[1] > st.obs_offset + st.n_obs:
        cap["hlast"] = h[0, st._act_slice(h.shape[1])].detach().float().cpu().numpy()   # [56, d]
    return None
st.layers[st.L - 1].register_forward_hook(last_hook)

def act(observation, desc):
    return np.asarray(get_action(cfg, model, observation, desc, processor=processor, action_head=action_head,
                                 proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
                                 use_film=cfg.use_film))

H_all, D_all, DH_all = [], [], []
suite = benchmark.get_benchmark_dict()[a.suite]()
for task_id in range(min(a.episodes, suite.n_tasks)):
    task = suite.get_task(task_id)
    env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    env.reset(); obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
    for _ in range(10):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    dn = False
    for t in range(a.steps):
        observation = {"full_image": get_libero_image(obs), "wrist_image": get_libero_wrist_image(obs),
                       "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                obs["robot0_gripper_qpos"]))}
        st.alpha = ALPHA
        a_on = act(observation, desc)                     # the rule at its strength
        H = cap["H"].copy()                               # entropy of that same forward, at layer ell
        h_on = cap["hlast"].copy()
        st.alpha = 0.0
        a_off = act(observation, desc)                    # identical forward, blend inert
        h_off = cap["hlast"].copy()
        st.alpha = ALPHA
        DH_all.append(np.linalg.norm(h_on - h_off, axis=1) / (np.linalg.norm(h_off, axis=1) + 1e-9))
        H_all.append(H)                                   # [56]
        D_all.append(np.abs(a_on - a_off).reshape(-1))    # [56] = 8 steps x 7 dims, chunk-major
        for x in a_on:                                    # execute the rule's actions (the arm's episode)
            obs, _, dn, _ = env.step(x.tolist())
            if dn:
                break
        if dn:
            break
    env.close()
    print(f"  task {task_id + 1} done, {len(H_all)} steps", flush=True)

H_all, D_all, DH_all = np.array(H_all), np.array(D_all), np.array(DH_all)   # [T, 56]
# scale every element by its own typical action magnitude so the seven dimensions are comparable
scale = np.abs(D_all).mean(axis=0) + 1e-9
res = {"suite": a.suite, "layer": ell, "alpha": a.alpha, "steps": int(H_all.shape[0]),
       "H": H_all.tolist(), "D": D_all.tolist(), "DH": DH_all.tolist(), "n_act": int(H_all.shape[1])}
r = np.corrcoef(H_all.reshape(-1), (D_all / scale).reshape(-1))[0, 1]
rh = np.corrcoef(H_all.reshape(-1), DH_all.reshape(-1))[0, 1]
print(f"steps={H_all.shape[0]}  corr(H, action displacement) = {r:.3f}   corr(H, last-layer hidden displacement) = {rh:.3f}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
