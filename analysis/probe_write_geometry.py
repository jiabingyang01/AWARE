"""What the write actually is, measured: the geometry of one blended FFN update.

Two questions the mechanism story rests on, neither of which the success rates can answer.

  1. Is the injected vector a different direction from the one the FFN would have produced, or only a
     rescaling of it?  If the angle between them is near zero the rule does nothing but scale the
     update and the subspace story is wrong.
  2. Does the blend move the action state toward the observation subspace?  Both states are captured in
     the SAME forward (pre-blend and post-blend at the same layer), so the comparison is paired and no
     trajectory divergence enters it.

Capture hooks are registered around the AWARE hook: one before it sees the FFN's own output, one after
it sees the blended output, because a forward hook that returns a value replaces the output for every
hook that follows.

  CUDA_VISIBLE_DEVICES=k python probe_write_geometry.py --suite libero_10 --layer 26 --out geom.json
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
ap.add_argument("--layer", type=int, default=26)
ap.add_argument("--alpha", default="0.03")
ap.add_argument("--out", default="")
a = ap.parse_args()

# the rule's own switches, identical to the runs the table reports
os.environ.update(AWARE_ALPHA=a.alpha, AWARE_TRIGGER="soft", AWARE_SOFT_DEPTH="ramp",
                  AWARE_SELECT="instruction")

import numpy as np, torch, glob
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image, quat2axisangle
from experiments.robot.robot_utils import get_action
from libero.libero import benchmark

key = {"libero_10": "libero-10", "libero_spatial": "libero-spatial",
       "libero_object": "libero-object", "libero_goal": "libero-goal"}[a.suite]
ckpt = sorted(glob.glob(f_paths.HF_HUB + "/models--moojink--openvla-7b-oft-finetuned-{key}/snapshots/*/"))[0]
cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name=a.suite, center_crop=True,
                     num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False)
model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

import importlib.util as ilu
spec = ilu.spec_from_file_location("aware_hooks", _paths.HOOKS_FILE)
uh = ilu.module_from_spec(spec); sys.modules["aware_hooks"] = uh; spec.loader.exec_module(uh)

lm = model.language_model if hasattr(model, "language_model") else model.llm_backbone.llm
st = uh.install(model, use_proprio=cfg.use_proprio)
layers = list(st.layers)
L = st.L
ell = min(a.layer, L - 1)

cap = {"pre": None, "post": None, "x": None}
mlp = layers[ell].mlp
# install() already ran, so this hook is called after the AWARE hook and sees the blended value; the
# pre-blend output is recovered by calling the module's forward directly, which skips every hook.
def post_hook(module, args, output):
    cap["post"] = output.detach()
    cap["x"] = args[0].detach()
    return None
mlp.register_forward_hook(post_hook)

rows = {"cos_fa": [], "rel_norm": [], "w": []}

def record():
    if cap["post"] is None or st.O is None:
        return
    sl = st._act_slice(cap["post"].shape[1])
    x = cap["x"][:, sl].float()                      # FFN input at the action tokens
    blended = cap["post"][:, sl].float()             # what the rule wrote
    with torch.no_grad():
        # .forward bypasses the hooks; calling the module would re-run the AWARE blend and this capture
        f = mlp.forward(cap["x"])[:, sl].float()     # what the FFN would have produced, recomputed
        O = st.O[0].float()                          # observation memory, layer-0 rows
        g = None if st.g is None else st.g[0].float()
        S = x[0] @ O.transpose(0, 1)
        if g is not None:
            S = S * g.unsqueeze(0)
        A = S @ O                                    # the injected direction, before the scale s
        s = f[0].abs().mean() / (A.abs().mean() + 1e-12)
        A = A * s
        cosfa = torch.nn.functional.cosine_similarity(f[0], A, dim=1)
        rows["cos_fa"] += cosfa.tolist()
        rows["rel_norm"] += (A.norm(dim=1) / f[0].norm(dim=1).clamp_min(1e-9)).tolist()
        num = (blended[0] - f[0])
        rows["w"] += (num.norm(dim=1) / (A.norm(dim=1) + 1e-9)).tolist()
    cap["post"] = None

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
        acts = get_action(cfg, model, observation, desc, processor=processor, action_head=action_head,
                          proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
                          use_film=cfg.use_film)
        record()
        for act in acts:
            obs, _, dn, _ = env.step(act.tolist())
            if dn:
                break
        if dn:
            break
    env.close()
    print(f"  task {task_id + 1} done, {len(rows['cos_fa'])} samples", flush=True)

res = {"suite": a.suite, "layer": ell, "alpha": a.alpha, "n": len(rows["cos_fa"]),
       "cos_fa": rows["cos_fa"], "rel_norm": rows["rel_norm"], "w": rows["w"]}
c = np.array(rows["cos_fa"])
print(f"cos(FFN output, injected): median {np.median(c):.3f}  |cos|<0.2 for {100*np.mean(np.abs(c)<0.2):.0f}% of writes")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
