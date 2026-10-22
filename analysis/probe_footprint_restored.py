"""Does the injection put observation content back into the action representation?

probe_obs_footprint_pertask.py measures how much of an action token's state the observation tokens of
the same layer can explain, and shows that share falling with depth. That motivates the depth factor,
but it says nothing about whether the rule's write actually restores any of it. This probe answers that
directly: the same forward is run twice on the same frame, once with the rule at its strength and once
with the blend inert, and the footprint is recorded at every layer in both. Two curves from one run, so
nothing but the rule differs between them.

The share is measured with the layer mean removed, exactly as in the aggregate probe, because the upper
layers of a decoder are anisotropic and an uncentred ratio drifts towards one for any pair of states.

    python experiments/probe_footprint_restored.py --steps 40 --episodes 6 --alpha 0.03 --out out.json
"""
# --- AWARE path bootstrap: every host path comes from aware/paths.py, overridable by env ---------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# ------------------------------------------------------------------------------------------------
import argparse, json, os, sys

os.environ.setdefault("LIBERO_CONFIG_PATH", _os.path.expanduser(_os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")))
sys.path.insert(0, _paths.OPENVLA_OFT)
sys.path.insert(0, _paths.LIBERO)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--suite", default="libero_10")
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--alpha", default="0.03")
ap.add_argument("--out", default="")
a = ap.parse_args()
os.environ.update(AWARE_ALPHA=a.alpha, AWARE_TRIGGER="soft", AWARE_SOFT_DEPTH="ramp",
                  AWARE_SELECT="instruction")

import numpy as np, torch, glob
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.libero.libero_utils import (get_libero_env, get_libero_image,
                                                   get_libero_wrist_image, quat2axisangle)
from experiments.robot.robot_utils import get_action
from libero.libero import benchmark

key = {"libero_10": "libero-10", "libero_spatial": "libero-spatial",
       "libero_object": "libero-object", "libero_goal": "libero-goal"}[a.suite]
ckpt = sorted(glob.glob(f_paths.HF_HUB + "/"
                        f"models--moojink--openvla-7b-oft-finetuned-{key}/snapshots/*/"))[0]
cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name=a.suite, center_crop=True,
                     num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False)
model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

import importlib.util as ilu
spec = ilu.spec_from_file_location("aware_hooks",
                                   _paths.HOOKS_FILE)
uh = ilu.module_from_spec(spec); sys.modules["aware_hooks"] = uh; spec.loader.exec_module(uh)
st = uh.install(model, use_proprio=cfg.use_proprio)
ALPHA = float(a.alpha)
L = st.L
N_OBS, N_ACT, OBS0 = st.n_obs, st.n_act, st.obs_offset


def proj_ratio(A, O):
    """Fraction of each row of A that the row space of O explains, as a norm ratio in [0, 1]."""
    G = O @ O.transpose(0, 1)
    G = G + (torch.diagonal(G).mean() * 1e-6) * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    C = A @ O.transpose(0, 1)
    X = torch.linalg.solve(G, C.transpose(0, 1))
    num = (C.transpose(0, 1) * X).sum(0).clamp_min(0).sqrt()
    return (num / A.norm(dim=1).clamp_min(1e-6)).mean().item()


cur = np.zeros(L)        # same-layer observation subspace
cur0 = np.zeros(L)       # layer-0 observation embeddings: the subspace the rule actually writes into
mem = {"O0": None, "mu0": None}


def pre0(module, args, kwargs):
    h = args[0] if args else kwargs["hidden_states"]
    if h.shape[1] > OBS0 + N_OBS:
        with torch.no_grad():
            x = h[0].float()
            mem["O0"] = x[OBS0:OBS0 + N_OBS]
            mem["mu0"] = x.mean(0)
    return None


def make_hook(l):
    def hook(module, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        if h.shape[1] <= OBS0 + N_OBS:                      # decode step, not the chunk forward
            return None
        with torch.no_grad():
            x = h[0].float()
            A = x[st._act_slice(h.shape[1])]
            O = x[OBS0:OBS0 + N_OBS]
            mu = x.mean(0)
            cur[l] = proj_ratio(A - mu, O - mu)
            if mem["O0"] is not None:
                cur0[l] = proj_ratio(A - mu, mem["O0"] - mem["mu0"])
        return None
    return hook


st.layers[0].register_forward_pre_hook(pre0, with_kwargs=True)


for l in range(L):
    st.layers[l].register_forward_hook(make_hook(l))


def act(observation, desc):
    return np.asarray(get_action(cfg, model, observation, desc, processor=processor,
                                 action_head=action_head, proprio_projector=proprio_projector,
                                 noisy_action_projector=noisy_action_projector, use_film=cfg.use_film))


on_all, off_all, on0_all, off0_all = [], [], [], []
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
                       "state": np.concatenate((obs["robot0_eef_pos"],
                                                quat2axisangle(obs["robot0_eef_quat"]),
                                                obs["robot0_gripper_qpos"]))}
        st.alpha = ALPHA
        a_on = act(observation, desc)
        on_all.append(cur.copy()); on0_all.append(cur0.copy())
        st.alpha = 0.0
        act(observation, desc)                              # same frame, blend inert
        off_all.append(cur.copy()); off0_all.append(cur0.copy())
        st.alpha = ALPHA
        for x in a_on:                                      # the rule's own episode
            obs, _, dn, _ = env.step(x.tolist())
            if dn:
                break
        if dn:
            break
    env.close()
    print(f"  task {task_id + 1} done, {len(on_all)} paired steps", flush=True)

on = np.array(on_all); off = np.array(off_all)              # [T, L]
on0 = np.array(on0_all); off0 = np.array(off0_all)
CHANCE = (N_OBS / on.shape[0] * 0 + (N_OBS / 4096.0)) ** 0.5   # random subspace of the same dimension
res = {"suite": a.suite, "alpha": a.alpha, "steps": int(on.shape[0]), "L": int(L),
       "n_obs": int(N_OBS), "chance": CHANCE,
       "footprint_on": on.mean(0).tolist(), "footprint_off": off.mean(0).tolist(),
       "footprint_on_std": on.std(0).tolist(), "footprint_off_std": off.std(0).tolist(),
       "layer0_on": on0.mean(0).tolist(), "layer0_off": off0.mean(0).tolist(),
       "layer0_on_std": on0.std(0).tolist(), "layer0_off_std": off0.std(0).tolist()}
d = on.mean(0) - off.mean(0)
d0 = on0.mean(0) - off0.mean(0)
print(f"steps={on.shape[0]}")
print(f"  same-layer subspace : mean lift over layers 16-31 {d[16:].mean():+.4f}, max {d.max():+.4f} at layer {int(d.argmax())}")
print(f"  layer-0 subspace    : mean lift over layers 16-31 {d0[16:].mean():+.4f}, max {d0.max():+.4f} at layer {int(d0.argmax())}")
for l in range(0, L, 4):
    print(f"  layer {l:2d}: same {off.mean(0)[l]:.3f}->{on.mean(0)[l]:.3f} ({d[l]:+.4f})   "
          f"layer0 {off0.mean(0)[l]:.3f}->{on0.mean(0)[l]:.3f} ({d0[l]:+.4f})")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
