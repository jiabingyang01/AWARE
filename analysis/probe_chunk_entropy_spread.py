"""How much the Action Entropy varies between elements of one chunk.

The rule's premise is that the elements of a single action chunk are at different stages of being
decided, which is the structure a threshold on the chunk average throws away. The evaluation logs only
the per-layer average, so this probe records the individual values: for every forward it keeps the
N_a per-element entropies at each layer and reports, per layer, the spread across elements relative to
their mean. Read-only; the hook returns nothing and the policy's actions are the baseline's.

  CUDA_VISIBLE_DEVICES=k python probe_chunk_entropy_spread.py --suite libero_10 --episodes 6
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
import numpy as np, torch, glob

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--suite", default="libero_10")
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--out", default="")
a = ap.parse_args()

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image, quat2axisangle
from experiments.robot.robot_utils import get_action
from libero.libero import benchmark

key = {"libero_10": "libero-10", "libero_spatial": "libero-spatial",
       "libero_object": "libero-object", "libero_goal": "libero-goal"}[a.suite]
ckpt = sorted(glob.glob(f_paths.HF_HUB + "/models--moojink--openvla-7b-oft-finetuned-{key}/snapshots/*/"))[0]
cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name=a.suite, num_trials_per_task=1,
                     center_crop=True, num_open_loop_steps=8, seed=7)
model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
lm = model.language_model
layers = list(lm.model.layers)
L = len(layers)
N_ACT, K = 56, 256
head, norm = lm.lm_head, lm.model.norm

per_layer = [[] for _ in range(L)]          # list of (min, max, mean, std) over the chunk
per_elem = np.zeros((L, N_ACT)); n_fwd = np.zeros(L)   # mean entropy of each element at each layer
quant = [[] for _ in range(L)]              # per-forward deciles across the chunk


def make_hook(l):
    def hook(module, args, kwargs, output):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] <= N_ACT + 1:
            return None
        with torch.no_grad():
            ha = (output[0] if isinstance(output, tuple) else output)[0, -(N_ACT + 1):-1]
            logits = head(norm(ha.to(head.weight.dtype))).float()
            top = logits.topk(K, dim=-1).values
            p = torch.softmax(top, dim=-1)
            ent = (-(p * p.clamp_min(1e-12).log()).sum(-1) / np.log(K)).cpu().numpy()
            per_layer[l].append((float(ent.min()), float(ent.max()), float(ent.mean()), float(ent.std())))
            per_elem[l] += ent; n_fwd[l] += 1
            quant[l].append(np.percentile(ent, [10, 25, 50, 75, 90]))
        return None
    return hook


for l, lyr in enumerate(layers):
    lyr.register_forward_hook(make_hook(l), with_kwargs=True)

suite = benchmark.get_benchmark_dict()[a.suite]()
for task_id in range(min(a.episodes, suite.n_tasks)):
    task = suite.get_task(task_id)
    env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    env.reset(); obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
    dn = False
    for _ in range(10):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    for _ in range(a.steps):
        observation = {"full_image": get_libero_image(obs), "wrist_image": get_libero_wrist_image(obs),
                       "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                obs["robot0_gripper_qpos"]))}
        acts = get_action(cfg, model, observation, desc, processor=processor, action_head=action_head,
                          proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
                          use_film=cfg.use_film)
        for act in acts:
            obs, _, dn, _ = env.step(act.tolist())
            if dn:
                break
        if dn:
            break
    env.close()
    print(f"  task {task_id + 1}/{a.episodes}", flush=True)

res = {"suite": a.suite, "n_act": N_ACT, "layers": [],
       "per_element": (per_elem / np.maximum(n_fwd, 1)[:, None]).tolist(),
       "deciles": [np.mean(q, axis=0).tolist() if q else None for q in quant]}
for l in range(L):
    if not per_layer[l]:
        continue
    arr = np.array(per_layer[l])
    res["layers"].append({"layer": l, "n": len(arr), "min": float(arr[:, 0].mean()),
                          "max": float(arr[:, 1].mean()), "mean": float(arr[:, 2].mean()),
                          "std": float(arr[:, 3].mean()),
                          "ratio_max_over_min": float((arr[:, 1] / np.maximum(arr[:, 0], 1e-6)).mean())})
print(json.dumps({d["layer"]: round(d["ratio_max_over_min"], 2) for d in res["layers"]}))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
