"""Per-task variant of probe_obs_footprint.py: the same measurement, recorded task by task.

The aggregate curve answers "does the observation fade with depth". This variant answers the question
the mechanism figure needs: does it fade MORE on the tasks where AWARE helps more. One number per task
(the footprint the action rows have lost by the deep layers) is paired against that task's gain, which
the evaluation ledgers already carry per task.

Measure how much of the action representation still lies in the observation subspace, layer by layer.

The companion probe (probe_attention_decay.py) measured attention share and found it non-monotonic, so
the depth factor l/(L-1) is not tracking attention. This probe measures the other candidate: the share
of each action row's representation that the observation tokens can still explain. That is the quantity
the rule actually manipulates, since the retrieved vector is built from the action row's inner products
with the observation rows. If the share falls with depth, the ramp puts the dose where the observation
has faded from the representation, which is the claim the paper needs.

Three read-only quantities per layer, averaged over action rows and control steps:
  same_layer   ||P_O(l) a_n|| / ||a_n||  -- explained by the observation rows AT THAT layer (what the
               rule reads from)
  layer0       ||P_O(0) a_n|| / ||a_n||  -- explained by the raw observation embeddings entering the
               stack, i.e. relevance to the original percept
  retr_cos     cos(a_n, sum_i <a_n,o_i> o_i) -- alignment between the action row and the exact direction
               the rule would inject there

Projection onto the row space of O uses ||P a||^2 = c^T G^{-1} c with G = O O^T and c = O a, so no basis
is ever formed. Nothing is written back, so the policy's behaviour is the baseline's.

  CUDA_VISIBLE_DEVICES=k python probe_obs_footprint.py --suite libero_10 --episodes 6 --out fp.json
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
N_OBS = 513 if cfg.use_proprio else 512
N_ACT = 56
OBS0 = 1

same = np.zeros(L); zero = np.zeros(L); rcos = np.zeros(L)
same_c = np.zeros(L); zero_c = np.zeros(L); aniso = np.zeros(L)
act_sim = np.zeros((L, L)); obs_sim = np.zeros((L, L))
count = 0
NSUB = 64                     # observation rows kept for the layer-by-layer similarity map
state = {"O0": None, "A": [], "Osub": [], "idx": None}


def proj_ratio(A, O):
    """Fraction of each row of A that the row space of O explains, as a norm ratio in [0, 1]."""
    G = O @ O.transpose(0, 1)
    G = G + (torch.diagonal(G).mean() * 1e-6) * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    C = A @ O.transpose(0, 1)                       # [n_act, m]
    X = torch.linalg.solve(G, C.transpose(0, 1))    # [m, n_act]
    num = (C.transpose(0, 1) * X).sum(0).clamp_min(0).sqrt()
    return (num / A.norm(dim=1).clamp_min(1e-6)).mean().item()


def make_hook(l):
    def hook(module, args, kwargs, output):
        global count
        h_in = args[0] if args else kwargs["hidden_states"]
        if h_in.shape[1] <= OBS0 + N_OBS:
            return None
        with torch.no_grad():
            if l == 0:
                state["O0"] = h_in[0, OBS0:OBS0 + N_OBS].float()
                state["mu0"] = h_in[0].float().mean(0)
            h = (output[0] if isinstance(output, tuple) else output)[0].float()
            A = h[-(N_ACT + 1):-1]
            O = h[OBS0:OBS0 + N_OBS]
            if l == 0:
                state["A"] = []; state["Osub"] = []
            if state["idx"] is None:
                state["idx"] = torch.linspace(0, N_OBS - 1, NSUB).long().to(O.device)
            state["A"].append(torch.nn.functional.normalize(A, dim=1))
            state["Osub"].append(torch.nn.functional.normalize(O[state["idx"]], dim=1))
            same[l] += proj_ratio(A, O)
            zero[l] += proj_ratio(A, state["O0"])
            # Late layers of a decoder are strongly anisotropic: every hidden state acquires the same
            # large shared component, which lifts any projection ratio towards one regardless of content.
            # Removing the layer mean before projecting is what separates a genuine return of observation
            # content from that artefact, so both versions are recorded and the anisotropy is measured
            # directly as the mean pairwise cosine between observation rows.
            mu = h.mean(0)
            Ac, Oc = A - mu, O - mu
            same_c[l] += proj_ratio(Ac, Oc)
            zero_c[l] += proj_ratio(Ac, state["O0"] - state["mu0"])
            On = torch.nn.functional.normalize(O[state["idx"]] if state["idx"] is not None else O, dim=1)
            M = On @ On.transpose(0, 1)
            aniso[l] += ((M.sum() - M.shape[0]) / (M.shape[0] * (M.shape[0] - 1))).item()
            S = A @ O.transpose(0, 1)                     # [n_act, m] inner products the rule uses
            R = S @ O                                     # the unnormalised retrieved vector
            rcos[l] += torch.nn.functional.cosine_similarity(A, R, dim=1).mean().item()
            if l == L - 1:
                # Two layer-by-layer similarity maps. The action map is the quantity other papers plot
                # to argue that consecutive layers are redundant; it is reported here only as the
                # reference point, because redundancy between layers says nothing about whether the
                # observation is still represented. The footprint curves above are what does.
                SA = torch.stack(state["A"])                    # [L, n_act, d]
                SO = torch.stack(state["Osub"])                 # [L, NSUB, d]
                act_sim[:] += torch.einsum("lnd,mnd->lm", SA, SA).div(SA.shape[1]).cpu().numpy()
                obs_sim[:] += torch.einsum("lnd,mnd->lm", SO, SO).div(SO.shape[1]).cpu().numpy()
                state["A"] = []; state["Osub"] = []
                count += 1
        return None
    return hook


for l, lyr in enumerate(layers):
    lyr.register_forward_hook(make_hook(l), with_kwargs=True)

suite = benchmark.get_benchmark_dict()[a.suite]()
per_task = []
for task_id in range(min(a.episodes, suite.n_tasks)):
    count_at_task_start = count
    same_c_at_task_start = same_c.copy()
    task = suite.get_task(task_id)
    env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    env.reset(); obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
    dn = False
    for _ in range(10):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    for t in range(a.steps):
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
    n_t = max(1, count - count_at_task_start)
    per_task.append({"task_id": task_id, "desc": desc, "forwards": int(n_t),
                     "same_layer_centered": ((same_c - same_c_at_task_start) / n_t).tolist()})
    print(f"  task {task_id + 1}/{a.episodes} done, {count} forwards", flush=True)

c = max(1, count)
res = {"suite": a.suite, "forwards": int(count), "per_task": per_task, "n_obs": N_OBS, "n_act": N_ACT,
       "same_layer_footprint": (same / c).tolist(),
       "layer0_footprint": (zero / c).tolist(),
       "retrieval_cos": (rcos / c).tolist(),
       "forgotten": (1.0 - same / c).tolist(),
       "same_layer_centered": (same_c / c).tolist(),
       "layer0_centered": (zero_c / c).tolist(),
       "obs_anisotropy": (aniso / c).tolist(),
       "action_layer_sim": (act_sim / c).tolist(),
       "obs_layer_sim": (obs_sim / c).tolist(),
       "linear_ramp": [l / (L - 1) for l in range(L)]}
print(json.dumps({k: [round(x, 4) for x in v] for k, v in res.items()
                  if k.endswith(("footprint", "cos", "centered", "anisotropy"))}, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
