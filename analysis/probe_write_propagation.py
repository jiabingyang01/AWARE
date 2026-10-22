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

"""What happens to the write as it travels up the stack, measured inside the network.

No success rate enters this probe, and nothing is tuned: alpha is fixed, the write is restricted to a
SINGLE layer (UAOR_START = UAOR_END = --write-layer), and the only thing that varies between runs is
WHAT is written. The question is whether the network treats the injected vector as signal or as noise.

For every control step the policy is run twice on the SAME observation, once with the write and once
with the blend inert, and at EVERY layer we record, for each of the 56 chunk elements:

  delta[t, l, n] = || h_on - h_off || / || h_off ||          how much of the write survives at layer l
  H_on / H_off[t, l, n]                                      Action Entropy of both forwards
  KL[t, l, n]     = KL( p_on || p_off )                      how much the element's own action
                                                             distribution was re-opened

Run once per content (three separate processes, because the hook reads UAOR_CONTENT at install time):

  obs    the real observation of this frame          (the rule)
  other  the observation of the previous episode     (a real scene, the wrong one)
  gauss  random directions with matched row norms    (off the observation manifold)

If the injected direction is processed as signal, delta should hold up or grow with depth for `obs`
and decay for `gauss`, with `other` in between. If all three decay the same way, the manifold story is
dead and the difference between them has to live somewhere else.

  CUDA_VISIBLE_DEVICES=1 python experiments/probe_write_propagation.py --content obs --write-layer 16
"""
# --- AWARE path bootstrap: every host path comes from aware/paths.py, overridable by env ---------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# ------------------------------------------------------------------------------------------------
import argparse, os, sys

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LIBERO_CONFIG_PATH", _os.path.expanduser(_os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")))
sys.path.insert(0, _paths.OPENVLA_OFT)
sys.path.insert(0, _paths.LIBERO)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--suite", default="libero_10")
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--write-layer", type=int, default=16)
ap.add_argument("--content", default="obs", choices=["obs", "other", "gauss"])
ap.add_argument("--alpha", default="0.03")
ap.add_argument("--out", default="")
a = ap.parse_args()
# the write is restricted to one layer, so the propagation above it is not contaminated by later writes
os.environ.update(AWARE_ALPHA=a.alpha, AWARE_TRIGGER="soft", AWARE_SOFT_DEPTH="ramp",
                  AWARE_SELECT="instruction", AWARE_CONTENT=a.content,
                  AWARE_START=str(a.write_layer), AWARE_END=str(a.write_layer),
                  UAOR_ALPHA=a.alpha, UAOR_TRIGGER="soft", UAOR_SOFT_DEPTH="ramp",
                  UAOR_SELECT="instruction", UAOR_CONTENT=a.content,
                  UAOR_START=str(a.write_layer), UAOR_END=str(a.write_layer))

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
print(f"[probe] content={a.content} write window = [{st.start}, {st.end}] of {st.L} layers, alpha={ALPHA}", flush=True)
assert st.start == st.end == a.write_layer, (st.start, st.end)

# ---- capture the action-token hidden state and its action-bin distribution at every layer -----------
capH, capP = {}, {}
def make_hook(li):
    def hook(module, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        if h.shape[1] > st.obs_offset + st.n_obs:                 # prefill only
            with torch.no_grad():
                ha = h[0, st._act_slice(h.shape[1])]              # [56, d]
                capH[li] = ha.detach().float().cpu().numpy()
                W = st._action_head_rows()
                logits = torch.nn.functional.linear(st.norm(ha), W).float()
                capP[li] = torch.softmax(logits, dim=-1).cpu().numpy()   # [56, 256]
        return None
    return hook
for li in range(st.L):
    st.layers[li].register_forward_hook(make_hook(li))

def act(observation, desc):
    return np.asarray(get_action(cfg, model, observation, desc, processor=processor, action_head=action_head,
                                 proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
                                 use_film=cfg.use_film))

def grab():
    return (np.stack([capH[l] for l in range(st.L)], 0),      # [L, 56, d]
            np.stack([capP[l] for l in range(st.L)], 0))      # [L, 56, 256]

DELTA, KLD, HON, HOFF = [], [], [], []
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
        capH.clear(); capP.clear(); st.alpha = ALPHA
        a_on = act(observation, desc); H_on, P_on = grab()
        capH.clear(); capP.clear(); st.alpha = 0.0
        act(observation, desc); H_off, P_off = grab()
        st.alpha = ALPHA
        # `other` needs a previous episode in the buffer; task 0 silently falls back to gauss, so drop it
        if not (a.content == "other" and task_id == 0):
            DELTA.append(np.linalg.norm(H_on - H_off, axis=-1) / (np.linalg.norm(H_off, axis=-1) + 1e-9))
            KLD.append((P_on * (np.log(P_on + 1e-12) - np.log(P_off + 1e-12))).sum(-1))
            e = lambda P: -(P * np.log(P + 1e-12)).sum(-1) / np.log(P.shape[-1])
            HON.append(e(P_on)); HOFF.append(e(P_off))
        for x in a_on:
            obs, _, dn, _ = env.step(x.tolist())
            if dn:
                break
        if dn:
            break
    env.close()
    print(f"  task {task_id + 1} done, {len(DELTA)} kept steps", flush=True)

D = np.asarray(DELTA, np.float32); K = np.asarray(KLD, np.float32)
HN = np.asarray(HON, np.float32); HF = np.asarray(HOFF, np.float32)
out = a.out or f"experiments/results/measurements/write_prop_{a.suite}_{a.content}_l{a.write_layer}.npz"
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out, delta=D, kl=K, H_on=HN, H_off=HF, content=a.content,
                    write_layer=a.write_layer, alpha=ALPHA, L=st.L, suite=a.suite)
print(f"\nwrote {out}  delta{D.shape}")
lay = np.arange(D.shape[1])
m = D.mean(axis=(0, 2))
print("relative deviation of the action state, by layer (write at %d):" % a.write_layer)
for l in lay[a.write_layer::4]:
    print(f"  layer {l:2d}: delta={m[l]:.5f}   KL={K[:, l, :].mean():.5f}   dH={(HN - HF)[:, l, :].mean():+.5f}")
print(f"  last layer: delta={m[-1]:.5f}   ratio to just-after-write = {m[-1] / (m[a.write_layer + 1] + 1e-12):.3f}")
