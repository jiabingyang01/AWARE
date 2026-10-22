"""What the instruction weighting actually points at, drawn on the observation.

Eq. (g) of the paper weights each observation token by how strongly the language rows of the same forward
retrieve it. This probe computes that weight exactly as the rule does --- at the input to layer 0, from all
language rows, with the same normalisation --- and, next to it, the weight the retrieval would use without
it: the share the action rows themselves assign to each observation token. The two maps are saved with the
third-person view they belong to, so the figure can put them side by side.

The distinction matters for the figure to be honest: the weight in the paper is a product of hidden states,
not a reading of the backbone's attention maps, and an earlier panel in this project drew the attention
anchor instead, which is a different quantity.

  CUDA_VISIBLE_DEVICES=k python probe_instruction_selection.py --suite libero_10 --tasks 0,2,6
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
sys.path.insert(0, _paths.OPENVLA_OFT); sys.path.insert(0, _paths.LIBERO)
import numpy as np, torch, glob

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--suite", default="libero_10")
ap.add_argument("--tasks", default="0,2,6")
ap.add_argument("--settle", type=int, default=12, help="control steps to run before the frame is taken")
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
L = len(lm.model.layers)
N_OBS = 513 if cfg.use_proprio else 512
N_ACT, OBS0, PATCH = 56, 1, 256
grab = {}
SPAN = {"start": None, "length": None}    # rows of the prompt that name the task


def instruction_span(task_description):
    """The task words inside the fixed prompt, the same slice the evaluation uses for the anchor.

    The template rows and the newline act as attention sinks and take a large share of any weight built
    from all language rows, so a weight meant to say what the instruction is about has to be read from
    these rows alone.
    """
    prefix = "In: What action should the robot take to"
    tok = processor.tokenizer
    start = len(tok(prefix).input_ids) - 1
    length = len(tok(f"{prefix} {task_description.lower()}").input_ids) - len(tok(prefix).input_ids)
    return start, length


def pre_hook(module, args, kwargs):
    """Layer 0 sees the sequence before any block has run, which is where the rule computes g."""
    h = args[0] if args else kwargs["hidden_states"]
    if h.shape[1] <= OBS0 + N_OBS:
        return None
    with torch.no_grad():
        x = h[0].float()
        O = x[OBS0:OBS0 + N_OBS]
        lang = x[OBS0 + N_OBS:-(N_ACT + 1)]
        act = x[-(N_ACT + 1):-1]
        r = torch.relu(lang @ O.transpose(0, 1)).sum(0)
        share = N_OBS * r / r.sum().clamp_min(1e-9)
        if SPAN["start"] is not None:
            s0, sl = SPAN["start"], SPAN["length"]
            sub = lang[s0:s0 + sl] if s0 + sl <= lang.shape[0] else lang
            rs = torch.relu(sub @ O.transpose(0, 1)).sum(0)
            grab["span"] = (N_OBS * rs / rs.sum().clamp_min(1e-9))[:PATCH].cpu().numpy()
        # Both are kept. The rule multiplies by the clamped weight, but the clamp pins every token at or
        # above the uniform share to one, which removes the contrast among exactly the tokens the
        # instruction retrieves most; the share before the clamp is what says where the instruction points.
        grab["g"] = share.clamp(max=1.0)[:PATCH].cpu().numpy()
        grab["share"] = share[:PATCH].cpu().numpy()
        grab["O0"] = O                     # the keys the rule retrieves against, fixed at this layer
        grab["gfull"] = share.clamp(max=1.0)
    return None


MID = 16
G_LAYERS = (0, 8, 16, 24)


def make_g_hook(li):
    """g as the rule computes it, but at a layer of our choosing.

    The rule computes it at the input to layer 0. There the language rows are still uncontextualised token
    embeddings, so their product with the projected vision features is not a semantic match; whether the
    weight looks different once the rows have been through the stack is worth knowing before the paper
    claims the instruction is selecting anything.
    """
    def hook(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] <= OBS0 + N_OBS:
            return None
        with torch.no_grad():
            x = h[0].float()
            O = x[OBS0:OBS0 + N_OBS]
            lang = x[OBS0 + N_OBS:-(N_ACT + 1)]
            r = torch.relu(lang @ O.transpose(0, 1)).sum(0)
            sh = N_OBS * r / r.sum().clamp_min(1e-9)
            grab[f"g{li}"] = sh.clamp(max=1.0)[:PATCH].cpu().numpy()
            grab[f"share{li}"] = sh[:PATCH].cpu().numpy()
            if SPAN["start"] is not None:
                s0, sl = SPAN["start"], SPAN["length"]
                sub = lang[s0:s0 + sl] if s0 + sl <= lang.shape[0] else lang
                rs = torch.relu(sub @ O.transpose(0, 1)).sum(0)
                grab[f"span{li}"] = (N_OBS * rs / rs.sum().clamp_min(1e-9))[:PATCH].cpu().numpy()
        return None
    return hook


def mid_hook(module, args, kwargs):
    """The action rows' own read of the observation, taken where the dose is substantial.

    At the input to layer 0 the action placeholders are near-identical embeddings, so their read is
    degenerate and says nothing about what the retrieval would pick; the comparison has to be made at a
    layer where the rows have diverged.
    """
    h = args[0] if args else kwargs["hidden_states"]
    if h.shape[1] <= OBS0 + N_OBS:
        return None
    with torch.no_grad():
        x = h[0].float()
        act = x[-(N_ACT + 1):-1]
        # The keys are the layer-0 observation, as in the rule; the query is the action row at this layer.
        # What the retrieval actually weights is this read multiplied by g, so both are kept and the
        # figure can show the difference the instruction makes rather than g on its own.
        O0 = grab["O0"]
        s = torch.relu(act @ O0.transpose(0, 1)).sum(0)
        gfull = grab.get("gfull")
        grab["read_nog"] = (s / s.sum().clamp_min(1e-9))[:PATCH].cpu().numpy()
        sg = s * gfull
        grab["read_g"] = (sg / sg.sum().clamp_min(1e-9))[:PATCH].cpu().numpy()
        grab["raw"] = grab["read_nog"]
    return None


lm.model.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
lm.model.layers[MID].register_forward_pre_hook(mid_hook, with_kwargs=True)
for _li in G_LAYERS:
    lm.model.layers[_li].register_forward_pre_hook(make_g_hook(_li), with_kwargs=True)

suite = benchmark.get_benchmark_dict()[a.suite]()
out = {"suite": a.suite, "tasks": []}
for task_id in [int(x) for x in a.tasks.split(",")]:
    task = suite.get_task(task_id)
    env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    SPAN["start"], SPAN["length"] = instruction_span(desc)
    env.reset(); obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
    for _ in range(10):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    for _ in range(max(1, a.settle // 8)):
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
    img = get_libero_image(obs)
    rec = {"task": task_id, "instruction": desc, "image": np.asarray(img),
           "g": grab["g"].copy(), "raw": grab["raw"].copy(), "share": grab["share"].copy(),
           "span": grab["span"].copy(),
           "read_nog": grab["read_nog"].copy(), "read_g": grab["read_g"].copy()}
    for _li in G_LAYERS:
        rec[f"g{_li}"] = grab[f"g{_li}"].copy()
        rec[f"share{_li}"] = grab[f"share{_li}"].copy()
        rec[f"span{_li}"] = grab[f"span{_li}"].copy()
    out["tasks"].append(rec)
    env.close()
    print(f"  task {task_id}: {desc}", flush=True)

if a.out:
    payload = {}
    for i, d in enumerate(out["tasks"]):
        payload[f"img_{i}"] = d["image"]; payload[f"g_{i}"] = d["g"]; payload[f"raw_{i}"] = d["raw"]
        payload[f"share_{i}"] = d["share"]; payload[f"span_{i}"] = d["span"]
        payload[f"read_nog_{i}"] = d["read_nog"]; payload[f"read_g_{i}"] = d["read_g"]
        for _li in G_LAYERS:
            payload[f"g{_li}_{i}"] = d[f"g{_li}"]
            payload[f"share{_li}_{i}"] = d[f"share{_li}"]
            payload[f"span{_li}_{i}"] = d[f"span{_li}"]
    payload["instructions"] = np.array([d["instruction"] for d in out["tasks"]])
    payload["task_ids"] = np.array([d["task"] for d in out["tasks"]])
    np.savez_compressed(a.out, **payload)
    print("wrote", a.out, "with", len(out["tasks"]), "tasks")
