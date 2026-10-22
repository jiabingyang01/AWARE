"""CPU equivalence test: aware_hooks (the released rule) vs a direct re-implementation of uaor.py semantics
on a tiny random Llama. Checks (a) identical trigger decisions per layer, (b) identical final hidden states.
Run with the openvla-oft interpreter (fork transformers 4.40)."""
# --- AWARE path bootstrap: every host path comes from aware/paths.py, overridable by env ---------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# ------------------------------------------------------------------------------------------------
import os, sys, types
import numpy as np, torch
sys.path.insert(0, _paths.OPENVLA_OFT)
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                  num_key_value_heads=4, vocab_size=300, max_position_embeddings=256)
lm = LlamaForCausalLM(cfg).eval()
T, N_OBS, N_ACT = 40, 10, 8
x = torch.randn(1, T, 64)
ALPHA, GAMMA = 0.05, 0.60   # gamma chosen so that some layers fire and some do not
os.environ.update({"UAOR_ALPHA": str(ALPHA), "UAOR_GAMMA": str(GAMMA), "UAOR_TOPK": "256"})

# ---------------- reference: uaor.py semantics, explicit loop ----------------
def entropy_ref(h):
    ha = h[:, -(N_ACT + 1):-1, :]
    logits = lm.lm_head(lm.model.norm(ha)).float()
    k = min(256, logits.shape[-1]); top = torch.topk(logits, k, dim=-1).values
    p = torch.softmax(top, -1); return float((-(p * p.clamp_min(1e-12).log()).sum(-1) / np.log(k)).mean())

with torch.no_grad():
    O = x[:, 1:1 + N_OBS, :]
    pos = torch.arange(T)[None]
    h = x.clone(); ref_trig = []
    gate_next = False
    for l, layer in enumerate(lm.model.layers):
        if gate_next:
            orig = layer.mlp.forward
            def blended(inp, _orig=orig):
                ffn = _orig(inp)
                A = (inp @ O.transpose(-1, -2)) @ O
                s = ffn.abs().mean() / A.abs().mean()
                return (1 - ALPHA) * ffn + ALPHA * s * A
            layer.mlp.forward = blended
        h = layer(h, position_ids=pos)[0]
        if gate_next:
            layer.mlp.forward = orig
        gate_next = False
        u = entropy_ref(h)
        if l <= len(lm.model.layers) - 2 and u > GAMMA:
            gate_next = True; ref_trig.append(l + 1)
    ref_out = lm.model.norm(h)

# ---------------- hooks ----------------
import importlib
uh = importlib.import_module("experiments.robot.aware_hooks")
st = uh.install_generic(lm, n_obs=N_OBS, n_act=N_ACT, shard_id=0, log_dir=None, obs_offset=1)
uh.mark_episode(0, 0)
with torch.no_grad():
    out = lm.model(inputs_embeds=x, position_ids=pos, output_hidden_states=True)
hook_out = out.last_hidden_state
hook_trig = [j for j, g in enumerate(st.cur_gate) if g is not None and (bool(g.item()) if torch.is_tensor(g) else bool(g))]
print("ref triggers :", ref_trig)
print("hook triggers:", hook_trig)
d = (hook_out - ref_out).abs().max().item() / ref_out.abs().max().item()
print("max rel diff final hidden:", f"{d:.3e}")
print("n_inject counted:", st.n_inject, " sync_free:", st.sync_free)
assert hook_trig == ref_trig, "trigger mismatch"
assert d < 1e-5, "output mismatch"
# gate must be tensor in sync-free mode and torch.where must have selected the anchored branch
print("PASS")
