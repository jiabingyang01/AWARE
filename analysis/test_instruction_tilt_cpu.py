"""Liveness of UAOR_SELECT on the same tiny model the equivalence test uses.

Confirms the weight neither collapses to one, which would rename the published operator, nor to
zero, which would silently disable the injection, and that the three selection settings give three
different hidden states.
"""
# --- AWARE path bootstrap: every host path comes from aware/paths.py, overridable by env ---------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from aware import paths as _paths
# ------------------------------------------------------------------------------------------------
import os, sys
import torch
sys.path.insert(0, _paths.OPENVLA_OFT)
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                  num_key_value_heads=4, vocab_size=300, max_position_embeddings=256)
lm = LlamaForCausalLM(cfg).eval()
ids = torch.randint(0, 300, (1, 40))

def run(select, trigger="all"):
    for k in [k for k in os.environ if k.startswith("UAOR_")]:
        os.environ.pop(k)
    os.environ.update(UAOR_ALPHA="0.05", UAOR_GAMMA="0.6", UAOR_TRIGGER=trigger, UAOR_SELECT=select)
    for name in [m for m in sys.modules if "aware_hooks" in m or "uaor_hooks" in m]:
        del sys.modules[name]
    import importlib
    uh = importlib.import_module("experiments.robot.aware_hooks")
    uh._install_on(lm, n_obs=10, n_act=8, shard_id=0, log_dir=None)
    with torch.no_grad():
        out = lm.model(ids).last_hidden_state.clone()
    summary = uh._STATE.summary()
    for layer in lm.model.layers:
        for attr in ("_forward_hooks", "_forward_pre_hooks"):
            getattr(layer, attr).clear()
            getattr(layer.mlp, attr).clear()
    return out, summary

base, s0 = run("none")
sel, s1 = run("instruction")
inv, s2 = run("instruction_inverted")
shu, s3 = run("instruction_shuffled")
lay, s4 = run("instruction_layerwise")
print("mean weight  instruction/inverted/shuffled:",
      round(s1["mean_selection_weight"], 4), round(s2["mean_selection_weight"], 4),
      round(s3["mean_selection_weight"], 4))
print("differs from published :", float((sel - base).abs().max()))
print("differs from inverted  :", float((sel - inv).abs().max()))
print("differs from shuffled  :", float((sel - shu).abs().max()))
print("layerwise mean weight  :", round(s4["mean_selection_weight"], 4))
print("layerwise vs published :", float((lay - base).abs().max()))
assert 0.0 < s1["mean_selection_weight"] < 1.0, "weight collapsed"
assert (sel - base).abs().max() > 0 and (sel - inv).abs().max() > 0 and (sel - shu).abs().max() > 0
assert (lay - base).abs().max() > 0 and 0.0 < s4["mean_selection_weight"] < 1.0
print("PASS")
