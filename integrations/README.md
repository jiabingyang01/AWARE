# Wiring AWARE into a backbone

**English** · [简体中文](README_zh.md)

AWARE is a set of forward hooks. It does not fork the policy and it does not touch weights. To put it
on a new backbone you only answer three questions:

1. **Where are the observation rows?** The memory the rule retrieves from. Usually the image patch
   tokens of the prefix, optionally plus a proprioceptive token.
2. **Where are the action rows?** The elements whose uncertainty is read and whose FFN output is
   blended. For a parallel-decoded chunk this is a block of `n_act` rows; for an autoregressive
   backbone it is a single row, the one producing the next action token (`AWARE_ACT_POS=last`).
3. **Which module holds the decoder layers?** The rule registers one forward hook per layer.

Everything else (entropy, depth ramp, instruction weighting, the blend) is shared and lives in
`aware/hooks.py`. No adapter re-implements the operator.

```python
from aware import install_generic
state = install_generic(language_model, n_obs=<#observation rows>, n_act=<#action rows>,
                        obs_offset=<index of the first observation row>)
```

Nothing happens unless `AWARE_ALPHA` is set, so the call is safe to leave in permanently.

---

## Switches

`AWARE_<NAME>`, with the pre-rename `UAOR_<NAME>` accepted as a fallback so that runs recorded before
the rename still reproduce. The new name wins when both are set.

| switch | default | meaning |
|---|---|---|
| `AWARE_ALPHA` | *(unset = off)* | blend strength; the only required switch |
| `AWARE_TRIGGER` | `entropy` | `soft` is the released rule (continuous weight). `entropy` is the CoRL-era hard threshold, `all`, `random`, `none` are controls |
| `AWARE_SOFT_DEPTH` | *(flat)* | `ramp` = weight scales with `l/(L-1)`, the released rule |
| `AWARE_SELECT` | `none` | `instruction` = tilt the retrieval by the instruction's own read of the observation |
| `AWARE_CONTENT` | `obs` | what is written. `other` = a different episode's observation, `gauss` = random directions with matched norms, `none` = damping only. Controls, not settings |
| `AWARE_SOFT_SIGNAL` | `entropy` | what the per-element weight is read from. `step`, `margin`, `inverted` are dose-matched controls |
| `AWARE_START` / `AWARE_END` | `1` / `L-1` | layer window; set both to the same value to write at one layer only |
| `AWARE_ACT_POS` | `chunk` | `last` for autoregressive backbones |

> **The released rule is `AWARE_TRIGGER=soft AWARE_SOFT_DEPTH=ramp AWARE_SELECT=instruction`.**
> Setting only `AWARE_ALPHA` falls back to the hard-threshold variant, which triggers on about one
> forward in a thousand: the log still prints `ACTIVE` and `alpha=0` is still bit-identical, but the
> output is indistinguishable from the untouched policy. Check `n_inject` and `soft_mean_weight` in
> the summary, never the `installed` line alone.

## Strengths used in the paper

| backbone | benchmark | alpha |
|---|---|---|
| OpenVLA-OFT | LIBERO Spatial / Object / Goal / Long | 0.04 / 0.05 / 0.10 / 0.03 |
| pi0 | LIBERO | 0.03 / 0.04 / 0.05 / 0.03 |
| pi0.5 | LIBERO | 0.01 / 0.08 / 0.01 / 0.07 |
| OpenVLA-7B | LIBERO | 0.03 / 0.02 / 0.01 / 0.01 |
| CogACT | SIMPLER Google Robot | 0.09 |
| SpatialVLA-4B | SIMPLER WidowX | 0.08 |
| LLaVA-VLA 0.5B | CALVIN ABC->D | 0.01 |

---

## Per-backbone notes

### OpenVLA-OFT

Action chunk is decoded in parallel, so `n_act = chunk x action_dim` (8 x 7 = 56) and the rows sit
just before the final position. `install` reads the layout off the model:

```python
from aware import install
state = install(vla, use_proprio=cfg.use_proprio, shard_id=cfg.shard_id, log_dir=...)
```

Call it once after the policy is built, and call `aware.mark_episode(task_id, episode_idx)` at the top
of every episode so that `AWARE_CONTENT=other` has a previous episode to read.


### openpi (pi0 / pi0.5), PyTorch path only

The flow-matching action expert runs at width 1024 while the unembedding is tied to the PaliGemma
tower at 2048, so the action distribution cannot be decoded from the expert's rows. AWARE therefore
attaches to the **prefix**: the action element is the last valid language row (`n_act = 1`) and the
memory is the image rows of the same prefix, which is bidirectional, so one write reaches every row at
the next layer. One line in `models_pytorch/pi0_pytorch.py`, at the end of `__init__`:

```python
from aware import pi0 as _aware_pi0
self._aware = _aware_pi0.maybe_install(self)
```

The JAX path is not supported: it hard-codes the sampling step count and never exposes the tower.

### CogACT

The action element is the cognition token, which is the EOS appended to the prompt and therefore the
last row; the memory is the patch block Prismatic inserts after `<BOS>`, so `obs_offset=1`.

```python
os.environ.setdefault("AWARE_ACT_POS", "last")
state = install_generic(vla.vlm.llm_backbone.llm,
                        n_obs=int(vla.vlm.vision_backbone.num_patches), n_act=1, obs_offset=1)
```

### SpatialVLA-4B

PaliGemma2-style: the processor puts the image tokens first, then `<bos>` + instruction + `\n`, and
the action is decoded autoregressively as spatial tokens. So `obs_offset=0` and the memory is
`image_seq_length x num_obs_steps` rows.

```python
os.environ.setdefault("AWARE_ACT_POS", "last")
n_obs = int(processor.image_seq_length) * int(processor.num_obs_steps)
state = install_generic(vla.language_model, n_obs=n_obs, n_act=1, obs_offset=0)
```

### LLaVA-VLA 0.5B

LLaVA-OneVision expands one `<image>` token into a block of 729 rows; the adapter finds that block and
asserts there is exactly one image token, then installs with `obs_offset` set to the block's start.
Actions are emitted as text tokens, so `AWARE_ACT_POS=last`. The wiring is in the evaluation server
(`llava/serve/flask_server_qwen.py`), which is where the model is held.

---

## A gate every new backbone must pass

Three checks, in this order. A backbone that has not passed them is not wired correctly, whatever the
success rate says.

1. **Identity.** `AWARE_ALPHA=0` must reproduce the untouched policy value for value.
2. **Layout proof.** Assert that the rows you declared really are the observation block and the action
   row, e.g. by checking the image-token count against the processor and by printing the row index the
   action is read from. Both existing autoregressive adapters do this and refuse to run if it fails.
3. **Effect.** With the released switch set, `n_inject` must be a large fraction of forwards and the
   output must actually differ from the untouched policy. `installed` and `ACTIVE` in the log prove
   nothing on their own.

`analysis/sv_aware_gate.py` is a worked example of all three on SpatialVLA.
