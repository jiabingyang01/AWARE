"""AWARE on pi0 / pi0.5, as forward hooks on the PaliGemma prefix tower of the openpi PyTorch path.

Where the rule attaches, and why
--------------------------------
pi0 predicts actions with a flow-matching expert whose rows live at width 1024, while the
unembedding is tied to the PaliGemma tower at width 2048, so the action distribution the rule reads
cannot be decoded from the expert's rows at all. What can be decoded is the prefix the expert is
conditioned on. The action element here is therefore the last valid language row of that prefix and
N_a = 1, and the observation memory is the image rows of the same prefix. The prefix uses
bidirectional attention, so a write at one row reaches every row at the next layer.

The rule itself is not re-implemented here. Entropy, the depth ramp, the instruction weighting and
the blend all come from ``aware_hooks._State``, the same object OpenVLA-OFT runs, so the two backbones
can only differ in where the rows are, never in what is computed.

Why the hooks sit on submodules
-------------------------------
openpi unrolls the decoder layer by hand (``gemma_pytorch.py``: input_layernorm, attention,
post_attention_layernorm, mlp), so a forward hook on a whole layer never fires; neither does one on
``paligemma_with_expert``, which ``pi0_pytorch.py`` calls through ``.forward()`` directly. These do
fire, and they are enough:

===========================================  ==============================================
``embed_tokens``            pre-forward      the language ids, hence the last non-pad row
``layers[m].input_layernorm`` pre-forward    residual stream entering m = output of m-1
``layers[m].mlp``           forward          where the blend is written
``norm``                    pre-forward      the final hidden state, for the per-forward counters
===========================================  ==============================================

The prefix tower runs once per policy query -- the denoising steps only run the expert -- so the
hooks fire once per query and the whole depth profile is read on that single pass.

Environment: exactly the variables OpenVLA-OFT uses, plus one that only this backbone needs.
With ``AWARE_ALPHA`` unset (``UAOR_ALPHA`` is still accepted) nothing is installed and the policy is
byte-identical to the released one.

``AWARE_PI0_OBS`` decides which image rows the observation memory contains. pi0 takes three camera
slots and LIBERO fills only two: ``right_wrist_0_rgb`` is ``np.zeros_like(base_image)`` and its
``image_mask`` is False, so the policy itself is told to ignore those 256 rows. Reading them anyway
spends a third of the retrieval on the SigLIP embedding of a black frame, and the clamp and the
rescale are computed over it too. ``masked`` (the default) drops any camera whose pixels are all
zero; ``all`` keeps them and is the control.

``AWARE_PI0_WRITE`` decides how wide the blend is written, which is a question OpenVLA-OFT never has
to answer. There the rule doses 56 action rows; here there is one action element, and writing only
its row touches one row in about eight hundred. ``action`` (the default) does that. ``prefix`` keeps
the weight exactly as the rule defines it -- read from the conditioning row's entropy, scaled by the
depth ramp, retrieved through the instruction weighting -- and applies it to every prefix row, which
is what the published operator does when a layer fires. The two differ in reach, not in the rule.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

import torch

log = logging.getLogger("aware_pi0")

_RULE = None
_STATE = None
_INSTALLED = False

_HOOKS_PATHS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hooks.py"),   # the packaged rule
    os.environ.get("AWARE_HOOKS_FILE", ""),                                   # explicit override
)


def _rule():
    """Load the shared rule (aware_hooks.py). Deliberately by path: a private copy would drift and the two backbones
    would stop running the same method without anything failing."""
    global _RULE
    if _RULE is None:
        for p in _HOOKS_PATHS:
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location("aware_hooks", p)
                mod = importlib.util.module_from_spec(spec)
                # Register before exec: torch.compile re-imports a traced function's module BY NAME,
                # and a module loaded only through importlib is not found that way. The pre-rename
                # name is registered too so a mixed import order still yields one module object.
                sys.modules["aware_hooks"] = mod
                sys.modules.setdefault("uaor_hooks", mod)
                spec.loader.exec_module(mod)
                _RULE = mod
                break
        else:
            raise FileNotFoundError(f"aware_hooks.py not found in {_HOOKS_PATHS}")
    return _RULE


def enabled() -> bool:
    return bool(os.environ.get("AWARE_ALPHA") or os.environ.get("UAOR_ALPHA"))


class _TiedHead(torch.nn.Module):
    """Logit lens for a tower whose unembedding is tied to the token embedding (Gemma)."""

    def __init__(self, emb: torch.nn.Module):
        super().__init__()
        self.weight = emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class _NormOut(torch.nn.Module):
    """openpi's GemmaRMSNorm returns ``(hidden, gate)``; the logit lens wants the tensor. The wrapper
    is read-only and is never the module the tower itself calls."""

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        return out[0] if isinstance(out, tuple) else out


class _ModelShim:
    def __init__(self, text_model, norm):
        self.layers = text_model.layers
        self.norm = norm


class _Shim:
    """``_State`` reads ``lm.model.layers``, ``lm.model.norm`` and ``lm.lm_head``."""

    def __init__(self, text_model, head):
        self.model = _ModelShim(text_model, _NormOut(text_model.norm))
        self.lm_head = head


_ALPHA_FILE = os.environ.get("AWARE_ALPHA_FILE", "")
_ALPHA_MTIME = None


def _refresh_alpha(st):
    """Re-read the strength if AWARE_ALPHA_FILE changed since the last call.

    Keyed on mtime_ns so an unchanged file costs a single stat. A malformed or missing file leaves
    the current strength alone rather than silently falling back to some default: a cell that
    cannot read its strength must keep running the one it was given, and the log says what it is.
    """
    global _ALPHA_MTIME
    if not _ALPHA_FILE:
        return
    try:
        m = os.stat(_ALPHA_FILE).st_mtime_ns
    except OSError:
        return
    if m == _ALPHA_MTIME:
        return
    try:
        v = float(open(_ALPHA_FILE).read().strip())
    except (OSError, ValueError):
        return
    _ALPHA_MTIME = m
    if v != st.alpha:
        old, st.alpha = st.alpha, v
        log.warning("[AWARE-pi0] alpha %.4f -> %.4f (from %s)", old, v, _ALPHA_FILE)


def install(pi0_model, *, shard_id: int = 0, log_dir: str | None = None):
    """Attach to a ``PI0Pytorch``. Idempotent; a no-op when AWARE_ALPHA is unset."""
    global _STATE, _INSTALLED
    if not enabled() or _INSTALLED:
        return _STATE
    uh = _rule()

    pg = pi0_model.paligemma_with_expert.paligemma
    text = pg.language_model
    head = getattr(pg, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        head = _TiedHead(text.embed_tokens)

    st = uh._State(_Shim(text, head), n_obs=0, n_act=1, shard_id=shard_id, log_dir=log_dir)
    uh._STATE = st                                  # mark_episode() and the atexit summary find it here
    _STATE = st
    st.obs_offset = 0
    if st.trigger != "soft":
        raise ValueError("AWARE on pi0 is the soft rule; set AWARE_TRIGGER=soft")
    write = os.environ.get("AWARE_PI0_WRITE", "action")
    if write not in ("action", "prefix"):
        raise ValueError(f"AWARE_PI0_WRITE must be action or prefix, got {write!r}")
    obs_mode = os.environ.get("AWARE_PI0_OBS", "masked")
    if obs_mode not in ("masked", "all"):
        raise ValueError(f"AWARE_PI0_OBS must be masked or all, got {obs_mode!r}")
    # last: the final valid language row, N_a = 1 (the historical choice).
    # lang: every valid language row, N_a = n_valid, each row weighted by its own entropy. The
    #       prefix is bidirectional, so no single row is a summary of the rest.
    rows_mode = os.environ.get("AWARE_PI0_ROWS", "last")
    if rows_mode not in ("last", "lang"):
        raise ValueError(f"AWARE_PI0_ROWS must be last or lang, got {rows_mode!r}")

    box = {"n_lang": None, "n_valid": None, "pads": [], "n_keep": None}

    def pre_vision(module, args):
        """One call per camera, in the order the prefix concatenates them. A slot pi0 pads is an
        all-zero frame, so the check is on the pixels rather than on a mask this hook cannot see."""
        x = args[0] if args else None
        if torch.is_tensor(x):
            box["pads"].append(bool(x.abs().max().item() == 0.0))
        return None

    def pre_embed_tokens(module, args):
        """The language ids arrive here before anything else in the prefix pass. Padding is id 0
        (openpi's tokenizer pads with False), so the last valid row is where the entropy is read.

        This is also the once-per-policy-call point where a file-driven strength is picked up."""
        _refresh_alpha(st)
        toks = args[0] if args else None
        if torch.is_tensor(toks) and toks.dim() == 2:
            box["n_lang"] = int(toks.shape[1])
            box["n_valid"] = int((toks[0] != 0).sum().item())
        return None

    def pre_layer(m: int):
        def hook(module, args):
            h = args[0] if args else None
            if not torch.is_tensor(h) or h.dim() != 3:
                return None
            if box["n_lang"] is None or h.shape[1] <= box["n_lang"]:
                return None                          # not the prefix pass (the expert runs alone)
            n_img = h.shape[1] - box["n_lang"]
            # n_img is the offset arithmetic; n_keep is how many of those rows the memory holds,
            # which differs once a padded camera is dropped. Layer 0 decides it, every layer uses it.
            st.n_obs = box["n_keep"] or n_img
            if rows_mode == "lang":
                st.act_index = n_img
                st.n_act = box["n_valid"]
            else:
                st.act_index = n_img + box["n_valid"] - 1
                st.n_act = 1
            st.lang_span = (n_img, n_img + box["n_valid"])
            if m == 0:
                pads = box["pads"]
                box["pads"] = []
                keep = slice(0, n_img)
                if obs_mode == "masked" and pads and any(pads) and n_img % len(pads) == 0:
                    per = n_img // len(pads)
                    idx = [i for k, p in enumerate(pads) if not p
                           for i in range(k * per, (k + 1) * per)]
                    if idx:
                        keep = torch.tensor(idx, device=h.device)
                        box["n_keep"] = len(idx)
                        st.n_obs = len(idx)
                st.O = h[:, keep, :].detach()
                if st.O_first_of_episode is None:
                    st.O_first_of_episode = st.O.clone()
                if st.select_layer == 0:
                    st.g = st.selection_weight(h, 0)
                st.fired_this_forward = False
                st.cur_u = [None] * st.L
                st.cur_gate = [None] * st.L
                for lyr in st.layers:
                    lyr.mlp._aware_gate = None
                return None
            _, gate = st.compute_gate(m - 1, h)       # input of layer m == output of layer m-1
            st.layers[m].mlp._aware_gate = gate
            st.cur_gate[m] = gate
            return None
        return hook

    def pre_norm(module, args):
        """The final hidden state: the last layer's entropy and the per-forward counters."""
        h = args[0] if args else None
        if not torch.is_tensor(h) or h.dim() != 3 or box["n_lang"] is None:
            return None
        if h.shape[1] <= box["n_lang"]:
            return None
        st.compute_gate(st.L - 1, h)
        st.n_forwards += 1
        st.forward_idx += 1
        for j, g in enumerate(st.cur_gate):
            if torch.is_tensor(g):          # n_act = 1 here, so the gate is [1, 1]: never numel > 1
                st.n_inject += 1
                st.per_layer_inject[j] += 1
        return None

    def mlp_hook(m: int):
        def hook(module, args, output):
            gate = getattr(module, "_aware_gate", None)
            if gate is None:
                return None
            if write == "action":
                return st.anchor_soft(args[0], output, gate, m)
            # Same weight, wider reach: the rows the slice covers become the whole prefix and the
            # single weight is repeated across them. act_index is restored immediately because the
            # entropy of the next layer is read from the conditioning row alone.
            T = output.shape[1]
            keep_i, keep_n = st.act_index, st.n_act
            st.act_index, st.n_act = 0, T
            try:
                return st.anchor_soft(args[0], output, gate.expand(-1, T), m)
            finally:
                st.act_index, st.n_act = keep_i, keep_n
        return hook

    tower = getattr(getattr(pg, "model", pg), "vision_tower", None)
    if tower is None:
        tower = getattr(pg, "vision_tower", None)
    if tower is not None:
        tower.register_forward_pre_hook(pre_vision)
    elif obs_mode == "masked":
        log.warning("[AWARE-pi0] no vision tower found: cannot tell a padded camera from a real "
                    "one, so the memory keeps every image row (AWARE_PI0_OBS=masked is inert here)")
    text.embed_tokens.register_forward_pre_hook(pre_embed_tokens)
    for m, lyr in enumerate(st.layers):
        lyr.input_layernorm.register_forward_pre_hook(pre_layer(m))
        lyr.mlp.register_forward_hook(mlp_hook(m))
    text.norm.register_forward_pre_hook(pre_norm)

    _INSTALLED = True
    log.warning("[AWARE-pi0] installed alpha=%.4f depth=%s select=%s write=%s obs=%s rows=%s L=%d",
                st.alpha, st.soft_depth, st.select, write, obs_mode, rows_mode, st.L)
    return st


def mark_episode(task_id: int, episode_idx: int):
    if _STATE is not None:
        _rule().mark_episode(task_id, episode_idx)


def summary() -> dict:
    return _STATE.summary() if _STATE is not None else {}


def maybe_install(model, *, shard_id: int = 0, log_dir: str | None = None):
    """The name pi0_pytorch.py uses for every env-guarded attachment in this repo."""
    return install(model, shard_id=shard_id, log_dir=log_dir or os.environ.get("AWARE_LOG") or os.environ.get("UAOR_LOG"))
