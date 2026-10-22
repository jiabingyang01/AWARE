"""Uncertainty-gated observation-subspace anchoring (UAOR / OSA) as inference-time hooks for OpenVLA-OFT.

Env-gated: when ``UAOR_ALPHA`` is unset nothing is installed and the policy is byte-identical to the
official one. When set, this reproduces the operator of the original ``uaor.py`` (the released UAOR
implementation) line by line, but as forward hooks so the fork's Llama code stays untouched:

  * after every decoder layer l (0..L-2) the residual stream at the action positions is early-decoded
    through ``model.norm`` + ``lm_head``; top-K softmax entropy normalised by ln K, averaged over the
    action tokens, is the layer's action entropy u_l;
  * if u_l > gamma_l and l is inside the layer window, the MLP of layer l+1 returns
        (1 - alpha) * FFN(x) + alpha * s * (x O^T O),   s = mean|FFN(x)| / mean|x O^T O|
    where O are the observation tokens (visual patches + proprio) read from the layer-0 input.
    (``uaor.py`` builds W1 = c1*O, W2 = c2*O^T and computes (x W1^T) W2^T with no activation in between,
    then rescales by mean|ffn|/mean|adapter|; the constants c1 c2 cancel inside s, so the two forms are
    numerically the same operator -- verified to 1e-7 relative error.)

Implementation note: the gate u_l > gamma is evaluated on the GPU and applied with ``torch.where`` so the
forward pass has a single host sync per step (for logging) instead of one per layer.

Switches (all optional):
  UAOR_ALPHA      blend ratio alpha (required to enable)
  UAOR_GAMMA      float threshold, or ``auto:<rate>`` = per-layer running quantile so that the long-run
                  trigger rate is <rate> (warm-up uses UAOR_GAMMA0, default 0.80; this mode syncs per layer)
  UAOR_START/END  1-indexed layer window as in uaor.py (defaults 1 and L-1 = full window)
  UAOR_TRIGGER    entropy | random | all | none     (random uses UAOR_RATE, default 0.2)
  UAOR_CONTENT    obs | other | gauss | none        (other = observation tokens of the previous episode,
                                                      gauss = random matrix with matched row norms,
                                                      none = pure (1-alpha) scaling, no anchoring)
  UAOR_ONCE=1     trigger at most once per forward (MemVR's look-twice; syncs per layer)
  UAOR_VISUAL_ONLY=1  drop the proprio token from O (MemVR-as-is)
  UAOR_ACT_POS    chunk (default: the 56 action placeholders) | last (the final token, MemVR's position)
  UAOR_TOPK       default 256
  UAOR_LOG        directory for per-forward jsonl logs (entropies, triggered layers)
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import time

import numpy as np
import torch

log = logging.getLogger("aware")

_STATE = None


def _env(name: str, default=None):
    """Read ``AWARE_<name>``, falling back to the pre-rename ``UAOR_<name>``.

    The method is AWARE. ``UAOR_`` is the name the switches had at CoRL submission; cells already on
    disk recorded it and the runners that produced them still pass it, so both are accepted and the
    new name wins when both are set.
    """
    v = os.environ.get("AWARE_" + name)
    if v is None:
        v = os.environ.get("UAOR_" + name)
    return default if v is None else v



def enabled() -> bool:
    return bool(_env("ALPHA"))


class _State:
    def __init__(self, lm, n_obs: int, n_act: int, shard_id: int, log_dir: str | None):
        self.layers = list(lm.model.layers)
        self.lm_model = lm.model
        self.norm = lm.model.norm
        self.head = lm.lm_head
        self.L = len(self.layers)
        self.n_obs = n_obs
        self.n_act = n_act
        self.alpha = float(_env("ALPHA", ""))
        g = _env("GAMMA", "0.80")
        if g.startswith("auto"):
            self.auto = True
            self.rate = float(g.split(":", 1)[1]) if ":" in g else 0.2
            self.gamma0 = float(_env("GAMMA0", "0.80"))
            self.gamma = None
        else:
            self.auto = False
            self.rate = None
            self.gamma = float(g)
        self.start = int(_env("START", "1"))
        self.end = int(_env("END", str(self.L - 1)))
        self.trigger = _env("TRIGGER", "entropy")
        self.rand_rate = float(_env("RATE", "0.2"))
        self.content = _env("CONTENT", "obs")
        self.once = _env("ONCE", "0") == "1"
        self.visual_only = _env("VISUAL_ONLY", "0") == "1"
        self.act_pos = _env("ACT_POS", "chunk")
        # Set per forward by a backbone adapter when the action element is not at a fixed offset
        # from the end of the sequence (pi0: last valid language row of a right-padded prefix).
        # None keeps the UAOR_ACT_POS rule, so OpenVLA-OFT is unchanged.
        self.act_index = None
        self.lang_span = None
        self._o32 = None            # float32 view of the observation memory, cached per forward
        self._gsl = None            # the instruction weight over the same rows, cached with it
        self._o32_key = None
        self.topk = int(_env("TOPK", "256"))
        self.warmup = int(_env("WARMUP", "200"))
        # What the retrieval is allowed to read. The original operator scores every observation token
        # by its similarity to the action hidden state, which already carries the decision the model
        # has committed to. "instruction" additionally weighs each observation token by how strongly
        # the language tokens themselves retrieve it, so the instruction says what to look for while
        # the hidden state says where the model currently is. The two controls keep the same weight
        # distribution and either move it elsewhere or apply it to the complement.
        # UAOR_TRIGGER=soft: no threshold. Each action element is blended with the retrieved
        # observation in proportion to its own normalised entropy (times alpha); every other position
        # is left untouched. UAOR_SOFT_POWER sharpens the weight (H**p), default 1.
        self.soft_power = float(_env("SOFT_POWER", "1.0"))
        self.soft_shuffle = _env("SOFT_SHUFFLE", "0") == "1"   # control: permute weights across elements
        # UAOR_SOFT_SIGNAL (control): what the per-element weight is read from.
        #   entropy : the element's own Action Entropy (the rule)
        #   step    : the index of the future step the element belongs to, 1..H. The later steps of a chunk
        #             carry more entropy, so this asks whether the entropy is only a proxy for position.
        #   margin  : one minus the element's top-1 probability, a confidence that is not an entropy.
        #   inverted: the entropy turned upside down, so the settled elements get the dose.
        # Alternatives are rescaled to the mean of the entropy they replace, so the dose is unchanged.
        self.soft_signal = _env("SOFT_SIGNAL", "entropy")
        # UAOR_SOFT_CENTER=1: weight = [H_n - mean_n H]_+ , i.e. only the elements that are more undecided
        # than the rest of their own action chunk; UAOR_SOFT_TOPK=k: keep the k most uncertain elements at
        # their entropy weight and zero the rest.
        self.soft_center = _env("SOFT_CENTER", "0") == "1"
        # UAOR_SOFT_UNIFORM=1 (control): every element gets the chunk-mean entropy, so the total dose is
        # the same as the per-element rule but nothing is assigned to particular elements.
        self.soft_uniform = _env("SOFT_UNIFORM", "0") == "1"
        # UAOR_SOFT_MIX=lambda: (1-lambda) * chunk mean + lambda * the element's own entropy. The chunk
        # mean is preserved for every lambda, so this moves how the dose is distributed and not how much
        # of it there is. lambda 0 reproduces UAOR_SOFT_UNIFORM, lambda 1 is the rule.
        self.soft_mix = float(_env("SOFT_MIX", "1.0"))
        self.soft_topk = int(_env("SOFT_TOPK", "0") or 0)
        # ---- mechanism variants screened for the ICRA version (every default is the paper's rule) ----
        # UAOR_SOFT_NORM: how the raw entropy becomes a weight.
        #   none   : alpha*H                                   (paper)
        #   excess : alpha*[H - mu]_+ / (1 - mu)   mu = running mean of H at this (layer, element): only
        #            elements more undecided than they usually are at this depth are dosed
        #   ratio  : alpha*H/mu                    above-typical elements get more, below-typical less
        #   zscore : alpha*clip((H - mu)/sigma, 0, 3)
        self.soft_norm = _env("SOFT_NORM", "none")
        self.soft_ema = float(_env("SOFT_EMA", "0.01"))
        self.mu = None; self.var = None; self.m2 = None; self.mu_count = None
        # UAOR_RETRIEVAL: what an action element reads from the observation memory.
        #   linear  : (x O^T) O                keys = values = layer-0 observation embeddings (paper)
        #   softmax : softmax(cos(x,o)/tau) O  an attention read over the observation, UAOR_TAU (0.1)
        #   layer   : (x O^T) X_obs            values = the observation rows as they stand at this layer
        #   ffn     : (x O^T) FFN(X_obs)       values = what this layer's FFN wrote for the observation rows
        self.retrieval = _env("RETRIEVAL", "linear")
        self.tau = float(_env("TAU", "0.1"))
        # UAOR_SOFT_STEP_MASK: 8 chars over the chunk's future steps (element n belongs to step n // 7 in
        # OpenVLA-OFT's step-major layout). UAOR_SOFT_DEPTH: flat | ramp (deeper layers get more, l/(L-1))
        # | iramp (shallower layers get more).
        self.step_mask = _env("SOFT_STEP_MASK", "")
        self.soft_depth = _env("SOFT_DEPTH", "flat")
        self._elem_mask = None
        self._soft_mass_t = None
        # UAOR_OBS_SUBSET: which observation tokens the retrieval may read. all (paper) | third (first camera's
        # patches) | wrist (second camera's patches) | noprop (both cameras, no proprioceptive token).
        self.obs_subset = _env("OBS_SUBSET", "all")
        # UAOR_OBS_BALANCE=1: retrieve from each camera (and the proprio token) separately and add the parts
        # at equal mean magnitude, so neither view dominates the read through its token norms.
        self.obs_balance = _env("OBS_BALANCE", "0") == "1"
        # ---- cost knobs. Both change the numbers, so both default to off and are validated on three
        # seeds before they are used for anything that goes into a table. ----
        # UAOR_HEAD_SLICE=action: read the language head only over the action vocabulary (the last 256
        # ids) instead of the whole 32k vocabulary. The entropy is then exactly over K=256 candidates
        # and no top-k is needed, which removes the dominant cost of the rule.
        self.head_slice = _env("HEAD_SLICE", "full")
        if self.head_slice not in ("full", "action"):
            raise ValueError(f"Unknown UAOR_HEAD_SLICE: {self.head_slice}")
        self.act_vocab = int(_env("ACTION_VOCAB", "256"))
        # OpenVLA pads the vocabulary to a multiple of 64 and de-tokenises against the
        # unpadded size, so the action ids stop this many rows short of the head's end.
        self.act_vocab_pad = int(_env("ACTION_VOCAB_PAD", "64"))
        self._logged_slice = False
        # UAOR_MIN_LAYER=k: leave layers below k untouched and do not even read the head there. With the
        # depth factor the weight at layer 1 is alpha/(L-1) times an entropy, i.e. a fraction of a
        # percent, so the early layers cost their share of the overhead for almost no effect.
        self.min_layer = int(_env("MIN_LAYER", "0"))
        # UAOR_SOFT_STEPGATE=1: scale the whole chunk's dose at layer l by 2*sigmoid(k*z), z = z-score of the
        # chunk-mean entropy u_l against its running statistics at the same depth (k = UAOR_SOFT_STEPGATE_K).
        # A step at which the model is more undecided than usual at this depth gets more, a confident step
        # less; the per-element assignment inside the chunk is unchanged and the mean dose stays about alpha.
        self.step_gate = _env("SOFT_STEPGATE", "0") == "1"
        self.step_gate_k = float(_env("SOFT_STEPGATE_K", "2.0"))
        self.u_mu = None; self.u_var = None; self.u_m2 = None; self.u_count = None
        # UAOR_WRITE: ffn (paper: blend the next layer's FFN output) | attn (blend the next layer's attention
        # output instead, i.e. as if the element had attended to the observation) | both.
        self.write = _env("WRITE", "ffn")
        if self.write not in ("ffn", "attn", "both"):
            raise ValueError(f"Unknown UAOR_WRITE: {self.write}")
        if self.obs_subset not in ("all", "third", "wrist", "noprop"):
            raise ValueError(f"Unknown UAOR_OBS_SUBSET: {self.obs_subset}")
        if self.soft_norm not in ("none", "excess", "ratio", "zscore"):
            raise ValueError(f"Unknown UAOR_SOFT_NORM: {self.soft_norm}")
        if self.retrieval not in ("linear", "softmax", "layer", "ffn"):
            raise ValueError(f"Unknown UAOR_RETRIEVAL: {self.retrieval}")
        if self.soft_depth not in ("flat", "ramp", "iramp"):
            raise ValueError(f"Unknown UAOR_SOFT_DEPTH: {self.soft_depth}")
        self.soft_mass = 0.0
        self.soft_count = 0
        self.select = _env("SELECT", "none")
        if self.select not in ("none", "instruction", "instruction_inverted", "instruction_shuffled",
                               "instruction_layerwise", "attention", "attention_inverted",
                               "attention_shuffled"):
            raise ValueError(f"Unknown UAOR_SELECT: {self.select}")
        self.g = None
        # UAOR_SELECT_LAYER=k: read the single instruction weight at the input to layer k instead of 0.
        self.select_layer = int(_env("SELECT_LAYER", "0"))
        self.select_power = float(_env("SELECT_POWER", "1.0"))
        self.select_rows = _env("SELECT_ROWS", "all")
        self.instruction_span = None
        self.g_sum, self.g_count = 0.0, 0
        self.obs_offset = 1
        # the sync-free fast path needs a fixed threshold and no "once" bookkeeping
        self.sync_free = (self.trigger == "entropy") and (not self.auto) and (not self.once)
        self.log_dir = log_dir
        self.shard_id = shard_id
        self.rng = np.random.default_rng(1234 + shard_id)
        self.hist = [[] for _ in range(self.L)]
        # per-forward scratch
        self.O = None
        self.O_prev_episode = None
        self.O_first_of_episode = None
        self.fired_this_forward = False
        self.cur_u = [None] * self.L          # tensors (0-d) or floats
        self.cur_gate = [None] * self.L       # gate applied at layer l (tensor/bool) for logging
        self.forward_idx = 0
        self.episode = None
        self.records = []
        self.n_forwards = 0
        self.n_inject = 0
        self.per_layer_inject = np.zeros(self.L, dtype=np.int64)
        self.per_layer_checked = np.zeros(self.L, dtype=np.int64)
        self._proved = False
        self.t_extra = 0.0

    # ---- thresholds ----------------------------------------------------------------------------
    def threshold(self, l: int) -> float:
        if not self.auto:
            return self.gamma
        h = self.hist[l]
        if len(h) < self.warmup:
            return self.gamma0
        return float(np.quantile(np.asarray(h[-5000:]), 1.0 - self.rate))

    def in_window(self, l: int) -> bool:
        # uaor.py: layer >= starting_layer-1 and layer <= ending_layer-1   (layer is 0-indexed)
        return (l >= self.start - 1) and (l <= self.end - 1)

    # ---- entropy (returns a 0-d GPU tensor; no host sync) -------------------------------------
    @torch.no_grad()
    def substitute_signal(self, gate, h):
        """Another per-element signal, rescaled to the mean of the entropy it replaces.

        Rescaling rather than using the raw quantity is what makes the comparison a comparison: the arms
        then differ in which elements are favoured and not in how much evidence is written in total.
        """
        if self.soft_signal == "step":
            n = gate.shape[-1]
            per = max(1, n // 8)                       # 8 future steps of 7 dimensions in OpenVLA-OFT
            idx = torch.arange(n, device=gate.device, dtype=gate.dtype) // per + 1.0
            alt = idx.unsqueeze(0).expand_as(gate)
        elif self.soft_signal == "margin":
            alt = self.action_margin_vec(h)
        elif self.soft_signal == "inverted":
            # The dose goes to the elements that have already settled. Same values, same total, opposite
            # direction: if the assignment is what carries the gain, this arm has to lose it.
            alt = gate.max(dim=1, keepdim=True).values - gate
        else:
            raise RuntimeError(f"unknown UAOR_SOFT_SIGNAL {self.soft_signal!r}")
        m0 = gate.mean(dim=1, keepdim=True)
        m1 = alt.mean(dim=1, keepdim=True).clamp_min(1e-6)
        return alt * (m0 / m1)

    @torch.no_grad()
    def compute_gate(self, l: int, h: torch.Tensor):
        """Action entropy of layer l's output and the gate that layer l+1's FFN consumes.

        Every backbone adapter calls this, so the rule itself has exactly one implementation and a
        new backbone can only differ in where the rows are, never in what is computed.
        """
        vec = None
        if self.trigger == "soft" and l <= self.L - 2 and self.in_window(l):
            vec = self.action_entropy_vec(h)            # [B, n_act]; its mean is the scalar entropy
            u = vec.mean()
        else:
            u = self.action_entropy(h)                  # 0-d tensor
        self.cur_u[l] = u
        if l > self.L - 2:
            return u, None
        gate = None
        if self.in_window(l) and not (self.once and self.fired_this_forward):
            self.per_layer_checked[l] += 1
            if self.trigger == "entropy":
                if self.sync_free:
                    gate = u > self.threshold(l)      # bool tensor, no sync
                else:
                    uf = float(u.item())
                    if self.auto:
                        self.hist[l].append(uf)
                        if len(self.hist[l]) > 20000:
                            del self.hist[l][:10000]
                    gate = bool(uf > self.threshold(l))
            elif self.trigger == "confident":
                # The mirror of the published rule. High action entropy marks layers whose
                # representation is still forming, and overwriting the FFN output there
                # replaces a computation in progress; the consolidated low-entropy layers are
                # where a decision that has drifted from the observation can still be
                # re-grounded. Same threshold, same alpha, opposite comparison.
                if self.sync_free:
                    gate = u < self.threshold(l)
                else:
                    uf = float(u.item())
                    if self.auto:
                        self.hist[l].append(uf)
                        if len(self.hist[l]) > 20000:
                            del self.hist[l][:10000]
                    gate = bool(uf < self.threshold(l))
            elif self.trigger == "soft":
                gate = vec if vec is not None else self.action_entropy_vec(h)   # [B, n_act], consumed by layer l+1's MLP
                if self.soft_signal != "entropy":
                    gate = self.substitute_signal(gate, h)
                if self.step_gate:
                    gate = gate * self.step_gate_scale(l, u)
                if self.soft_norm != "none":
                    gate = self.normalise_soft(l, gate)
                if self.soft_uniform:
                    gate = gate.mean(dim=1, keepdim=True).expand_as(gate).clone()
                if self.soft_mix != 1.0:
                    mu = gate.mean(dim=1, keepdim=True)
                    gate = mu + self.soft_mix * (gate - mu)
                if self.soft_center:
                    gate = (gate - gate.mean(dim=1, keepdim=True)).clamp_min(0)
                if self.soft_topk > 0 and self.soft_topk < gate.shape[1]:
                    thr = torch.topk(gate, self.soft_topk, dim=1).values[:, -1:]
                    gate = torch.where(gate >= thr, gate, torch.zeros_like(gate))
                if self.soft_power != 1.0:
                    gate = gate.clamp_min(0) ** self.soft_power
                if self.soft_shuffle:
                    gate = gate[:, torch.randperm(gate.shape[1], device=gate.device)]
            elif self.trigger == "random":
                gate = bool(self.rng.random() < self.rand_rate)
            elif self.trigger == "all":
                gate = True
            else:                                  # none: measurement only
                gate = False
            if isinstance(gate, bool) and gate:
                self.fired_this_forward = True
            if torch.is_tensor(gate) and gate.dtype.is_floating_point:
                self.fired_this_forward = True
        return u, gate

    def _act_slice(self, n_tokens: int) -> slice:
        """Rows the rule reads and writes. act_index, when an adapter has set it, wins over act_pos."""
        if self.act_index is not None:
            return slice(self.act_index, self.act_index + self.n_act)
        if self.act_pos == "last":
            return slice(n_tokens - 1, n_tokens)
        return slice(n_tokens - (self.n_act + 1), n_tokens - 1)

    def _action_head_rows(self) -> torch.Tensor:
        """The lm_head rows that are action bins.

        OpenVLA pads the vocabulary to a multiple of 64 and de-tokenises against the UNPADDED size,
        so the action ids stop `pad` rows short of the end of the head. Slicing the last n_bins rows
        instead takes `pad` rows that can never be an action and drops that many real bins.
        """
        n = self.head.weight.shape[0]
        end = n - self.act_vocab_pad
        start = end - self.act_vocab
        if start < 0:
            raise ValueError(f"action vocab [{start},{end}) does not fit a head of {n} rows")
        if not self._logged_slice:
            self._logged_slice = True
            print(f"[AWARE] head_slice=action reads lm_head rows [{start}, {end}) "
                  f"of {n} (pad={self.act_vocab_pad}, bins={self.act_vocab})", flush=True)
        return self.head.weight[start:end]

    def action_entropy_vec(self, h: torch.Tensor) -> torch.Tensor:
        """Normalised action entropy of every action element, [B, n_act] in [0, 1]; the mean of this is
        the scalar UAOR thresholds. Keeping the vector is what lets the reinjection be applied per
        element instead of switching a whole layer."""
        ha = h[:, self._act_slice(h.shape[1])]
        if self.head_slice == "action":
            # only the action ids can win at the layers where the weight is not negligible, so the head
            # is applied to that slice alone: one [n_act, d] x [d, 256] product instead of [d, 32k]+top-k
            W = self._action_head_rows()
            logits = torch.nn.functional.linear(self.norm(ha), W).float()
            k = logits.shape[-1]
            p = torch.softmax(logits, dim=-1)
        else:
            logits = self.head(self.norm(ha)).float()
            k = min(self.topk, logits.shape[-1])
            p = torch.softmax(torch.topk(logits, k, dim=-1).values, dim=-1)
        return -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / float(np.log(k))

    def action_margin_vec(self, h: torch.Tensor) -> torch.Tensor:
        """One minus the top-1 probability of every action element, [B, n_act] in [0, 1].

        A confidence that is not an entropy: it reads only the winner and ignores how the rest of the mass
        is spread. Used as the alternative signal, it asks whether the shape of the distribution matters or
        only whether one candidate leads.
        """
        ha = h[:, self._act_slice(h.shape[1])]
        if self.head_slice == "action":
            W = self._action_head_rows()
            logits = torch.nn.functional.linear(self.norm(ha), W).float()
            p = torch.softmax(logits, dim=-1)
        else:
            logits = self.head(self.norm(ha)).float()
            k = min(self.topk, logits.shape[-1])
            p = torch.softmax(torch.topk(logits, k, dim=-1).values, dim=-1)
        return 1.0 - p.max(dim=-1).values

    @torch.no_grad()
    def normalise_soft(self, l: int, H: torch.Tensor) -> torch.Tensor:
        """Turn the raw per-element entropy into a relative one with running statistics of the same
        (layer, element), so that the plateau of the depth profile (about 0.8 through the middle of the
        stack for every element at every step) stops acting as a uniform dose. No host sync."""
        if self.mu is None:
            self.mu = torch.zeros(self.L, H.shape[1], device=H.device, dtype=torch.float32)
            self.var = torch.ones(self.L, H.shape[1], device=H.device, dtype=torch.float32)
            self.m2 = torch.zeros(self.L, H.shape[1], device=H.device, dtype=torch.float32)
            self.mu_count = [0] * self.L
        hb = H.float().mean(0)
        n = self.mu_count[l]
        if n < self.warmup:                                  # Welford during the warm-up
            d = hb - self.mu[l]
            self.mu[l] += d / (n + 1)
            self.m2[l] += d * (hb - self.mu[l])
            if n + 1 == self.warmup:
                self.var[l] = self.m2[l] / max(1, n)
        else:                                                # then an EMA that keeps following the run
            r = self.soft_ema
            d = hb - self.mu[l]
            self.mu[l] += r * d
            self.var[l] = (1.0 - r) * self.var[l] + r * d * d
        self.mu_count[l] = n + 1
        if n < self.warmup:
            return H                                         # warm-up: the paper's rule
        mu = self.mu[l].unsqueeze(0)
        if self.soft_norm == "excess":
            return ((H - mu) / (1.0 - mu).clamp_min(1e-3)).clamp_min(0.0)
        if self.soft_norm == "ratio":
            return H / mu.clamp_min(1e-3)
        return ((H - mu) / (self.var[l].sqrt().unsqueeze(0) + 1e-6)).clamp(0.0, 3.0)   # zscore

    @torch.no_grad()
    def step_gate_scale(self, l: int, u: torch.Tensor) -> torch.Tensor:
        """2*sigmoid(k * zscore(u_l)) with running per-layer statistics of the chunk-mean entropy; 1 during warm-up."""
        if self.u_mu is None:
            dev = u.device
            self.u_mu = torch.zeros(self.L, device=dev); self.u_var = torch.ones(self.L, device=dev)
            self.u_m2 = torch.zeros(self.L, device=dev); self.u_count = [0] * self.L
        uf = u.float()
        n = self.u_count[l]
        if n < self.warmup:
            d = uf - self.u_mu[l]
            self.u_mu[l] += d / (n + 1)
            self.u_m2[l] += d * (uf - self.u_mu[l])
            if n + 1 == self.warmup:
                self.u_var[l] = self.u_m2[l] / max(1, n)
        else:
            r = self.soft_ema
            d = uf - self.u_mu[l]
            self.u_mu[l] += r * d
            self.u_var[l] = (1.0 - r) * self.u_var[l] + r * d * d
        self.u_count[l] = n + 1
        if n < self.warmup:
            return torch.ones((), device=u.device)
        z = (uf - self.u_mu[l]) / (self.u_var[l].sqrt() + 1e-6)
        return 2.0 * torch.sigmoid(self.step_gate_k * z)

    @torch.no_grad()
    def action_entropy(self, h: torch.Tensor) -> torch.Tensor:
        ha = h[:, self._act_slice(h.shape[1])]
        if self.head_slice == "action":
            logits = torch.nn.functional.linear(self.norm(ha), self._action_head_rows()).float()
            k = logits.shape[-1]
            p = torch.softmax(logits, dim=-1)
        else:
            logits = self.head(self.norm(ha)).float()
            k = min(self.topk, logits.shape[-1])
            top = torch.topk(logits, k, dim=-1).values
            p = torch.softmax(top, dim=-1)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / float(np.log(k))
        return ent.mean()

    # ---- instruction-derived selection over the observation memory ----------------------------
    @torch.no_grad()
    def selection_weight(self, h: torch.Tensor, obs_offset: int):
        """Query the observation memory with the language tokens instead of the action state.

        This is the same retrieval the operator already performs, run once per forward with the
        instruction as the query, so it needs no attention internals, no extra parameters and no
        second pass. The share each observation token receives is compared against the uniform
        reference, which leaves no threshold and no fitted constant. The magnitude of the injection
        is unaffected because anchor() rescales the retrieval to the FFN output it blends with; only
        the shape of the retrieval changes.
        """
        if self.select in ("none", "instruction_layerwise") or self.select.startswith("attention") \
                or self.O is None:
            return None
        if self.lang_span is not None:
            language = h[:, self.lang_span[0]:self.lang_span[1], :]
        else:
            language = h[:, obs_offset + self.n_obs: -(self.n_act + 1), :]
        if language.shape[1] == 0:
            raise ValueError("Expected language rows between the observation and the action placeholders")
        if self.select_rows == "instruction":
            if self.instruction_span is None:
                raise RuntimeError("UAOR_SELECT_ROWS=instruction needs set_instruction_span() before the forward")
            s0, sl = self.instruction_span
            if s0 + sl > language.shape[1] or sl <= 0:
                raise RuntimeError(f"instruction span {self.instruction_span} outside {language.shape[1]} language rows")
            language = language[:, s0:s0 + sl, :]
        return self._weight_from(language)

    @torch.no_grad()
    def _weight_from(self, language: torch.Tensor):
        scores = torch.relu(language.float() @ self.O.float().transpose(-1, -2)).sum(1)
        share = scores / scores.sum(-1, keepdim=True).clamp_min(1e-12)
        w = (share * scores.shape[-1]).clamp(max=1.0)
        if self.select_power != 1.0:
            w = w.clamp_min(0) ** self.select_power
        if self.select == "instruction_inverted":
            w = 1.0 - w
        elif self.select == "instruction_shuffled":
            w = w.roll(w.shape[-1] // 2, dims=-1)
        self.g_sum += float(w.mean())
        self.g_count += 1
        return w

    @torch.no_grad()
    def _attention_weight(self, gate: torch.Tensor):
        """Selection from the model's own language-to-visual attention at the current layer.

        The layer-0 embedding retrieval turned out to be nearly uniform and to follow patch norms
        (share entropy 7.8-7.9 of 8 bits, correlation with patch norm 0.45-0.70 on captured states),
        so it does not say what the instruction is about. The trained cross-modal attention does; it is
        published per layer by the visual_feedback anchor provider and consumed here for the same
        layer's FFN. Visual tokens take the anchor, the proprio token is left at weight one.
        """
        visual = gate
        if self.select == "attention_inverted":
            visual = 1.0 - visual
        elif self.select == "attention_shuffled":
            per_view = visual.shape[-1] // 2
            visual = visual.reshape(visual.shape[0], 2, per_view).roll(per_view // 2, dims=2).reshape_as(visual)
        g = torch.ones(visual.shape[0], self.n_obs, device=visual.device, dtype=visual.dtype)
        g[:, :visual.shape[-1]] = visual
        self.g_sum += float(visual.mean())
        self.g_count += 1
        return g

    # ---- anchoring operator ------------------------------------------------------------------
    @torch.no_grad()
    def anchor(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        O = self.O
        if O is None or self.content == "none":
            return ((1.0 - self.alpha) * out.float()).to(out.dtype)
        if self.content == "other":
            O = self.O_prev_episode if self.O_prev_episode is not None else self._gauss_like(O)
        elif self.content == "gauss":
            O = self._gauss_like(O)
        O32 = O.float()
        x32 = x.float()
        g = self.g
        if self.select.startswith("attention"):
            gate = getattr(self.lm_model, "_instruction_anchor_gate", None)
            if gate is None:
                raise RuntimeError("UAOR_SELECT=attention needs the visual_feedback anchor provider "
                                   "(VFB_BETA=1 VFB_MODE=anchor_provider) installed on the same model")
            if gate.shape[-1] > self.n_obs:
                raise RuntimeError("Anchor covers more tokens than the observation memory")
            g = self._attention_weight(gate)
        if self.select == "instruction_layerwise" and x.shape[1] > self.obs_offset + self.n_obs:
            # The instruction as it stands at this layer, querying the original observation memory.
            _lang = x[:, self.obs_offset + self.n_obs: -(self.n_act + 1), :].float()
            if self.select_rows == "instruction":
                if self.instruction_span is None:
                    raise RuntimeError("UAOR_SELECT_ROWS=instruction needs set_instruction_span()")
                _s0, _sl = self.instruction_span
                if _sl <= 0 or _s0 + _sl > _lang.shape[1]:
                    raise RuntimeError(f"instruction span {self.instruction_span} outside {_lang.shape[1]} rows")
                _lang = _lang[:, _s0:_s0 + _sl, :]
            g = self._weight_from(_lang)
        scores = x32 @ O32.transpose(-1, -2)             # <x, o_i> for every observation token
        if g is not None:
            scores = scores * g.unsqueeze(1)              # instruction-derived weight over o_i
        A = scores @ O32                                  # sum_i g_i <x,o_i> o_i
        s = out.float().abs().mean() / (A.abs().mean() + 1e-12)
        new = (1.0 - self.alpha) * out.float() + self.alpha * s * A
        return new.to(out.dtype)

    @torch.no_grad()
    def anchor_soft(self, x: torch.Tensor, out: torch.Tensor, weights: torch.Tensor, l: int = 0) -> torch.Tensor:
        """Blend only the action elements, each with its own strength alpha*H_n.

        Retrieval follows anchor() (same memory O, unnormalised lookup by default) and the scale s is
        taken on the rows that are written, so the only difference from the published operator is where
        and how much: the 56 action rows instead of every row, and a continuous weight instead of a
        threshold. Everything is computed on the action rows only; the other rows are returned as is.
        """
        O = self.O
        if O is None:
            return out
        if self.content == "other":                   # control: the observation of a different episode
            O = self.O_prev_episode if self.O_prev_episode is not None else self._gauss_like(O)
        elif self.content == "gauss":                 # control: random directions with the same norms
            O = self._gauss_like(O)
        sl = self._act_slice(out.shape[1])
        osl = self.obs_slice()
        # Cast once per forward, not once per layer: O and g do not change between layers, and on a
        # backbone that decodes its action tokens one at a time this runs 256 times per control step.
        key = (id(O), osl.start, osl.stop)
        if self._o32_key != key:
            self._o32_key = key
            self._o32 = O[:, osl].float()
            self._gsl = None if self.g is None else self.g[:, osl].unsqueeze(1)
        O32 = self._o32
        xa, oa = x[:, sl].float(), out[:, sl].float()                      # [B, n_act, d]
        if self.retrieval == "softmax":
            sim = torch.nn.functional.normalize(xa, dim=-1) @ torch.nn.functional.normalize(O32, dim=-1).transpose(-1, -2)
            A = torch.softmax(sim / self.tau, dim=-1) @ O32
        else:
            scores = xa @ O32.transpose(-1, -2)                             # <x, o_i> for every observation token
            if self.g is not None and self.select.startswith("instruction"):
                scores = scores * self._gsl                                 # instruction-derived weight over o_i
            if self.retrieval == "layer":
                V = x[:, self.obs_offset:self.obs_offset + self.n_obs][:, osl].float()
            elif self.retrieval == "ffn":
                V = out[:, self.obs_offset:self.obs_offset + self.n_obs][:, osl].float()
            else:
                V = O32
            if self.obs_balance and self.obs_subset == "all":
                n_img = (self.n_obs - 1) // 2 if self.n_obs % 2 == 1 else self.n_obs // 2
                parts = [(0, n_img), (n_img, 2 * n_img)] + ([(2 * n_img, self.n_obs)] if self.n_obs > 2 * n_img else [])
                A = 0.0
                for a0, a1 in parts:
                    Ap = scores[:, :, a0:a1] @ V[:, a0:a1]
                    A = A + Ap / (Ap.abs().mean() + 1e-12)
            else:
                A = scores @ V
        # Scale on the rows that are written. The action placeholders enter as zeros and their FFN
        # outputs stay small, so a scale taken over every position over-injects them by a large
        # factor; the first run with the global scale collapsed LIBERO-Long to 116/500.
        if self.content == "none":                    # control: damp the FFN output, inject nothing
            A = torch.zeros_like(oa); s = oa.new_zeros(())
        else:
            s = oa.abs().mean() / (A.abs().mean() + 1e-12)
        w = self.alpha * weights.to(oa.dtype)
        if self.step_mask or self.soft_depth != "flat":
            w = w * self.element_scale(l, weights.shape[1], oa.device)
        w = w.clamp(0.0, 1.0).unsqueeze(-1)                                 # [B, n_act, 1]
        # Write the action rows in place. The hook's return value replaces the module output and
        # autograd is off, so cloning the whole [B, N, d] tensor once per layer only to overwrite 56 of
        # its rows is pure memory traffic; the arithmetic is identical either way.
        out[:, sl] = ((1.0 - w) * oa + w * s * A).to(out.dtype)
        m = w.mean()                                                        # kept on device: no per-layer sync
        self._soft_mass_t = m if self._soft_mass_t is None else self._soft_mass_t + m
        self.soft_count += 1
        return out

    def obs_slice(self) -> slice:
        """Rows of the observation memory the retrieval reads (OpenVLA-OFT order: camera-1 patches, camera-2
        patches, proprioceptive token)."""
        if self.obs_subset == "all":
            return slice(0, self.n_obs)
        n_img = (self.n_obs - 1) // 2 if self.n_obs % 2 == 1 else self.n_obs // 2
        if self.obs_subset == "third":
            return slice(0, n_img)
        if self.obs_subset == "wrist":
            return slice(n_img, 2 * n_img)
        return slice(0, 2 * n_img)                                          # noprop

    def element_scale(self, l: int, n: int, device) -> torch.Tensor:
        """Optional mask over the chunk's future steps times a per-layer depth factor."""
        # rebuilt when the element count changes as well as the device: on pi0 with
        # AWARE_PI0_ROWS=lang the count is the instruction length and differs per task
        if (self._elem_mask is None or self._elem_mask.device != device
                or self._elem_mask.shape[0] != n):
            m = torch.ones(n, device=device, dtype=torch.float32)
            if self.step_mask:
                steps = [c == "1" for c in self.step_mask]
                per_step = max(1, n // len(steps))
                m = torch.tensor([1.0 if steps[min(i // per_step, len(steps) - 1)] else 0.0 for i in range(n)],
                                 device=device, dtype=torch.float32)
            self._elem_mask = m
        depth = 1.0
        if self.soft_depth == "ramp":
            depth = l / max(1, self.L - 1)
        elif self.soft_depth == "iramp":
            depth = 1.0 - l / max(1, self.L - 1)
        return self._elem_mask.unsqueeze(0) * depth

    def _gauss_like(self, O: torch.Tensor) -> torch.Tensor:
        g = torch.randn_like(O.float())
        g = g / (g.norm(dim=-1, keepdim=True) + 1e-12) * O.float().norm(dim=-1, keepdim=True)
        return g.to(O.dtype)

    # ---- logging -------------------------------------------------------------------------------
    def flush(self):
        if not self.log_dir or not self.records:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, f"aware_shard{self.shard_id}.jsonl"), "a") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
        self.records = []

    def summary(self) -> dict:
        return {
            "n_forwards": int(self.n_forwards),
            "n_inject": int(self.n_inject),
            "inject_rate_per_checked_layer": float(self.n_inject / max(1, self.per_layer_checked.sum())),
            "per_layer_inject": self.per_layer_inject.tolist(),
            "per_layer_checked": self.per_layer_checked.tolist(),
            "extra_seconds": self.t_extra,
            "final_thresholds": [self.threshold(l) for l in range(self.L)] if self.auto else self.gamma,
            "sync_free": self.sync_free,
            "select": self.select,
            "soft_mean_weight": (float(self._soft_mass_t) / self.soft_count) if self.soft_count else None,
            "soft_norm": self.soft_norm, "retrieval": self.retrieval, "soft_depth": self.soft_depth,
            "soft_step_mask": self.step_mask, "obs_subset": self.obs_subset, "obs_balance": self.obs_balance,
            "head_slice": self.head_slice, "min_layer": self.min_layer,
            "step_gate": self.step_gate, "write": self.write,
            "soft_mu_layer": self.mu.mean(1).tolist() if self.mu is not None else None,
            "soft_mu_elem": self.mu.mean(0).tolist() if self.mu is not None else None,
            "mean_selection_weight": (self.g_sum / self.g_count) if self.g_count else None,
        }


# ------------------------------------------------------------------------------------------------
def set_instruction_span(start: int, length: int):
    """Tell the installed hook which language rows name the task, as the evaluation computes them."""
    if _STATE is not None:
        _STATE.instruction_span = (int(start), int(length))


def mark_episode(task_id: int, episode_idx: int):
    """Call once per episode before the first policy query (episode boundary for the logs and for
    UAOR_CONTENT=other)."""
    st = _STATE
    if st is None:
        return
    st.flush()
    if st.O_first_of_episode is not None:
        st.O_prev_episode = st.O_first_of_episode
    st.O_first_of_episode = None
    st.episode = [int(task_id), int(episode_idx)]
    st.forward_idx = 0


def _install_on(lm, *, n_obs: int, n_act: int, shard_id: int, log_dir: str | None, obs_offset: int = 1):
    global _STATE
    st = _State(lm, n_obs=n_obs, n_act=n_act, shard_id=shard_id, log_dir=log_dir)
    _STATE = st

    st.obs_offset = obs_offset

    def make_select_hook(k):
        """Read g at the input to layer k. The observation memory st.O stays the layer-0 one, so only the
        query side of the weight moves; that is the difference the experiment is about."""
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if h.shape[1] > obs_offset + st.n_obs:
                st.g = st.selection_weight(h, obs_offset)
            return None
        return hook

    def pre_hook_layer0(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        t0 = time.perf_counter()
        if h.shape[1] > obs_offset + st.n_obs:            # prefill (skip 1-token decode steps)
            st.O = h[:, obs_offset:obs_offset + st.n_obs, :].detach()
            if st.O_first_of_episode is None:
                st.O_first_of_episode = st.O.clone()
            if st.select_layer == 0:
                st.g = st.selection_weight(h, obs_offset)
        st.fired_this_forward = False
        st.cur_u = [None] * st.L
        st.cur_gate = [None] * st.L
        for lyr in st.layers:
            lyr.mlp._aware_gate = None
            lyr.self_attn._aware_gate = None
        st.t_extra += time.perf_counter() - t0
        return None

    def make_layer_hook(l: int):
        def hook(module, args, output):
            t0 = time.perf_counter()
            h = output[0] if isinstance(output, (tuple, list)) else output
            if st.trigger == "soft" and l + 1 < st.min_layer:
                st.layers[l + 1].mlp._aware_gate = None if l <= st.L - 2 else None
                st.t_extra += time.perf_counter() - t0
                return None
            u, gate = st.compute_gate(l, h)
            if l <= st.L - 2:
                st.layers[l + 1].mlp._aware_gate = gate
                st.layers[l + 1].self_attn._aware_gate = gate
                st.cur_gate[l + 1] = gate
            else:
                # last layer: one host sync for the whole forward (logging + counters)
                st.n_forwards += 1
                st.forward_idx += 1
                us = torch.stack([v if torch.is_tensor(v) else torch.tensor(float("nan"), device=h.device)
                                  for v in st.cur_u]).tolist()
                trig = []
                for j, g in enumerate(st.cur_gate):
                    if g is None:
                        continue
                    gb = (bool(g.item()) if g.numel() == 1 else True) if torch.is_tensor(g) else bool(g)
                    if gb:
                        trig.append(j)
                        st.n_inject += 1
                        st.per_layer_inject[j] += 1
                if st.log_dir:
                    st.records.append({"ep": st.episode, "k": st.forward_idx,
                                       "u": [round(v, 4) for v in us], "trig": trig})
                    if len(st.records) >= 200:
                        st.flush()
            st.t_extra += time.perf_counter() - t0
            return None
        return hook

    def make_mlp_hook(l: int):
        def hook(module, args, output):
            gate = getattr(module, "_aware_gate", None)
            if gate is None or (isinstance(gate, bool) and not gate):
                return None
            t0 = time.perf_counter()
            x = args[0]
            if torch.is_tensor(gate) and gate.dtype.is_floating_point:
                if st.write == "attn":
                    return None                           # this configuration writes the attention output instead
                # per-element soft blend on the action positions only; alpha*H_n replaces alpha
                new = st.anchor_soft(x, output, gate, l)
            else:
                anchored = st.anchor(x, output)
                if torch.is_tensor(gate):
                    new = torch.where(gate, anchored, output)
                else:
                    new = anchored
            if not st._proved:
                st._proved = True
                log.warning("[AWARE] ACTIVE alpha=%.3f gamma=%s trigger=%s content=%s window=[%d,%d] "
                            "n_obs=%d n_act=%d act_pos=%s once=%s sync_free=%s first_layer=%d",
                            st.alpha, _env("GAMMA", "0.80"), st.trigger, st.content,
                            st.start, st.end, st.n_obs, st.n_act, st.act_pos, st.once, st.sync_free, l)
            st.t_extra += time.perf_counter() - t0
            return new
        return hook

    def make_attn_hook(l: int):
        def hook(module, args, kwargs, output):
            gate = getattr(module, "_aware_gate", None)
            if gate is None or not (torch.is_tensor(gate) and gate.dtype.is_floating_point) or st.write == "ffn":
                return None
            t0 = time.perf_counter()
            x = args[0] if args else kwargs["hidden_states"]
            attn_out = output[0] if isinstance(output, (tuple, list)) else output
            new = st.anchor_soft(x, attn_out, gate, l)
            if not st._proved:
                st._proved = True
                log.warning("[AWARE] ACTIVE (attention write) alpha=%.3f trigger=%s write=%s first_layer=%d",
                            st.alpha, st.trigger, st.write, l)
            st.t_extra += time.perf_counter() - t0
            if isinstance(output, tuple):
                return (new,) + tuple(output[1:])
            if isinstance(output, list):
                return [new] + list(output[1:])
            return new
        return hook

    st.layers[0].register_forward_pre_hook(pre_hook_layer0, with_kwargs=True)
    if st.select_layer > 0:
        st.layers[st.select_layer].register_forward_pre_hook(
            make_select_hook(st.select_layer), with_kwargs=True)
    for l in range(st.L):
        st.layers[l].register_forward_hook(make_layer_hook(l))
        if l >= 1:
            st.layers[l].mlp._aware_gate = None
            st.layers[l].self_attn._aware_gate = None
            st.layers[l].mlp.register_forward_hook(make_mlp_hook(l))
            st.layers[l].self_attn.register_forward_hook(make_attn_hook(l), with_kwargs=True)

    cfg = {k: v for k, v in os.environ.items() if k.startswith("UAOR_")}
    cfg.update({"n_obs": n_obs, "n_act": n_act, "L": st.L, "shard_id": shard_id, "sync_free": st.sync_free})
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"uaor_config_shard{shard_id}.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    log.warning("[AWARE] installed: %s", json.dumps(cfg))

    def _exit():
        st.flush()
        if log_dir:
            with open(os.path.join(log_dir, f"uaor_summary_shard{shard_id}.json"), "w") as f:
                json.dump(st.summary(), f, indent=1)
        log.warning("[AWARE] summary: %s", json.dumps({k: v for k, v in st.summary().items()
                                                       if k not in ("per_layer_inject", "per_layer_checked")}))
    atexit.register(_exit)
    return st


def install(vla, *, use_proprio: bool, shard_id: int = 0, log_dir: str | None = None):
    """OpenVLA-OFT entry point. Idempotent. No-op when UAOR_ALPHA is unset."""
    if not enabled() or _STATE is not None:
        return _STATE
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    n_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    visual_only = _env("VISUAL_ONLY", "0") == "1"
    n_obs = n_patches + (1 if (use_proprio and not visual_only) else 0)
    return _install_on(vla.language_model, n_obs=n_obs, n_act=ACTION_DIM * NUM_ACTIONS_CHUNK,
                       shard_id=shard_id, log_dir=log_dir, obs_offset=1)


def install_generic(lm, *, n_obs: int, n_act: int = 1, shard_id: int = 0, log_dir: str | None = None,
                    obs_offset: int = 1):
    """Any HF LlamaForCausalLM-like module (``lm.model.layers``, ``lm.model.norm``, ``lm.lm_head``), e.g.
    the OpenVLA-7B autoregressive policy: n_obs=256 patches after <BOS>, n_act=1 with UAOR_ACT_POS=last."""
    if not enabled() or _STATE is not None:
        return _STATE
    return _install_on(lm, n_obs=n_obs, n_act=n_act, shard_id=shard_id, log_dir=log_dir, obs_offset=obs_offset)
