# AWARE: Action-Entropy-Weighted and Attention-Guided Retrieval of Observation Evidence for Vision-Language-Action Models

**English** · [简体中文](README_zh.md) · [**Project Page**](https://jiabingyang01.github.io/AWARE/)

**A**ction-Entropy-**W**eighted and **A**ttention-Guided **R**etrieval of Observation **E**vidence (**AWARE**) is a training-free, plug-and-play inference module for Vision-Language-Action (VLA) models. At every decoder layer, AWARE tracks per-element predictive uncertainty via **Action Entropy** and retrieves instruction-grounded observation evidence into each action element's Feed-Forward Network (FFN) output in proportion to its uncertainty and network depth—requiring no fine-tuning, no extra sensors, no architectural changes, and zero learnable parameters.

```
                   /         l              \
w(l)  =  alpha  *  |  -----------  *  H(l)   |          per action element n, per layer l
  t,n              \    L - 1          t,n  /

FFN'(h)  =  ( 1 - w )  *  FFN(h)   +   w * s * A(h)
```

- `H` is the element's normalized **Action Entropy**, obtained by projecting the layer's hidden state through the frozen language modeling head (`K = 256`).
- `A(h)` is a linear associative memory lookup over the encoded observation tokens `O`, weighted once per forward pass by instruction-to-observation cross-modal alignment `g`.
- `s` matches the mean activation magnitude of the native FFN output to that of the retrieved features.

Nothing is installed unless `AWARE_ALPHA` is set. When `AWARE_ALPHA` is unset, the policy remains byte-identical to the untouched base checkpoint.

---

## Quick Start

```python
from aware import install                  # OpenVLA-OFT (parallel chunk decoding)
state = install(vla, use_proprio=True)

from aware import install_generic          # Causal / dual-system backbones (OpenVLA, SpatialVLA, CogACT, LLaVA-VLA)
state = install_generic(language_model, n_obs=256, n_act=1, obs_offset=0)

from aware import pi0                      # openpi pi0 / pi0.5 (PyTorch path)
state = pi0.maybe_install(model)
```

Enable AWARE at inference time via environment variables:

```bash
export AWARE_ALPHA=0.03 \
       AWARE_TRIGGER=soft AWARE_SOFT_DEPTH=ramp AWARE_SELECT=instruction
```

See [`integrations/README.md`](integrations/README.md) for per-backbone wiring details, switch configurations, and verification gates.

---

## Main Results

### LIBERO Benchmark (500 episodes per suite)

| Method | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|
| OpenVLA (7B) | 82.8 | 85.6 | 78.4 | 49.6 | 74.1 |
| **+ AWARE (Ours)** | **86.4** | **87.0** | **79.0** | **57.2** | **77.4** |
| $\pi_0$ (3B) | 96.4 | 97.2 | 93.8 | 81.2 | 92.2 |
| **+ AWARE (Ours)** | **98.2** | **98.6** | **94.4** | **84.0** | **93.8** |
| $\pi_{0.5}$ (3B) | 98.2 | 98.6 | 97.6 | 92.6 | 96.8 |
| **+ AWARE (Ours)** | **99.2** | **99.6** | **98.4** | **94.0** | **97.8** |
| OpenVLA-OFT (7B) | 98.0 | 98.2 | 97.6 | 93.8 | 96.9 |
| **+ AWARE (Ours)** | **99.2** | **98.6** | **98.6** | **96.0** | **98.1** |

- **SIMPLER (Visual Matching)**: CogACT (7B) on Google Robot improves from 73.7% to **76.9%** (+8.3% on *Open Top Drawer and Place Apple*); SpatialVLA (4B) on WidowX improves from 42.7% to **44.8%**.
- **CALVIN ABC$\rightarrow$D**: LLaVA-VLA (0.5B) average completed chain length increases from 3.49 to **3.63**, with full 5-subtask completion rising from 47.0% to **50.4%**.
- **Real-World Franka Research 3**: Consistently improves 5 VLA backbones across 5 tabletop manipulation tasks over 2,000 physical rollouts (+4.0% to +8.0% average gain, up to +17.5% on a single task).

---

## Repository Structure

```text
aware/          Core AWARE package (hooks.py implements the operator; pi0.py adapts openpi).
integrations/   Wiring guides, switch reference, and verification gates for each VLA family.
eval/           Evaluation launchers for LIBERO, SIMPLER, and CALVIN.
analysis/       Forward-pass diagnostic probes (observation footprint decay, action entropy wavefront).
figures/        Scripts used to generate the paper's figures.
docs/           Interactive project website (served via GitHub Pages).
```

---

## Requirements

- Python $\ge$ 3.10, PyTorch $\ge$ 2.0, `numpy`.
- The `aware` hooks depend only on `torch` and `numpy`, attaching non-invasively to the host policy's existing environment (OpenVLA-OFT, openpi PyTorch, CogACT, SpatialVLA, LLaVA-VLA).

---

## Citation

```bibtex
@article{yang2026aware,
  title   = {AWARE: Action-Entropy-Weighted and Attention-Guided Retrieval of Observation Evidence for Vision-Language-Action Models},
  author  = {Yang, Jiabing and Chen, Yixiang and Xu, Yuan and Li, Peiyan and Wu, Xiangnan and Wen, Zichen and Fang, Bowen and Yu, Tao and Zhang, Zhengbo and Li, Yingda and Wang, Kai and Liu, Jing and Liu, Nianfeng and Huang, Yan and Wang, Liang},
  journal = {arXiv preprint arXiv:2602.18020},
  year    = {2026}
}
```
