<div align="center">

# AWARE: Action-Entropy-Weighted and Attention-Guided Retrieval of Observation Evidence for Vision-Language-Action Models

一种面向视觉-语言-动作（VLA）模型的**免训练、即插即用**推理期框架：在每一层追踪逐元素**动作熵（Action Entropy）**，并通过模型原生前馈网络（FFN）动态取回指令引导的观测证据——*无需微调、无需额外传感器、不引入任何可学习参数*。

[**\[📄论文\]**](https://arxiv.org/abs/2602.18020) &nbsp;&nbsp; [**\[🏠项目主页\]**](https://jiabingyang01.github.io/AWARE/) &nbsp;&nbsp; [**\[🎬对比视频\]**](https://jiabingyang01.github.io/AWARE/#rollouts) &nbsp;&nbsp; [**\[🔌接入指南\]**](integrations/README_zh.md) &nbsp;&nbsp; [**\[🌐English\]**](README.md)

[![Training-Free](https://img.shields.io/badge/Inference-Training--Free-7030a0?style=flat-square)](https://jiabingyang01.github.io/AWARE/)
[![Backbones](https://img.shields.io/badge/VLA_Backbones-7_Models_(0.5B--7B)-21918c?style=flat-square)](#--实验结果)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-LIBERO_%7C_SIMPLER_%7C_CALVIN_%7C_Real--World-3c7d2c?style=flat-square)](#--实验结果)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE.md)

</div>

---

## 🔥 最新动态

* **`2026.10`** 🌟 **AWARE** 核心代码、交互式[项目主页](https://jiabingyang01.github.io/AWARE/)、前向机理探针及涵盖 **7 种 VLA 基座（LIBERO、SIMPLER、CALVIN 与 Franka 真机）** 的评测套件正式开源！

## 👀 目录

- [方法概览](#-方法概览)
- [快速开始与骨干接入](#-快速开始与骨干接入)
- [评测与诊断探针](#-评测与诊断探针)
- [实验结果](#-实验结果)
- [仓库结构](#-仓库结构)
- [致谢](#-致谢)
- [引用](#-引用)

## 📋 方法概览

![AWARE 总体框架](docs/assets/img/aware_pipeline.webp)

在 VLA 模型的前向推理过程中，随着层深增加会出现两个显著的表征瓶颈：
1. **观测子空间衰减（Observation Subspace Decay）**：动作表征中由观测 Token 解释的子空间占比随深度单调下降，在所有 LIBERO 子集上到第 26 层均衰减至初始值的一半左右。
2. **不确定性跨步异步收敛（Decision Wavefront）**：在并行预测的动作块（$H=8, D_a=7, N_a=56$）中，近期时间步在中层即收敛，而远期时间步直到深层仍保持高预测熵。

<div align="center">
  <img src="docs/assets/img/fig_footprint.svg" width="88%" alt="AWARE 经验动机"/>
</div>

基于 Transformer 前馈网络（FFN）本质上是键值关联记忆（Key-Value Associative Memory）的洞察，**AWARE** 让每个动作元素在推理期自主调节其观测特征取回：
- **1. 读取（逐元素动作熵）**：在第 $\ell$ 层输出处，将每个动作元素的隐状态投射过冻结的语言头（Logit Lens），计算归一化动作熵 $\mathcal{H}^{(\ell)}_{t,n} \in [0, 1]$（$K=256$）。
- **2. 加权（指令引导观测门控）**：利用指令与视觉 Patch 的非负对齐度计算无参重要性权重 $\boldsymbol{g}_t$，过滤无关背景干扰。
- **3. 取回（FFN 关联记忆查找）**：在第 $\ell+1$ 层，以进入 FFN 的隐状态为查询，通过两次轻量矩阵乘法取回观测证据：$\textsc{inj}_{t,n}^{(\ell+1)} = s\,\boldsymbol{O}_t^{\top}\operatorname{diag}(\boldsymbol{g}_t)\,\boldsymbol{O}_t\,\boldsymbol{h}_{t,n}^{(\ell+1)}$。
- **4. 写入（连续自适应混合）**：按元素熵与相对层深 $w^{(\ell+1)}_{t,n} = \alpha\,\tfrac{\ell}{L-1}\,\mathcal{H}^{(\ell)}_{t,n}$ 线性混合进 FFN 输出，精准增强深层高不确定性维度，同时保持已收敛维度不变。

## 🔧 快速开始与骨干接入

**环境要求**：Python $\ge$ 3.10，PyTorch $\ge$ 2.0，`numpy`。`aware/` 钩子除宿主策略已有环境外**不引入任何额外依赖**。

```python
# OpenVLA-OFT（并行动作块解码，N_a = 56）
from aware import install
state = install(vla, use_proprio=True)

# 自回归与双系统骨干（OpenVLA, SpatialVLA, CogACT, LLaVA-VLA）
from aware import install_generic
state = install_generic(language_model, n_obs=256, n_act=1, obs_offset=0)

# openpi（pi0 / pi0.5，PyTorch 路径）
from aware import pi0
state = pi0.maybe_install(model)
```

推理时通过环境变量启用：

```bash
export AWARE_ALPHA=0.03 \
       AWARE_TRIGGER=soft \
       AWARE_SOFT_DEPTH=ramp \
       AWARE_SELECT=instruction
```

详细接线说明、开关配置与三步自检流程请参阅 [`integrations/README_zh.md`](integrations/README_zh.md)。

## 🧪 评测与诊断探针

```bash
# LIBERO 评测
python eval/run_libero_oft.py
bash eval/ovla_cell_aware.sh
python eval/run_aware_pi0.py --model pi0 --suite libero_10 --alpha 0.03 --seed 7

# SIMPLER (Visual Matching) 评测 — CogACT & SpatialVLA
python eval/run_aware_cogact.py --alpha 0.09 --seed 7
ARM=aware ALPHA=0.08 bash eval/sv_aware_cell.sh

# CALVIN (ABC -> D) 评测 — LLaVA-VLA (0.5B)
ARM=aware ALPHA=0.01 bash eval/calvin_cell_v5.sh
```

## 📈 实验结果

| 方法 | 发表会议 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | **LIBERO 平均** |
|---|---|---:|---:|---:|---:|---:|
| OpenVLA (7B) | *CoRL'24* | 82.8 | 85.6 | 78.4 | 49.6 | 74.1 |
| **+ AWARE (Ours)** | — | **86.4** (+3.6) | **87.0** (+1.4) | **79.0** (+0.6) | **57.2** (+7.6) | **77.4** (+3.3) |
| $\pi_0$ (3B) | *RSS'25* | 96.4 | 97.2 | 93.8 | 81.2 | 92.2 |
| **+ AWARE (Ours)** | — | **98.2** (+1.8) | **98.6** (+1.4) | **94.4** (+0.6) | **84.0** (+2.8) | **93.8** (+1.6) |
| $\pi_{0.5}$ (3B) | *CoRL'25* | 98.2 | 98.6 | 97.6 | 92.6 | 96.8 |
| **+ AWARE (Ours)** | — | **99.2** (+1.0) | **99.6** (+1.0) | **98.4** (+0.8) | **94.0** (+1.4) | **97.8** (+1.0) |
| OpenVLA-OFT (7B) | *RSS'25* | 98.0 | 98.2 | 97.6 | 93.8 | 96.9 |
| **+ AWARE (Ours)** | — | **99.2** (+1.2) | **98.6** (+0.4) | **98.6** (+1.0) | **96.0** (+2.2) | **98.1** (+1.2) |

![真机实验平台与任务](docs/assets/img/real_world_setup.webp)

* **SIMPLER Visual Matching**：Google Robot 上 CogACT 平均成功率 73.7% $\rightarrow$ **76.9%**（多阶段任务 *Open Top Drawer and Place Apple* **+8.3%**）；WidowX 上 SpatialVLA 42.7% $\rightarrow$ **44.8%**。
* **CALVIN ABC$\rightarrow$D**：LLaVA-VLA (0.5B) 在第 3/4/5 步子任务上分别提升 **+4.2% / +5.0% / +3.4%**，五步全通率从 47.0% $\rightarrow$ **50.4%**，平均完成长度 3.49 $\rightarrow$ **3.63**。
* **Franka Research 3 真机评测（2,000 次物理试验）**：5 项操作任务 $\times$ 5 种 VLA 基座平均提升 **+4.0% ~ +8.0%**，单任务最高提升 **+17.5%**；单步环境交互仅增加 **2.6 ms**（保持 **68.1 Hz** 实时控制吞吐）。

## 🗂️ 仓库结构

```text
AWARE/
├── aware/                  # 核心 AWARE 算子与适配器包 (hooks.py, pi0.py)
├── integrations/           # 各 VLA 骨干接线指南、环境变量开关与三步验收标准
├── eval/                   # LIBERO / SIMPLER / CALVIN 评测启动脚本
├── analysis/               # 前向表征探针（观测子空间占比、动作熵决策波前、门控验证）
├── figures/                # 论文图表生成脚本
└── docs/                   # 交互式项目主页源码 (https://jiabingyang01.github.io/AWARE/)
```

## 🙏 致谢

感谢以下优秀开源项目与基准社区：
[OpenVLA](https://github.com/openvla/openvla) ·
[OpenVLA-OFT](https://github.com/moojink/openvla-oft) ·
[openpi ($\pi_0 / \pi_{0.5}$)](https://github.com/Physical-Intelligence/openpi) ·
[CogACT](https://github.com/microsoft/CogACT) ·
[SpatialVLA](https://github.com/SpatialVLA/SpatialVLA) ·
[LLaVA-VLA](https://github.com/LLaVA-VL/LLaVA-NeXT) ·
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) ·
[SimplerEnv](https://github.com/simpler-env/SimplerEnv) ·
[CALVIN](https://github.com/mees/calvin)。

本仓库采用 **MIT License** 开源（详见 [`LICENSE.md`](LICENSE.md)）。

## 📝 引用

```bibtex
@article{yang2026aware,
  title   = {AWARE: Action-Entropy-Weighted and Attention-Guided Retrieval of Observation Evidence for Vision-Language-Action Models},
  author  = {Yang, Jiabing and Chen, Yixiang and Xu, Yuan and Li, Peiyan and Wen, Zichen and Fang, Bowen and Cui, Chenhang and Yu, Tao and Ma, Qisen and Wang, Kai and He, Ziheng and Ye, Wen and Wu, Xiangnan and Li, Yingda and Huang, Yan and Wang, Liang},
  journal = {arXiv preprint arXiv:2602.18020},
  year    = {2026}
}
```
