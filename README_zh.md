# AWARE：面向视觉-语言-动作模型的动作熵加权与注意力引导观测证据取回

[English](README.md) · **简体中文** · [**项目主页**](https://jiabingyang01.github.io/AWARE/)

**A**ction-Entropy-**W**eighted and **A**ttention-Guided **R**etrieval of Observation **E**vidence (**AWARE**) 是一种面向视觉-语言-动作（VLA）策略的**免训练、即插即用**推理期模块。在每个 Transformer 解码层中，AWARE 通过逐元素**动作熵（Action Entropy）**追踪预测不确定性，利用前馈网络（FFN）的键值关联记忆性质，按元素熵与层深比例将指令引导的观测特征动态注入高不确定性动作维度——**无需微调、无需额外传感器、不改动模型架构、不引入任何可学习参数**。

```
                   /         l              \
w(l)  =  alpha  *  |  -----------  *  H(l)   |          对每个动作元素 n、每一层 l
  t,n              \    L - 1          t,n  /

FFN'(h)  =  ( 1 - w )  *  FFN(h)   +   w * s * A(h)
```

---

## 快速开始

```python
from aware import install                  # OpenVLA-OFT（并行动作块解码）
state = install(vla, use_proprio=True)

from aware import install_generic          # 自回归 / 双系统骨干（OpenVLA, SpatialVLA, CogACT, LLaVA-VLA）
state = install_generic(language_model, n_obs=256, n_act=1, obs_offset=0)

from aware import pi0                      # openpi pi0 / pi0.5（PyTorch 路径）
state = pi0.maybe_install(model)
```

推理时通过环境变量启用发布版规则：

```bash
export AWARE_ALPHA=0.03 \
       AWARE_TRIGGER=soft AWARE_SOFT_DEPTH=ramp AWARE_SELECT=instruction
```

各骨干模型的详细接线说明、超参表与三步验收检查见 [`integrations/README_zh.md`](integrations/README_zh.md)。

---

## 目录结构

```text
aware/          核心方法包（hooks.py 为核心算子，pi0.py 为 openpi 适配器）
integrations/   各 VLA 骨干接线指南、开关说明与自检流程
eval/           LIBERO、SIMPLER 与 CALVIN 评测启动脚本
analysis/       前向诊断探针（观测子空间占比衰减、动作熵决策波前等）
figures/        论文图表生成脚本
docs/           交互式项目主页源码（通过 GitHub Pages 部署）
```
