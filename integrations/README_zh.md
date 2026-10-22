# 把 AWARE 接到一个骨干上

[English](README.md) · **简体中文**

AWARE 是一组前向钩子。它不 fork 策略,也不碰权重。要把它接到新骨干上,你只需要回答三个问题:

1. **观测行在哪里?** 也就是规则取回内容的那块记忆。通常是前缀里的图像 patch token,可选再加一个本体感受 token。
2. **动作行在哪里?** 也就是被读取不确定性、被混合 FFN 输出的那些元素。并行解码的动作块是一段 `n_act` 行;
   自回归骨干则是单独一行,即产生下一个动作 token 的那一行(`AWARE_ACT_POS=last`)。
3. **解码层挂在哪个模块下?** 规则会给每一层注册一个前向钩子。

其余一切(熵、深度 ramp、指令加权、混合)都是共用的,住在 `aware/hooks.py` 里。**没有任何适配器重新实现算子。**

```python
from aware import install_generic
state = install_generic(language_model, n_obs=<观测行数>, n_act=<动作行数>,
                        obs_offset=<第一行观测的下标>)
```

不设 `AWARE_ALPHA` 就什么都不会发生,所以这行调用可以一直留在代码里。

---

## 开关

命名为 `AWARE_<NAME>`,同时接受改名前的 `UAOR_<NAME>` 作为回退,这样改名之前跑出来的结果仍然可复现。
两者都设时以新名为准。

| 开关 | 默认 | 含义 |
|---|---|---|
| `AWARE_ALPHA` | *(不设 = 关闭)* | 混合强度;唯一必需的开关 |
| `AWARE_TRIGGER` | `entropy` | `soft` 是发布版规则(连续权重)。`entropy` 是 CoRL 时期的硬阈值,`all`、`random`、`none` 是对照 |
| `AWARE_SOFT_DEPTH` | *(平)* | `ramp` = 权重随 `l/(L-1)` 缩放,即发布版规则 |
| `AWARE_SELECT` | `none` | `instruction` = 按指令自身对观测的读取来倾斜取回 |
| `AWARE_CONTENT` | `obs` | 写什么。`other` = 另一条 episode 的观测,`gauss` = 模长匹配的随机方向,`none` = 只衰减不写入。**这些是对照,不是可选配置** |
| `AWARE_SOFT_SIGNAL` | `entropy` | 逐元素权重从哪里读。`step`、`margin`、`inverted` 是同剂量对照 |
| `AWARE_START` / `AWARE_END` | `1` / `L-1` | 层窗口;两者设成同一个值即只在单层写入 |
| `AWARE_ACT_POS` | `chunk` | 自回归骨干设 `last` |

> **发布版规则是 `AWARE_TRIGGER=soft AWARE_SOFT_DEPTH=ramp AWARE_SELECT=instruction`。**
> 只设 `AWARE_ALPHA` 会退回硬阈值变体,它大约每一千次前向才触发一次:日志照样打印 `ACTIVE`,`alpha=0` 也照样
> 逐位相同,但输出与未干预的策略无从区分。**要看 summary 里的 `n_inject` 和 `soft_mean_weight`,
> 绝不能只看 `installed` 那一行。**

## 论文中使用的强度

| 骨干 | 基准 | alpha |
|---|---|---|
| OpenVLA-OFT | LIBERO Spatial / Object / Goal / Long | 0.04 / 0.05 / 0.10 / 0.03 |
| pi0 | LIBERO | 0.03 / 0.04 / 0.05 / 0.03 |
| pi0.5 | LIBERO | 0.01 / 0.08 / 0.01 / 0.07 |
| OpenVLA-7B | LIBERO | 0.03 / 0.02 / 0.01 / 0.01 |
| CogACT | SIMPLER Google Robot | 0.09 |
| SpatialVLA-4B | SIMPLER WidowX | 0.08 |
| LLaVA-VLA 0.5B | CALVIN ABC→D | 0.01 |

---

## 各骨干的接线说明

### OpenVLA-OFT

动作块是并行解码的,所以 `n_act = chunk × action_dim`(8 × 7 = 56),这些行紧挨在最后一个位置之前。
`install` 会自己从模型上把布局读出来:

```python
from aware import install
state = install(vla, use_proprio=cfg.use_proprio, shard_id=cfg.shard_id, log_dir=...)
```

策略建好后调用一次即可;另外在每条 episode 开头调用 `aware.mark_episode(task_id, episode_idx)`,
这样 `AWARE_CONTENT=other` 才有上一条 episode 可读。


### openpi(pi0 / pi0.5),仅 PyTorch 路

流匹配动作专家的宽度是 1024,而 unembedding 与 PaliGemma tower 绑定在 2048,所以动作分布**无法**从专家的行里解码。
因此 AWARE 挂在**前缀**上:动作元素是最后一个有效的语言行(`n_act = 1`),记忆是同一前缀里的图像行;前缀是双向注意力,
所以在一行上的写入会在下一层到达每一行。改动是 `models_pytorch/pi0_pytorch.py` 里 `__init__` 末尾的一行:

```python
from aware import pi0 as _aware_pi0
self._aware = _aware_pi0.maybe_install(self)
```

**JAX 路不支持**:它把采样步数写死,而且从不暴露 tower。

### CogACT

动作元素是 cognition token,也就是追加在 prompt 后面的 EOS,因此是最后一行;记忆是 Prismatic 在 `<BOS>` 之后
插入的 patch 块,所以 `obs_offset=1`。

```python
os.environ.setdefault("AWARE_ACT_POS", "last")
state = install_generic(vla.vlm.llm_backbone.llm,
                        n_obs=int(vla.vlm.vision_backbone.num_patches), n_act=1, obs_offset=1)
```

### SpatialVLA-4B

PaliGemma2 风格:processor 把图像 token 放在最前,然后是 `<bos>` + 指令 + `\n`,动作以 spatial token 自回归解码。
所以 `obs_offset=0`,记忆是 `image_seq_length × num_obs_steps` 行。

```python
os.environ.setdefault("AWARE_ACT_POS", "last")
n_obs = int(processor.image_seq_length) * int(processor.num_obs_steps)
state = install_generic(vla.language_model, n_obs=n_obs, n_act=1, obs_offset=0)
```

### LLaVA-VLA 0.5B

LLaVA-OneVision 把一个 `<image>` token 展开成 729 行的一块;适配器会找到这一块,断言图像 token 恰好只有一个,
再以该块起点作为 `obs_offset` 安装。动作以文本 token 发出,所以用 `AWARE_ACT_POS=last`。接线写在评测服务端
(`llava/serve/flask_server_qwen.py`),因为模型是握在那里的。

---

## 每个新骨干都必须过的闸门

三项检查,按此顺序。**没过这三关的骨干就是没接对,无论成功率多好看。**

1. **恒等。** `AWARE_ALPHA=0` 必须逐值复现未干预的策略。
2. **布局自证。** 断言你声明的那些行确实是观测块和动作行,例如把图像 token 数与 processor 对一遍、
   把动作读取所在的行号打印出来。现有两个自回归适配器都这么做,不符就拒绝运行。
3. **有效。** 在发布版开关下,`n_inject` 必须占前向次数的很大一部分,并且输出必须**确实**与未干预的策略不同。
   日志里的 `installed` 和 `ACTIVE` 本身什么都证明不了。

`analysis/sv_aware_gate.py` 是这三关在 SpatialVLA 上的完整示例。
