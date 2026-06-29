# XLA Custom Call 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 设计目标

XLA Custom Call 是 OneDNN-Flash 区别于普通 TensorFlow Custom Op 的关键路径。

目标：

> 保持 FlexAttention pattern 不被 XLA 拆成 materialized `QK -> softmax -> PV`，而是作为整体 lowering 到 OneDNN-Flash Runtime。

---

## 2. 最小闭环范围

第一阶段只做：

- f32。
- fixed shape。
- causal or no mask。
- score_mod 只支持 scale。
- CPU backend。
- 输出与 TF reference 一致。

---

## 3. 目标流程

```text
TensorFlow function
  -> XLA HLO
  -> OneDNN-Flash CustomCall HLO
  -> XLA runtime calls onednn_flash_attention_entry
  -> RuntimePlanner
  -> AttentionExecutionEngine
  -> oneDNN backend
```

---

## 4. CustomCall ABI 草案

```text
onednn_flash_attention_entry(
  void* out,
  const void* q,
  const void* k,
  const void* v,
  const OneDNN-FlashOpaqueConfig* config,
  size_t config_size
)
```

Opaque config 包含：

```text
shape
strides
dtype
mask type
score_mod type
tile policy
backend policy
```

---

## 5. HLO 验证

必须能 dump HLO 并检查：

- 存在 `custom-call`。
- 不存在完整 `M x N` probability tensor 的长期 materialization。
- CustomCall opaque 中包含 OneDNN-Flash config。

---

## 6. 风险

| 风险 | 应对 |
|---|---|
| TF/XLA CustomCall ABI 复杂 | 先 fixed shape demo |
| 与 TF Custom Op 注册冲突 | 独立测试进程 |
| XLA 不易识别 pattern | 先显式调用 wrapper，后做 pattern pass |
| debug 困难 | 强制 HLO dump + config dump |

---

## 7. 第一阶段交付

1. XLA CustomCall 调研文档。
2. fixed-shape smoke prototype。
3. HLO dump 文件。
4. correctness test。
5. eager Custom Op vs XLA CustomCall benchmark。
