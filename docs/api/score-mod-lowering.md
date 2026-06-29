# ScoreMod Lowering 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 设计目标

ScoreMod Lowering 模块负责把用户侧 FlexAttention 的 `score_mod` 语义转换成 FlashOne Runtime 与 oneDNN 后端可执行的结构化计划。

核心目标：

> 能下沉到 oneDNN JIT/post-ops 的 score_mod 必须下沉；不能下沉的部分必须显式标记 fallback，不允许隐藏在普通 C++ callback 中假装完成。

---

## 2. 输入与输出

### 2.1 输入

```text
ScoreModSpec:
  type: none | scale | additive_bias | alibi | composed | callback
  dtype: f32 | bf16
  shape context:
    B,H,M,N,D,Dv
  parameters:
    scale
    bias tensor descriptor
    alibi slopes descriptor
    composition list
```

### 2.2 输出

```text
ScoreModPlan:
  lowered_ops:
    - op kind
    - backend target: onednn_post_ops | flashone_epilogue | reference_callback
    - tensor/broadcast metadata
  post_ops_signature
  requires_score_tile_materialization: bool
  fallback_required: bool
  fallback_reason: optional string
  numeric_policy:
    accumulation_dtype
    exp_dtype
    mask_value
```

---

## 3. score_mod 分类

| 等级 | 类型 | 示例 | 目标后端 | 优先级 |
|---|---|---|---|---|
| S0 | identity | `score` | no-op | P0 |
| S1 | scale | `score * c` | oneDNN post-op / attr | P0 |
| S2 | additive bias | `score + bias[i,j]` | oneDNN binary add / precomputed bias tile | P0 |
| S3 | ALiBi | `score + slope[h] * (i-j)` | generated bias tile + add | P1 |
| S4 | unary eltwise | `tanh(score)` | oneDNN eltwise post-op | P1 |
| S5 | composed | `scale + bias + mask` | mixed plan | P1 |
| S6 | arbitrary callback | Python/C++ callback | reference fallback | P2 |

---

## 4. Lowering 策略

### 4.1 scale

候选实现：

1. oneDNN primitive attr output scale。
2. oneDNN eltwise linear post-op。
3. FlashOne epilogue fallback。

选择规则：

```text
if oneDNN primitive path supports scale attr for current dtype/layout:
    lower to oneDNN
else:
    lower to FlashOne epilogue
```

验收：

- QK score tile 与 reference diff 在 tolerance 内。
- benchmark 比 C++ 后处理不差，或差距可解释。

### 4.2 additive bias

候选实现：

1. oneDNN binary add post-op，bias tile same-shape。
2. oneDNN binary add post-op，支持 broadcast。
3. FlashOne epilogue add fallback。

关键问题：

- 当前 oneDNN 版本是否支持 matmul binary post-op。
- bias memory descriptor 是否能表达当前 tile/broadcast。
- ukernel BRGEMM 是否支持同等 post-op；如果不支持，QK 可能先走 `dnnl::matmul`。

### 4.3 ALiBi

ALiBi 本质是结构化 additive bias：

```text
bias[h,i,j] = slope[h] * (i - j)
```

Lowering 方案：

```text
ALiBiSpec
  -> BiasTileGenerator
  -> additive_bias plan
  -> oneDNN binary add or FlashOne epilogue
```

初期不追求零 materialization；先验证语义链路。

### 4.4 mask

mask 不直接作为 ScoreMod 的唯一模块，但 causal/boundary mask 会影响 score epilogue。

策略：

- 整块无效：由 BlockMaskPlan tile skip。
- 边界 tile：转成 additive mask value，例如 `-inf` 或大负数。
- 若 oneDNN post-op 不适合表达，则 FlashOne epilogue fallback。

---

## 5. 与 oneDNN post-ops 的关系

ScoreMod Lowering 不直接调用 oneDNN API，而是生成 `PostOpsPlan`：

```text
PostOpsPlan:
  ops:
    - scale
    - binary_add bias
    - eltwise tanh
  required_external_tensors:
    - bias tile
  unsupported_ops:
    - callback
```

`OneDnnPostOpsBuilder` 再把它转换成 oneDNN primitive attr/post_ops。

---

## 6. fallback 规则

fallback 必须显式可观测。

```text
FallbackReason:
  unsupported_score_mod_type
  unsupported_dtype
  unsupported_oneDNN_post_op
  unsupported_broadcast
  requires_python_callback
  numerical_policy_not_supported
```

禁止：

- 静默从 oneDNN post-op 退回 C++ callback。
- benchmark 只报告 fallback 后的性能，却标记为 oneDNN path。

---

## 7. 测试设计

### 7.1 correctness tests

| 测试 | shape | score_mod | 验证 |
|---|---|---|---|
| scale only | small/edge | `score * scale` | vs reference |
| additive bias | same-shape tile | `score + bias` | vs reference |
| causal boundary | partial tile | mask + scale | vs reference |
| ALiBi generated | multi-head | slope[h]*(i-j) | vs numpy |
| unsupported callback | any | callback | fallback reason |

### 7.2 benchmark

至少比较：

```text
QK + C++ scale/bias epilogue
QK + oneDNN post-ops scale/bias
QK + fallback callback
```

记录：

- shape。
- tile size。
- dtype。
- backend。
- post_ops_signature。
- fallback reason。

---

## 8. 第一阶段交付

Stage 1 只交付：

1. `ScoreModSpec` / `ScoreModPlan` 设计落地。
2. scale lowering。
3. additive bias lowering。
4. fallback reason。
5. correctness + benchmark。

ALiBi 与 composed score_mod 放到 Stage 2。
