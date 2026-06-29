# oneDNN Integration 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 模块目标

oneDNN Integration 模块负责把 OneDNN-Flash 的执行计划映射到 oneDNN 的高性能执行能力：

- JIT primitive。
- matmul primitive。
- ukernel BRGEMM。
- post-ops。
- transform/pack。
- primitive/kernel cache。
- stream/threading/memory wrapper 管理。

该模块的目标不是简单封装 GEMM，而是最大化利用 oneDNN 的 JIT 和 post-ops 特性。

---

## 2. 模块边界

### 2.1 输入

```text
OneDnnKernelRequest:
  op_kind: qk | pv | transform
  shape: M,N,K,Dv,batch
  dtype: f32 | bf16
  layout: row_major | transposed | packed
  score_mod_plan(optional)
  post_ops_plan(optional)
  cache_key
  workspace pointers
```

### 2.2 输出

```text
OneDnnKernelResult:
  output tile pointer filled
  cache hit/miss metadata
  fallback reason(optional)
  execution status
```

### 2.3 不负责

- 不解析 Python score_mod。
- 不决定全局 tile traversal。
- 不直接依赖 TensorFlow Tensor 对象。
- 不维护 online softmax 全局状态。

---

## 3. 子模块

### 3.1 OneDnnPostOpsBuilder

职责：把 `ScoreModPlan` 中可下沉部分转成 oneDNN post-ops。

初始支持：

| ScoreMod | oneDNN 映射 | 备注 |
|---|---|---|
| scale | output scale / eltwise linear | 优先验证 |
| additive bias | binary add 或预生成 bias tile | 取决于 oneDNN API 支持 |
| unary tanh/sigmoid | eltwise post-op | P1 |
| ALiBi | structured bias tile + binary add | P1 |
| arbitrary callback | 不支持 | fallback |

输出：

```text
PostOpsBuildResult:
  supported_ops
  unsupported_ops
  primitive_attr
  fallback_required
```

### 3.2 OneDnnMatmulKernel

职责：使用 `dnnl::matmul` primitive 执行 QK/PV tile，优先用于 post-ops 语义验证。

设计重点：

- primitive cache keyed by shape + dtype + post_ops signature。
- memory wrapper 尽量复用。
- stream wait 粒度需要评估。
- 支持 QK score tile 输出到 workspace。

### 3.3 OneDnnBrgemmKernel

职责：使用 `dnnl::ukernel::brgemm` 执行低开销 tile GEMM。

当前已有实验能力：

- scratchpad reuse。
- K-split batch-reduce。
- delayed hw context release。
- transformed-K 实验路径。

需要系统化：

- 统一进入 RuntimePlan。
- 明确何时使用 BRGEMM，何时退回 matmul primitive。
- 若 post-ops 无法表达，不应强行选 BRGEMM。

### 3.4 OneDnnTransformKernel

职责：处理 B tile pack/transpose/materialization。

当前结论：

- oneDNN BRGEMM ukernel 无法直接表达 row-major K `[N,D]` 作为 `B=[D,N]` 的任意 strided transposed view。
- 当前 transformed-K 路径使用 `dnnl::ukernel::transform(pack_type::trans)`，当 out_ld 为 `16/32/48/64` 时可用。
- 其他边界 tile 使用 reference materialization fallback。

需要补齐：

- transform cache key。
- fallback 统计。
- shape sweep 评估 transform 成本。

### 3.5 OneDnnPrimitiveCache

职责：缓存 oneDNN primitive/ukernel/transform。

cache key 必须包含：

```text
op_kind
shape
dtype
ISA
layout
post_ops_signature
batch_size / reduce_block
threading mode
```

---

## 4. 执行路径选择

### 4.1 QK path

```text
if ScoreModPlan has post_ops that matmul supports:
    use OneDnnMatmulKernel + post_ops
elif BRGEMM supported and no required post_ops:
    use OneDnnBrgemmKernel
else:
    use reference/strided fallback
```

### 4.2 PV path

```text
if BRGEMM supports shape/layout:
    use OneDnnBrgemmKernel
else:
    use OneDnnMatmulKernel
```

### 4.3 Transform path

```text
if B layout directly consumable:
    no transform
elif oneDNN transform supports shape:
    use OneDnnTransformKernel
else:
    use reference materialization fallback
```

---

## 5. 第一阶段实现任务

### Task 1：PostOps capability probe

目标：确认当前 oneDNN 版本对 matmul post-ops 的支持边界。

输出：

- small C++ smoke。
- 文档记录：scale/additive bias/eltwise/binary 是否支持。

### Task 2：QK scale post-op

目标：把 `score *= scale` 从 OneDNN-Flash C++ 后处理下沉到 oneDNN post-op。

验收：

- correctness diff <= 当前 reference tolerance。
- C++ tests pass。
- benchmark 对比 C++ scale 后处理。

### Task 3：QK additive bias post-op

目标：支持 bias tile add。

验收：

- 支持 broadcast 或 same-shape bias tile。
- 至少一个 ALiBi-like structured bias 测试。

---

## 6. 验证标准

每个 oneDNN path 必须有：

1. Reference correctness test。
2. Unsupported shape fallback test。
3. Cache hit test 或 cache metadata dump。
4. Benchmark：median + p90。
5. 文档记录限制。

---

## 7. 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| ukernel post-op 能力弱 | BRGEMM 无法承接 score_mod | 用 matmul primitive 验证语义，BRGEMM 只做无 post-op 快路径 |
| transform 成本抵消收益 | transformed-K 不稳定 | shape sweep + planner 选择 |
| oneDNN API 版本差异 | 本地/目标机行为不同 | capability probe + compile-time/runtime gate |
| stream/memory wrapper 开销大 | TF eager 路径慢 | XLA Custom Call + wrapper reuse |

---

## 8. 当前结论

下一步不应继续直接扩大 BRGEMM 微优化，而应先实现 oneDNN post-ops 能力验证，尤其是 QK 的 `scale + additive_bias` 下沉。这是回到 OneDNN-Flash 最初设计原则的关键动作。

---

## 9. Capability Probe 输出格式

Stage 1 必须先生成 oneDNN capability probe 报告，再实现正式路径。

Probe 关注：

```text
matmul primitive:
  scale/output_scale support
  binary add post-op support
  eltwise linear/tanh support
  runtime args support
  same-shape bias support
  broadcast bias support

ukernel brgemm:
  post-op support boundary
  transform support boundary
  supported N/out_ld values
```

建议输出：

```json
{
  "onednn_version": "...",
  "primitive": "matmul",
  "dtype": "f32",
  "post_ops": {
    "scale": {"supported": true, "method": "..."},
    "binary_add_same_shape": {"supported": true, "method": "..."},
    "binary_add_broadcast": {"supported": false, "reason": "..."},
    "eltwise_tanh": {"supported": true, "method": "..."}
  },
  "notes": []
}
```

### 9.1 Stage 1 边界

Stage 1 的 oneDNN 主路径只承诺 `dnnl::matmul + post_ops`。

BRGEMM 在 Stage 1 中只作为：

- 已有性能 baseline。
- layout/transform 能力参考。
- 后续 Stage 的候选快路径。

不能为了 BRGEMM 局部性能收益改变 ScoreModPlan/BlockMaskPlan 的系统边界。
