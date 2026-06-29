# OneDNN-Flash 系统级设计方案

> 状态：Draft v0.1
> 日期：2026-06-27
> 作者：小西 / OpenClaw
> 项目：OneDNN-Flash = Flash Attention on oneDNN
> 适用范围：后续实现必须先对齐本文档，再进入模块编码与性能优化。

---

## 0. 设计原则

OneDNN-Flash 的核心原则不是“自己手写一个 attention kernel”，也不是“把若干 GEMM 调用替换成 oneDNN”。

OneDNN-Flash 的核心原则是：

> 以 TensorFlow/XLA 保留图级语义，以 OneDNN-Flash 描述 FlexAttention 执行计划，以 oneDNN 充分发挥 JIT、BRGEMM、post-ops、primitive cache、AMX/SVE 后端能力，形成 CPU 端可融合、可调度、可扩展的 FlexAttention/FlexInteraction 系统。

因此后续开发必须遵守以下约束：

1. **系统方案先行**：任何 kernel 优化前，必须明确它属于哪一层、服务哪个设计假设、如何验证。
2. **oneDNN-native 优先**：优先使用 oneDNN JIT/BRGEMM/post-ops/transform/cache；只有在 oneDNN 无法表达时，才用 OneDNN-Flash 自有代码补齐。
3. **不 materialize N² 中间矩阵**：完整 score/probability 矩阵不得作为默认路径落内存。
4. **score_mod/BlockMask 是一等公民**：不能只优化 dense causal attention，必须保留 FlexAttention 扩展点。
5. **XLA/TF 集成是主线，不是附属 benchmark**：eager Custom Op 只是验证路径，最终目标是 graph/XLA 级融合。
6. **每个阶段必须可验证**： correctness、性能、fallback、限制都要有文档和测试。

---

## 1. 项目定位与目标

### 1.1 项目定位

OneDNN-Flash 是 CPU 端 FlexAttention 执行框架，目标是填补 GPU FlexAttention/FlashAttention 在 CPU 侧的空白。

Phase 1 聚焦 FlexAttention：

```text
O = attention(Q, K, V, score_mod, block_mask)
```

Phase 2 扩展为 FlexInteraction：

```text
aggregate(transform(interact(A, B)), C)
```

用于推荐系统特征交叉、DLRM interaction、DeepFM FM、DSSM matching 等场景。

### 1.2 技术目标

| 目标 | 说明 |
|---|---|
| 图级融合 | XLA 保持 attention/flex interaction pattern 不被拆散 |
| oneDNN JIT 执行 | 内层计算尽量由 oneDNN JIT/BRGEMM/post-ops 承接 |
| Flash tiling | 避免完整 `N x N` score/probability 矩阵 |
| score_mod 下沉 | scale/bias/ALiBi/mask 等尽量进入 oneDNN post-ops 或 fused epilogue |
| BlockMask tile skip | 块级跳过无效 tile，降低实际 FLOPs |
| 多 ISA | x64 AMX/AVX-512，后续 ARM SVE/鲲鹏930 |
| 可回退 | oneDNN/XLA 不可用时保留 reference/TF Custom Op 路径 |

### 1.3 非目标

短期不做：

- 训练反向传播。
- GPU 后端。
- 任意 Python lambda 的运行时解释。
- 无边界地手写自有 JIT 替代 oneDNN。
- 在没有系统验证的情况下追逐单个 microbenchmark 指标。

---

## 2. 总体架构

### 2.1 分层架构

```text
┌─────────────────────────────────────────────────────────────┐
│ Python / TensorFlow API                                     │
│ onednn_flash_attention(q, k, v, score_mod, block_mask, options) │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Graph Capture / XLA Lowering Layer                           │
│ - pattern recognition                                        │
│ - score_mod lowering                                         │
│ - BlockMask lowering                                         │
│ - Custom Call emission                                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ OneDNN-Flash Runtime Planner                                     │
│ - shape/dtype/ISA dispatch                                   │
│ - tile policy                                                │
│ - backend selection                                          │
│ - primitive/kernel cache key                                 │
│ - fallback selection                                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Attention Execution Engine                                   │
│ - Q tile scheduler                                           │
│ - K/V tile scheduler                                         │
│ - BlockMask tile skip                                        │
│ - online softmax state                                       │
│ - output accumulation                                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ oneDNN Integration Layer                                     │
│ - matmul primitive + post-ops                                │
│ - ukernel BRGEMM                                             │
│ - transform/pack                                             │
│ - primitive cache                                            │
│ - threadpool/stream/memory wrapper management                │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ Hardware Backend                                             │
│ - AMX / AVX-512 / AVX2                                       │
│ - ARM SVE                                                    │
│ - cache/memory hierarchy                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 当前工程与目标架构的关系

当前仓库已有：

- standalone C++ attention core。
- Python/Numpy golden。
- backend seam。
- score bias / BlockMask 原型。
- oneDNN matmul tile backend。
- oneDNN BRGEMM ukernel 实验路径。
- TensorFlow CPU Custom Op MVP。
- TensorFlow E2E benchmark。

但目标架构仍缺：

- XLA lowering / Custom Call。
- score_mod 到 oneDNN post-ops 的系统映射。
- BlockMask tile scheduler 的完整设计。
- bf16 + AMX 主路径。
- oneDNN post-ops 能力边界验证。
- runtime planner/cache key 的稳定抽象。

---

## 3. 数据流设计

### 3.1 Dense causal attention 数据流

```text
Input:
  Q[B,H,M,D]
  K[B,H,N,D]
  V[B,H,N,Dv]

For each batch/head:
  For each Q tile Q_i:
    init online state: m, l, acc

    For each K/V tile K_j,V_j:
      if BlockMask says skip(Q_i, K_j):
        continue

      S_ij = Q_i @ K_j^T
      S_ij = score_mod(S_ij, i, j)
      S_ij = apply dense/boundary mask(S_ij)
      update online softmax state(m, l, acc, S_ij, V_j)

    O_i = acc / l
```

### 3.2 oneDNN-native 目标数据流

```text
Q/K tile
  -> oneDNN BRGEMM/matmul JIT
       post_ops: scale + additive_bias + simple score_mod + mask injection
  -> OneDNN-Flash online softmax recurrence
  -> oneDNN BRGEMM/matmul PV accumulation
  -> output tile
```

重要说明：

- `scale/additive_bias/部分 score_mod` 应优先进入 oneDNN post-ops。
- `online softmax recurrence` 由于跨 K-block 维护 row max/sum，短期仍由 OneDNN-Flash execution engine 管理。
- `BlockMask tile skip` 在 scheduler 层做，不应等到 post-op 才 mask。
- `boundary mask` 可作为 post-op/additive mask 或 OneDNN-Flash fallback 处理。

---

## 4. 模块设计

### 4.1 Python / TensorFlow API 模块

#### 职责

- 提供用户入口。
- 选择 eager Custom Op / graph XLA 路径。
- 将用户参数标准化成 OneDNN-Flash runtime options。

#### 初始 API

```python
def onednn_flash_attention(
    q,
    k,
    v,
    *,
    causal: bool = True,
    score_mod=None,
    block_mask=None,
    query_block_size=None,
    key_block_size=None,
    tile_kernel="auto",
    qk_tile_layout="auto",
    dtype_policy="auto",
):
    ...
```

#### 设计要求

- API 不暴露底层 oneDNN 细节。
- `score_mod` 先支持枚举/结构化形式，不立即支持任意 Python callback。
- `block_mask` 使用结构化描述，方便 lowering。
- 所有 auto 策略必须可打印/可记录，便于 benchmark 复现。

---

### 4.2 ScoreMod 表达与 lowering 模块

#### 职责

把用户侧 score modification 表达成后端可执行的 epilogue/post-op 计划。

#### 支持等级

| 等级 | 类型 | 示例 | 后端策略 |
|---|---|---|---|
| S0 | scale | `score * scale` | oneDNN attr/output scale 或 post-op |
| S1 | additive bias | `score + bias[i,j]` | binary/add post-op 或预生成 tile bias |
| S2 | ALiBi | `score + slope[h]*(i-j)` | structured bias generator + post-op/fallback |
| S3 | simple unary | tanh/sigmoid/relu | oneDNN eltwise post-op |
| S4 | arbitrary function | Python callback | reference/fallback，不作为性能主路径 |

#### 初始交付

- `ScaleScoreMod`
- `AdditiveBiasScoreMod`
- `AlibiScoreMod` 设计，不一定第一阶段全实现。
- `ScoreModPlan`：记录哪些部分下沉 oneDNN，哪些部分 fallback。

#### 关键约束

- 不能把 `score_mod` 直接固化成 C++ callback 后就算完成。
- 必须能回答：这个 score_mod 是否可被 oneDNN post-ops 表达？如果不能，fallback 代价是什么？

---

### 4.3 BlockMask 模块

#### 职责

表达并执行 block-level sparse attention。

#### 数据结构

```text
BlockMask
  - block_m
  - block_n
  - layout: dense / causal / sliding_window / prefix / custom_sparse
  - block predicate: may_compute(q_block, k_block)
  - boundary predicate: valid(q_index, k_index)
```

#### 执行层级

1. **tile skip**：整个 tile 不参与 QK/PV。
2. **boundary mask**：causal/sliding window 的边界 tile 内做元素级 mask。
3. **score mask**：最终转成 additive `-inf` 或 predicate。

#### 初始交付

- CausalBlockMask。
- SlidingWindowBlockMask。
- benchmark 展示 skip tile 数量、实际 FLOPs 降低、延迟收益。

---

### 4.4 Runtime Planner 模块

#### 职责

根据 shape、dtype、ISA、score_mod、mask、构建选定执行计划。

#### 输入

```text
RuntimePlanInput:
  B,H,M,N,D,Dv
  dtype
  causal / block_mask
  score_mod signature
  hardware capability
  user options
```

#### 输出

```text
RuntimePlan:
  q_block_size
  k_block_size
  tile traversal order
  qk_kernel_kind
  pv_kernel_kind
  score_mod_plan
  block_mask_plan
  workspace_size
  cache_key
  fallback_reason(optional)
```

#### 设计要求

- `auto` 策略必须稳定可复现。
- cache key 必须包含 shape、dtype、ISA、score_mod signature、mask type、tile sizes、backend kind。
- planner 不直接执行 kernel，只生成 plan。

---

### 4.5 Attention Execution Engine 模块

#### 职责

执行 Flash tiling 和 online softmax recurrence。

#### 子模块

| 子模块 | 职责 |
|---|---|
| TileScheduler | 遍历 Q/K/V tiles，调用 BlockMask skip |
| QKExecutor | 调用 oneDNN/ref 计算 score tile |
| ScoreEpilogue | 执行未下沉 oneDNN 的 score_mod/mask |
| OnlineSoftmax | 维护 row max/sum/acc |
| PVExecutor | 调用 oneDNN/ref 完成 value accumulation |
| WorkspaceManager | 管理 tile buffers、scratchpad、transform buffers |

#### 设计要求

- 不能在 K-loop 内做 heap allocation。
- Q/K/V/O 尽量 zero-copy 使用框架传入指针。
- 所有临时 buffer 由 workspace 预分配。
- attention engine 不应依赖 TensorFlow 类型，只依赖 raw pointer + shape + plan。

---

### 4.6 oneDNN Integration 模块

#### 职责

封装所有 oneDNN 调用，并隐藏 primitive/ukernel 差异。

#### 子模块

| 子模块 | 职责 |
|---|---|
| OneDnnMatmulKernel | 使用 `dnnl::matmul` primitive，支持 post-ops |
| OneDnnBrgemmKernel | 使用 `dnnl::ukernel::brgemm`，低开销 tile GEMM |
| OneDnnTransformKernel | B packing/transpose/materialization |
| OneDnnPostOpsBuilder | 根据 ScoreModPlan 构造 primitive_attr/post_ops |
| OneDnnPrimitiveCache | 缓存 primitive/JIT/ukernel |
| OneDnnThreadingAdapter | stream/threadpool/TF intra-op 适配 |

#### matmul primitive vs ukernel BRGEMM 取舍

| 路径 | 优点 | 缺点 | 使用场景 |
|---|---|---|---|
| `dnnl::matmul` | post-ops 支持成熟，语义完整 | primitive/memory wrapper 开销更大 | score_mod/post-ops 验证、复杂 epilogue |
| `ukernel::brgemm` | 低开销，更接近 JIT microkernel | post-op/pack API 受限 | 稳定 shape、高性能核心路径 |
| reference | 简单、可验证 | 慢 | correctness fallback |

#### 关键设计原则

- 先用 `dnnl::matmul + post_ops` 验证 score_mod 下沉语义。
- 再评估迁移到 `ukernel::brgemm` 或混合路径。
- 不能为了 BRGEMM 牺牲系统语义；BRGEMM 是执行后端，不是架构中心。

---

### 4.7 TensorFlow Custom Op 模块

#### 职责

提供非 XLA 路径和测试/benchmark 入口。

#### 当前状态

已有 `OneDNN-FlashAttention` CPU op，支持：

- `causal`
- `query_block_size`
- `key_block_size`
- `use_onednn`
- `tile_kernel`
- `qk_tile_layout`

#### 后续设计

新增结构化 attrs：

```text
score_mod_type
score_mod_params
block_mask_type
block_mask_params
dtype_policy
backend_policy
```

#### 限制

- 不把 eager Custom Op 当最终性能目标。
- Custom Op 是 XLA Custom Call 前的工程验证层。

---

### 4.8 XLA Custom Call 模块

#### 职责

保持 graph-level attention pattern，避免被拆成普通 matmul/softmax/matmul。

#### 目标流程

```text
TensorFlow graph
  -> XLA HLO
  -> pattern/lowering pass identifies OneDNN-Flash attention
  -> emits CustomCall("onednn_flash_attention")
  -> OneDNN-Flash runtime planner
  -> oneDNN execution
```

#### 最小闭环

1. 固定 shape/f32/causal。
2. 无 score_mod 或只支持 scale。
3. XLA HLO 中出现 OneDNN-Flash CustomCall。
4. runtime 执行结果与 TF reference 一致。
5. benchmark 对比 eager Custom Op 和原生 TF graph。

#### 成功标准

- attention 语义在 HLO 中没有被拆散为 materialized `N x N` score path。
- CustomCall 能接入 OneDNN-Flash runtime planner。
- 至少一个固定 shape 正确运行。

---

## 5. 后端选择策略

### 5.1 初始 backend policy

```text
if XLA CustomCall available and supported(score_mod, block_mask, dtype):
    use xla_custom_call
elif TensorFlow Custom Op available:
    use tf_custom_op
else:
    use reference/python fallback
```

### 5.2 tile kernel policy

```text
if score_mod requires post_ops not available in ukernel:
    QK uses dnnl::matmul + post_ops
elif shape stable and BRGEMM supported:
    QK uses ukernel::brgemm
else:
    QK uses reference/oneDNN matmul fallback

PV:
    prefer ukernel::brgemm or dnnl::matmul depending shape/cache overhead
```

### 5.3 fallback 必须可观测

每次 fallback 需要能记录：

```text
fallback_reason:
  - unsupported dtype
  - unsupported score_mod
  - unsupported mask
  - oneDNN unavailable
  - XLA unavailable
  - ukernel pack unsupported
```

---

## 6. 开发阶段与交付物

### Stage 0：系统设计冻结

交付：

- 本文档。
- 模块边界图。
- 开发流程与验收标准。
- 当前代码与目标架构的差距清单。

禁止事项：

- 不再直接追加 microkernel 优化。
- 不再无设计地新增 attr/enum。

---

### Stage 1：oneDNN post-ops 能力验证

目标：验证最初原则中“充分发挥 post-ops”的可行性。

交付：

- `OneDnnPostOpsBuilder` 设计与实现。
- QK tile 支持 `scale + additive_bias` post-op。
- correctness tests：对比 reference score path。
- benchmark：post-op path vs OneDNN-Flash C++ 后处理。
- 文档：哪些 score_mod 可下沉，哪些不可。

验收：

- C++ tests pass。
- Python/TF tests pass。
- 至少一个 benchmark 显示 post-op path 不劣于 C++ 后处理，或明确解释原因。

---

### Stage 2：ScoreModPlan + BlockMaskPlan

目标：把 score_mod 和 mask 从 ad-hoc 参数变成系统 plan。

交付：

- `ScoreModPlan`。
- `BlockMaskPlan`。
- causal/sliding window tile skip。
- fallback reason 机制。
- benchmark 输出 skip tile 数量。

---

### Stage 3：Runtime Planner

目标：统一 shape/dtype/backend/tile/cache key 策略。

交付：

- `RuntimePlan`。
- `PlanCacheKey`。
- auto tile policy。
- backend selection policy。
- plan dump/debug string。

---

### Stage 4：XLA Custom Call 最小闭环

目标：证明 OneDNN-Flash 不是 eager Custom Op，而是可 graph-level 集成。

交付：

- fixed-shape XLA CustomCall prototype。
- HLO dump。
- correctness test。
- eager vs graph benchmark。

---

### Stage 5：bf16 + AMX 主路径

目标：回到原始 TDR：bf16 + AMX 是性能主路径。

交付：

- bf16 Q/K/V/O。
- AMX capability detection。
- oneDNN bf16 primitive/BRGEMM path。
- perf counters / VTune 或 perf 报告。

---

## 7. 验证体系

### 7.1 correctness

每个模块必须至少有：

- reference 对照。
- deterministic random seed。
- max_abs_diff / max_rel_diff。
- causal/boundary cases。
- small shape + edge tile shape。

### 7.2 performance

benchmark 必须记录：

- shape：B/H/M/N/D/Dv。
- dtype。
- tile size。
- backend。
- score_mod/mask。
- warmup/repeat。
- median/p50/p90。
- fallback reason。
- oneDNN cache hit/miss。

### 7.3 integration

每轮交付至少跑：

```bash
cmake --build <build-dir> -j
ctest --test-dir <build-dir> --output-on-failure
PYTHONPATH=$PWD/python:$PWD python3 -m pytest -q tests/python tests/tensorflow
```

XLA 阶段新增：

```bash
# dump HLO and check CustomCall
TF_XLA_FLAGS=... python3 tests/xla/...
```

---

## 8. 当前代码差距清单

| 领域 | 当前状态 | 差距 |
|---|---|---|
| Standalone core | 已有 | 需要按 RuntimePlan 重构入口 |
| oneDNN matmul | 已有 | post-ops builder 不完整 |
| oneDNN BRGEMM | 实验路径已有 | 与 score_mod/post-ops 融合关系未设计完 |
| transform/pack | 实验 transformed-K | 需要纳入 OneDnnTransformKernel 模块 |
| ScoreMod | 有 ScoreBiasFn/reference | 缺结构化 ScoreModPlan/lowering |
| BlockMask | 有原型 | 缺 tile scheduler 级完整 plan |
| TensorFlow Custom Op | 已有 MVP | attrs 需要系统化，不能继续散增 |
| XLA Custom Call | 未做 | 下一阶段主线之一 |
| bf16/AMX | 未做 | 原始目标主路径缺失 |
| 文档 | 有 project-plan/design | 需要以本文档作为系统设计基线 |

---

## 9. 近期执行建议

后续不要直接继续 `brgemm_transformed_k` shape sweep 作为主线，而是按以下顺序执行：

1. 冻结本设计文档并评审。
2. 新增 `docs/modules/`，拆出各模块设计。
3. 先做 `OneDnnPostOpsBuilder` 设计与 QK `scale + additive_bias` post-op 验证。
4. 再把现有 `qk_tile_layout`、BRGEMM、transform 路径收编到 RuntimePlan，而不是继续作为散落 enum。
5. 启动 XLA Custom Call 最小闭环设计。

---

## 10. 设计结论

OneDNN-Flash 的系统核心是：

```text
FlexAttention semantics
  -> structured score_mod / block_mask
  -> RuntimePlan
  -> oneDNN JIT/post-ops/BRGEMM execution
  -> XLA/TF graph integration
```

BRGEMM、transformed-K、scratchpad reuse 都是重要底层能力，但不是架构主线本身。

后续开发必须围绕“oneDNN JIT/post-ops 能承接多少 FlexAttention 语义”这个核心问题推进。
