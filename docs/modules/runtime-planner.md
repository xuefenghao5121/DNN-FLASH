# Runtime Planner 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 设计目标

Runtime Planner 是 FlashOne 的执行决策中心。它负责把输入 shape、dtype、score_mod、block_mask、硬件能力和用户选项转换为稳定可复现的执行计划。

核心目标：

> 所有 backend/tile/layout/fallback 决策必须集中在 Runtime Planner，不再散落在 Python wrapper、TensorFlow op、C++ attention loop、oneDNN kernel 中。

---

## 2. 输入

```text
RuntimePlanInput:
  shape:
    B,H,M,N,D,Dv
  dtype:
    input_dtype
    accumulation_dtype
    output_dtype
  mask:
    BlockMaskSpec
  score_mod:
    ScoreModSpec
  options:
    query_block_size optional
    key_block_size optional
    backend_policy: auto | reference | onednn_matmul | onednn_brgemm
    qk_layout_policy: auto | strided_k | copied_transposed | brgemm_transformed_k
    use_xla: bool
  hardware:
    has_onednn
    has_brgemm
    has_amx
    has_avx512
    has_sve
```

---

## 3. 输出

```text
RuntimePlan:
  shape
  tile_policy:
    q_block_size
    k_block_size
    traversal_order
  score_mod_plan
  block_mask_plan
  qk_kernel_plan:
    backend
    layout
    post_ops_plan
    transform_plan
  pv_kernel_plan:
    backend
    layout
  workspace_plan:
    score_tile_bytes
    probs_tile_bytes
    output_acc_bytes
    transform_bytes
    scratchpad_bytes
  cache_key
  fallback:
    required
    reason
  debug_string
```

---

## 4. 规划步骤

```text
1. Normalize shape/dtype/options
2. Build ScoreModPlan
3. Build BlockMaskPlan
4. Select tile sizes
5. Select QK backend/layout
6. Select PV backend/layout
7. Build workspace plan
8. Build cache key
9. Emit debug/fallback metadata
```

---

## 5. Tile 策略

初始策略：

```text
if user explicitly sets q/k block:
    use user values
else if M,N <= 64:
    q=64,k=64
else:
    q=32,k=64
```

后续策略应纳入：

- dtype。
- D/Dv。
- cache size。
- AMX tile shape。
- BlockMask sparsity。
- post-ops cost。

---

## 6. Backend 策略

### 6.1 QK backend

```text
if score_mod_plan has required oneDNN post_ops:
    prefer onednn_matmul
elif has_brgemm and layout supported:
    prefer onednn_brgemm
elif has_onednn:
    use onednn_matmul
else:
    reference
```

### 6.2 PV backend

```text
if has_brgemm and shape/layout supported:
    onednn_brgemm
elif has_onednn:
    onednn_matmul
else:
    reference
```

---

## 7. QK layout 策略

候选：

| Layout | 优点 | 缺点 |
|---|---|---|
| `strided_k` | zero-copy row-major K | oneDNN primitive 可能不如 contiguous B |
| `copied_transposed` | contiguous B | 每 tile copy/transpose |
| `brgemm_transformed_k` | 利用 oneDNN transform | 仍需 materialization，edge fallback |

选择原则：

- TensorFlow Custom Op 当前默认 `strided_k`。
- BRGEMM 若无法表达 strided transposed K，则使用 transform 或 copied fallback。
- 最终策略由 benchmark 数据和 capability probe 决定。

---

## 8. Cache Key

```text
PlanCacheKey:
  B,H,M,N,D,Dv
  dtype tuple
  q_block,k_block
  mask signature
  score_mod signature
  qk backend/layout/post_ops signature
  pv backend/layout
  ISA
  oneDNN version
```

cache key 必须能解释性能差异，不能只按 shape 缓存。

---

## 9. Debug/可观测性

每个 plan 应能 dump：

```text
FlashOne RuntimePlan:
  shape: B=...,H=...,M=...,N=...,D=...,Dv=...
  dtype: ...
  q_block/k_block: ...
  score_mod: ... lowered_to=...
  mask: ... skipped_tiles=...
  qk: backend=..., layout=..., post_ops=...
  pv: backend=..., layout=...
  workspace: ... bytes
  cache_key: ...
  fallback: none | reason
```

---

## 10. 第一阶段交付

1. `RuntimePlan` 数据结构。
2. plan builder。
3. debug string。
4. 将现有 `qk_tile_layout`、tile heuristic、backend selection 迁入 planner。
5. tests 覆盖 auto/user override/fallback。

---

## 11. Cache Key 分层边界

Runtime Planner 不应把所有缓存都混成一个 key。系统中至少存在四类 key：

### 11.1 RuntimePlanCacheKey

用于缓存完整执行计划。

```text
RuntimePlanCacheKey:
  semantic_shape: B,H,M,N,D,Dv
  dtype_policy
  score_mod_signature
  block_mask_signature
  user_policy_signature
  hardware_capability_signature
```

包含完整语义，因为它决定 tile、fallback、workspace、debug 输出。

### 11.2 OneDnnKernelCacheKey

用于缓存 oneDNN primitive / ukernel。

```text
OneDnnKernelCacheKey:
  op_kind: matmul | brgemm | transform
  tile_shape: M_tile,N_tile,K_tile
  dtype_tuple
  layout_signature
  post_ops_signature
  isa_signature
  oneDNN_version
```

不应包含无关的 batch/head 信息，除非 primitive 本身依赖这些维度。

### 11.3 TransformCacheKey

用于缓存 K/V transform kernel 或 packed layout 描述。

```text
TransformCacheKey:
  source_layout
  packed_layout
  tile_shape
  dtype
  pack_type
  isa_signature
```

### 11.4 BenchmarkConfigKey

用于性能报告归档。

```text
BenchmarkConfigKey:
  commit
  build_config
  runtime_plan_cache_key
  environment_signature
```

### 11.5 边界规则

- RuntimePlanner 可以引用 kernel cache key，但不直接构造 primitive。
- oneDNN Integration 可以构造 kernel cache key，但不能改变 score_mod/block_mask 语义。
- benchmark key 只用于报告，不参与运行时选择。
