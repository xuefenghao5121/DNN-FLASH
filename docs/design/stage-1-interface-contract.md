# Stage 1 Interface Contract

> 状态：Draft v0.1
> 日期：2026-06-27
> 适用范围：Stage 1.0 ~ Stage 1.4
> 依赖：`docs/system-design.md`、`docs/system-design-freeze-checklist.md`、`docs/stage-1-postops-validation-plan.md`

---

## 0. 目的

本文档冻结 Stage 1 的最小接口契约，避免实现阶段把决策散落到 Python wrapper、TensorFlow op、oneDNN kernel 或 benchmark 脚本中。

Stage 1 只验证：

```text
score_mod semantics
  -> ScoreModPlan
  -> RuntimePlan
  -> oneDNN matmul post-ops capability
  -> QK score tile path
  -> fallback/debug/benchmark observability
```

---

## 1. Public vs Internal 边界

### 1.1 Public headers

Stage 1 允许进入 public header 的内容：

```text
include/onednn_flash/runtime_plan.hpp
include/onednn_flash/score_mod_plan.hpp
```

只放稳定语义结构：

- enum class。
- POD-like config struct。
- plan input/output struct。
- debug/fallback string helpers。

禁止 public header 暴露：

- oneDNN C++ 类型。
- TensorFlow 类型。
- primitive cache 实现。
- ukernel/BRGEMM transform 细节。
- benchmark-only 字段。

### 1.2 Internal headers

Stage 1 内部实现可以使用：

```text
src/onednn_flash/runtime_plan_internal.hpp
src/onednn_flash/onednn_postops_internal.hpp
src/onednn_flash/onednn_capability_internal.hpp
```

可包含：

- oneDNN primitive/post-op 适配。
- kernel cache key。
- capability probe helper。
- debug serialization helper。

### 1.3 TensorFlow/Python 边界

- Python wrapper 只构造用户语义和 policy。
- TensorFlow Custom Op 只把 attrs/input 转成 RuntimePlanInput。
- backend/layout/post_ops 最终决策必须来自 RuntimePlanner。

---

## 2. Core enums

### 2.1 DataType

```cpp
enum class DataType {
  F32,
  BF16,        // reserved, not Stage 1 active path
  Unsupported,
};
```

Stage 1 active path: `F32` only.

### 2.2 ScoreModKind

```cpp
enum class ScoreModKind {
  None,
  Scale,
  AdditiveBias,
  ScaleAdditiveBias,
  Unsupported,
};
```

### 2.3 BiasKind

```cpp
enum class BiasKind {
  None,
  SameShapeTile,
  BroadcastRow,
  BroadcastCol,
  AlibiGenerated,     // reserved, not Stage 1 required
  Unsupported,
};
```

Stage 1 required path: `None`, `SameShapeTile`.

### 2.4 BlockMaskKind

```cpp
enum class BlockMaskKind {
  None,
  Causal,
  Unsupported,
};
```

### 2.5 QkBackendKind

```cpp
enum class QkBackendKind {
  Reference,
  OneDnnMatmul,
  OneDnnBrgemmBaseline,   // baseline only, not Stage 1 main path
};
```

### 2.6 QkLayoutKind

```cpp
enum class QkLayoutKind {
  RowMajorKStrided,
  CopiedTransposedK,
  BrgemmTransformedKBaseline,
};
```

No new layout enum may be added in Stage 1 without updating the design documents.

### 2.7 LoweringStatus

```cpp
enum class LoweringStatus {
  LoweredToOneDnnPostOps,
  PartiallyLowered,
  OneDNN-FlashEpilogue,
  ReferenceFallback,
  Unsupported,
};
```

---

## 3. FallbackReason

Stage 1 uses a closed initial enum:

```cpp
enum class FallbackReason {
  None,
  OneDnnDisabled,
  OneDnnUnavailable,
  UnsupportedDType,
  UnsupportedShape,
  UnsupportedScoreMod,
  UnsupportedBlockMask,
  UnsupportedPostOp,
  UnsupportedBiasLayout,
  UnsupportedBroadcast,
  UnsupportedEdgeTile,
  NumericalSafetyFallback,
  DebugForcedReference,
};
```

Rules:

- `FallbackReason::None` means the selected plan is the intended fast path.
- Any non-None reason must be visible in debug metadata and benchmark JSON.
- Multiple reasons may be stored as a vector internally, but the public Stage 1 plan exposes one primary reason plus optional debug details.

---

## 4. ScoreModPlan

```cpp
struct ScoreModPlan {
  ScoreModKind kind;
  DataType dtype;

  // Valid for Scale and ScaleAdditiveBias.
  bool has_scale;
  float scale_value;

  // Valid for AdditiveBias and ScaleAdditiveBias.
  BiasKind bias_kind;
  bool has_bias;

  LoweringStatus lowering_status;
  FallbackReason fallback_reason;

  // Stable string used by cache/report/debug.
  std::string signature;
};
```

### 4.1 Stage 1 supported signatures

```text
none
scale:f32
additive_bias:same_shape_tile:f32
scale_additive_bias:same_shape_tile:f32
unsupported:<reason>
```

### 4.2 Lowering rules

| Input | oneDNN support | LoweringStatus | Fallback |
|---|---|---|---|
| none | N/A | LoweredToOneDnnPostOps | None |
| scale | supported | LoweredToOneDnnPostOps | None |
| scale | unsupported | OneDNN-FlashEpilogue | UnsupportedPostOp |
| additive same-shape | supported | LoweredToOneDnnPostOps | None |
| additive same-shape | unsupported | OneDNN-FlashEpilogue | UnsupportedPostOp |
| broadcast bias | not Stage 1 | OneDNN-FlashEpilogue | UnsupportedBroadcast |
| callback/custom | not Stage 1 | ReferenceFallback | UnsupportedScoreMod |

---

## 5. BlockMaskPlan

```cpp
struct BlockMaskPlan {
  BlockMaskKind kind;
  bool has_boundary_tiles;
  bool can_skip_tiles;
  bool requires_mask_tile_generator;
  FallbackReason fallback_reason;
  std::string signature;
};
```

Stage 1 rules:

- `None`: dense path.
- `Causal`: tile skip/dense/boundary allowed.
- causal boundary tile mask is handled by OneDNN-Flash epilogue, not oneDNN post-op.

### 5.1 Causal boundary tile mask data structure

Stage 1 uses an additive mask tile descriptor:

```cpp
struct MaskTileDescriptor {
  bool required;
  int q_start;
  int k_start;
  int q_size;
  int k_size;
  float masked_value;   // recommended: -infinity or sufficiently negative finite value
  bool all_masked_row_possible;
};
```

Implementation may materialize a temporary same-shape mask tile internally, but this materialization is not part of the public API.

All-masked row behavior:

```text
output[row, :] = 0
softmax_l[row] = 0
debug.all_masked_row = true
```

---

## 6. RuntimePlanInput

```cpp
struct RuntimePlanInput {
  int batch;
  int heads;
  int query_length;
  int key_length;
  int head_dim;
  int value_dim;

  DataType q_dtype;
  DataType k_dtype;
  DataType v_dtype;
  DataType out_dtype;

  ScoreModKind requested_score_mod;
  bool has_scale;
  float scale_value;
  BiasKind requested_bias_kind;

  BlockMaskKind requested_block_mask;

  // Policies. Auto is represented by default values.
  bool force_reference;
  bool enable_onednn;
  bool enable_debug;
};
```

No TensorFlow or oneDNN type is allowed in `RuntimePlanInput`.

---

## 7. RuntimePlan

```cpp
struct RuntimePlan {
  RuntimePlanInput input;

  ScoreModPlan score_mod_plan;
  BlockMaskPlan block_mask_plan;

  QkBackendKind qk_backend;
  QkLayoutKind qk_layout;
  LoweringStatus qk_lowering_status;

  bool uses_onednn_post_ops;
  bool requires_onednn_flash_epilogue;
  bool requires_mask_tile_generator;

  FallbackReason fallback_reason;

  std::string runtime_plan_signature;
  std::string debug_summary;
};
```

Rules:

- `RuntimePlan` is the only source of backend/layout/post-op decision.
- TensorFlow op must not override `qk_backend` after plan creation.
- oneDNN Integration may reject a requested plan only by returning a capability/fallback result, not by silently choosing a different semantic path.

---

## 8. Cache keys

### 8.1 Public semantic key

```cpp
struct RuntimePlanCacheKey {
  std::string semantic_shape;
  std::string dtype_signature;
  std::string score_mod_signature;
  std::string block_mask_signature;
  std::string policy_signature;
};
```

Allowed in public header.

### 8.2 Internal kernel key

```cpp
struct OneDnnKernelCacheKey {
  std::string op_kind;
  std::string tile_shape;
  std::string dtype_signature;
  std::string layout_signature;
  std::string post_ops_signature;
  std::string isa_signature;
  std::string onednn_version;
};
```

Internal only. Must not include B/H unless primitive requires it.

### 8.3 Internal transform key

```cpp
struct TransformCacheKey {
  std::string source_layout;
  std::string packed_layout;
  std::string tile_shape;
  std::string dtype;
  std::string pack_type;
  std::string isa_signature;
};
```

Internal only.

---

## 9. Capability probe contract

Probe test file:

```text
tests/cpp/test_onednn_postops_capability.cpp
```

Optional probe binary name:

```text
onednn_flash_onednn_postops_capability
```

Report output:

```text
docs/backend/onednn-postops-capability-2026-06-27.md
```

Machine-readable output:

```text
benchmarks/results/stage-1-postops/onednn-postops-capability.json
```

Required probe rows:

```text
matmul.scale
matmul.binary_add_same_shape
matmul.binary_add_broadcast_row
matmul.binary_add_broadcast_col
matmul.eltwise_linear
ukernel_brgemm.post_ops_boundary
ukernel_transform.supported_out_ld
```

Each row records:

```json
{
  "name": "matmul.scale",
  "supported": true,
  "method": "primitive_attr|post_op|runtime_arg|unsupported",
  "compile_status": "passed|failed|not_tested",
  "run_status": "passed|failed|not_tested",
  "reason": null
}
```

---

## 10. Benchmark/report contract

Benchmark output directory:

```text
benchmarks/results/stage-1-postops/
```

Stage report path:

```text
docs/reports/stage-1-postops-validation-report.md
```

Required benchmark files:

```text
benchmarks/results/stage-1-postops/cpp-qk-postops.json
benchmarks/results/stage-1-postops/cpp-qk-postops.csv
benchmarks/results/stage-1-postops/tf-eager-qk-postops.json
benchmarks/results/stage-1-postops/tf-eager-qk-postops.csv
```

Required report sections:

```text
1. Environment
2. Capability Probe Results
3. Correctness Results
4. Performance Results
5. Fallback Summary
6. Interpretation Boundary
7. Next-stage Recommendation
```

Interpretation boundary is mandatory: C++ microbenchmark cannot be used to claim TensorFlow graph/XLA system speedup.

---

## 11. Minimal tests

Stage 1.0 tests:

```text
tests/cpp/test_runtime_plan.cpp
tests/cpp/test_score_mod_plan.cpp
```

Stage 1.1 tests:

```text
tests/cpp/test_onednn_postops_capability.cpp
```

Stage 1.2/1.3 tests:

```text
tests/cpp/test_qk_postops.cpp
```

TensorFlow smoke remains isolated if it loads a shared library registering `OneDNN-FlashAttention`.

---

## 12. Freeze status

This contract closes the design questions for:

- RuntimePlan minimum fields.
- ScoreModPlan minimum fields.
- FallbackReason initial enum.
- capability probe output path.
- benchmark/report output path.
- cache key public/internal boundary.
- causal boundary tile mask descriptor.

Implementation can start only after `docs/system-design-freeze-checklist.md` is updated to reference this contract and the user confirms Stage 1.0 implementation may begin.
