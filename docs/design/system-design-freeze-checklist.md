# OneDNN-Flash System Design Freeze Checklist

> 状态：Stage 1.4 gate closed for review
> 日期：2026-06-28
> 目的：在进入实现前冻结系统设计边界，防止再次陷入局部 kernel/BRGEMM 优化。

---

## 0. 冻结原则

系统设计冻结不是说设计永不变，而是说：

1. 后续实现必须先落在清晰的系统边界内。
2. 任何突破边界的改动必须先回到设计文档评审。
3. Stage 1 的目标是验证系统链路，而不是追逐局部 benchmark。

当前系统基线：

```text
docs/system-design.md
```

Stage 1 执行基线：

```text
docs/stage-1-postops-validation-plan.md
```

自审与修正记录：

```text
docs/design-self-review-2026-06-27.md
```

Stage 1 接口契约：

```text
docs/stage-1-interface-contract.md
```

Stage 1 gate closure report:

```text
docs/reports/stage-1-gate-closure-2026-06-28.md
```

---

## 1. 文档冻结条件

进入实现前，以下文档必须存在且通过 `git diff --check`：

| 文档 | 状态 | 作用 | 冻结条件 |
|---|---|---|---|
| `docs/system-design.md` | Draft | 系统设计主基线 | 架构层级、主路径、非目标清晰 |
| `docs/modules/README.md` | Draft | 模块索引 | 标注各模块状态与依赖 |
| `docs/modules/python-tf-api.md` | Draft | Python/TF API 边界 | 用户语义参数、调度参数、debug 参数分层 |
| `docs/modules/score-mod-lowering.md` | Draft | score_mod lowering | 支持/不支持/降级路径清晰 |
| `docs/modules/block-mask-planner.md` | Draft | BlockMask 设计 | tile skip/dense/boundary 与 MaskTileGenerator 边界清晰 |
| `docs/modules/runtime-planner.md` | Draft | RuntimePlan 设计 | plan input/output/cache/fallback 边界清晰 |
| `docs/modules/attention-engine.md` | Draft | 执行引擎 | QK/softmax/PV/workspace 责任清晰 |
| `docs/modules/onednn-integration.md` | Draft | oneDNN 后端 | matmul/post-ops/BRGEMM/transform/cache 边界清晰 |
| `docs/modules/tensorflow-custom-op.md` | Draft | TF Custom Op | eager 验证路径与 ABI 限制清晰 |
| `docs/modules/xla-custom-call.md` | Draft | XLA 设计 | 明确 Stage 1 不实现，只保留兼容性 |
| `docs/modules/verification-benchmark.md` | Draft | 验证体系 | correctness/benchmark/fallback schema 清晰 |
| `docs/modules/development-flow.md` | Draft | 开发流程 | 设计->实现->验证->报告 gate 清晰 |
| `docs/stage-1-postops-validation-plan.md` | Draft | Stage 1 执行计划 | 任务拆分、非目标、文件边界、测试门禁清晰 |
| `docs/stage-1-interface-contract.md` | Draft | Stage 1 接口契约 | RuntimePlan/ScoreModPlan/FallbackReason/输出路径/cache 边界清晰 |
| `docs/design-self-review-2026-06-27.md` | Draft | 自审记录 | P0/P1/P2 问题和整改路径清晰 |

---

## 2. 必须关闭的 P0/P1 设计问题

### P0: 文档基线冲突

- [x] `docs/design.md` 已标注为 historical/MVP note。
- [x] `docs/project-plan.md` 已标注为 planning draft。
- [x] 明确冲突时以 `docs/system-design.md` 为准。

### P0: H2 技术假设过强

- [x] 已取消 “oneDNN BRGEMM + post-ops 表达 online softmax” 作为硬假设。
- [x] 已改为 oneDNN 承接 QK/PV 与部分 score_mod。
- [x] online softmax 跨 K-block recurrence 归 OneDNN-Flash Execution Engine 管理。

### P0: Stage 1 缺 Minimal RuntimePlan skeleton

- [x] `docs/stage-1-postops-validation-plan.md` 已加入 Stage 1.0。
- [x] `docs/stage-1-interface-contract.md` 已确认 `ScoreModPlan` / `RuntimePlan` 最小字段。
- [x] `docs/stage-1-interface-contract.md` 已确认 fallback/debug 输出字段。

### P1: oneDNN capability probe 未定义

- [x] `docs/modules/onednn-integration.md` 已加入 capability probe 输出格式。
- [x] `docs/stage-1-interface-contract.md` 已确认 probe 测试文件名和输出路径。

### P1: cache key 未分层

- [x] `docs/modules/runtime-planner.md` 已定义四类 key。
- [x] `docs/stage-1-interface-contract.md` 已确认 public/internal cache key 边界。

### P1: MaskTileGenerator 未定义

- [x] `docs/modules/block-mask-planner.md` 已定义职责边界。
- [x] `docs/stage-1-interface-contract.md` 已确认 Stage 1 causal boundary tile mask descriptor。

### P1: benchmark schema 未落地

- [x] `docs/modules/verification-benchmark.md` 已加入 JSON schema 草案。
- [x] `docs/stage-1-interface-contract.md` 已确认 benchmark 输出目录。
- [x] `docs/stage-1-interface-contract.md` 已确认 report template 路径。

### P1: TensorFlow attrs JSON 化风险

- [x] Stage 1 plan 已规定不大改 TF Custom Op ABI。
- [x] `docs/stage-1-interface-contract.md` 已确认仅允许内部接入 RuntimePlan/debug，Python wrapper 不成为 backend 决策中心。

---

## 3. Stage 1 冻结范围

Stage 1 允许做：

```text
ScoreModPlan / RuntimePlan minimal skeleton
oneDNN matmul post-ops capability probe
QK scale post-op validation
QK same-shape additive bias post-op validation
fallback/debug metadata
C++ correctness tests
TensorFlow eager smoke tests
benchmark JSON/CSV/report
```

Stage 1 已完成/落盘：

```text
include/onednn_flash/runtime_plan.hpp
include/onednn_flash/score_mod_plan.hpp
src/onednn_flash/runtime_plan.cpp
src/onednn_flash/score_mod_plan.cpp
tests/cpp/test_runtime_plan.cpp
tests/cpp/test_onednn_postops_capability.cpp
src/onednn_flash/qk_score_tile.cpp
src/onednn_flash/qk_score_tile_internal.hpp
tests/cpp/test_qk_score_tile.cpp
benchmarks/bench_qk_postops.cpp
benchmarks/results/stage-1-postops/onednn-postops-capability.json
benchmarks/results/stage-1-postops/cpp-qk-postops.json
benchmarks/results/stage-1-postops/cpp-qk-postops.csv
docs/backend/onednn-postops-capability-2026-06-27.md
docs/reports/stage-1-postops-validation-report.md
docs/reports/stage-1-gate-closure-2026-06-28.md
```

Stage 1 不允许做：

```text
XLA CustomCall implementation
full ALiBi/sliding-window/custom sparse
bf16/AMX main path
online softmax algorithm rewrite
PV accumulation rewrite
new BRGEMM transform strategy as main work
shape-by-shape BRGEMM sweep as main work
```

---

## 4. 模块边界冻结

### 4.1 Python/TF API

冻结边界：

- Python/TF API 只表达用户语义和可选 policy。
- 不在 wrapper 中硬编码 backend/layout 决策。
- debug 输出来自 RuntimePlan，不由 wrapper 自己拼。

### 4.2 ScoreMod Lowering

冻结边界：

- Stage 1 仅支持 `none`、`scale`、`additive_bias`、`scale + additive_bias`。
- 不能表达的 score_mod 必须 fallback 到 OneDNN-Flash epilogue/reference。
- callback/custom score_mod 不作为性能路径。

### 4.3 BlockMask Planner

冻结边界：

- Stage 1 仅支持 no mask / causal。
- causal boundary mask 由 OneDNN-Flash epilogue 处理。
- 不把 boundary mask 强塞进 oneDNN post-op。

### 4.4 Runtime Planner

冻结边界：

- RuntimePlanner 是 backend/layout/post_ops 决策中心。
- TensorFlow op、Python wrapper、oneDNN kernel 不允许绕过 RuntimePlanner 各自决策。
- plan cache 与 kernel cache 必须分层。

### 4.5 Attention Engine

冻结边界：

- Stage 1 只替换 QK score tile 生成和 score epilogue 前半部分。
- OnlineSoftmax/PV 主循环不大改。
- Workspace 生命周期不因局部优化而破坏。

### 4.6 oneDNN Integration

冻结边界：

- Stage 1 主路径是 `dnnl::matmul + post_ops`。
- BRGEMM 是 baseline/候选快路径，不是 Stage 1 架构中心。
- oneDNN 不可用时 reference path 必须可构建。

### 4.7 TensorFlow Custom Op

冻结边界：

- Stage 1 不做 ABI 大改。
- 只允许增加内部 RuntimePlan/debug 接入。
- 重复注册同名 op 的测试仍使用 isolated subprocess。

### 4.8 XLA CustomCall

冻结边界：

- Stage 1 不实现 XLA CustomCall。
- 只维护设计兼容性。
- Stage 2 前单独完成 XLA ABI/FFI 调研。

---

## 5. 允许进入实现的 Gate

只有以下条件都满足，才允许从设计进入实现：

- [x] `git diff --check` 通过。
- [x] 文档基线冲突已解决。
- [x] Stage 1 范围和非目标明确。
- [x] Stage 1.0 Minimal Plan Skeleton 字段确认。
- [x] oneDNN capability probe 输出路径确认。
- [x] benchmark JSON/report 输出路径确认。
- [x] fallback reason 枚举初稿确认。
- [x] 用户确认可以从设计阶段进入 Stage 1 实现阶段。

---

## 6. 实现前必须回答的问题

### RuntimePlan 最小字段

```text
shape
score_mod_plan
block_mask_plan
qk_backend_plan
qk_layout_plan
post_ops_plan
fallback_reason
debug_flags
```

### ScoreModPlan 最小字段

```text
kind: none | scale | additive_bias | scale_additive_bias | unsupported
scale_value(optional)
bias_kind(optional)
bias_layout(optional)
oneDNN_lowering_status
fallback_reason
```

### FallbackReason 初稿

```text
None
OneDnnDisabled
OneDnnUnavailable
UnsupportedDType
UnsupportedScoreMod
UnsupportedBlockMask
UnsupportedPostOp
UnsupportedBiasLayout
UnsupportedBroadcast
UnsupportedEdgeTile
NumericalSafetyFallback
DebugForcedReference
```

### 输出路径建议

```text
docs/backend/onednn-postops-capability-2026-06-27.md
benchmarks/results/stage-1-postops/*.json
benchmarks/results/stage-1-postops/*.csv
docs/reports/stage-1-postops-validation-report.md
```

---

## 7. 冻结后的变更流程

如果实现中发现设计不成立：

1. 停止继续编码。
2. 更新对应模块设计文档。
3. 更新本 checklist。
4. 标注影响范围。
5. 再继续实现。

禁止：

- 在实现中悄悄扩大 Stage 1 范围。
- 为了单个 benchmark 修改系统边界。
- 把局部实验结果写成系统结论。
- 绕过文档直接新增架构概念。

---

## 8. 当前冻结判断

截至 Stage 1.4 gate closure：

```text
系统设计方向：已冻结用于 Stage 1 review
Stage 1 范围：已冻结
实现入口：已打开并完成 Stage 1.0/1.1/1.2/1.3 最小闭环
Stage 1.4：已生成 gate closure 与提交分组建议
```

已验证：

```bash
cmake --build build-brgemm-main -j2
ctest --test-dir build-brgemm-main --output-on-failure
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
git diff --check
```

结果：

```text
CTest: 11/11 passed
Python/TensorFlow pytest: 17 passed
git diff --check: clean
```

推荐下一步：

```text
按 docs/reports/stage-1-gate-closure-2026-06-28.md 分组 review/提交。
优先提交设计包 -> RuntimePlan/ScoreModPlan -> capability probe -> QK post-ops path。
BRGEMM exploratory context 单独延后或拆分，避免混淆 Stage 1 主线。
```
