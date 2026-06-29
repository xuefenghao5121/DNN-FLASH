# Stage 1: oneDNN Post-Ops Validation Plan

> 状态：Draft v0.1
> 日期：2026-06-27
> 依赖基线：`docs/system-design.md`、`docs/design-self-review-2026-06-27.md`

---

## 0. 阶段目标

Stage 1 的目标不是追逐局部 BRGEMM 性能，而是验证 FlashOne 系统主线中的关键链路：

```text
用户 score_mod 语义
  -> ScoreModPlan
  -> RuntimePlan
  -> oneDNN post-ops capability
  -> QK score tile execution
  -> fallback/debug/benchmark 可观测
```

Stage 1 成功标准：

> 能明确回答 oneDNN 当前版本能承接多少 `scale + additive_bias` score_mod 语义，不能承接的部分为什么 fallback，以及这条路径在 C++/TensorFlow eager 层的 correctness/performance 表现。

---

## 1. 非目标

本阶段明确不做：

- 不继续扩大 `brgemm_transformed_k` shape sweep。
- 不把 BRGEMM 作为架构中心。
- 不实现 XLA Custom Call。
- 不做完整 ALiBi/sliding-window/custom sparse。
- 不做 bf16/AMX 主路线。
- 不重写 online softmax/PV 主循环。
- 不引入新的局部 kernel 优化作为主线。

---

## 2. 任务拆分

### Stage 1.0: Minimal Plan Skeleton

目标：建立最小 `ScoreModPlan` 与 `RuntimePlan`，防止 post-ops 逻辑再次散落。

交付：

```text
ScoreModSpec
ScoreModPlan
RuntimePlanInput
RuntimePlan
FallbackReason
Debug string / dump
```

范围：

- `score_mod = none | scale | additive_bias | scale + additive_bias`
- `block_mask = none | causal`
- `dtype = f32`
- `backend_policy = auto | reference | onednn_matmul`
- `qk_layout_policy` 保留现有默认，不改变 Python 默认行为。

验收：

- 单元测试覆盖 plan 生成。
- fallback reason 可打印。
- 不改实际性能路径也可先通过。

---

### Stage 1.1: oneDNN Capability Probe

目标：实测当前 oneDNN 版本 post-ops 支持边界。

Probe 内容：

```text
dnnl::matmul:
  scale/output scale
  binary add same-shape bias
  binary add broadcast bias
  eltwise linear/tanh
  runtime args

ukernel::brgemm:
  post-op support boundary
  transform support boundary
  supported out_ld/N
```

交付：

```text
docs/backend/onednn-postops-capability-2026-06-27.md
build artifact / test output summary
JSON-like capability table
```

验收：

- 不能只根据文档推断，必须有 smoke test 或编译/运行证据。
- unsupported 项必须有原因。

---

### Stage 1.2: QK Scale Post-Op

目标：验证 `score = QK * scale` 是否可下沉到 oneDNN matmul/post-op 路径。

设计边界：

- 只替换 QK score tile 生成或 score epilogue 前半段。
- 不修改 online softmax recurrence。
- 不修改 PV accumulation。
- 若 oneDNN 不支持，走 FlashOne epilogue fallback。

验收：

- correctness vs reference。
- benchmark 记录 backend/layout/score_mod/fallback。
- fallback reason 可观测。

---

### Stage 1.3: QK Additive Bias Post-Op

目标：验证 `score = QK * scale + bias` 的 same-shape bias tile 路径。

设计边界：

- 初期只支持 same-shape bias tile。
- broadcast bias 可 probe，但不作为必须实现。
- causal boundary mask 不强行下沉；仍由 FlashOne epilogue 处理。

验收：

- scale + additive bias correctness。
- bias tile 生命周期清晰。
- benchmark 区分 bias materialization 成本。

---

### Stage 1.4: Report & Gate

目标：输出阶段报告，决定是否进入 RuntimePlan 深化或 XLA CustomCall 调研。

交付：

```text
docs/reports/stage-1-postops-validation-report.md
benchmark JSON/CSV
fallback summary
next-stage recommendation
```

验收：

- 所有测试通过。
- benchmark schema 字段齐全。
- 结论明确区分 C++ microbenchmark 与 TensorFlow eager。

---

## 3. 文件改动边界

允许改动：

```text
include/flashone/*plan*.hpp
src/flashone/*plan*.cpp
include/flashone/score_mod*.hpp
src/flashone/score_mod*.cpp
include/flashone/onednn_*postops*.hpp
src/flashone/onednn_*postops*.cpp
tests/cpp/*plan*test*.cpp
tests/cpp/*onednn*postops*.cpp
benchmarks/*
docs/backend/*postops-capability*.md
docs/reports/*stage-1*.md
```

谨慎改动：

```text
tensorflow_ops/flashone_attention_op.cc
python/flashone_tf/ops.py
src/flashone/attention.cpp
src/flashone/batched_attention.cpp
```

原则：只接入 RuntimePlan/debug，不做 ABI 大改。

禁止改动主线：

```text
不为 Stage 1 修改 XLA CustomCall 不存在的接口。
不为 Stage 1 大改 BRGEMM transform 路径。
不把 Python wrapper 变成 backend 决策中心。
```

---

## 4. 测试门禁

每个实现小步至少运行：

```bash
git diff --check
cmake --build build-brgemm-main -j2
ctest --test-dir build-brgemm-main --output-on-failure
PYTHONPATH=$PWD/python:$PWD python3 -m pytest -q tests/python tests/tensorflow
```

如果 build dir cache 过旧，使用 fresh build dir。

---

## 5. Benchmark 门禁

Stage 1 benchmark 至少覆盖：

```text
shape:
  B=1,H=1,M=N=64,D=Dv=32
  B=1,H=1,M=N=128,D=Dv=64
  B=1,H=4,M=N=128,D=Dv=64

score_mod:
  none
  scale
  scale + additive_bias

backend:
  reference
  onednn_matmul
  existing brgemm baseline if available, only as baseline
```

必须记录：

- backend。
- qk layout。
- score_mod signature。
- fallback reason。
- oneDNN version。
- C++ vs TensorFlow eager 层级。

---

## 6. 阶段退出条件

Stage 1 完成后必须能回答：

1. 当前 oneDNN matmul 是否支持 scale post-op？怎么支持？
2. 当前 oneDNN matmul 是否支持 same-shape additive bias？怎么传 runtime arg？
3. broadcast bias 是否支持？如果不支持，原因是什么？
4. ukernel BRGEMM 在 post-op 上有什么限制？
5. scale/additive_bias 下沉后 correctness 是否稳定？
6. 性能瓶颈来自哪里：post-op、memory wrapper、bias materialization、TensorFlow eager overhead，还是别的？
7. 下一阶段应该深化 RuntimePlan、进入 XLA ABI 调研，还是回到 oneDNN backend 改造？

---

## 7. 防局部优化护栏

任何 Stage 1 代码改动如果满足以下任一条件，必须先回到设计文档更新：

- 引入新的 layout enum。
- 修改 BRGEMM transform 策略。
- 改变 Python 默认 backend/layout 行为。
- 大改 online softmax/PV recurrence。
- 绕过 RuntimePlan 直接在 TensorFlow op 或 oneDNN kernel 里决策。
- benchmark 结论只基于 C++ microbenchmark 却推导系统收益。

本阶段的核心问题始终是：

> oneDNN JIT/post-ops 能承接多少 FlexAttention score_mod 语义？

不是：

> 某个 BRGEMM 变体在某个 shape 上能不能再快一点。
