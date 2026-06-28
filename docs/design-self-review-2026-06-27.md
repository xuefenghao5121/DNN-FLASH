# FlashOne 设计自审报告

> 日期：2026-06-27
> 范围：`docs/system-design.md` 与 `docs/modules/*.md`
> 目标：检查当前设计包是否真正回到 “oneDNN JIT/post-ops/XLA 系统方案” 主线，并找出进入 Stage 1 前必须修正的问题。

---

## 0. 自审结论

当前设计包已经把方向从“继续局部 BRGEMM 微优化”拉回到系统方案：

```text
TensorFlow/XLA 图级语义
  -> FlashOne RuntimePlan
  -> ScoreMod/BlockMask lowering
  -> oneDNN JIT/post-ops/BRGEMM 执行
  -> correctness + benchmark 验证
```

这个方向是正确的。

但当前文档仍处在 Draft 状态，存在 5 类关键问题：

1. **新旧设计文档有冲突**：`docs/design.md` 和 `docs/project-plan.md` 仍保留早期 MVP/假设描述，与新 `system-design.md` 的系统主线存在不一致。
2. **Stage 1 依赖关系还不够严格**：post-ops 验证前至少需要一个最小 `RuntimePlan/ScoreModPlan` skeleton，否则实现会再次散落。
3. **oneDNN post-ops 能力边界需要先 probe**：不能假设 matmul/BRGEMM 都能承接 scale/additive bias/mask。
4. **XLA Custom Call 设计仍偏概念**：需要基于当前 TensorFlow/XLA 版本确认 ABI/FFI/注册方式。
5. **benchmark 和 fallback 可观测性还没有落成 schema**：设计里要求记录，但还没定义实际 JSON schema 与 report 模板。

因此建议：

> 先做一次文档修正与 Stage 1 执行计划冻结，再进入任何代码实现。

---

## 1. 对总体设计方向的审查

### 1.1 正确点

当前 `docs/system-design.md` 已经明确：

- FlashOne 不是手写 attention kernel。
- FlashOne 不是简单替换 GEMM。
- oneDNN JIT/post-ops/BRGEMM/primitive cache 是核心执行能力。
- XLA/TF graph integration 是主线，eager Custom Op 只是验证路径。
- score_mod 与 BlockMask 是一等公民。
- BRGEMM/transformed-K 是底层能力，不是架构中心。

这与项目最初设计原则一致。

### 1.2 仍需修正的地方

当前新设计和旧文档之间有明显张力。

`docs/design.md` 的原则是：

```text
Start with a small, verifiable, standalone attention core.
Do not hide algorithmic uncertainty behind TensorFlow/XLA/oneDNN integration.
```

这个原则在项目启动阶段是合理的，但现在容易误导后续开发继续围绕 standalone core 做局部推进。

`docs/project-plan.md` 中 H2 写的是：

```text
oneDNN BRGEMM + post-ops 能表达 online softmax
```

这句话过强。当前更准确的说法应该是：

```text
oneDNN JIT/BRGEMM/post-ops 承接 QK/PV 和部分 score_mod；
online softmax 的跨 K-block recurrence 短期由 FlashOne Execution Engine 管理。
```

### 1.3 结论

需要新增文档状态说明：

- `docs/system-design.md` 是当前系统设计基线。
- `docs/design.md` 是早期 MVP 设计，需要标注为 historical/MVP 或更新。
- `docs/project-plan.md` 中 H2 需要修正为更现实的 post-ops 能力验证假设。

---

## 2. 模块边界审查

### 2.1 Python / TensorFlow API

状态：方向正确，但需要更明确地区分三类参数：

1. 用户语义参数：`score_mod`、`block_mask`。
2. 调度策略参数：tile size、backend policy、layout policy。
3. debug/benchmark 参数。

风险：

- 如果 Python API 继续直接选择 tile/backend，会绕过 RuntimePlanner。
- 如果 `score_mod_params_json` 直接暴露给用户，后续 ABI 可能难维护。

建议：

- Python API 只构造 `RuntimePlanInput`。
- `auto` 决策只能在 RuntimePlanner 中发生。
- `return_debug=True` 应输出 planner 结果，而不是 wrapper 自己的决策。

---

### 2.2 ScoreMod Lowering

状态：方向正确，是系统设计的关键模块。

关键风险：

#### 风险 A：scale 的 oneDNN 表达方式不能先验假设

`score * scale` 是 softmax 前的 score scaling。候选表达包括：

- matmul primitive attr output scale。
- post-op eltwise linear。
- matmul alpha-like 参数，如果 API 支持。
- FlashOne epilogue fallback。

但不同 oneDNN API/primitive/ukernel 支持不一致，必须先做 capability probe。

#### 风险 B：additive bias 需要 runtime tensor 参数

如果用 oneDNN binary add post-op，需要明确：

- bias tile 的 memory descriptor。
- binary post-op runtime arg 传递方式。
- same-shape 与 broadcast 的支持边界。
- 每 tile bias materialization 的开销。

#### 风险 C：mask 与 bias 不应混在一起

boundary mask 可以用 additive large-negative value，但 mask 的生成逻辑属于 BlockMask/MaskTileGenerator，不应直接塞进 ScoreMod。

建议：

- Stage 1 只做 `scale` 和 same-shape `additive_bias`。
- ALiBi 只做 generated bias tile smoke，不作为主 benchmark 目标。
- callback score_mod 明确永远不是性能主路径。

---

### 2.3 BlockMask Planner

状态：tile skip / boundary / score mask 三层划分正确。

缺口：

- 还没有定义 `MaskTileGenerator`。
- 没有定义 all-masked row 的数值行为。
- sliding window/prefix 的优先级应后移，避免 Stage 1 范围膨胀。

建议：

- Stage 1 只支持 no mask / causal。
- causal 分为：full skip、full dense、boundary。
- boundary mask 先由 FlashOne epilogue 处理，不强行下沉 oneDNN。

---

### 2.4 Runtime Planner

状态：这是当前最重要的骨架模块。

关键问题：

`system-design.md` 把 Stage 1 定为 oneDNN post-ops 能力验证，但如果没有最小 RuntimePlanner，post-ops 实现可能再次直接写进 oneDNN kernel 或 TensorFlow op。

建议调整 Stage 1 顺序：

```text
Stage 1.0: Minimal RuntimePlan / ScoreModPlan skeleton
Stage 1.1: oneDNN post-ops capability probe
Stage 1.2: QK scale post-op
Stage 1.3: QK additive bias post-op
Stage 1.4: benchmark + fallback report
```

另一个问题：cache key 需要分层。

当前文档里 `PlanCacheKey` 包含 B/H/M/N/D 等全量信息，但 oneDNN kernel cache 未必需要 B/H。应拆分：

```text
RuntimePlanCacheKey
OneDnnKernelCacheKey
TransformCacheKey
BenchmarkConfigKey
```

---

### 2.5 Attention Execution Engine

状态：边界基本清晰。

缺口：

- QKExecutor/PVExecutor 接口还没有明确 shape/layout runtime args。
- OnlineSoftmax 与 PV 的关系需要更精确：当前实现可能是 `prob tile -> PV`，未来可优化为 fused recurrence，但设计要允许两者。
- 全 mask row 行为需要定义。

建议：

- Stage 1 不改 online softmax 结构。
- 只把 QK score tile 生成路径接入 post-op 实验。
- 不在 Stage 1 同时改 PV/softmax。

---

### 2.6 oneDNN Integration

状态：方向正确，尤其是“matmul primitive 先验证 post-ops 语义，BRGEMM 是快路径而非架构中心”。

关键风险：

1. `ukernel::brgemm` post-op 能力可能明显弱于 `dnnl::matmul`。
2. `dnnl::matmul` post-op 支持和 runtime args 需要实测。
3. oneDNN transform 当前只支持特定 out_ld；edge tile fallback 会影响结果解释。
4. stream/memory wrapper 开销可能掩盖 post-op 收益。

建议：

- Stage 1 只承诺 `dnnl::matmul + post_ops`。
- BRGEMM 只作为 baseline/对照，不作为 Stage 1 主实现。
- capability probe 结果必须落文档。

---

### 2.7 TensorFlow Custom Op

状态：定位正确：验证路径，不是最终唯一性能路径。

风险：

- JSON attr 在 TensorFlow op 中可能导致版本维护和 graph cache 问题。
- 同一进程重复注册同名 op 的限制已有历史问题，测试必须继续用 isolated subprocess。
- 如果 Custom Op attrs 与 Python wrapper、XLA opaque config 不统一，会形成三套语义。

建议：

- Stage 1 不大改 Custom Op ABI，只在内部准备 RuntimePlanInput。
- 新 attrs 等 RuntimePlan skeleton 稳定后再加。
- debug 输出先走 benchmark JSON，不急着作为 Tensor 输出。

---

### 2.8 XLA Custom Call

状态：方向正确，但目前仍是概念设计。

缺口：

- 需要确认当前 TensorFlow/XLA 版本支持的 CustomCall/FFI 方式。
- 需要明确是 TensorFlow custom call、XLA FFI、PJRT custom call，还是 TF op + XLA lowering。
- `FlashOneOpaqueConfig` 的 ABI 需要版本号、alignment、endianness、schema。

建议：

- XLA 不进入 Stage 1 实现。
- Stage 1 只保证所有设计与 XLA CustomCall 兼容。
- Stage 2/3 前先做 XLA ABI 调研文档。

---

### 2.9 Verification & Benchmark

状态：原则正确，但还缺实际 schema。

建议新增：

```text
benchmarks/schema/flashone-benchmark.schema.json
benchmarks/reports/templates/stage-report.md
```

benchmark 必须区分：

- C++ microbenchmark。
- TensorFlow eager Custom Op。
- TensorFlow graph。
- XLA CustomCall。

并记录 fallback reason，否则性能数据不可解释。

---

## 3. Stage 1 可执行性审查

当前 Stage 1 目标是：

```text
oneDNN post-ops 能力验证
```

这是正确的，但需要压缩范围。

### 3.1 推荐 Stage 1 范围

只做：

1. Minimal `ScoreModPlan`。
2. Minimal `RuntimePlan`。
3. oneDNN matmul post-ops capability probe。
4. QK `scale` post-op。
5. QK same-shape `additive_bias` post-op。
6. fallback/debug metadata。
7. C++ tests + TensorFlow smoke + benchmark。

暂不做：

- ALiBi 完整优化。
- sliding window/prefix/custom sparse。
- BRGEMM post-op 主路径。
- XLA CustomCall 实现。
- bf16/AMX 主路径。
- RuntimePlanner 完整自动调优。

### 3.2 Stage 1 成功标准

Stage 1 成功不是“性能立刻超过 TF graph”，而是证明：

```text
FlashOne score_mod 语义
  -> ScoreModPlan
  -> oneDNN post-ops capability
  -> QK tile execution
  -> correctness/benchmark/fallback 可观测
```

如果 post-ops 性能不佳，也可以接受，只要能明确原因：

- post-op API 不支持。
- memory wrapper 开销过大。
- bias tile materialization 成本过高。
- TensorFlow eager overhead 掩盖收益。

---

## 4. 发现问题清单

| ID | 严重级别 | 问题 | 影响 | 建议 |
|---|---|---|---|---|
| DR-001 | P0 | `docs/design.md` 与 `docs/system-design.md` 的开发重心不一致 | 后续实现可能继续按 MVP standalone 思路推进 | 标注旧文档为 historical 或更新为指向 system-design |
| DR-002 | P0 | `docs/project-plan.md` H2 对 “BRGEMM + post-ops 表达 online softmax” 表述过强 | 技术假设不准确，可能导致错误实现目标 | 改为 oneDNN 承接 QK/PV 和部分 score_mod，online recurrence 由 FlashOne 管理 |
| DR-003 | P0 | Stage 1 未显式要求 Minimal RuntimePlan skeleton 先行 | post-ops 代码可能再次散落 | Stage 1.0 先建 RuntimePlan/ScoreModPlan skeleton |
| DR-004 | P1 | oneDNN post-ops 能力未先 probe | 可能设计了 API 不支持的路径 | 新增 capability probe 文档和 smoke test |
| DR-005 | P1 | cache key 未分层 | plan cache/kernel cache 混杂，影响复用 | 拆分 RuntimePlanCacheKey / OneDnnKernelCacheKey / TransformCacheKey |
| DR-006 | P1 | mask tile generator 未定义 | boundary mask 与 score_mod 容易混乱 | 新增 MaskTileGenerator 小节 |
| DR-007 | P1 | Benchmark schema 未落地 | 性能结论不可复现 | 增加 JSON schema 和 stage report template |
| DR-008 | P1 | TensorFlow attrs JSON 化风险未评估 | 可能影响 graph cache/ABI 稳定 | Stage 1 不改 ABI，先内部 RuntimePlanInput |
| DR-009 | P2 | XLA CustomCall ABI 仍偏概念 | 后续实现可能踩版本坑 | Stage 2 前单独调研 TF/XLA FFI |
| DR-010 | P2 | all-masked row 行为未定义 | edge case correctness 风险 | 在 Attention Engine 文档中定义数值行为 |

---

## 5. 建议整改顺序

### Step 1：修正文档基线

- 更新 `docs/design.md`：标注为早期 MVP 设计，当前系统基线见 `docs/system-design.md`。
- 更新 `docs/project-plan.md`：修正 H2 假设。

### Step 2：补 Stage 1 执行计划

新增：

```text
docs/stage-1-postops-validation-plan.md
```

内容包括：

- Stage 1.0 ~ 1.4 任务拆分。
- 文件改动范围。
- 测试命令。
- benchmark 命令。
- 验收标准。
- 不做事项。

### Step 3：补关键设计缺口

- Runtime Planner：cache key 分层。
- BlockMask：MaskTileGenerator。
- Verification：benchmark JSON schema 草案。
- oneDNN Integration：capability probe 输出格式。

### Step 4：再进入实现

实现顺序：

```text
Minimal RuntimePlan/ScoreModPlan
  -> oneDNN post-ops capability probe
  -> QK scale post-op
  -> QK additive bias post-op
  -> tests/benchmark/report
```

---

## 6. 自审最终判断

当前设计包可以作为系统设计基线，但还不能直接进入大规模实现。

可以进入的下一步是：

> 文档整改 + Stage 1 执行计划冻结。

只有完成以下条件后，才建议写代码：

- [ ] 旧 `docs/design.md` 与 `docs/project-plan.md` 不再误导。
- [ ] `docs/stage-1-postops-validation-plan.md` 完成。
- [ ] Stage 1 明确先做 Minimal RuntimePlan/ScoreModPlan skeleton。
- [ ] oneDNN post-ops capability probe 的输入输出定义完成。
- [ ] benchmark/fallback 记录格式完成。
