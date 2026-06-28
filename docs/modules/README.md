# FlashOne 模块设计索引

> 状态：Draft v0.1
> 日期：2026-06-27

本文档是 `docs/system-design.md` 的模块化拆分索引。后续每个模块进入实现前，必须先补齐对应设计文档。

## 模块列表

| 模块 | 设计文档 | 状态 | 下一步 |
|---|---|---|---|
| Python / TensorFlow API | `python-tf-api.md` | 已写初版 | 统一 API、attrs、fallback 语义 |
| ScoreMod lowering | `score-mod-lowering.md` | 已写初版 | scale/additive bias/ALiBi 到 post-ops 的映射 |
| BlockMask planner | `block-mask-planner.md` | 已写初版 | causal/sliding window tile skip |
| Runtime planner | `runtime-planner.md` | 已写初版 | shape/dtype/backend/tile/cache key |
| Attention engine | `attention-engine.md` | 已写初版 | scheduler、online softmax、workspace |
| oneDNN integration | `onednn-integration.md` | 已写初版 | post-ops、BRGEMM、transform、cache |
| TensorFlow Custom Op | `tensorflow-custom-op.md` | 已写初版 | MVP attrs 系统化 |
| XLA Custom Call | `xla-custom-call.md` | 已写初版 | fixed-shape 最小闭环 |
| Verification & benchmark | `verification-benchmark.md` | 已写初版 | correctness/perf/integration gates |
| Development flow | `development-flow.md` | 已写初版 | 设计先行、实现、验证、提交门禁 |

## 开发规则

1. 每个模块实现前必须先有模块设计。
2. 模块设计必须包含：职责、输入输出、关键数据结构、fallback、测试、性能指标。
3. 实现 PR/commit 必须引用对应模块设计章节。
4. 如果实现过程中发现设计不成立，先更新设计，再改代码。


---

## Design Freeze Documents

| Document | Role |
|---|---|
| `../system-design.md` | Current system design baseline |
| `../design-self-review-2026-06-27.md` | Self-review issues and remediation plan |
| `../stage-1-postops-validation-plan.md` | Stage 1 execution boundary |
| `../stage-1-interface-contract.md` | Stage 1 public/internal interface contract |
| `../system-design-freeze-checklist.md` | Gate before implementation |

## Current Freeze Boundary

Implementation must not start until `../system-design-freeze-checklist.md` gates are satisfied.

Stage 1 is limited to:

- Minimal `ScoreModPlan` / `RuntimePlan` skeleton.
- oneDNN matmul post-ops capability probe.
- QK `scale` post-op validation.
- QK same-shape `additive_bias` post-op validation.
- fallback/debug/benchmark observability.

Stage 1 explicitly excludes:

- XLA CustomCall implementation.
- BRGEMM transformed-K shape sweep as a main task.
- online softmax/PV algorithm rewrite.
- bf16/AMX main path.
- Python wrapper becoming the backend decision center.
