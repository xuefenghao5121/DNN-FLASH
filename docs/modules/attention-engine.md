# Attention Execution Engine 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 设计目标

Attention Execution Engine 负责按照 RuntimePlan 执行 Flash-style tiled attention。

它不负责选择 backend，也不负责解析 TensorFlow/Python 参数；它只执行 plan。

---

## 2. 核心循环

```text
for each batch/head:
  for each q_tile:
    init online softmax state
    for each k_tile:
      if block_mask_plan.skip(q_tile,k_tile):
        continue
      qk_tile = QKExecutor(plan.qk).run(Q_tile, K_tile)
      ScoreEpilogue.apply_unlowered_ops(qk_tile)
      OnlineSoftmax.update(qk_tile, V_tile)
    write output tile
```

---

## 3. 子模块

| 子模块 | 职责 |
|---|---|
| TileScheduler | 生成 Q/K tile traversal，应用 tile skip |
| QKExecutor | 调用 reference/oneDNN 计算 score tile |
| ScoreEpilogue | 执行未下沉到 oneDNN 的 score_mod/mask |
| OnlineSoftmax | 维护 m/l/acc recurrence |
| PVExecutor | 执行 probability/value accumulation |
| WorkspaceManager | 统一管理 score/prob/acc/transform/scratchpad |

---

## 4. Online Softmax 状态

每个 query row 维护：

```text
m[row]
l[row]
acc[row, dv]
```

更新公式：

```text
m_new = max(m_old, max(score_block))
alpha = exp(m_old - m_new)
p = exp(score_block - m_new)
l_new = l_old * alpha + sum(p)
acc_new = acc_old * alpha + p @ V_tile
```

---

## 5. Workspace 原则

- K loop 内禁止 heap allocation。
- Q/K/V/O 指针从外部传入，尽量 zero-copy。
- score/prob/acc/transform/scratchpad 由 workspace 预分配。
- workspace size 由 RuntimePlan 提前计算。

---

## 6. 与 oneDNN 的边界

Execution Engine 只依赖抽象接口：

```text
QKKernel::run(q_ptr, k_ptr, score_ptr, workspace, post_ops_runtime_args)
PVKernel::run(prob_ptr, v_ptr, acc_ptr, workspace)
```

不直接构造 oneDNN primitive，不直接管理 oneDNN cache。

---

## 7. 数值策略

初始：

- f32 input/output/accumulation。
- mask value 使用大负数，避免 NaN。
- 空 tile / 全 mask row 需要定义行为。

后续：

- bf16 input。
- f32 accumulation。
- output bf16/f32 policy。

---

## 8. 测试设计

| 测试 | 验证 |
|---|---|
| no mask dense | vs standard reference |
| causal | boundary tile correctness |
| partial tiles | M/N/Dv 非 block 对齐 |
| all skipped guard | 不 NaN/不崩溃 |
| workspace reuse | K loop 无 allocation |
| backend swap | reference/oneDNN 输出一致 |

---

## 9. 第一阶段交付

1. 将现有 attention options 与 workspace 接入 RuntimePlan。
2. TileScheduler 与 BlockMaskPlan 对接。
3. QK/PV executor 抽象化。
4. 保持旧 API overload 兼容。

---

## 10. Stage 1 执行边界

Stage 1 不重构完整 online softmax/PV 流程，只允许替换 QK score tile 的生成与 score epilogue 前半部分。

允许改动：

- QKExecutor 接收 RuntimePlan 中的 backend/layout/post_ops plan。
- ScoreEpilogue 处理 oneDNN 未承接的 scale/bias/mask。
- Debug metadata 记录实际 backend 与 fallback。

禁止改动：

- 不引入新的 online softmax 算法变体。
- 不重写 PV accumulation 主循环。
- 不为了 BRGEMM 局部收益改变 RuntimePlan 语义。
- 不把 XLA CustomCall 与 Stage 1 混在一起实现。

all-masked row 行为遵循 `BlockMask Planner` 的定义：输出零向量，不产生 NaN。
