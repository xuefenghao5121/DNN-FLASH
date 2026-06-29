# TensorFlow Custom Op 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 设计目标

TensorFlow Custom Op 是 OneDNN-Flash 在 XLA Custom Call 完成前的工程验证路径，同时也是非 XLA 环境下的 fallback runtime。

目标：

- 为 TensorFlow eager/graph 提供可运行入口。
- 复用 RuntimePlanner 与 AttentionExecutionEngine。
- 不把 Custom Op 设计成最终唯一高性能路径。
- attrs 与 Python API、XLA opaque config 保持语义一致。

---

## 2. 当前状态

当前已有 `OneDNN-FlashAttention` CPU op，支持：

- `causal`
- `query_block_size`
- `key_block_size`
- `use_onednn`
- `tile_kernel`
- `qk_tile_layout`

这些 attrs 是迭代过程中逐步加入的，需要收编到 RuntimePlan，不应继续无限扩散。

---

## 3. 目标 Op 语义

```text
OneDNN-FlashAttention(
  q: Tensor[B,H,M,D],
  k: Tensor[B,H,N,D],
  v: Tensor[B,H,N,Dv],
  optional bias/mask tensors
) -> o: Tensor[B,H,M,Dv]
```

要求：

- Q/K/V 只读，不复制。
- O 由 TensorFlow allocate_output，一次分配，OneDNN-Flash direct write。
- 临时 workspace 由 OneDNN-Flash 管理，不通过 TensorFlow 暴露。

---

## 4. attrs 设计

建议 attrs：

```text
backend_policy: string
q_block_size: int
k_block_size: int
score_mod_type: string
score_mod_params_json: string
block_mask_type: string
block_mask_params_json: string
qk_layout_policy: string
dtype_policy: string
debug: bool
```

兼容策略：

- 保留旧 attrs。
- Python wrapper 将旧参数转换成新结构。
- C++ op 内部统一生成 `RuntimePlanInput`。

---

## 5. 数据路径

```text
TensorFlow Tensor
  -> raw pointer + shape extraction
  -> RuntimePlanInput
  -> RuntimePlanner
  -> AttentionExecutionEngine
  -> output direct write
```

禁止：

- 在 op 边界无必要复制 Q/K/V。
- 在每个 tile 构造高成本对象但无 cache/reuse。
- 在 TensorFlow op 内散落 backend selection 逻辑。

---

## 6. Debug 能力

调试模式输出：

- selected RuntimePlan。
- fallback reason。
- block mask statistics。
- oneDNN cache status。

TensorFlow op 可先通过 Python wrapper 返回 debug 字符串，或写入 benchmark JSON。

---

## 7. 测试设计

| 测试 | 验证 |
|---|---|
| basic causal | vs TF reference |
| explicit tile | attr 生效 |
| qk layout policy | strided/copied/transformed 兼容 |
| score_mod scale | RuntimePlan 生成正确 |
| fallback | unsupported attr 有明确错误或 fallback |
| isolated subprocess | 避免同进程重复注册同名 op |

---

## 8. 第一阶段交付

1. Custom Op attrs 与 RuntimePlanInput 对齐。
2. Python wrapper 兼容旧参数。
3. 增加 debug plan 输出。
4. 保持现有 TensorFlow tests 通过。
