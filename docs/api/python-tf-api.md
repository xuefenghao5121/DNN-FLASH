# Python / TensorFlow API 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 设计目标

Python / TensorFlow API 是 FlashOne 的用户入口，负责把用户侧调用转换成结构化运行时配置。

目标：

- API 简洁，隐藏 oneDNN 细节。
- score_mod 和 block_mask 使用可 lowering 的结构化表示。
- eager Custom Op 与未来 XLA Custom Call 共用同一套语义参数。
- 所有 `auto` 决策必须可打印、可复现。

---

## 2. 目标 API

```python
def flashone_attention(
    q,
    k,
    v,
    *,
    causal: bool | None = None,
    score_mod=None,
    block_mask=None,
    query_block_size: int | None = None,
    key_block_size: int | None = None,
    backend: str = "auto",
    tile_kernel: str = "auto",
    qk_tile_layout: str = "auto",
    dtype_policy: str = "auto",
    return_debug: bool = False,
):
    ...
```

---

## 3. 结构化 score_mod API

初期不支持任意 Python lambda 作为性能路径。

建议：

```python
Scale(value)
AdditiveBias(tensor)
Alibi(slopes)
Composed([Scale(...), AdditiveBias(...)])
```

任意 callback：

```python
CallbackScoreMod(fn)
```

必须触发 fallback，并在 debug 中说明。

---

## 4. 结构化 block_mask API

```python
CausalMask()
SlidingWindowMask(left, right=0)
PrefixMask(prefix_length)
SparseBlockMask(block_table)
```

`causal=True` 是 `block_mask=CausalMask()` 的简写。

---

## 5. Debug 输出

当 `return_debug=True`：

```python
out, debug = flashone_attention(..., return_debug=True)
print(debug.runtime_plan)
```

debug 至少包含：

- selected backend。
- tile sizes。
- qk layout。
- score_mod lowering。
- block mask statistics。
- fallback reason。

---

## 6. TensorFlow Op attrs

Custom Op attrs 应与 RuntimePlan 对齐，而不是无限扩散。

建议 attrs：

```text
backend_policy
q_block_size
k_block_size
score_mod_type
score_mod_params_json
block_mask_type
block_mask_params_json
qk_layout_policy
dtype_policy
return_debug
```

---

## 7. 第一阶段交付

1. 保持现有 API 兼容。
2. 新增结构化 score_mod/block_mask 的 Python 类型。
3. 新增 debug plan dump。
4. 将 wrapper 中已有 tile heuristic 迁移到 RuntimePlanner。
