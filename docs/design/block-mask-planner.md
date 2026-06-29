# BlockMask Planner 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 设计目标

BlockMask Planner 负责把 FlexAttention 的 mask 语义转换成 tile-level 执行计划。

核心目标：

> mask 优先用于 tile skip，而不是只在 score tile 生成后做元素级置负无穷。

---

## 2. mask 层级

OneDNN-Flash 中 mask 分三层：

1. **Tile skip**：整个 Q/K block 无效，直接跳过 QK 与 PV。
2. **Boundary predicate**：边界 tile 内部分元素无效。
3. **Score epilogue mask**：用 additive mask value 处理剩余无效元素。

---

## 3. 输入与输出

### 3.1 输入

```text
BlockMaskSpec:
  type: none | causal | sliding_window | prefix | custom_sparse
  q_block_size
  k_block_size
  M,N
  parameters:
    window_left
    window_right
    prefix_length
    sparse block table
```

### 3.2 输出

```text
BlockMaskPlan:
  block_m
  block_n
  tile_skip_policy
  boundary_policy
  score_mask_policy
  mask_value
  statistics:
    total_tiles
    skipped_tiles_estimate
    boundary_tiles_estimate
```

---

## 4. 支持类型

### 4.1 none

所有 tile 均计算，无 boundary mask。

### 4.2 causal

语义：

```text
valid(i,j) = j <= i
```

Tile skip：

```text
if k_tile_start > q_tile_end:
    skip
```

Boundary：

```text
if k_tile_end > q_tile_start and k_tile_start <= q_tile_end:
    boundary causal mask
```

### 4.3 sliding window

语义：

```text
valid(i,j) = i - window_left <= j <= i + window_right
```

Tile skip：

```text
if k_tile_end < q_tile_start - window_left:
    skip
if k_tile_start > q_tile_end + window_right:
    skip
```

### 4.4 prefix

用于 prefix-LM 或 packed prompt。

语义：

```text
j < prefix_length or j <= i
```

### 4.5 custom_sparse

使用稀疏 block table。

初期只设计，不作为 Stage 1 实现目标。

---

## 5. 与 ScoreMod 的关系

BlockMaskPlan 产生两类结果：

1. tile skip 信息交给 `TileScheduler`。
2. boundary mask 信息交给 `ScoreModPlan`/`ScoreEpilogue`，转成 additive mask 或 predicate。

边界 tile 的元素级 mask 不应污染 oneDNN kernel selection；它是 epilogue 问题。

---

## 6. 统计与可观测性

每次 plan dump 应输出：

```text
mask_type
q_blocks
k_blocks
total_tiles
skipped_tiles
boundary_tiles
dense_tiles
effective_tile_ratio
```

benchmark 必须记录这些指标，否则无法说明 BlockMask 是否真的带来收益。

---

## 7. 测试设计

| 测试 | 输入 | 验证 |
|---|---|---|
| causal small | M=N=8, block=4 | skip/boundary 分类正确 |
| causal non-square | M=6,N=10 | edge tile 正确 |
| sliding window | window=2 | skip 数量正确 |
| prefix | prefix_length=4 | prefix 区域可见 |
| all skipped guard | custom_sparse | 输出保持数值安全 |

---

## 8. 第一阶段交付

Stage 1 交付：

1. causal BlockMaskPlan。
2. plan statistics dump。
3. boundary mask correctness。
4. tile skip benchmark。

Stage 2 再做 sliding window/prefix/custom sparse。

---

## 9. MaskTileGenerator 边界设计

BlockMask Planner 只决定 tile 级分类；具体边界 tile 的元素级 mask 由 `MaskTileGenerator` 生成。

```text
MaskTileGenerator:
  input:
    BlockMaskPlan
    q_tile_range
    k_tile_range
  output:
    mask_tile kind:
      none
      additive_same_shape
      predicate_bitmap
    mask_value
    fallback_reason(optional)
```

### 9.1 职责边界

| 组件 | 职责 |
|---|---|
| BlockMaskPlanner | 判断 tile skip / dense / boundary |
| MaskTileGenerator | 为 boundary tile 生成元素级 mask 描述 |
| ScoreEpilogue | 应用不能下沉的 mask |
| OneDnnPostOpsBuilder | 只处理 oneDNN 可表达的 additive/binary mask |

### 9.2 Stage 1 限制

Stage 1 只支持：

- no mask。
- causal mask。
- boundary tile 由 OneDNN-Flash epilogue 处理。

Stage 1 不把 causal boundary mask 强行下沉 oneDNN post-op，避免过早陷入 post-op/broadcast 细节。

### 9.3 all-masked row 行为

如果某个 query row 没有任何有效 key：

```text
output[row, :] = 0
softmax_l[row] = 0
fallback/debug records all_masked_row = true
```

禁止产生 NaN。
