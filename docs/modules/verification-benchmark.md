# Verification & Benchmark 模块设计

> 状态：Draft v0.1
> 日期：2026-06-27
> 上级文档：`docs/system-design.md`

---

## 1. 设计目标

验证体系必须保证每次系统演进都可解释、可复现、可回退。

---

## 2. Correctness 门禁

每个阶段必须跑：

```bash
cmake --build <build-dir> -j
ctest --test-dir <build-dir> --output-on-failure
PYTHONPATH=$PWD/python:$PWD python3 -m pytest -q tests/python tests/tensorflow
```

新增模块必须包含：

- reference 对照。
- edge tile。
- fallback path。
- deterministic seed。
- max_abs_diff / max_rel_diff。

---

## 3. Benchmark 记录字段

每条 benchmark 记录必须包含：

```text
commit
build dir
oneDNN version
CPU/ISA
shape: B,H,M,N,D,Dv
dtype
tile sizes
backend
qk layout
score_mod signature
block_mask signature
fallback reason
warmup/repeat
median/p50/p90/min/max
```

---

## 4. 性能分层

性能报告分三层：

1. **C++ microbenchmark**：验证 kernel/tiling 本身。
2. **TensorFlow eager Custom Op**：验证框架边界开销。
3. **TensorFlow graph/XLA**：验证最终系统路径。

不能用 C++ microbenchmark 替代系统结论。

---

## 5. 回归判定

| 类型 | 判定 |
|---|---|
| correctness | 任一测试失败即阻断 |
| whitespace | `git diff --check` 必须通过 |
| performance | 关键 shape 退化 >10% 必须解释 |
| fallback | 新 fallback reason 必须记录 |
| docs | 新模块/新策略必须更新设计文档 |

---

## 6. 报告模板

```text
## Summary
## Correctness
## Performance
## Fallbacks
## Regressions
## Next actions
```

---

## 7. 第一阶段交付

1. benchmark JSON schema。
2. benchmark report template。
3. CI/local checklist。
4. 将现有 benchmark 输出补充 plan/debug 字段。

---

## 8. Benchmark JSON Schema 草案

每次 benchmark 至少输出以下结构：

```json
{
  "schema_version": "flashone.benchmark.v1",
  "commit": "<git-sha>",
  "build": {
    "type": "debug|release",
    "flashone_enable_onednn": true,
    "flashone_has_onednn_brgemm": true
  },
  "environment": {
    "cpu": "...",
    "isa": "...",
    "onednn_version": "...",
    "tensorflow_version": "..."
  },
  "shape": {
    "B": 1,
    "H": 1,
    "M": 128,
    "N": 128,
    "D": 64,
    "Dv": 64
  },
  "runtime_plan": {
    "q_block": 32,
    "k_block": 64,
    "qk_backend": "onednn_matmul|onednn_brgemm|reference",
    "qk_layout": "strided_k|copied_transposed|brgemm_transformed_k",
    "score_mod_signature": "scale|additive_bias|...",
    "block_mask_signature": "none|causal|...",
    "fallback_reason": null
  },
  "timing_ms": {
    "median": 0.0,
    "p90": 0.0,
    "min": 0.0,
    "max": 0.0
  },
  "correctness": {
    "max_abs_diff": 0.0,
    "max_rel_diff": 0.0
  }
}
```

### 8.1 报告边界

- C++ microbenchmark 只能说明 kernel/tile 层结果。
- TensorFlow eager benchmark 说明 Custom Op 边界开销。
- TensorFlow graph/XLA benchmark 才能作为系统路径结论。
- 若发生 fallback，必须在图表和结论中显式标注。
