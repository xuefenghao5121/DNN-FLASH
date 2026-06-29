# OneDNN-Flash 开发流程与验收门禁

> 状态：Draft v0.1
> 日期：2026-06-27

---

## 1. 总流程

```text
需求/方向
  -> 系统设计文档
  -> 模块设计文档
  -> 评审/冻结设计
  -> 小步实现
  -> correctness 验证
  -> benchmark 验证
  -> 文档更新
  -> commit
```

---

## 2. 禁止流程

禁止：

- 未更新设计文档直接写核心模块。
- 未说明 fallback reason 就引入 fallback。
- 只用 C++ microbenchmark 宣称 TensorFlow/XLA 系统收益。
- 为追求 BRGEMM 局部收益牺牲 score_mod/BlockMask 语义。
- 将 build artifacts 或 third_party 大目录误提交。

---

## 3. 每阶段交付格式

每个阶段必须交付：

```text
1. 设计变更
2. 实现变更
3. 测试结果
4. benchmark 结果
5. 限制与 fallback
6. 下一步
```

---

## 4. Stage 1 开发任务

Stage 1：oneDNN post-ops 能力验证。

### 任务 1：capability probe

交付：

- oneDNN matmul post-ops smoke。
- 记录当前 oneDNN 版本支持的 post-op 类型。

### 任务 2：ScoreModPlan skeleton

交付：

- scale/additive_bias plan。
- fallback reason。
- tests。

### 任务 3：QK scale post-op

交付：

- correctness。
- benchmark。
- fallback。

### 任务 4：QK additive bias post-op

交付：

- same-shape bias tile。
- ALiBi-like generated bias tile smoke。

---

## 5. 验收命令

```bash
git diff --check
cmake --build build-brgemm-main -j2
ctest --test-dir build-brgemm-main --output-on-failure
PYTHONPATH=$PWD/python:$PWD python3 -m pytest -q tests/python tests/tensorflow
```

如果 build dir 不存在或 cache 过旧，使用 fresh build dir。

---

## 6. 设计评审检查清单

- 是否符合 oneDNN-native 原则？
- 是否保留 score_mod/BlockMask 扩展点？
- 是否避免 N² materialization？
- 是否有 fallback reason？
- 是否有 benchmark plan？
- 是否与 XLA Custom Call 主线兼容？
