# FlexAttention CPU 加速方案 — 项目计划

> 作者: 小西 (OpenClaw) | 日期: 2026-06-25 | 状态: 待审核

---

## 一、项目概述

### 1.1 目标

基于 TensorFlow + oneDNN + XLA 技术栈，设计并实现 CPU 平台的 FlexAttention 框架，提供 Flash 级别的 Attention 计算性能和灵活的 score_mod 抽象。

### 1.2 核心价值

- **填补 CPU FlashAttention 空白**：GPU 有 FlashAttention/FlexAttention，CPU 没有任何 Flash 级方案
- **复用已有基础设施**：oneDNN BRGEMM + JIT + post-ops，TF 线程池 + 内存分配，XLA 图级融合
- **数学天然匹配**：FlashAttention 的 PV 累加 = BRGEMM 的 batch-reduce 定义
- **ARM 可扩展**：oneDNN 有 SVE 后端，鲲鹏930 可部署

### 1.3 技术栈

| 层级 | 技术 | 角色 |
|------|------|------|
| 框架 | TensorFlow 2.x | 算子注册、线程调度、部署生态 |
| 编译器 | XLA (jit_compile=True) | 图级融合、score_mod 自动融合、buffer aliasing |
| 计算库 | oneDNN v3.14+ | BRGEMM JIT 内核、post-ops、ISA 自适应 |
| 桥接 | XLA Custom Call | 保持 attention 模式不被 XLA 拆散 |
| 硬件 | x64 (AMX/AVX-512) + ARM64 (SVE) | 目标平台 |

---

## 二、两阶段规划

### Phase 1: FlexAttention (3 个月)

#### Month 1: 原型验证 (2026-07-01 ~ 2026-07-31)

**目标：** 验证 5 个核心假设

| # | 假设 | 验证方式 | 失败应对 |
|---|------|---------|---------|
| H1 | XLA Custom Call 能保持 attention 不被拆散 | 写原型, 检查 HLO IR | 改用纯 oneDNN primitive (TF Custom Op) |
| H2 | oneDNN BRGEMM + post-ops 能表达 online softmax | 扩展 post-op, 跑通 | 自己写 JIT 内核 (基于 Xbyak) |
| H3 | Flash tiling 在 CPU 上比标准 attention 快 ≥3x | 性能对比测试 | 分析瓶颈, 调整 tile size |
| H4 | AMX tile 利用率达到 ≥60% | VTune/perf 分析 | 调整 tile 配置 |
| H5 | Primitive cache 在固定 shape 下 100% 命中 | 多次推理测延迟 | 增加预热策略 |

**交付物：**
- Causal FlashAttention 原型 (bf16, AMX, causal mask)
- 假设验证报告
- 性能基准数据

**任务分解：**

```
Week 1-2: 环境搭建 + 最小原型
  • oneDNN BRGEMM API 调研 + 编译测试
  • XLA Custom Call 框架搭建
  • 最小 Flash Attention 原型 (硬编码 causal, bf16, AMX)
  • 验证 H1: 检查 XLA 是否拆散 attention

Week 3: Online Softmax 实现
  • 扩展 oneDNN post-ops: row_reduce (max/sum)
  • 实现 online softmax update 逻辑
  • 验证 H2: 数值正确性 + 寄存器内完成

Week 4: 性能验证
  • 性能基准: vs tf.nn.scaled_dot_product_attention
  • VTune 分析: AMX 利用率, cache miss, 线程效率
  • 验证 H3, H4, H5
  • 撰写验证报告
```

#### Month 2: 完整 FlexAttention (2026-08-01 ~ 2026-08-31)

**目标：** 实现完整的 FlexAttention API

**功能清单：**

| 功能 | 优先级 | 说明 |
|------|--------|------|
| Causal mask | P0 | 因果注意力 |
| Sliding window mask | P0 | 滑动窗口 |
| Scale (1/sqrt(d)) | P0 | 标准缩放 |
| ALiBi bias | P1 | 相对位置偏置 |
| Tanh gating | P1 | 门控注意力 |
| BlockMask (block 级跳过) | P0 | 稀疏注意力优化 |
| AVX-512 后端 | P0 | 非 AMX 平台支持 |
| bf16 数据类型 | P0 | 主力精度 |
| f32 数据类型 | P1 | 高精度场景 |
| int8 量化 | P2 | 量化推理 |

**任务分解：**

```
Week 5-6: Score Mod 系统
  • XLA epilogue → oneDNN post-ops 翻译层
  • score_mod 组件: scale, add_bias, tanh, sigmoid, exp
  • mask 组件: causal, sliding_window, prefix, custom
  • 组合测试: scale + ALiBi + causal

Week 7-8: BlockMask + 多后端
  • BlockMask 表示 (block 级稀疏描述)
  • Tile 级 mask 跳过逻辑
  • AVX-512 JIT 后端 (无 AMX 平台)
  • K-Super-Blocking (L2 cache 感知)
  • 端到端测试: Transformer 模型推理
```

#### Month 3: 集成 + 性能调优 (2026-09-01 ~ 2026-09-30)

**目标：** 生产可用 + ARM 适配

**任务分解：**

```
Week 9-10: TF 集成
  • TF Custom Op 注册 (非 XLA 路径)
  • XLA Custom Call 注册 (XLA 路径)
  • Primitive cache 集成 (TF_ONEDNN_OBJECT_CACHE)
  • 线程池配置 (DNNL_CPU_RUNTIME=THREADPOOL)
  • 内存分配器集成 (TF BFC allocator)

Week 11: 性能调优
  • Tile size autotune
  • 线程并行策略优化
  • 内存布局优化 (blocked layout 传播)
  • 性能对比: vs PyTorch SDPA, vs 标准 TF attention
  • 目标: ≥3x 加速 (vs 标准 attention)

Week 12: ARM SVE 适配
  • oneDNN SVE BRGEMM 验证 (鲲鹏930)
  • SVE 版 online softmax
  • 鲲鹏930 性能测试
  • 撰写 Phase 1 总结报告
```

**Phase 1 交付物：**

```
1. flex_attention Python API
   flex_attention(Q, K, V, score_mod=..., block_mask=...)
   
2. 支持的 score_mod:
   • scale (1/sqrt(d))
   • ALiBi (相对位置偏置)
   • tanh gating
   • 自定义组合

3. 支持的 mask:
   • causal (因果)
   • sliding_window (滑动窗口)
   • prefix (前缀)
   • 组合 mask (AND/OR)

4. 支持的数据类型: bf16, f32
5. 支持的硬件: x64 (AMX/AVX-512), ARM64 (SVE)
6. 性能: ≥3x vs 标准 attention (固定 shape)
7. 集成: TF Custom Op + XLA Custom Call
8. 文档: API 文档 + 性能报告 + 技术白皮书
```

---

### Phase 2: FlexInteraction (2-3 个月, Phase 1 验证通过后)

#### Month 4: 抽象泛化 (2026-10-01 ~ 2026-10-31)

**目标：** 将 FlexAttention 泛化为 FlexInteraction

**三阶段抽象：**

```
interact(A, B):      matmul / hadamard / custom
transform(S):        softmax / identity / sigmoid / custom  
aggregate(S, C):     matmul / triu / sum / topk / custom
```

**新增组件：**

| 组件 | 类型 | 用途 |
|------|------|------|
| identity transform | transform | DLRM/DeepFM (无 softmax) |
| triu aggregate | aggregate | DLRM 交互层 (取上三角) |
| sum_triu aggregate | aggregate | DeepFM FM 层 (上三角求和) |
| topk aggregate | aggregate | DSSM 匹配 (取 top-K) |
| symmetric interact | interact | DLRM (A=B, 跳过下三角) |

#### Month 5: 推荐模型适配 (2026-11-01 ~ 2026-11-30)

**目标：** DLRM / DeepFM / DSSM 端到端加速

**任务：**

```
Week 1-2: DLRM 交互层
  • symmetric tiling (跳过 50% 计算)
  • triu 提取 (寄存器内 post-op)
  • 端到端 DLRM 推理对比

Week 3: DeepFM FM 层
  • sum_triu aggregate 实现
  • 端到端 DeepFM 推理对比

Week 4: DSSM 匹配层
  • batch dot product → BRGEMM
  • topk aggregate
  • 端到端 DSSM 推理对比
```

#### Month 6: 集成 + 论文 (2026-12-01 ~ 2026-12-31)

**目标：** 生产可用 + 学术产出

**交付物：**

```
1. flex_interact(A, B, C, config=...) API
2. DLRM 加速: ~30x (交互层), ~5-8x (端到端)
3. DeepFM 加速: ~20x (FM 层), ~4-6x (端到端)
4. DSSM 加速: ~5x (匹配层), ~3x (端到端)
5. 论文初稿
```

---

## 三、技术架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  用户代码 (Python)                                           │
│  @tf.function(jit_compile=True)                             │
│  def attention(Q, K, V, slopes):                            │
│      S = tf.matmul(Q, K, transpose_b=True) * scale          │
│      S = S + slopes * relative_pos  # ALiBi                 │
│      S = tf.where(causal_mask, S, -1e9)                     │
│      O = flash_softmax_matmul(S, V)  # Custom Call          │
│      return O                                               │
├─────────────────────────────────────────────────────────────┤
│  XLA HLO 级优化                                              │
│  • 自动融合: scale + ALiBi + mask → epilogue                 │
│  • buffer aliasing: S 和 P 复用内存                          │
│  • 识别 attention 模式 → 生成 Custom Call (不拆散)            │
├─────────────────────────────────────────────────────────────┤
│  Custom Call 适配层                                          │
│  • 解析 XLA epilogue → oneDNN post-ops 链                    │
│  • 查询/创建 attention primitive (带 cache)                   │
│  • 传递 tile 配置 + mask 描述                                 │
├─────────────────────────────────────────────────────────────┤
│  oneDNN Attention Primitive                                  │
│  • Flash-style tiling (Q-tile × K-tile)                     │
│  • BRGEMM QK (AMX/AVX-512 JIT)                             │
│  • Post-ops 注入 (score_mod + mask, 寄存器内)                │
│  • Online softmax (寄存器内归约)                              │
│  • BRGEMM PV (AMX/AVX-512 JIT, 累加模式)                    │
│  • BlockMask tile 级跳过                                     │
│  • K-Super-Blocking (L2 cache 感知)                         │
├─────────────────────────────────────────────────────────────┤
│  硬件层                                                      │
│  • AMX tile (x64) / SVE vector (ARM64)                      │
│  • TBB 线程池 (与 TF intra-op 共享)                          │
│  • Primitive cache (JIT 编译结果缓存)                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| score_mod 表达 | TF 原生算子 + XLA 自动融合 | 不需要自己写 DSL 编译器 |
| 防止 XLA 拆散 | Custom Call (opaque) | 最简单, 不侵入 XLA 内部 |
| 内核生成 | oneDNN BRGEMM JIT | 复用已有 AMX/AVX-512 优化 |
| softmax 实现 | 扩展 oneDNN post-ops | 寄存器内完成, 不写内存 |
| mask 跳过 | BlockMask + tile 级 | 比 dense mask 节省 50%+ 计算 |
| 线程管理 | DNNL_CPU_RUNTIME=THREADPOOL | 与 TF 共享线程池, 避免超订阅 |

### 3.3 性能目标

| 场景 | 标准 attention | FlexAttention (目标) | 加速比 |
|------|---------------|---------------------|--------|
| Causal, seq=2048, d=128, bf16, AMX | ~15ms | ~0.5ms | 30x |
| Causal, seq=4096, d=128, bf16, AMX | ~60ms | ~2ms | 30x |
| Causal+ALiBi, seq=4096 | ~65ms | ~2.2ms | 30x |
| Sliding window(1024), seq=4096 | ~60ms | ~0.8ms | 75x |
| Causal, seq=2048, AVX-512 (无AMX) | ~25ms | ~3ms | 8x |
| Causal, seq=2048, ARM SVE | ~40ms | ~8ms | 5x |

---

## 四、Phase 1 原型验证详细方案

### 4.1 开发环境

```
硬件:
  • 开发机: Intel Xeon (AMX 支持), 32核+, 64GB+ RAM
  • ARM 测试机: 鲲鹏930 (SVE 256-bit), 120核

软件:
  • OS: Ubuntu 22.04 / openEuler 24.03 (ARM)
  • TensorFlow: 2.15+ (支持 XLA Custom Call)
  • oneDNN: v3.14+ (从源码编译, 启用 DNNL_EXPERIMENTAL_UKERNEL)
  • 编译器: GCC 12+ / Clang 15+
  • 性能分析: VTune / perf / Linux perf-tools
```

### 4.2 原型代码结构

```
flex-attention/
├── CMakeLists.txt
├── src/
│   ├── attention_primitive.hpp     # Attention 原语封装
│   ├── attention_primitive.cpp
│   ├── brgemm_wrapper.hpp          # oneDNN BRGEMM 封装
│   ├── brgemm_wrapper.cpp
│   ├── online_softmax.hpp          # Online softmax 实现
│   ├── online_softmax.cpp
│   ├── score_mod_injector.hpp      # Score modifier 注入
│   ├── mask_strategy.hpp           # Mask 跳过策略
│   ├── tile_config.hpp             # Tile 大小推导
│   ├── xla_custom_call.cpp         # XLA Custom Call 桥接
│   └── tf_op.cpp                   # TF Custom Op 注册
├── tests/
│   ├── test_correctness.py         # 数值正确性测试
│   ├── test_performance.py         # 性能基准测试
│   └── test_xla_fusion.py          # XLA 融合验证
├── benchmarks/
│   ├── bench_attention.py          # Attention 性能基准
│   └── bench_transformer.py        # Transformer 端到端
└── docs/
    └── design.md                   # 设计文档
```

### 4.3 验证测试用例

```
测试矩阵:

| 测试 | seq_len | d | mask | score_mod | 数据类型 | 验证内容 |
|------|---------|---|------|-----------|---------|---------|
| T1   | 512     | 64 | none | scale     | bf16    | 基础正确性 |
| T2   | 2048    | 128| causal | scale   | bf16    | Causal mask |
| T3   | 4096    | 128| causal | scale+ALiBi | bf16 | ALiBi 偏置 |
| T4   | 4096    | 128| sliding | scale   | bf16    | 滑动窗口 |
| T5   | 2048    | 128| causal | scale   | f32     | f32 精度 |
| T6   | 8192    | 256| causal | scale   | bf16    | 大序列 |
| T7   | 2048    | 128| causal | tanh_gate | bf16  | 自定义 score_mod |

性能基准:
  • 对比对象: tf.nn.scaled_dot_product_attention (oneDNN 后端)
  • 指标: 延迟 (ms), 吞吐 (tokens/s), 内存峰值 (MB)
  • 硬件: AMX 平台 + AVX-512 平台 + ARM SVE 平台
```

### 4.4 里程碑决策点

```
Month 1 结束 → 决策门 G1:
  ┌─ 5 个假设全部验证? ─┐
  │                     │
         Yes            No
          │              │
          ↓              ↓
    继续 Month 2-3    排查问题, 调整路线:
                     • XLA 拆散 → 纯 oneDNN primitive
                     • post-ops 不足 → 自己写 JIT
                     • 性能不达标 → 分析瓶颈

Month 3 结束 → 决策门 G2:
  ┌─ 性能 ≥3x? ─┐
  │              │
     Yes        No
      │          │
      ↓          ↓
  进入 Phase 2  继续优化 / 重新评估

Phase 2 Month 5 结束 → 决策门 G3:
  ┌─ DLRM 交互层 ≥10x? ─┐
  │                      │
     Yes                No
      │                  │
      ↓                  ↓
  继续推荐模型适配    评估推荐场景可行性
```

---

## 五、风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| XLA 拆散 attention 模式 | 中 | 高 | Custom Call (opaque), 或退化为纯 oneDNN primitive |
| oneDNN post-ops 无法表达 online softmax | 中 | 高 | 自己扩展 post-ops (修改 oneDNN 源码) 或写独立 JIT |
| AMX tile 利用率低 | 低 | 中 | Tile size autotune + VTune 分析 |
| ARM SVE BRGEMM 不成熟 | 高 | 中 | Phase 1 聚焦 x64, ARM 在 Month 3 验证 |
| TF 版本兼容性 | 低 | 中 | 明确支持 TF 2.15+, 测试矩阵覆盖 |
| 性能不达 3x | 低 | 高 | 先确认计算/内存瓶颈, 调整 tiling 策略 |

---

## 六、资源需求

### 6.1 人力资源

| 角色 | 人数 | Phase 1 | Phase 2 |
|------|------|---------|---------|
| 架构师 | 1 | 全程 | 全程 |
| C++ 开发 (oneDNN) | 1 | Month 1-3 | Month 4-5 |
| TF/XLA 集成 | 1 | Month 2-3 | Month 4-6 |
| 性能测试 | 1 | Month 1, 3 | Month 5-6 |
| ARM 适配 | 1 | Month 3 | - |

### 6.2 硬件需求

| 设备 | 用途 | 数量 |
|------|------|------|
| Intel Xeon (AMX) 服务器 | 开发 + 测试 | 1 |
| Intel Xeon (AVX-512) 服务器 | 非 AMX 测试 | 1 |
| 鲲鹏930 (ARM SVE) 服务器 | ARM 测试 | 1 (已有) |

---

## 七、预期产出

### Phase 1 产出

```
技术:
  ✅ flex_attention(Q, K, V, score_mod=..., block_mask=...) API
  ✅ 支持: causal / sliding / ALiBi / tanh_gate
  ✅ 数据类型: bf16, f32
  ✅ 硬件: x64 (AMX/AVX-512), ARM64 (SVE)
  ✅ TF Custom Op + XLA Custom Call 集成
  ✅ 性能: ≥3x vs 标准 attention

文档:
  ✅ 设计文档
  ✅ API 文档
  ✅ 性能报告
  ✅ 技术白皮书
```

### Phase 2 产出

```
技术:
  ✅ flex_interact(A, B, C, config=...) 通用 API
  ✅ DLRM 交互层加速 (~30x)
  ✅ DeepFM FM 层加速 (~20x)
  ✅ DSSM 匹配层加速 (~5x)

学术:
  ✅ 论文初稿: "FlexInteraction: A Unified Acceleration Framework for Attention and Feature Interaction on CPU"
```

---

## 八、时间线总览

```
2026-07  Phase 1 Month 1: 原型验证
         └─ 决策门 G1: 5 个假设验证

2026-08  Phase 1 Month 2: 完整 FlexAttention
         └─ score_mod + BlockMask + AVX-512

2026-09  Phase 1 Month 3: 集成 + 调优 + ARM
         └─ 决策门 G2: 性能 ≥3x?

2026-10  Phase 2 Month 4: FlexInteraction 抽象
         └─ interact/transform/aggregate 三阶段

2026-11  Phase 2 Month 5: 推荐模型适配
         └─ DLRM / DeepFM / DSSM
         └─ 决策门 G3: DLRM ≥10x?

2026-12  Phase 2 Month 6: 集成 + 论文
         └─ 端到端验证 + 论文初稿
```

---

*本计划为初版, 将根据 Phase 1 验证结果动态调整。*
