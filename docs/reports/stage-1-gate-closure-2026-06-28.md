# OneDNN-Flash Stage 1.4 Gate Closure and Diff Grouping

> Date: 2026-06-28
> Purpose: close the Stage 1 post-ops validation gate and prevent mixed commits from hiding design/implementation boundaries.

## 1. Current Baseline

Current HEAD before this worktree is:

```text
5c6e98c Add configurable QK tile layouts
```

Current worktree is intentionally large because it contains:

1. pre-Stage-1 BRGEMM/local oneDNN exploratory changes;
2. system design package;
3. Stage 1.0 RuntimePlan / ScoreModPlan skeleton;
4. Stage 1.1 oneDNN post-ops capability probe;
5. Stage 1.2/1.3 QK score-tile post-ops implementation and benchmark/report.

Do **not** commit the whole worktree as one patch.

## 2. Validation Status

Latest full validation passed:

```bash
cmake --build build-brgemm-main -j2
ctest --test-dir build-brgemm-main --output-on-failure
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
git diff --check
```

Observed results:

```text
CTest: 11/11 passed
Python/TensorFlow pytest: 17 passed
git diff --check: clean
```

Stage 1 benchmark artifacts:

```text
benchmarks/results/stage-1-postops/onednn-postops-capability.json
benchmarks/results/stage-1-postops/cpp-qk-postops.json
benchmarks/results/stage-1-postops/cpp-qk-postops.csv
docs/reports/stage-1-postops-validation-report.md
```

## 3. Proposed Commit Groups

### Commit A — System design package

Purpose: make the design baseline reviewable independently from code.

Files:

```text
docs/design.md
docs/project-plan.md
docs/system-design.md
docs/design-self-review-2026-06-27.md
docs/system-design-freeze-checklist.md
docs/stage-1-postops-validation-plan.md
docs/stage-1-interface-contract.md
docs/modules/README.md
docs/modules/attention-engine.md
docs/modules/block-mask-planner.md
docs/modules/development-flow.md
docs/modules/onednn-integration.md
docs/modules/python-tf-api.md
docs/modules/runtime-planner.md
docs/modules/score-mod-lowering.md
docs/modules/tensorflow-custom-op.md
docs/modules/verification-benchmark.md
docs/modules/xla-custom-call.md
```

Suggested message:

```text
Document OneDNN-Flash system design and Stage 1 post-ops plan
```

Gate:

```bash
git diff --check <files above>
```

### Commit B — Local oneDNN/BRGEMM baseline context

Purpose: isolate BRGEMM/local oneDNN exploratory backend support from the Stage 1 post-ops path.

Files:

```text
.gitignore
cmake/FindLocalDnnl.cmake
include/onednn_flash/onednn_brgemm_tile_kernel.hpp
src/onednn_flash/onednn_brgemm_tile_kernel.cpp
include/onednn_flash/attention.hpp
include/onednn_flash/attention_workspace.hpp
include/onednn_flash/tile_kernel.hpp
src/onednn_flash/attention.cpp
src/onednn_flash/tile_kernel.cpp
benchmarks/bench_attention.cpp
docs/backend/onednn-brgemm-notes.md
tests/cpp/test_tile_kernel.cpp
python/onednn_flash_tf/ops.py
tensorflow_ops/onednn_flash_attention_op.cc
tests/tensorflow/test_onednn_flash_tf_op.py
```

Important boundary:

- This is **not** the Stage 1 post-op main path.
- BRGEMM remains baseline/candidate context.
- Any performance claims must stay local to the affected benchmark/test.

Suggested message:

```text
Add oneDNN BRGEMM baseline context and transformed-K experiment
```

Gate:

```bash
cmake --build build-brgemm-main -j2
ctest --test-dir build-brgemm-main --output-on-failure
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
```

Risk notes:

- This group touches TensorFlow op attrs (`tile_kernel`, `qk_tile_layout`) and may be considered ABI/API expansion.
- If reviewers want Stage 1 to stay completely independent, commit B can be delayed or split again.

### Commit C — Stage 1.0 RuntimePlan / ScoreModPlan skeleton

Purpose: introduce the public semantic planning API frozen by the interface contract.

Files:

```text
CMakeLists.txt
include/onednn_flash/runtime_plan.hpp
include/onednn_flash/score_mod_plan.hpp
src/onednn_flash/runtime_plan.cpp
src/onednn_flash/score_mod_plan.cpp
tests/cpp/test_runtime_plan.cpp
```

Suggested message:

```text
Add Stage 1 RuntimePlan and ScoreModPlan skeleton
```

Gate:

```bash
cmake --build build-brgemm-main -j2
./build-brgemm-main/onednn_flash_runtime_plan_tests
ctest --test-dir build-brgemm-main --output-on-failure
```

### Commit D — Stage 1.1 oneDNN post-ops capability probe

Purpose: record machine-checked oneDNN post-op support boundary.

Files:

```text
CMakeLists.txt
tests/cpp/test_onednn_postops_capability.cpp
docs/backend/onednn-postops-capability-2026-06-27.md
benchmarks/results/stage-1-postops/onednn-postops-capability.json
```

Suggested message:

```text
Probe oneDNN matmul post-ops capability for Stage 1
```

Gate:

```bash
cmake --build build-brgemm-main -j2
./build-brgemm-main/onednn_flash_onednn_postops_capability
ctest --test-dir build-brgemm-main --output-on-failure
```

### Commit E — Stage 1.2/1.3 QK post-ops path and report

Purpose: connect RuntimePlan/ScoreModPlan to a minimal QK score-tile execution path.

Files:

```text
CMakeLists.txt
src/onednn_flash/qk_score_tile.cpp
src/onednn_flash/qk_score_tile_internal.hpp
tests/cpp/test_qk_score_tile.cpp
benchmarks/bench_qk_postops.cpp
benchmarks/results/stage-1-postops/cpp-qk-postops.json
benchmarks/results/stage-1-postops/cpp-qk-postops.csv
docs/reports/stage-1-postops-validation-report.md
```

Suggested message:

```text
Validate Stage 1 QK score tile post-ops path
```

Gate:

```bash
cmake --build build-brgemm-main -j2
./build-brgemm-main/onednn_flash_qk_score_tile_tests
./build-brgemm-main/onednn_flash_qk_postops_bench --warmup 2 --repeat 5 \
  --output-json benchmarks/results/stage-1-postops/cpp-qk-postops.json \
  --output-csv benchmarks/results/stage-1-postops/cpp-qk-postops.csv
ctest --test-dir build-brgemm-main --output-on-failure
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
git diff --check
```

## 4. Commit Ordering Recommendation

Recommended order:

```text
A -> C -> D -> E -> B
```

Rationale:

- The design package explains the Stage 1 boundary first.
- RuntimePlan/ScoreModPlan is the semantic base.
- Capability probe provides evidence.
- QK post-op path consumes both.
- BRGEMM exploratory context should land last or separately so it does not look like the Stage 1 center.

If the goal is a smaller PR, use:

```text
A -> C -> D -> E
```

and defer Commit B.

## 5. Stage 1.4 Gate Answers

1. Does oneDNN matmul support scale post-op?
   Yes. Stage 1 uses `eltwise_linear(scale, 0)` and capability probe passed compile/run.

2. Does oneDNN matmul support same-shape additive bias?
   Yes. Stage 1 uses binary add post-op with same-shape bias memory as runtime post-op arg.

3. Does broadcast bias support become Stage 1 fast path?
   No. Probe shows support, but RuntimePlan keeps broadcast row/col as fallback until design promotes it.

4. What is the BRGEMM ukernel post-op boundary?
   BRGEMM ukernel is available as baseline/candidate context, but Stage 1 post-op path does not use ukernel BRGEMM post-ops.

5. Is correctness stable for scale/additive bias lowering?
   Yes. `test_qk_score_tile.cpp` and `cpp-qk-postops` records match reference with `max_abs_diff <= 1e-6`.

6. What is the main performance caveat?
   Small tiles are overhead-sensitive because the current QK post-op path constructs oneDNN engine/stream/primitive/memory wrappers per call. Larger 128x128x64 tiles show clear oneDNN advantage at isolated C++ QK-tile level.

7. What should happen next?
   Close commits first. Then do a focused RuntimePlan/oneDNN integration hardening pass: primitive cache key layering, memory wrapper reuse, and stream/engine lifetime cleanup. XLA CustomCall ABI should wait until this is stable.

## 6. Risks / Non-goals

- Do not claim TensorFlow graph or XLA speedup from C++ QK-tile results.
- Do not promote broadcast bias to fast path without design update.
- Do not make BRGEMM the Stage 1 main path.
- Do not hide TensorFlow op ABI changes inside the post-ops commit.
- Do not mix generated benchmark JSON with unrelated historical benchmark outputs.

## 7. Final Gate Status

```text
System design: closed for Stage 1
Stage 1.0 skeleton: implemented and tested
Stage 1.1 capability probe: implemented and tested
Stage 1.2/1.3 QK post-ops path: implemented, benchmarked, reported
Stage 1.4 closure: this document defines commit groups and next gate
```

Stage 1 is ready for review once commits are split according to the groups above.
