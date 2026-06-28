# oneDNN Post-ops Capability Probe

> Date: 2026-06-27
> Scope: FlashOne Stage 1.1 capability probe only; not a performance benchmark.

## Results

| Name | Supported | Method | Compile | Run | Reason |
|---|---:|---|---|---|---|
| `matmul.scale` | yes | `post_op` | `passed` | `passed` |  |
| `matmul.binary_add_same_shape` | yes | `post_op` | `passed` | `passed` |  |
| `matmul.binary_add_broadcast_row` | yes | `post_op` | `passed` | `passed` |  |
| `matmul.binary_add_broadcast_col` | yes | `post_op` | `passed` | `passed` |  |
| `matmul.eltwise_linear` | yes | `post_op` | `passed` | `passed` |  |
| `ukernel_brgemm.post_ops_boundary` | no | `unsupported` | `passed` | `not_tested` | oneDNN ukernel BRGEMM API is available, but Stage 1 does not expose post-op support for ukernel BRGEMM; use dnnl::matmul post-ops instead |
| `ukernel_transform.supported_out_ld` | yes | `runtime_arg` | `passed` | `passed` | supported out_ld values observed by integration contract: 16,32,48,64 |

## Stage 1 Interpretation

- Stage 1 main path remains `dnnl::matmul + post_ops`.
- BRGEMM ukernel is capability/baseline context only, not the Stage 1 post-op path.
- Broadcast rows are probed for evidence, but Stage 1 RuntimePlan still treats broadcast bias as fallback until explicitly promoted by design.
- This probe does not claim TensorFlow graph/XLA speedup.
