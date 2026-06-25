#include "flashone/batched_attention.hpp"

#include <cmath>
#include <exception>
#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT

REGISTER_OP("FlashOneAttention")
    .Input("q: float")
    .Input("k: float")
    .Input("v: float")
    .Output("out: float")
    .Attr("causal: bool = true")
    .Attr("query_block_size: int = 16")
    .Attr("key_block_size: int = 32")
    .Attr("use_onednn: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle q;
        shape_inference::ShapeHandle k;
        shape_inference::ShapeHandle v;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &q));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &k));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &v));

        shape_inference::DimensionHandle batch = c->Dim(q, 0);
        shape_inference::DimensionHandle heads = c->Dim(q, 1);
        shape_inference::DimensionHandle query_tokens = c->Dim(q, 2);
        shape_inference::DimensionHandle value_dim = c->Dim(v, 3);
        c->set_output(0, c->MakeShape({batch, heads, query_tokens, value_dim}));
        return OkStatus();
    });

class FlashOneAttentionOp final : public OpKernel {
public:
    explicit FlashOneAttentionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("causal", &causal_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("query_block_size", &query_block_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("key_block_size", &key_block_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_onednn", &use_onednn_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& q = ctx->input(0);
        const Tensor& k = ctx->input(1);
        const Tensor& v = ctx->input(2);

        OP_REQUIRES(ctx, q.dims() == 4, errors::InvalidArgument("q must be rank 4 [B,H,M,D]"));
        OP_REQUIRES(ctx, k.dims() == 4, errors::InvalidArgument("k must be rank 4 [B,H,N,D]"));
        OP_REQUIRES(ctx, v.dims() == 4, errors::InvalidArgument("v must be rank 4 [B,H,N,Dv]"));
        OP_REQUIRES(ctx, q.dim_size(0) == k.dim_size(0) && q.dim_size(0) == v.dim_size(0),
                    errors::InvalidArgument("batch dimensions must match"));
        OP_REQUIRES(ctx, q.dim_size(1) == k.dim_size(1) && q.dim_size(1) == v.dim_size(1),
                    errors::InvalidArgument("head dimensions must match"));
        OP_REQUIRES(ctx, q.dim_size(3) == k.dim_size(3),
                    errors::InvalidArgument("q/k head dimensions must match"));
        OP_REQUIRES(ctx, k.dim_size(2) == v.dim_size(2),
                    errors::InvalidArgument("k/v key-token dimensions must match"));
        OP_REQUIRES(ctx, query_block_size_ > 0 && key_block_size_ > 0,
                    errors::InvalidArgument("block sizes must be positive"));

        TensorShape out_shape({q.dim_size(0), q.dim_size(1), q.dim_size(2), v.dim_size(3)});
        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

        flashone::BatchedAttentionShape shape{
            static_cast<std::size_t>(q.dim_size(0)),
            static_cast<std::size_t>(q.dim_size(1)),
            static_cast<std::size_t>(q.dim_size(2)),
            static_cast<std::size_t>(k.dim_size(2)),
            static_cast<std::size_t>(q.dim_size(3)),
            static_cast<std::size_t>(v.dim_size(3)),
        };

        flashone::AttentionOptions options;
        options.scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
        options.causal = causal_;
        options.query_block_size = static_cast<std::size_t>(query_block_size_);
        options.key_block_size = static_cast<std::size_t>(key_block_size_);
#ifdef FLASHONE_HAS_ONEDNN
        if (use_onednn_) {
            options.qk_tile_kernel = flashone::TileKernelKind::OneDnn;
            options.pv_tile_kernel = flashone::TileKernelKind::OneDnn;
        }
#endif

        try {
            flashone::flash_attention_batched_qk_pv_tile(q.flat<float>().data(),
                                                         k.flat<float>().data(),
                                                         v.flat<float>().data(),
                                                         out->flat<float>().data(),
                                                         shape,
                                                         options);
        } catch (const std::exception& e) {
            ctx->CtxFailure(errors::Internal("FlashOneAttention failed: ", e.what()));
        }
    }

private:
    bool causal_ = true;
    int query_block_size_ = 16;
    int key_block_size_ = 32;
    bool use_onednn_ = true;
};

REGISTER_KERNEL_BUILDER(Name("FlashOneAttention").Device(DEVICE_CPU), FlashOneAttentionOp);
