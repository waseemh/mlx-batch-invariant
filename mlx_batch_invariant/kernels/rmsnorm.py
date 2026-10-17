"""Batch-invariant RMSNorm.

One threadgroup of ``TG`` threads per row, regardless of how many rows are in
the launch. Each thread sums a fixed stride of squares in fp32, then the
threadgroup reduces through a fixed-shape tree. Nothing about the reduction
depends on the batch size, so the result for a row is bitwise identical
whether it is alone or one of thousands.

Stock ``mx.fast.rms_norm`` already looked invariant in our probes; this
kernel exists so the contract is *guaranteed* rather than observed.
"""

from __future__ import annotations

import mlx.core as mx

TG = 256

_SOURCE = f"""
    threadgroup float red[{TG}];
    const int D = x_shape[x_ndim - 1];
    const int row = threadgroup_position_in_grid.x;
    const int t = thread_position_in_threadgroup.x;
    const device T* xr = x + (size_t)row * D;

    // 1) per-thread partial sum, fixed stride, fixed order
    float ss = 0.0f;
    for (int i = t; i < D; i += {TG}) {{
        float v = (float)xr[i];
        ss = fma(v, v, ss);
    }}
    red[t] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) fixed-shape tree reduction
    for (int s = {TG} / 2; s > 0; s >>= 1) {{
        if (t < s) red[t] += red[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    const float inv = metal::precise::rsqrt(red[0] / (float)D + eps[0]);

    // 3) normalize + scale
    for (int i = t; i < D; i += {TG}) {{
        float v = (float)xr[i] * inv;
        out[(size_t)row * D + i] = (T)(v * (float)w[i]);
    }}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="bi_rms_norm",
            input_names=["x", "w", "eps"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _kernel


def batch_invariant_rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    D = x.shape[-1]
    if weight is None:
        weight = mx.ones((D,), dtype=x.dtype)
    rows = max(1, x.size // D)
    x2 = x.reshape(rows, D)
    eps_arr = mx.array([eps], dtype=mx.float32)
    out = _get_kernel()(
        inputs=[x2, weight, eps_arr],
        template=[("T", x.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(rows, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(x.shape)
