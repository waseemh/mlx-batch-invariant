"""Batch-invariant affine quantized matmul: y[M, N] = x[M, K] @ W[N, K].T

``W`` is MLX's packed affine format: ``bits``-wide unsigned codes, packed
low-to-high inside uint32 words (``32 // bits`` codes per word), with one
``scale`` and ``bias`` per ``group_size`` consecutive codes of a row:

    w[n, k] = code(n, k) * scales[n, k // group_size] + biases[n, k // group_size]

Same contract as ``kernels/matmul.py``: one tile config for every M, no
split-K, and every output element accumulated in fp32 by one thread walking
``k = 0..K-1`` in order. Dequantization happens on the fly when the weight
slab is staged into threadgroup memory, in fp32, so the reduction tree is
fixed regardless of batch size or row position.

Stock ``mx.quantized_matmul`` differs between batch 1 and batch 64 on the same
row (measured 0.125 abs in fp16 at K=N=4096, 4-bit, group 64); this kernel
does not.
"""

from __future__ import annotations

import mlx.core as mx

BM, BN, BK = 64, 64, 16
TM, TN = 4, 4
TG_X, TG_Y = BN // TN, BM // TM

_SOURCE = f"""
    const int M = x_shape[0];
    const int K = x_shape[1];
    const int N = w_shape[0];
    const int words_per_row = w_shape[1];
    const int groups_per_row = scales_shape[1];
    constexpr int PER_WORD = 32 / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;

    const int tg_m = threadgroup_position_in_grid.y;
    const int tg_n = threadgroup_position_in_grid.x;
    const int tx = thread_position_in_threadgroup.x;
    const int ty = thread_position_in_threadgroup.y;
    const int lid = ty * {TG_X} + tx;

    const int row0 = tg_m * {BM};
    const int col0 = tg_n * {BN};

    threadgroup float Xs[{BM} * {BK}];
    threadgroup float Ws[{BK} * {BN}];

    float acc[{TM}][{TN}];
    for (int i = 0; i < {TM}; ++i)
        for (int j = 0; j < {TN}; ++j)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += {BK}) {{
        for (int e = lid; e < {BM} * {BK}; e += {TG_X * TG_Y}) {{
            int i = e / {BK}, kk = e % {BK};
            int m = row0 + i, k = k0 + kk;
            Xs[e] = (m < M && k < K) ? (float)x[(size_t)m * K + k] : 0.0f;
        }}
        // stage W[col0:col0+BN, k0:k0+BK] dequantized, laid out [kk][j]
        for (int e = lid; e < {BK} * {BN}; e += {TG_X * TG_Y}) {{
            int kk = e / {BN}, j = e % {BN};
            int k = k0 + kk, n = col0 + j;
            float val = 0.0f;
            if (k < K && n < N) {{
                uint word = w[(size_t)n * words_per_row + k / PER_WORD];
                uint code = (word >> (BITS * (k % PER_WORD))) & MASK;
                int g = k / GS;
                float s = (float)scales[(size_t)n * groups_per_row + g];
                float b = (float)biases[(size_t)n * groups_per_row + g];
                val = fma((float)code, s, b);
            }}
            Ws[e] = val;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < {BK}; ++kk) {{
            float xreg[{TM}];
            float wreg[{TN}];
            for (int i = 0; i < {TM}; ++i)
                xreg[i] = Xs[(ty * {TM} + i) * {BK} + kk];
            for (int j = 0; j < {TN}; ++j)
                wreg[j] = Ws[kk * {BN} + tx * {TN} + j];
            for (int i = 0; i < {TM}; ++i)
                for (int j = 0; j < {TN}; ++j)
                    acc[i][j] = fma(xreg[i], wreg[j], acc[i][j]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (int i = 0; i < {TM}; ++i) {{
        int m = row0 + ty * {TM} + i;
        if (m >= M) continue;
        for (int j = 0; j < {TN}; ++j) {{
            int n = col0 + tx * {TN} + j;
            if (n >= N) continue;
            out[(size_t)m * N + n] = (T)acc[i][j];
        }}
    }}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="bi_quantized_matmul",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _kernel


SUPPORTED_BITS = (4, 8)
SUPPORTED_GROUPS = (32, 64, 128)


def supports(x, w, scales, biases, transpose, group_size, bits, mode) -> bool:
    return (
        mode == "affine"
        and transpose
        and biases is not None
        and bits in SUPPORTED_BITS
        and group_size in SUPPORTED_GROUPS
        and isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating)
        and w.dtype == mx.uint32 and w.ndim == 2
        and scales.ndim == 2 and biases.ndim == 2
        and x.shape[-1] % group_size == 0
        and w.shape[1] * (32 // bits) == x.shape[-1]
    )


def batch_invariant_quantized_matmul(
    x: mx.array, w: mx.array, scales: mx.array, biases: mx.array,
    *, group_size: int, bits: int,
) -> mx.array:
    lead = x.shape[:-1]
    K = x.shape[-1]
    N = w.shape[0]
    x2 = x.reshape(-1, K) if x.ndim != 2 else x
    M = x2.shape[0]
    dt = x.dtype
    if scales.dtype != dt:
        scales = scales.astype(dt)
    if biases.dtype != dt:
        biases = biases.astype(dt)
    grid = ((N + BN - 1) // BN * TG_X, (M + BM - 1) // BM * TG_Y, 1)
    out = _get_kernel()(
        inputs=[x2, w, scales, biases],
        template=[("T", dt), ("BITS", bits), ("GS", group_size)],
        grid=grid,
        threadgroup=(TG_X, TG_Y, 1),
        output_shapes=[(M, N)],
        output_dtypes=[dt],
    )[0]
    return out.reshape(*lead, N)
