"""Batch-invariant matmul: C[M, N] = A[M, K] @ B[K, N].

Design (mirrors the Thinking Machines recipe):

* One tile configuration (BM x BN x BK) for every shape. Stock GEMMs pick
  split-K or different tiles when M is small, which changes the reduction
  order of each output element as the batch grows. We never do that.
* Each output element is accumulated in fp32 by exactly one thread, walking
  k = 0, 1, ..., K-1 in order. The reduction tree for C[m, n] therefore
  depends only on K, never on M, N, or which threadgroup it landed in.
* B is read through its strides, so a transposed weight view (what
  ``nn.Linear`` passes) is consumed in place without a contiguous copy.

This is a correct, invariant, *unoptimized* kernel: ~an order of magnitude
below MLX's tuned GEMM. The point of the stub is the contract, not speed.
"""

from __future__ import annotations

import mlx.core as mx

BM = 64  # rows of C per threadgroup
BN = 64  # cols of C per threadgroup
BK = 16  # K slab staged in threadgroup memory per iteration
TM = 4   # rows of C per thread
TN = 4   # cols of C per thread
TG_X = BN // TN  # 16
TG_Y = BM // TM  # 16

_SOURCE = f"""
    // ---- geometry -------------------------------------------------------
    const int M = a_shape[0];
    const int K = a_shape[1];
    const int N = b_shape[1];
    const int a_sm = a_strides[0], a_sk = a_strides[1];
    const int b_sk = b_strides[0], b_sn = b_strides[1];

    const int tg_m = threadgroup_position_in_grid.y;  // which BM block
    const int tg_n = threadgroup_position_in_grid.x;  // which BN block
    const int tx = thread_position_in_threadgroup.x;  // 0..{TG_X - 1}
    const int ty = thread_position_in_threadgroup.y;  // 0..{TG_Y - 1}
    const int lid = ty * {TG_X} + tx;                 // 0..{TG_X * TG_Y - 1}

    const int row0 = tg_m * {BM};
    const int col0 = tg_n * {BN};

    threadgroup float As[{BM} * {BK}];
    threadgroup float Bs[{BK} * {BN}];

    float acc[{TM}][{TN}];
    for (int i = 0; i < {TM}; ++i)
        for (int j = 0; j < {TN}; ++j)
            acc[i][j] = 0.0f;

    // ---- main loop: fixed order over K, no split-K ----------------------
    for (int k0 = 0; k0 < K; k0 += {BK}) {{
        // cooperative load of A[row0:row0+BM, k0:k0+BK]
        for (int e = lid; e < {BM} * {BK}; e += {TG_X * TG_Y}) {{
            int i = e / {BK}, kk = e % {BK};
            int m = row0 + i, k = k0 + kk;
            As[e] = (m < M && k < K) ? (float)a[m * a_sm + k * a_sk] : 0.0f;
        }}
        // cooperative load of B[k0:k0+BK, col0:col0+BN]
        for (int e = lid; e < {BK} * {BN}; e += {TG_X * TG_Y}) {{
            int kk = e / {BN}, j = e % {BN};
            int k = k0 + kk, n = col0 + j;
            Bs[e] = (k < K && n < N) ? (float)b[k * b_sk + n * b_sn] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // every thread: sequential fma over the slab, fixed order
        for (int kk = 0; kk < {BK}; ++kk) {{
            float areg[{TM}];
            float breg[{TN}];
            for (int i = 0; i < {TM}; ++i)
                areg[i] = As[(ty * {TM} + i) * {BK} + kk];
            for (int j = 0; j < {TN}; ++j)
                breg[j] = Bs[kk * {BN} + tx * {TN} + j];
            for (int i = 0; i < {TM}; ++i)
                for (int j = 0; j < {TN}; ++j)
                    acc[i][j] = fma(areg[i], breg[j], acc[i][j]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // ---- store -------------------------------------------------------------
    for (int i = 0; i < {TM}; ++i) {{
        int m = row0 + ty * {TM} + i;
        if (m >= M) continue;
        for (int j = 0; j < {TN}; ++j) {{
            int n = col0 + tx * {TN} + j;
            if (n >= N) continue;
            out[m * N + n] = (T)acc[i][j];
        }}
    }}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="bi_matmul",
            input_names=["a", "b"],
            output_names=["out"],
            source=_SOURCE,
            ensure_row_contiguous=False,
        )
    return _kernel


def batch_invariant_matmul(a: mx.array, b: mx.array) -> mx.array:
    """``a @ b`` for 2-D ``a`` and ``b`` with a batch-invariant reduction order.

    ``a`` may have extra leading dims; they are flattened into M. ``b`` must be
    2-D (a transposed view is fine). Output dtype follows the promoted input
    dtype; accumulation is always fp32.
    """
    if b.ndim != 2:
        raise ValueError("batch_invariant_matmul: b must be 2-D")
    if a.ndim < 1:
        raise ValueError("batch_invariant_matmul: a must be at least 1-D")
    lead = a.shape[:-1]
    K = a.shape[-1]
    if K != b.shape[0]:
        raise ValueError(f"shape mismatch {a.shape} @ {b.shape}")
    N = b.shape[1]
    out_dtype = mx.result_type(a, b) if hasattr(mx, "result_type") else a.dtype
    a2 = a.reshape(-1, K) if a.ndim != 2 else a
    if a2.dtype != out_dtype:
        a2 = a2.astype(out_dtype)
    if b.dtype != out_dtype:
        b = b.astype(out_dtype)
    M = a2.shape[0]

    grid = ((N + BN - 1) // BN * TG_X, (M + BM - 1) // BM * TG_Y, 1)
    out = _get_kernel()(
        inputs=[a2, b],
        template=[("T", out_dtype)],
        grid=grid,
        threadgroup=(TG_X, TG_Y, 1),
        output_shapes=[(M, N)],
        output_dtypes=[out_dtype],
    )[0]
    return out.reshape(*lead, N)
