"""Batch-invariant decode attention (T_q == 1) with a fixed KV split size.

Why the split *size* is fixed and not the split *count*: a decode step must
read the whole KV cache, so some parallelism along the KV axis is required.
Stock kernels pick "n splits" from the batch size and sequence length, which
changes which keys get reduced together as the batch grows. We instead chunk
the KV axis into blocks of exactly ``SPLIT_SIZE`` keys. Key ``j`` always lands
in block ``j // SPLIT_SIZE`` at position ``j % SPLIT_SIZE``, and the blocks are
combined in index order. Growing the batch, or appending tokens to the cache,
cannot change the reduction tree of an existing (batch, head) pair.

Two kernels:

1. ``partial``: one thread per (batch*head, split). Online softmax over the
   keys of that split, sequentially, in fp32. Emits (m, l, acc[D]).
2. ``combine``: one thread per (batch*head, d). Walks the splits in order and
   merges (m, l, acc) with the standard rescaling. Emits the normalized output.

Supported: ``mask`` is ``None`` or ``"causal"`` (a no-op for a single query
against a full cache), no attention sinks, GQA/MQA via integer head ratio.
Everything else falls back to the stock kernel (counted by the mode so you
can see it). Prefill (T_q > 1) is a TODO; see README.
"""

from __future__ import annotations

import mlx.core as mx

SPLIT_SIZE = 256

_PARTIAL_SRC = """
    const int BH      = q_shape[0];        // B * Hq
    const int Hq      = hp[0];
    const int Hkv     = hp[1];
    const int L       = k_shape[2];
    const int nsplit  = hp[2];
    const float scale = scl[0];

    const int gid = thread_position_in_grid.x;
    if (gid >= BH * nsplit) return;
    const int bh    = gid / nsplit;
    const int split = gid % nsplit;
    const int b     = bh / Hq;
    const int h     = bh % Hq;
    const int hk    = h / (Hq / Hkv);

    const device T* qr = q + (size_t)bh * D;
    const device T* kb = k + ((size_t)b * Hkv + hk) * (size_t)L * D;
    const device T* vb = v + ((size_t)b * Hkv + hk) * (size_t)L * D;

    float qreg[D];
    for (int d = 0; d < D; ++d) qreg[d] = (float)qr[d] * scale;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[D];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;

    const int j0 = split * SPLIT;
    const int j1 = metal::min(L, j0 + SPLIT);
    for (int j = j0; j < j1; ++j) {
        const device T* kr = kb + (size_t)j * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s = fma(qreg[d], (float)kr[d], s);
        float m_new = metal::max(m, s);
        float corr  = metal::exp(m - m_new);
        float p     = metal::exp(s - m_new);
        l = fma(l, corr, p);
        const device T* vr = vb + (size_t)j * D;
        for (int d = 0; d < D; ++d) acc[d] = fma(acc[d], corr, p * (float)vr[d]);
        m = m_new;
    }

    const size_t po = ((size_t)bh * nsplit + split);
    pm[po] = m;
    pl[po] = l;
    for (int d = 0; d < D; ++d) pacc[po * D + d] = acc[d];
"""

_COMBINE_SRC = """
    const int BH     = pm_shape[0];
    const int nsplit = pm_shape[1];
    const int gid = thread_position_in_grid.x;
    if (gid >= BH * D) return;
    const int bh = gid / D;
    const int d  = gid % D;

    float m = -INFINITY, l = 0.0f, a = 0.0f;
    for (int i = 0; i < nsplit; ++i) {
        const size_t po = (size_t)bh * nsplit + i;
        float mi = pm[po];
        float m_new = metal::max(m, mi);
        float c_old = metal::exp(m - m_new);
        float c_new = metal::exp(mi - m_new);
        l = fma(l, c_old, pl[po] * c_new);
        a = fma(a, c_old, pacc[po * D + d] * c_new);
        m = m_new;
    }
    out[(size_t)bh * D + d] = (T)(a / l);
"""

_partial = None
_combine = None


def _kernels():
    global _partial, _combine
    if _partial is None:
        _partial = mx.fast.metal_kernel(
            name="bi_decode_attn_partial",
            input_names=["q", "k", "v", "hp", "scl"],
            output_names=["pm", "pl", "pacc"],
            source=_PARTIAL_SRC,
        )
        _combine = mx.fast.metal_kernel(
            name="bi_decode_attn_combine",
            input_names=["pm", "pl", "pacc"],
            output_names=["out"],
            source=_COMBINE_SRC,
        )
    return _partial, _combine


def supports(q: mx.array, k: mx.array, v: mx.array, mask, sinks) -> bool:
    return (
        q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        and q.shape[2] == 1
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
        and sinks is None
        and q.shape[1] % k.shape[1] == 0
        and k.shape == v.shape
        and q.shape[-1] == k.shape[-1]
    )


def batch_invariant_decode_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, split_size: int = SPLIT_SIZE
) -> mx.array:
    B, Hq, Tq, D = q.shape
    _, Hkv, L, _ = k.shape
    assert Tq == 1
    nsplit = (L + split_size - 1) // split_size
    BH = B * Hq
    dt = q.dtype
    if k.dtype != dt:
        k = k.astype(dt)
    if v.dtype != dt:
        v = v.astype(dt)
    q2 = q.reshape(BH, D)
    hp = mx.array([Hq, Hkv, nsplit], dtype=mx.int32)
    scl = mx.array([scale], dtype=mx.float32)
    partial, combine = _kernels()

    n = BH * nsplit
    pm, pl, pacc = partial(
        inputs=[q2, k, v, hp, scl],
        template=[("T", dt), ("D", D), ("SPLIT", split_size)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 64), 1, 1),
        output_shapes=[(BH, nsplit), (BH, nsplit), (BH, nsplit, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    n2 = BH * D
    out = combine(
        inputs=[pm, pl, pacc],
        template=[("T", dt), ("D", D)],
        grid=(n2, 1, 1),
        threadgroup=(min(n2, 256), 1, 1),
        output_shapes=[(BH, D)],
        output_dtypes=[dt],
    )[0]
    return out.reshape(B, Hq, 1, D)
