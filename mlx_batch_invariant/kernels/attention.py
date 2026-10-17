"""Batch-invariant attention with a fixed KV split size.

Why the split *size* is fixed and not the split *count*: attention must read
the whole KV cache, so some parallelism along the KV axis is required. Stock
kernels pick "n splits" from batch size and sequence length, which changes
which keys get reduced together as the batch grows. We instead chunk the KV
axis into blocks of exactly ``SPLIT_SIZE`` keys. Key ``j`` always lands in
block ``j // SPLIT_SIZE`` at position ``j % SPLIT_SIZE``, and blocks are
merged in index order. Growing the batch, or appending tokens to the cache,
cannot change the reduction tree of an existing (batch, head, query) row.

Three kernels sharing one inner loop (``_BLOCK_LOOP``) and one merge formula
(``_MERGE``) so that **decode and prefill produce the same bits for the same
query**:

* ``partial`` (decode): one thread per (batch*head, split). Online softmax
  over that split's keys. Emits (m, l, acc[D]).
* ``combine`` (decode): one thread per (batch*head, d). Merges splits in order.
* ``prefill``: one thread per (batch*head, query). Runs the same per-split
  loop and the same merge, sequentially, for the keys that query may see.
  No partial buffer, so memory is O(B*H*T*D) regardless of KV length.

Causal convention matches ``mx.fast.scaled_dot_product_attention``: with a
KV cache of length L and T queries, query i attends keys 0..L-T+i.

Supported: ``mask`` is ``None`` or ``"causal"``, no attention sinks, GQA/MQA
via integer head ratio. Array masks fall back to the stock kernel (counted).
"""

from __future__ import annotations

import mlx.core as mx

SPLIT_SIZE = 256

# Online softmax over keys [j0, j1) of one (b, hk) stream into (m, l, acc).
# Identical text in every kernel => identical instruction sequence.
_BLOCK_LOOP = """
        float m = -INFINITY;
        float l = 0.0f;
        float acc[D];
        for (int d = 0; d < D; ++d) acc[d] = 0.0f;
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
"""

# Merge a split's (m, l, acc) into the running (M, Ls, A). Scalar form is used
# by the decode combine kernel (one d per thread); vector form by prefill.
_MERGE_SCALARS = """
            float m_new = metal::max(M, m);
            float c_old = metal::exp(M - m_new);
            float c_new = metal::exp(m - m_new);
            Ls = fma(Ls, c_old, l * c_new);
"""

_PARTIAL_SRC = f"""
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

    const int j0 = split * SPLIT;
    const int j1 = metal::min(L, j0 + SPLIT);
    {{
{_BLOCK_LOOP}
        const size_t po = ((size_t)bh * nsplit + split);
        pm[po] = m;
        pl[po] = l;
        for (int d = 0; d < D; ++d) pacc[po * D + d] = acc[d];
    }}
"""

_COMBINE_SRC = f"""
    const int BH     = pm_shape[0];
    const int nsplit = pm_shape[1];
    const int gid = thread_position_in_grid.x;
    if (gid >= BH * D) return;
    const int bh = gid / D;
    const int d  = gid % D;

    float M = -INFINITY, Ls = 0.0f, A = 0.0f;
    for (int i = 0; i < nsplit; ++i) {{
        const size_t po = (size_t)bh * nsplit + i;
        const float m = pm[po];
        const float l = pl[po];
        {{
{_MERGE_SCALARS}
            A = fma(A, c_old, pacc[po * D + d] * c_new);
            M = m_new;
        }}
    }}
    out[(size_t)bh * D + d] = (T)(A / Ls);
"""

_PREFILL_SRC = f"""
    const int Hq      = hp[0];
    const int Hkv     = hp[1];
    const int Tq      = hp[2];
    const int causal  = hp[3];
    const int L       = k_shape[2];
    const float scale = scl[0];
    const int BHT = q_shape[0];            // B * Hq * Tq

    const int gid = thread_position_in_grid.x;
    if (gid >= BHT) return;
    const int qi = gid % Tq;
    const int bh = gid / Tq;
    const int b  = bh / Hq;
    const int h  = bh % Hq;
    const int hk = h / (Hq / Hkv);

    const device T* qr = q + (size_t)gid * D;
    const device T* kb = k + ((size_t)b * Hkv + hk) * (size_t)L * D;
    const device T* vb = v + ((size_t)b * Hkv + hk) * (size_t)L * D;

    float qreg[D];
    for (int d = 0; d < D; ++d) qreg[d] = (float)qr[d] * scale;

    const int nkeys  = causal ? (L - Tq + qi + 1) : L;
    const int nsplit = (nkeys + SPLIT - 1) / SPLIT;

    float M = -INFINITY, Ls = 0.0f;
    float A[D];
    for (int d = 0; d < D; ++d) A[d] = 0.0f;

    for (int split = 0; split < nsplit; ++split) {{
        const int j0 = split * SPLIT;
        const int j1 = metal::min(nkeys, j0 + SPLIT);
{_BLOCK_LOOP}
        {{
{_MERGE_SCALARS}
            for (int d = 0; d < D; ++d) A[d] = fma(A[d], c_old, acc[d] * c_new);
            M = m_new;
        }}
    }}
    for (int d = 0; d < D; ++d) out[(size_t)gid * D + d] = (T)(A[d] / Ls);
"""

_partial = _combine = _prefill = None


def _kernels():
    global _partial, _combine, _prefill
    if _partial is None:
        _partial = mx.fast.metal_kernel(
            name="bi_attn_partial",
            input_names=["q", "k", "v", "hp", "scl"],
            output_names=["pm", "pl", "pacc"],
            source=_PARTIAL_SRC,
        )
        _combine = mx.fast.metal_kernel(
            name="bi_attn_combine",
            input_names=["pm", "pl", "pacc"],
            output_names=["out"],
            source=_COMBINE_SRC,
        )
        _prefill = mx.fast.metal_kernel(
            name="bi_attn_prefill",
            input_names=["q", "k", "v", "hp", "scl"],
            output_names=["out"],
            source=_PREFILL_SRC,
        )
    return _partial, _combine, _prefill


def supports(q: mx.array, k: mx.array, v: mx.array, mask, sinks) -> bool:
    return (
        q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
        and sinks is None
        and q.shape[1] % k.shape[1] == 0
        and k.shape == v.shape
        and q.shape[-1] == k.shape[-1]
        and q.shape[0] == k.shape[0]
        and (mask is None or q.shape[2] <= k.shape[2])
    )


def _prep(q, k, v):
    dt = q.dtype
    if k.dtype != dt:
        k = k.astype(dt)
    if v.dtype != dt:
        v = v.astype(dt)
    return dt, k, v


def batch_invariant_decode_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, split_size: int = SPLIT_SIZE
) -> mx.array:
    """T_q == 1. Two launches: per-split partials, then an ordered combine."""
    B, Hq, Tq, D = q.shape
    _, Hkv, L, _ = k.shape
    assert Tq == 1
    dt, k, v = _prep(q, k, v)
    nsplit = (L + split_size - 1) // split_size
    BH = B * Hq
    q2 = q.reshape(BH, D)
    hp = mx.array([Hq, Hkv, nsplit], dtype=mx.int32)
    scl = mx.array([scale], dtype=mx.float32)
    partial, combine, _ = _kernels()

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


def batch_invariant_prefill_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool,
    split_size: int = SPLIT_SIZE,
) -> mx.array:
    """Any T_q. One thread per query row, same split loop + merge as decode."""
    B, Hq, Tq, D = q.shape
    _, Hkv, L, _ = k.shape
    dt, k, v = _prep(q, k, v)
    BHT = B * Hq * Tq
    q2 = q.reshape(BHT, D)
    hp = mx.array([Hq, Hkv, Tq, 1 if causal else 0], dtype=mx.int32)
    scl = mx.array([scale], dtype=mx.float32)
    _, _, prefill = _kernels()
    out = prefill(
        inputs=[q2, k, v, hp, scl],
        template=[("T", dt), ("D", D), ("SPLIT", split_size)],
        grid=(BHT, 1, 1),
        threadgroup=(min(BHT, 64), 1, 1),
        output_shapes=[(BHT, D)],
        output_dtypes=[dt],
    )[0]
    return out.reshape(B, Hq, Tq, D)


def batch_invariant_attention(q, k, v, *, scale: float, mask=None) -> mx.array:
    """Dispatch: decode path for T_q == 1, prefill path otherwise."""
    if q.shape[2] == 1:
        return batch_invariant_decode_attention(q, k, v, scale=scale)
    return batch_invariant_prefill_attention(q, k, v, scale=scale, causal=(mask == "causal"))
