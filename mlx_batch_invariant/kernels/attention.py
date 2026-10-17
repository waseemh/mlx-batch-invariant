"""Batch-invariant attention with a fixed KV split size, lane-parallel.

Canonical order per (batch, head, query):

* keys are processed in fixed splits of ``SPLIT_SIZE`` (256) in index order;
  key ``j`` always lands in split ``j // 256`` at the same position, so
  neither batch size nor cache growth changes an existing row's reduction;
* the score ``q.k`` is computed as 32 lane partials (lane ``l`` owns dims
  ``[l*DL, (l+1)*DL)``, accumulated sequentially with fma) folded by a fixed
  xor-butterfly tree and taken from lane 0;
* inside a split, the online-softmax recurrence runs key by key:
  ``m' = max(m, s); corr = exp(m - m'); p = exp(s - m'); l = fma(l, corr, p);
  acc = fma(acc, corr, p * v)``, from exact zeros;
* split partials ``(m, l, acc)`` are merged left to right with
  ``c_old = exp(M - M'); c_new = exp(m - M'); L = fma(L, c_old, l * c_new);
  A = fma(A, c_old, acc * c_new)``.

Two kernels execute that sequence; they differ only in what shares a K/V
row load:

* ``decode`` (T_q == 1): one simdgroup per (batch, kv head, head chunk,
  split). The ``G`` query heads of a kv head share each K/V row. Emits
  (m, l, acc) per (query head, split); ``combine`` folds the splits.
* ``prefill``: one simdgroup per (batch, head, block of ``QB`` queries); the
  queries share each K/V row and the split fold happens in registers.
  Causal masking is per query inside the key loop, which is bitwise the
  same as stopping the loop early (an empty or short split merges as a
  no-op).

Supported: ``mask`` None or ``"causal"``, no sinks, D % 32 == 0, GQA/MQA via
integer head ratio. Array masks fall back to stock (counted).
"""

from __future__ import annotations

import mlx.core as mx

SPLIT_SIZE = 256
QB = 4          # queries per prefill simdgroup
MAX_G = 8       # query heads per decode simdgroup
MIN_UNITS = 256 # below this many simdgroups, stop sharing K/V across heads to get parallelism
UNITS_PER_TG = 4

# ---- shared per-key update: identical text in decode and prefill --------------
# Expects: kr/vr (device row pointers), qreg[NQ][DL], m[NQ], l[NQ], acc[NQ][DL],
# and a predicate `active(iq)` deciding whether query iq sees this key.
_KEY_UPDATE = """
            float kreg[DL];
            float vreg[DL];
            @UNROLL@
            for (int d = 0; d < DL; ++d) kreg[d] = (float)kr[lane * DL + d];
            @UNROLL@
            for (int d = 0; d < DL; ++d) vreg[d] = (float)vr[lane * DL + d];
            @UNROLL@
            for (int iq = 0; iq < NQ; ++iq) {
                float part = 0.0f;
                @UNROLL@
                for (int d = 0; d < DL; ++d) part = fma(qreg[iq][d], kreg[d], part);
                part += simd_shuffle_xor(part, 16);
                part += simd_shuffle_xor(part, 8);
                part += simd_shuffle_xor(part, 4);
                part += simd_shuffle_xor(part, 2);
                part += simd_shuffle_xor(part, 1);
                const float s = simd_shuffle(part, 0);
                if (ACTIVE(iq)) {
                    float m_new = metal::max(m[iq], s);
                    float corr  = metal::exp(m[iq] - m_new);
                    float p     = metal::exp(s - m_new);
                    l[iq] = fma(l[iq], corr, p);
                    @UNROLL@
                    for (int d = 0; d < DL; ++d) acc[iq][d] = fma(acc[iq][d], corr, p * vreg[d]);
                    m[iq] = m_new;
                }
            }
"""

_MERGE = """
                float M_new = metal::max(M[iq], m[iq]);
                float c_old = metal::exp(M[iq] - M_new);
                float c_new = metal::exp(m[iq] - M_new);
                Ls[iq] = fma(Ls[iq], c_old, l[iq] * c_new);
                @UNROLL@
                for (int d = 0; d < DL; ++d) A[iq][d] = fma(A[iq][d], c_old, acc[iq][d] * c_new);
                M[iq] = M_new;
"""

_DECODE_SRC = """
    #define NQ GU
    #define ACTIVE(iq) ((iq) < gcount)
    const int Hq     = hp[0];
    const int Hkv    = hp[1];
    const int nsplit = hp[2];
    const int nchunk = hp[3];          // head chunks per kv head
    const int L      = k_shape[2];
    const float scale = scl[0];
    const int G = Hq / Hkv;

    const int unit = thread_position_in_grid.x / 32;
    const int lane = thread_index_in_simdgroup;
    const int units = k_shape[0] * Hkv * nchunk * nsplit;
    if (unit >= units) return;
    int u = unit;
    const int split = u % nsplit; u /= nsplit;
    const int chunk = u % nchunk; u /= nchunk;
    const int hk    = u % Hkv;    u /= Hkv;
    const int b     = u;
    const int h0 = hk * G + chunk * GU;
    const int gcount = metal::min(GU, G - chunk * GU);

    const device T* kb = k + ((size_t)b * Hkv + hk) * (size_t)L * D;
    const device T* vb = v + ((size_t)b * Hkv + hk) * (size_t)L * D;

    float qreg[GU][DL];
    @UNROLL@
    for (int iq = 0; iq < GU; ++iq) {
        const device T* qr = q + ((size_t)b * Hq + h0 + metal::min(iq, gcount - 1)) * D;
        @UNROLL@
        for (int d = 0; d < DL; ++d) qreg[iq][d] = (float)qr[lane * DL + d] * scale;
    }
    float m[GU], l[GU], acc[GU][DL];
    @UNROLL@
    for (int iq = 0; iq < GU; ++iq) {
        m[iq] = -INFINITY; l[iq] = 0.0f;
        @UNROLL@
        for (int d = 0; d < DL; ++d) acc[iq][d] = 0.0f;
    }

    const int j0 = split * SPLIT;
    const int j1 = metal::min(L, j0 + SPLIT);
    for (int j = j0; j < j1; ++j) {
        const device T* kr = kb + (size_t)j * D;
        const device T* vr = vb + (size_t)j * D;
@KEY_UPDATE@
    }

    @UNROLL@
    for (int iq = 0; iq < GU; ++iq) {
        if (iq < gcount) {
            const size_t po = ((size_t)b * Hq + h0 + iq) * nsplit + split;
            if (lane == 0) { pm[po] = m[iq]; pl[po] = l[iq]; }
            @UNROLL@
            for (int d = 0; d < DL; ++d) pacc[po * D + lane * DL + d] = acc[iq][d];
        }
    }
    #undef NQ
    #undef ACTIVE
"""

_COMBINE_SRC = """
    const int BH     = pm_shape[0];
    const int nsplit = pm_shape[1];
    const int gid = thread_position_in_grid.x;
    if (gid >= BH * D) return;
    const int bh = gid / D;
    const int d  = gid % D;

    float M = -INFINITY, Ls = 0.0f, A = 0.0f;
    for (int i = 0; i < nsplit; ++i) {
        const size_t po = (size_t)bh * nsplit + i;
        const float m = pm[po];
        const float l = pl[po];
        const float acc = pacc[po * D + d];
        float M_new = metal::max(M, m);
        float c_old = metal::exp(M - M_new);
        float c_new = metal::exp(m - M_new);
        Ls = fma(Ls, c_old, l * c_new);
        A = fma(A, c_old, acc * c_new);
        M = M_new;
    }
    out[(size_t)bh * D + d] = (T)(A / Ls);
"""

_PREFILL_SRC = """
    #define NQ QB
    #define ACTIVE(iq) (j < nkeys[iq])
    const int Hq     = hp[0];
    const int Hkv    = hp[1];
    const int Tq     = hp[2];
    const int causal = hp[3];
    const int L      = k_shape[2];
    const float scale = scl[0];
    const int nblk = (Tq + QB - 1) / QB;

    const int unit = thread_position_in_grid.x / 32;
    const int lane = thread_index_in_simdgroup;
    const int units = k_shape[0] * Hq * nblk;
    if (unit >= units) return;
    int u = unit;
    const int blk = u % nblk; u /= nblk;
    const int h   = u % Hq;   u /= Hq;
    const int b   = u;
    const int hk  = h / (Hq / Hkv);
    const int q0  = blk * QB;

    const device T* kb = k + ((size_t)b * Hkv + hk) * (size_t)L * D;
    const device T* vb = v + ((size_t)b * Hkv + hk) * (size_t)L * D;

    int nkeys[QB];
    float qreg[QB][DL];
    @UNROLL@
    for (int iq = 0; iq < QB; ++iq) {
        const int qi = metal::min(q0 + iq, Tq - 1);
        nkeys[iq] = (q0 + iq < Tq) ? (causal ? (L - Tq + qi + 1) : L) : 0;
        const device T* qr = q + (((size_t)b * Hq + h) * Tq + qi) * D;
        @UNROLL@
        for (int d = 0; d < DL; ++d) qreg[iq][d] = (float)qr[lane * DL + d] * scale;
    }
    int nkeys_max = 0;
    @UNROLL@
    for (int iq = 0; iq < QB; ++iq) nkeys_max = metal::max(nkeys_max, nkeys[iq]);
    const int nsplit = (nkeys_max + SPLIT - 1) / SPLIT;

    float M[QB], Ls[QB], A[QB][DL];
    float m[QB], l[QB], acc[QB][DL];
    @UNROLL@
    for (int iq = 0; iq < QB; ++iq) {
        M[iq] = -INFINITY; Ls[iq] = 0.0f;
        @UNROLL@
        for (int d = 0; d < DL; ++d) A[iq][d] = 0.0f;
    }

    for (int split = 0; split < nsplit; ++split) {
        @UNROLL@
        for (int iq = 0; iq < QB; ++iq) {
            m[iq] = -INFINITY; l[iq] = 0.0f;
            @UNROLL@
            for (int d = 0; d < DL; ++d) acc[iq][d] = 0.0f;
        }
        const int j0 = split * SPLIT;
        const int j1 = metal::min(nkeys_max, j0 + SPLIT);
        for (int j = j0; j < j1; ++j) {
            const device T* kr = kb + (size_t)j * D;
            const device T* vr = vb + (size_t)j * D;
@KEY_UPDATE@
        }
        @UNROLL@
        for (int iq = 0; iq < QB; ++iq) {
            if (j0 < nkeys[iq]) {   // split touched this query; otherwise the merge is a no-op
@MERGE@
            }
        }
    }

    @UNROLL@
    for (int iq = 0; iq < QB; ++iq) {
        if (q0 + iq < Tq) {
            const size_t o = (((size_t)b * Hq + h) * Tq + q0 + iq) * D;
            @UNROLL@
            for (int d = 0; d < DL; ++d) out[o + lane * DL + d] = (T)(A[iq][d] / Ls[iq]);
        }
    }
    #undef NQ
    #undef ACTIVE
"""

UNROLL = "#pragma clang loop unroll(full)"


def _src(s: str) -> str:
    return s.replace("@KEY_UPDATE@", _KEY_UPDATE).replace("@MERGE@", _MERGE).replace("@UNROLL@", UNROLL)


_decode = _combine = _prefill = None
_params: dict = {}


def _param(vals, dtype):
    """Small constant arrays, cached so a decode step does not re-upload them per layer."""
    key = (tuple(vals), dtype)
    if key not in _params:
        _params[key] = mx.array(list(vals), dtype=dtype)
    return _params[key]


def _kernels():
    global _decode, _combine, _prefill
    if _decode is None:
        _decode = mx.fast.metal_kernel(
            name="bi_attn_decode",
            input_names=["q", "k", "v", "hp", "scl"],
            output_names=["pm", "pl", "pacc"],
            source=_src(_DECODE_SRC),
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
            source=_src(_PREFILL_SRC),
        )
    return _decode, _combine, _prefill


def supports(q: mx.array, k: mx.array, v: mx.array, mask, sinks) -> bool:
    return (
        q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
        and sinks is None
        and q.shape[1] % k.shape[1] == 0
        and k.shape == v.shape
        and q.shape[-1] == k.shape[-1]
        and q.shape[-1] % 32 == 0
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


def _tg(units: int):
    return (min(units, UNITS_PER_TG) * 32, 1, 1)


def batch_invariant_decode_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, split_size: int = SPLIT_SIZE
) -> mx.array:
    """T_q == 1. Per-split partials (G heads share K/V rows), then an ordered combine."""
    B, Hq, Tq, D = q.shape
    _, Hkv, L, _ = k.shape
    assert Tq == 1 and D % 32 == 0
    dt, k, v = _prep(q, k, v)
    G = Hq // Hkv
    nsplit = (L + split_size - 1) // split_size
    # Heads per unit: share K/V rows across the group when there is enough work
    # to fill the GPU anyway; otherwise give every head its own unit. The
    # per-head arithmetic is identical either way.
    GU = min(G, MAX_G)
    if B * Hkv * nsplit * ((G + GU - 1) // GU) < MIN_UNITS:
        GU = 1
    nchunk = (G + GU - 1) // GU
    BH = B * Hq
    units = B * Hkv * nchunk * nsplit
    hp = _param([Hq, Hkv, nsplit, nchunk], mx.int32)
    scl = _param([float(scale)], mx.float32)
    decode, combine, _ = _kernels()
    pm, pl, pacc = decode(
        inputs=[q.reshape(BH, D), k, v, hp, scl],
        template=[("T", dt), ("D", D), ("DL", D // 32), ("GU", GU), ("SPLIT", split_size)],
        grid=(units * 32, 1, 1),
        threadgroup=_tg(units),
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
    """Any T_q. One simdgroup per block of QB queries; same split loop and merge as decode."""
    B, Hq, Tq, D = q.shape
    _, Hkv, L, _ = k.shape
    assert D % 32 == 0
    dt, k, v = _prep(q, k, v)
    nblk = (Tq + QB - 1) // QB
    units = B * Hq * nblk
    hp = _param([Hq, Hkv, Tq, 1 if causal else 0], mx.int32)
    scl = _param([float(scale)], mx.float32)
    _, _, prefill = _kernels()
    out = prefill(
        inputs=[q, k, v, hp, scl],
        template=[("T", dt), ("D", D), ("DL", D // 32), ("QB", QB), ("SPLIT", split_size)],
        grid=(units * 32, 1, 1),
        threadgroup=_tg(units),
        output_shapes=[(B, Hq, Tq, D)],
        output_dtypes=[dt],
    )[0]
    return out


def batch_invariant_attention(q, k, v, *, scale: float, mask=None) -> mx.array:
    """Dispatch: decode path for T_q == 1, prefill path otherwise."""
    if q.shape[2] == 1:
        return batch_invariant_decode_attention(q, k, v, scale=scale)
    return batch_invariant_prefill_attention(q, k, v, scale=scale, causal=(mask == "causal"))
