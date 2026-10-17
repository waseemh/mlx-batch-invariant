"""EXPERIMENTAL: batch-invariant attention on the simdgroup MMA, fixed 256-key splits.

Status: correct and bitwise batch/chunk/decode-vs-prefill invariant (tests
exercise it), but slower than the active lane-parallel kernel in
``attention.py`` on every shape measured (decode 1.2-1.7x of stock vs
0.9-1.6x; prefill 4.3x vs 3.9x). Not wired into the mode. Kept as the
starting point for an MMA-based prefill, with two lessons baked in: the
all-masked-block NaN guard, and that larger key blocks (16, 32) were slower
here from register pressure.


One kernel body serves prefill and decode. A unit is one simdgroup owning an
8-row query tile that shares one K/V stream:

* **prefill** rows are 8 consecutive queries of one head;
* **decode** rows are up to 8 query heads of one kv head (grouped-query
  attention fills the tile for free; the single query of each head is one
  row). With several KV splits each unit handles one split and writes
  (m, l, acc) partials that ``combine`` folds; with one split it writes the
  output directly, which is the same fold applied once.

Canonical order per query row, which both paths execute identically:

* keys in blocks of 8 from key 0, inside fixed splits of 256;
* scores ``S = Q·Kᵀ`` by ``simdgroup_multiply_accumulate`` over D in steps of
  8 (operands in the input dtype, fp32 accumulate), then ``s = S * scale`` in
  fp32; masked keys (beyond the row's causal limit or the cache) get -inf;
* ``m' = max(m, max over the block)``; ``corr = exp(m - m')``;
  ``p = exp(s - m')``; ``l = fma(l, corr, psum)`` where ``psum`` is the fixed
  tree ``((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7))``;
* ``acc = acc * corr`` elementwise, then ``acc += P·V`` by the MMA with P in
  the input dtype;
* split partials merge left to right: ``c_old = exp(M - M')``,
  ``c_new = exp(m - M')``, ``L = fma(L, c_old, l*c_new)``,
  ``A = fma(A, c_old, acc*c_new)``.

Masked keys contribute exact zeros to ``psum`` and to the MMA, which is why a
query's bits do not depend on how many keys its tile-mates see or on
whether it was computed during prefill or a later decode step
(``harness.sweep`` checks both).

The row rescale uses the lane layout of ``simdgroup_float8x8`` observed on
this GPU (``tests`` assert it): a lane holds two adjacent columns of row
``((lane>>1)&3) + ((lane>>4)&1)*4``.

Supported: ``mask`` None or ``"causal"``, no sinks, D % 8 == 0, D <= 256,
GQA/MQA via integer head ratio. Array masks fall back to stock (counted).
"""

from __future__ import annotations

import mlx.core as mx

SPLIT_SIZE = 256
UNROLL = "#pragma clang loop unroll(full)"

# ROWS: 0 = prefill (queries of one head), 1 = decode (heads of one kv head)
# OUTP: 0 = fold splits in-kernel and write the output, 1 = write one split's partials
_SRC = """
    constexpr int ND  = D / 8;       // d blocks
    constexpr int NK  = KB / 8;      // key sub-blocks per block
    constexpr int LDQ = D + 8;
    constexpr int LDP = KB + 8;
    constexpr int CPL = KB / 4;      // score columns per lane in the softmax step
    const int Hq = hp[0], Hkv = hp[1], Tq = hp[2], causal = hp[3], nsplit = hp[4], nchunk = hp[5];
    const int L = k_shape[2];
    const float scale = scl[0];
    const int lane = thread_index_in_simdgroup;
    const int unit = threadgroup_position_in_grid.x;

    threadgroup T Qs[8 * LDQ];
    threadgroup T Ks[KB * LDQ];
    threadgroup T Vs[KB * LDQ];
    threadgroup T Ps[8 * LDP];
    threadgroup float Ss[8 * KB];
    threadgroup float rm[8];
    threadgroup float rl[8];
    threadgroup float rc[8];
    threadgroup float accs[8 * D];

    // ---- rows of this unit -------------------------------------------------
    int b, hk, split0, split1, nkmax_unit = 0;
    threadgroup int row_hq[8];
    threadgroup int row_qi[8];
    threadgroup int nkeys[8];
    if (ROWS == 0) {
        const int nblk = (Tq + 7) / 8;
        int u = unit;
        const int blk = u % nblk; u /= nblk;
        const int h = u % Hq; u /= Hq;
        b = u; hk = h / (Hq / Hkv);
        if (lane < 8) {
            const int r = lane;
            const int qi = blk * 8 + r;
            const bool valid = qi < Tq;
            row_hq[r] = h; row_qi[r] = valid ? qi : 0;
            nkeys[r] = valid ? (causal ? (L - Tq + qi + 1) : L) : 0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        @UNROLL@
        for (int r = 0; r < 8; ++r) nkmax_unit = metal::max(nkmax_unit, nkeys[r]);
        split0 = 0; split1 = (nkmax_unit + SPLIT - 1) / SPLIT;
    } else {
        int u = unit;
        const int split = u % nsplit; u /= nsplit;
        const int chunk = u % nchunk; u /= nchunk;
        hk = u % Hkv; u /= Hkv; b = u;
        const int G = Hq / Hkv;
        const int h0 = hk * G + chunk * 8;
        if (lane < 8) {
            const int r = lane;
            const bool valid = (chunk * 8 + r) < G;
            row_hq[r] = valid ? h0 + r : h0; row_qi[r] = 0;
            nkeys[r] = valid ? L : 0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        nkmax_unit = L;
        split0 = split; split1 = split + 1;
    }

    // ---- stage Q once ----------------------------------------------------------
    for (int e = lane; e < 8 * D; e += 32) {
        const int r = e / D, d = e % D;
        Qs[r * LDQ + d] = (nkeys[r] > 0)
            ? q[(((size_t)b * Hq + row_hq[r]) * Tq + row_qi[r]) * D + d] : (T)0;
    }
    const device T* kb = k + ((size_t)b * Hkv + hk) * (size_t)L * D;
    const device T* vb = v + ((size_t)b * Hkv + hk) * (size_t)L * D;

    // running fold state (OUTP == 0) lives in threadgroup memory to keep registers free
    threadgroup float Mt[8];
    threadgroup float Lt[8];
    threadgroup float co[8];
    threadgroup float cn[8];
    threadgroup float At[8 * D];
    if (lane < 8) { Mt[lane] = -INFINITY; Lt[lane] = 0.0f; }
    for (int e = lane; e < 8 * D; e += 32) At[e] = 0.0f;

    const int srow = lane / 4;             // row this lane handles in the softmax step
    const int scol = (lane % 4) * CPL;     // its first column
    const int mrow = ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4;   // row of this lane's MMA elements

    simdgroup_float8x8 acc[ND];

    for (int split = split0; split < split1; ++split) {
        @UNROLL@
        for (int db = 0; db < ND; ++db) acc[db] = simdgroup_float8x8(0.0f);
        if (lane < 8) { rm[lane] = -INFINITY; rl[lane] = 0.0f; }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const int jend = metal::min(nkmax_unit, split * SPLIT + SPLIT);
        for (int j0 = split * SPLIT; j0 < jend; j0 += KB) {
            // stage K and V block rows j0..j0+KB-1 (zero beyond the cache)
            for (int e = lane; e < KB * (D / 4); e += 32) {
                const int r = e / (D / 4), d4 = e % (D / 4);
                const int j = j0 + r;
                const vec<T, 4> kv4 = (j < L) ? *(const device vec<T, 4>*)(kb + (size_t)j * D + d4 * 4) : vec<T, 4>(0);
                const vec<T, 4> vv4 = (j < L) ? *(const device vec<T, 4>*)(vb + (size_t)j * D + d4 * 4) : vec<T, 4>(0);
                *(threadgroup vec<T, 4>*)(Ks + r * LDQ + d4 * 4) = kv4;
                *(threadgroup vec<T, 4>*)(Vs + r * LDQ + d4 * 4) = vv4;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // S = Q . K^T over D, one 8x8 tile per key sub-block
            {
                simdgroup_float8x8 S[NK];
                @UNROLL@
                for (int kb = 0; kb < NK; ++kb) S[kb] = simdgroup_float8x8(0.0f);
                @UNROLL@
                for (int db = 0; db < ND; ++db) {
                    simdgroup_matrix<T, 8, 8> qf;
                    simdgroup_load(qf, Qs + db * 8, LDQ);
                    @UNROLL@
                    for (int kb = 0; kb < NK; ++kb) {
                        simdgroup_matrix<T, 8, 8> kf;
                        simdgroup_load(kf, Ks + kb * 8 * LDQ + db * 8, LDQ, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(S[kb], qf, kf, S[kb]);
                    }
                }
                @UNROLL@
                for (int kb = 0; kb < NK; ++kb) simdgroup_store(S[kb], Ss + kb * 8, KB);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax step: this lane's row, CPL consecutive columns
            {
                const int r = srow;
                float sv[CPL];
                float bm = -INFINITY;
                @UNROLL@
                for (int c = 0; c < CPL; ++c) {
                    const bool ok = (j0 + scol + c) < nkeys[r];
                    sv[c] = ok ? Ss[r * KB + scol + c] * scale : -INFINITY;
                    bm = metal::max(bm, sv[c]);
                }
                bm = metal::max(bm, simd_shuffle_xor(bm, 1));
                bm = metal::max(bm, simd_shuffle_xor(bm, 2));
                const float m_old = rm[r];
                const float m_new = metal::max(m_old, bm);
                // A block whose keys are all masked for this row leaves m at -inf;
                // exp(-inf - -inf) would be NaN. corr = 1 makes the block an exact
                // no-op, so a row's bits do not depend on its tile-mates' key ranges.
                const float corr = (m_new == -INFINITY) ? 1.0f : metal::exp(m_old - m_new);
                float t2 = 0.0f;
                @UNROLL@
                for (int c = 0; c < CPL; ++c) {
                    const float pc = (sv[c] == -INFINITY) ? 0.0f : metal::exp(sv[c] - m_new);
                    t2 = t2 + pc;                       // sequential within the lane
                    Ps[r * LDP + scol + c] = (T)pc;
                }
                t2 = t2 + simd_shuffle_xor(t2, 1);      // then a fixed tree across the row's 4 lanes
                const float psum = t2 + simd_shuffle_xor(t2, 2);
                const float l_new = fma(rl[r], corr, psum);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if ((lane & 3) == 0) { rm[r] = m_new; rl[r] = l_new; rc[r] = corr; }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // acc = acc * corr (row-wise), then acc += P . V
            {
                const float c = rc[mrow];
                @UNROLL@
                for (int db = 0; db < ND; ++db) {
                    acc[db].thread_elements()[0] *= c;
                    acc[db].thread_elements()[1] *= c;
                }
                simdgroup_matrix<T, 8, 8> pf[NK];
                @UNROLL@
                for (int kb = 0; kb < NK; ++kb) simdgroup_load(pf[kb], Ps + kb * 8, LDP);
                @UNROLL@
                for (int db = 0; db < ND; ++db) {
                    @UNROLL@
                    for (int kb = 0; kb < NK; ++kb) {
                        simdgroup_matrix<T, 8, 8> vf;
                        simdgroup_load(vf, Vs + kb * 8 * LDQ + db * 8, LDQ);
                        simdgroup_multiply_accumulate(acc[db], pf[kb], vf, acc[db]);
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        // stage the split's acc
        @UNROLL@
        for (int db = 0; db < ND; ++db) simdgroup_store(acc[db], accs + db * 8, D);
        simdgroup_barrier(mem_flags::mem_threadgroup);

@PARTIAL_BLOCK@
        // fold this split into (Mt, Lt, At): the combine kernel's formula
        if (lane < 8) {
            const int r = lane;
            const float m = rm[r], l = rl[r];
            const float M_new = metal::max(Mt[r], m);
            co[r] = metal::exp(Mt[r] - M_new);
            cn[r] = metal::exp(m - M_new);
            Lt[r] = fma(Lt[r], co[r], l * cn[r]);
            Mt[r] = M_new;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = lane; e < 8 * D; e += 32) {
            const int r = e / D;
            At[e] = fma(At[e], co[r], accs[e] * cn[r]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

@FINAL_BLOCK@
"""

_PARTIAL_BLOCK = """
        {
            const int split = split0;
            @UNROLL@
            for (int r = 0; r < 8; ++r) {
                if (nkeys[r] > 0) {
                    const size_t po = ((size_t)b * Hq + row_hq[r]) * nsplit + split;
                    if (lane == 0) { pm[po] = rm[r]; pl[po] = rl[r]; }
                    for (int d = lane; d < D; d += 32) pacc[po * D + d] = accs[r * D + d];
                }
            }
            return;
        }
"""

_FINAL_BLOCK = """
    for (int e = lane; e < 8 * D; e += 32) {
        const int r = e / D, d = e % D;
        if (nkeys[r] > 0)
            out[(((size_t)b * Hq + row_hq[r]) * Tq + row_qi[r]) * D + d] = (T)(At[e] / Lt[r]);
    }
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

_kernels: dict = {}
_params: dict = {}


def _param(vals, dtype):
    key = (tuple(vals), dtype)
    if key not in _params:
        _params[key] = mx.array(list(vals), dtype=dtype)
    return _params[key]


def _kernel(rows: int, outp: int):
    key = (rows, outp)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"bi_attn_mma_r{rows}_o{outp}",
            input_names=["q", "k", "v", "hp", "scl"],
            output_names=["out"] if outp == 0 else ["pm", "pl", "pacc"],
            source=(_SRC.replace("@PARTIAL_BLOCK@", _PARTIAL_BLOCK if outp == 1 else "")
                        .replace("@FINAL_BLOCK@", _FINAL_BLOCK if outp == 0 else "")
                        .replace("@UNROLL@", UNROLL)),
        )
    return _kernels[key]


def _combine():
    if "combine" not in _kernels:
        _kernels["combine"] = mx.fast.metal_kernel(
            name="bi_attn_combine",
            input_names=["pm", "pl", "pacc"],
            output_names=["out"],
            source=_COMBINE_SRC,
        )
    return _kernels["combine"]


def supports(q: mx.array, k: mx.array, v: mx.array, mask, sinks) -> bool:
    return (
        q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
        and sinks is None
        and q.shape[1] % k.shape[1] == 0
        and k.shape == v.shape
        and q.shape[-1] == k.shape[-1]
        and q.shape[-1] % 8 == 0 and q.shape[-1] <= 256
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


KEY_BLOCK = 8    # keys per block (16 and 32 measured slower): S tiles, one softmax step and one rescale per block


def _template(dt, D, split_size, rows, outp):
    return [("T", dt), ("D", D), ("SPLIT", split_size), ("KB", KEY_BLOCK), ("ROWS", rows), ("OUTP", outp)]


def batch_invariant_decode_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, split_size: int = SPLIT_SIZE
) -> mx.array:
    """T_q == 1: tile rows are the query heads of one kv head."""
    B, Hq, Tq, D = q.shape
    _, Hkv, L, _ = k.shape
    assert Tq == 1
    dt, k, v = _prep(q, k, v)
    G = Hq // Hkv
    nchunk = (G + 7) // 8
    nsplit = (L + split_size - 1) // split_size
    units = B * Hkv * nchunk * nsplit
    hp = _param([Hq, Hkv, Tq, 0, nsplit, nchunk], mx.int32)
    scl = _param([float(scale)], mx.float32)
    if nsplit == 1:
        return _kernel(1, 0)(
            inputs=[q, k, v, hp, scl],
            template=_template(dt, D, split_size, 1, 0),
            grid=(units * 32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(B, Hq, 1, D)], output_dtypes=[dt],
        )[0]
    BH = B * Hq
    pm, pl, pacc = _kernel(1, 1)(
        inputs=[q, k, v, hp, scl],
        template=_template(dt, D, split_size, 1, 1),
        grid=(units * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(BH, nsplit), (BH, nsplit), (BH, nsplit, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    n2 = BH * D
    out = _combine()(
        inputs=[pm, pl, pacc],
        template=[("T", dt), ("D", D)],
        grid=(n2, 1, 1), threadgroup=(min(n2, 256), 1, 1),
        output_shapes=[(BH, D)], output_dtypes=[dt],
    )[0]
    return out.reshape(B, Hq, 1, D)


def batch_invariant_prefill_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float, causal: bool,
    split_size: int = SPLIT_SIZE,
) -> mx.array:
    """Any T_q: tile rows are 8 consecutive queries of one head; splits fold in-kernel."""
    B, Hq, Tq, D = q.shape
    _, Hkv, L, _ = k.shape
    dt, k, v = _prep(q, k, v)
    nblk = (Tq + 7) // 8
    units = B * Hq * nblk
    hp = _param([Hq, Hkv, Tq, 1 if causal else 0, 1, 1], mx.int32)
    scl = _param([float(scale)], mx.float32)
    return _kernel(0, 0)(
        inputs=[q, k, v, hp, scl],
        template=_template(dt, D, split_size, 0, 0),
        grid=(units * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(B, Hq, Tq, D)], output_dtypes=[dt],
    )[0]


def batch_invariant_attention(q, k, v, *, scale: float, mask=None) -> mx.array:
    if q.shape[2] == 1:
        return batch_invariant_decode_attention(q, k, v, scale=scale)
    return batch_invariant_prefill_attention(q, k, v, scale=scale, causal=(mask == "causal"))
