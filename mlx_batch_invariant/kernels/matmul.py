"""Batch-invariant matmul: C[M, N] = A[M, K] @ B[K, N].

Canonical reduction order (the contract every launch obeys):

* K is cut into fixed groups of ``GROUP_K`` elements (group ``g`` covers
  ``k in [g*GROUP_K, (g+1)*GROUP_K)`` regardless of M, N or launch shape).
* Within a group, k advances in slabs of ``BK`` and, inside a slab, through
  the hardware ``simdgroup_multiply_accumulate`` in steps of 8, with operands
  in the input dtype (half, bfloat or float) and an fp32 accumulator. The
  group partial starts from an exact zero. (The quantized kernel stages its
  dequantized operands as fp32 and uses the fp32 MMA; each op is only
  required to be consistent with itself.)
* The output is the left fold ``((0 + p_0) + p_1) + ...`` of group partials,
  as plain fp32 adds.

Several launch shapes produce bitwise-identical results because they execute
the same sequence of IEEE operations per element:

* **staging layout**: both slabs are staged ``[row][k]`` with a padded
  leading dimension and B is loaded with the transposed fragment load. Pure
  data movement; the MMA sees the same operands.
* **tile height**: 64-row tiles for large M, 8-row tiles for M <= 32 (so a
  decode step does not pad one row to sixty-four). An element's MMA sequence
  depends only on its k slabs and its row within an 8x8 fragment, which is
  ``m % 8`` in both.
* **fused vs split**: large M folds all groups inside one threadgroup; small
  M launches one threadgroup per (tile, K group), writes fp32 group partials,
  and a second kernel does the left fold. This is the "fixed split size" idea
  from the Thinking Machines post applied to K: parallelism along K is fine
  as long as chunk boundaries never move with M.

The choice of shape is M-dependent; the arithmetic is not. ``harness.sweep``
checks bits across both boundaries (batch 1 vs 64 crosses tile height and
fused/split at once).

B is read through its strides, so the transposed weight view ``nn.Linear``
passes is consumed in place.
"""

from __future__ import annotations

import mlx.core as mx

GROUP_K = 256      # fixed K chunk; never depends on M
BN = 64            # tile width
BK = 32            # staged K slab
SPLIT_M_MAX = 32   # M <= this: skinny tiles + one K group per threadgroup

TILE_FULL = (64, 2, 4)    # (BM, simdgroups along M, along N): 256 threads, 4x2 frags each
TILE_SKINNY = (8, 1, 4)   # 128 threads, 1x2 frags each
PAD = 4                   # extra floats per staged row, so fragment rows hit different banks

UNROLL = "#pragma clang loop unroll(full)"


def cfg(tile, st_bytes=4):
    """Kernel-source parameters. ``st_bytes`` is the size of the staged
    element type (``ST``): 4 when slabs are staged as fp32, 2 for half/bfloat."""
    BM, SGM, SGN = tile
    threads = SGM * SGN * 32
    LD = BK + PAD  # both slabs are staged [row][k], k contiguous; LD % 4 == 0 keeps rows vector-aligned
    slab_floats = ((BM + BN) * LD * st_bytes + 3) // 4
    return dict(
        BM=BM, SGM=SGM, SGN=SGN, BN=BN, BK=BK, GROUP_K=GROUP_K, LD=LD,
        FM=BM // (8 * SGM), FN=BN // (8 * SGN),
        THREADS=threads, PER_THREAD=BM * BN // threads,
        SMEMF=max(slab_floats, BM * BN),
    )


def _fill(src: str, c: dict) -> str:
    for k, v in c.items():
        src = src.replace(f"@{k}@", str(v))
    assert "@" not in src, "unfilled token in kernel source"
    return src


_TILE_BODY = """
    const int M = a_shape[0];
    const int K = a_shape[1];
    const int a_sm = a_strides[0], a_sk = a_strides[1];
    const int ngroups = (K + @GROUP_K@ - 1) / @GROUP_K@;

    const int tg_n = threadgroup_position_in_grid.x;
    const int tg_m = threadgroup_position_in_grid.y;
    const int tg_z = threadgroup_position_in_grid.z;
    const int lid  = thread_index_in_threadgroup;
    const int sg   = simdgroup_index_in_threadgroup;
    const int sg_m = sg / @SGN@, sg_n = sg % @SGN@;

    const int row0 = tg_m * @BM@;
    const int col0 = tg_n * @BN@;
    const int g0 = (MODE == 1) ? tg_z : 0;
    const int g1 = (MODE == 1) ? tg_z + 1 : ngroups;

    // Slabs are staged as @ST@ in [row][k] layout with leading dim LD; the
    // result tile (fp32) aliases the same memory after the slab loop.
    threadgroup float smem[@SMEMF@];
    threadgroup @ST@* As = (threadgroup @ST@*)smem;
    threadgroup @ST@* Bs = As + @BM@ * @LD@;
    threadgroup float* Cs = smem;

    // 4-wide loads need unit stride along k, K % 4 == 0 and 8/16-byte row starts.
    const bool veca = (a_sk == 1) && (K % 4 == 0) && (a_sm % 4 == 0);

    float total[@PER_THREAD@];
    @UNROLL@
    for (int t = 0; t < @PER_THREAD@; ++t) total[t] = 0.0f;
    simdgroup_float8x8 accg[@FM@][@FN@];

    for (int g = g0; g < g1; ++g) {
        @UNROLL@
        for (int i = 0; i < @FM@; ++i)
            @UNROLL@
            for (int j = 0; j < @FN@; ++j)
                accg[i][j] = simdgroup_float8x8(0.0f);
        const int kg0 = g * @GROUP_K@;
        const int kg1 = metal::min(K, kg0 + @GROUP_K@);
        for (int k0 = kg0; k0 < kg1; k0 += @BK@) {
            if (veca) {
                for (int e = lid; e < @BM@ * (@BK@ / 4); e += @THREADS@) {
                    int i = e / (@BK@ / 4), k4 = e % (@BK@ / 4);
                    int m = row0 + i, k = k0 + k4 * 4;
                    vec<@ST@, 4> v = (m < M && k < K)
                        ? vec<@ST@, 4>(*(const device vec<T, 4>*)(a + (size_t)m * a_sm + k))
                        : vec<@ST@, 4>(0);
                    *(threadgroup vec<@ST@, 4>*)(As + i * @LD@ + k4 * 4) = v;
                }
            } else {
                for (int e = lid; e < @BM@ * @BK@; e += @THREADS@) {
                    int i = e / @BK@, kk = e % @BK@;
                    int m = row0 + i, k = k0 + kk;
                    As[i * @LD@ + kk] = (m < M && k < K) ? (@ST@)a[(size_t)m * a_sm + (size_t)k * a_sk] : (@ST@)0;
                }
            }
@STAGE_B@
            threadgroup_barrier(mem_flags::mem_threadgroup);

            @UNROLL@
            for (int kk = 0; kk < @BK@; kk += 8) {
                simdgroup_matrix<@ST@, 8, 8> af[@FM@];
                simdgroup_matrix<@ST@, 8, 8> bf[@FN@];
                @UNROLL@
                for (int i = 0; i < @FM@; ++i)
                    simdgroup_load(af[i], As + (sg_m * @FM@ * 8 + i * 8) * @LD@ + kk, @LD@);
                @UNROLL@
                for (int j = 0; j < @FN@; ++j)
                    simdgroup_load(bf[j], Bs + (sg_n * @FN@ * 8 + j * 8) * @LD@ + kk, @LD@, ulong2(0, 0), true);
                @UNROLL@
                for (int i = 0; i < @FM@; ++i)
                    @UNROLL@
                    for (int j = 0; j < @FN@; ++j)
                        simdgroup_multiply_accumulate(accg[i][j], af[i], bf[j], accg[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Stage the group partial and fold it with a plain fp32 add: the exact
        // same operation the split-mode combine kernel performs.
        @UNROLL@
        for (int i = 0; i < @FM@; ++i)
            @UNROLL@
            for (int j = 0; j < @FN@; ++j)
                simdgroup_store(accg[i][j], Cs + (sg_m * @FM@ * 8 + i * 8) * @BN@ + sg_n * @FN@ * 8 + j * 8, @BN@);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        @UNROLL@
        for (int t = 0; t < @PER_THREAD@; ++t)
            total[t] = total[t] + Cs[lid + t * @THREADS@];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    @UNROLL@
    for (int t = 0; t < @PER_THREAD@; ++t) {
        int e = lid + t * @THREADS@;
        int i = e / @BN@, j = e % @BN@;
        int m = row0 + i, n = col0 + j;
        if (m < M && n < N) {
            if (MODE == 1) out[((size_t)tg_z * M + m) * N + n] = (OUT)Cs[e];  // raw group partial
            else           out[(size_t)m * N + n] = (OUT)total[t];
        }
    }
"""

# Consecutive threads must read consecutive addresses. For the nn.Linear case
# (b = W.T, so b_sk == 1) that means walking k fastest, 4 at a time when
# aligned; for a row-major [K, N] operand it means walking n fastest.
_STAGE_B_DENSE = """
            if (vecb) {
                for (int e = lid; e < @BN@ * (@BK@ / 4); e += @THREADS@) {
                    int j = e / (@BK@ / 4), k4 = e % (@BK@ / 4);
                    int n = col0 + j, k = k0 + k4 * 4;
                    vec<@ST@, 4> v = (n < N && k < K)
                        ? vec<@ST@, 4>(*(const device vec<T, 4>*)(b + (size_t)n * b_sn + k))
                        : vec<@ST@, 4>(0);
                    *(threadgroup vec<@ST@, 4>*)(Bs + j * @LD@ + k4 * 4) = v;
                }
            } else if (b_sk == 1) {
                for (int e = lid; e < @BK@ * @BN@; e += @THREADS@) {
                    int j = e / @BK@, kk = e % @BK@;
                    int k = k0 + kk, n = col0 + j;
                    Bs[j * @LD@ + kk] = (k < K && n < N) ? (@ST@)b[(size_t)n * b_sn + k] : (@ST@)0;
                }
            } else {
                for (int e = lid; e < @BK@ * @BN@; e += @THREADS@) {
                    int kk = e / @BN@, j = e % @BN@;
                    int k = k0 + kk, n = col0 + j;
                    Bs[j * @LD@ + kk] = (k < K && n < N) ? (@ST@)b[(size_t)k * b_sk + (size_t)n * b_sn] : (@ST@)0;
                }
            }
"""

_DENSE_HEAD = """
    const int N = b_shape[1];
    const int b_sk = b_strides[0], b_sn = b_strides[1];
    const bool vecb = (b_sk == 1) && (b_shape[0] % 4 == 0) && (b_sn % 4 == 0);
"""

_COMBINE_SRC = """
    const int G = part_shape[0];
    const size_t MN = (size_t)part_shape[1] * part_shape[2];
    const size_t idx = thread_position_in_grid.x;
    if (idx >= MN) return;
    float total = 0.0f;
    for (int g = 0; g < G; ++g) total = total + part[(size_t)g * MN + idx];
    out[idx] = (OUT)total;
"""


def tile_source(head: str, stage_b: str, tile, st: str = "float", st_bytes: int = 4) -> str:
    """``st`` is the staged element type: "T" (input dtype; half/bfloat MMA
    with fp32 accumulate) or "float" (fp32 MMA). ``st_bytes`` is its size,
    which sizes the shared memory."""
    c = dict(cfg(tile, st_bytes), UNROLL=UNROLL, ST=st)
    body = _TILE_BODY.replace("@STAGE_B@", stage_b)
    return head + _fill(body, c)


_dense: dict = {}
_combine = None


def _get_dense(tile, nbytes):
    """Dense kernel for a tile config and input element size (2: half/bfloat, 4: float)."""
    key = (tile, nbytes)
    if key not in _dense:
        _dense[key] = mx.fast.metal_kernel(
            name=f"bi_matmul_bm{tile[0]}_s{nbytes}",
            input_names=["a", "b"],
            output_names=["out"],
            source=tile_source(_DENSE_HEAD, _STAGE_B_DENSE, tile, st="T", st_bytes=nbytes),
            ensure_row_contiguous=False,
        )
    return _dense[key]


def get_combine():
    global _combine
    if _combine is None:
        _combine = mx.fast.metal_kernel(
            name="bi_matmul_combine",
            input_names=["part"],
            output_names=["out"],
            source=_COMBINE_SRC,
        )
    return _combine


def launch_tiled(get_kernel, inputs, template, M, N, K, out_dtype):
    """Run a tile kernel (``get_kernel(tile)``) in the launch shape for M."""
    ngroups = (K + GROUP_K - 1) // GROUP_K
    skinny = M <= SPLIT_M_MAX
    tile = TILE_SKINNY if skinny else TILE_FULL
    c = cfg(tile)
    kernel = get_kernel(tile)
    grid_xy = ((N + BN - 1) // BN * c["THREADS"], (M + c["BM"] - 1) // c["BM"])
    tg = (c["THREADS"], 1, 1)
    if not (skinny and ngroups > 1):
        return kernel(
            inputs=inputs,
            template=template + [("OUT", out_dtype), ("MODE", 0)],
            grid=(grid_xy[0], grid_xy[1], 1),
            threadgroup=tg,
            output_shapes=[(M, N)],
            output_dtypes=[out_dtype],
        )[0]
    part = kernel(
        inputs=inputs,
        template=template + [("OUT", mx.float32), ("MODE", 1)],
        grid=(grid_xy[0], grid_xy[1], ngroups),
        threadgroup=tg,
        output_shapes=[(ngroups, M, N)],
        output_dtypes=[mx.float32],
    )[0]
    n = M * N
    return get_combine()(
        inputs=[part],
        template=[("OUT", out_dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[out_dtype],
    )[0]


def batch_invariant_matmul(a: mx.array, b: mx.array) -> mx.array:
    """``a @ b`` for 2-D ``b`` with a batch-invariant reduction order.

    ``a`` may have extra leading dims; they are flattened into M. ``b`` must be
    2-D (a transposed view is fine). Accumulation is fp32; output dtype is the
    promoted input dtype.
    """
    if b.ndim != 2:
        raise ValueError("batch_invariant_matmul: b must be 2-D")
    lead = a.shape[:-1]
    K = a.shape[-1]
    if K != b.shape[0]:
        raise ValueError(f"shape mismatch {a.shape} @ {b.shape}")
    N = b.shape[1]
    out_dtype = mx.result_type(a, b)
    a2 = a.reshape(-1, K) if a.ndim != 2 else a
    if a2.dtype != out_dtype:
        a2 = a2.astype(out_dtype)
    if b.dtype != out_dtype:
        b = b.astype(out_dtype)
    M = a2.shape[0]
    nbytes = out_dtype.size
    out = launch_tiled(lambda tile: _get_dense(tile, nbytes), [a2, b], [("T", out_dtype)], M, N, K, out_dtype)
    return out.reshape(*lead, N)
