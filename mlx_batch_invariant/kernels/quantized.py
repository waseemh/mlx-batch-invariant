"""Batch-invariant affine quantized matmul: y[M, N] = x[M, K] @ W[N, K].T

``W`` is MLX's packed affine format: ``bits``-wide unsigned codes packed
low-to-high inside uint32 words (``32 // bits`` codes per word), with one
``scale`` and ``bias`` per ``group_size`` consecutive codes of a row:

    w[n, k] = code(n, k) * scales[n, k // group_size] + biases[n, k // group_size]

This reuses the dense tile body from ``matmul.py`` (same K groups, same slab
order, same MMA sequence, same fold, same launch shapes), with one change:
the B slab is dequantized in fp32 while it is staged. Each thread unpacks a
whole uint32 word, so the global reads are coalesced word loads and the
threadgroup stores are k-contiguous.

Stock ``mx.quantized_matmul`` differs between batch 1 and batch 2 on the same
row (measured 0.125 abs in fp16 at K=N=4096, 4-bit, group 64); this kernel
does not.
"""

from __future__ import annotations

import mlx.core as mx

from . import matmul as _mm

SUPPORTED_BITS = (4, 8)
SUPPORTED_GROUPS = (32, 64, 128)

_QUANT_HEAD = """
    const int N = w_shape[0];
    const int words_per_row = w_shape[1];
    const int groups_per_row = scales_shape[1];
    constexpr int PER_WORD = 32 / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr int WPS = @BK@ / PER_WORD;   // words per row per slab
"""

# Every word's codes share one (scale, bias): group_size is a multiple of
# PER_WORD, and a slab never straddles a group (BK <= 32 <= group_size).
_STAGE_B_QUANT = """
            for (int e = lid; e < @BN@ * WPS; e += @THREADS@) {
                int j = e / WPS, wi = e % WPS;
                int n = col0 + j;
                int kbase = k0 + wi * PER_WORD;
                threadgroup float* dst = Bs + j * @LD@ + wi * PER_WORD;
                if (n < N && kbase < K) {
                    uint word = w[(size_t)n * words_per_row + (size_t)(kbase / PER_WORD)];
                    int g = kbase / GS;
                    float s = (float)scales[(size_t)n * groups_per_row + g];
                    float bia = (float)biases[(size_t)n * groups_per_row + g];
                    @UNROLL@
                    for (int q = 0; q < PER_WORD; ++q)
                        dst[q] = fma((float)((word >> (BITS * q)) & MASK), s, bia);
                } else {
                    @UNROLL@
                    for (int q = 0; q < PER_WORD; ++q) dst[q] = 0.0f;
                }
            }
"""

_kernels: dict = {}


def _get_kernel(tile):
    if tile not in _kernels:
        c = dict(_mm.cfg(tile, 4), UNROLL=_mm.UNROLL, ST="float")
        head = _mm._fill(_QUANT_HEAD, c)
        _kernels[tile] = mx.fast.metal_kernel(
            name=f"bi_quantized_matmul_bm{tile[0]}",
            input_names=["a", "w", "scales", "biases"],
            output_names=["out"],
            source=_mm.tile_source(head, _mm._fill(_STAGE_B_QUANT, c), tile, st="float", st_bytes=4),
            ensure_row_contiguous=False,
        )
    return _kernels[tile]


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
    w = mx.contiguous(w)
    scales = mx.contiguous(scales.astype(dt) if scales.dtype != dt else scales)
    biases = mx.contiguous(biases.astype(dt) if biases.dtype != dt else biases)
    out = _mm.launch_tiled(
        _get_kernel, [x2, w, scales, biases], [("T", dt), ("BITS", bits), ("GS", group_size)],
        M, N, K, dt,
    )
    return out.reshape(*lead, N)
