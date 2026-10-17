"""Functional entry points with explicit fallback accounting.

Each op returns the batch-invariant result when the kernel supports the
inputs, and otherwise calls the stock MLX op and records a fallback. The
mode exposes the counters so a run can be audited: "deterministic" with ten
silent fallbacks is not deterministic.
"""

from __future__ import annotations

from collections import Counter

import mlx.core as mx

from .kernels import (
    batch_invariant_attention,
    batch_invariant_matmul,
    batch_invariant_quantized_matmul,
    batch_invariant_rms_norm,
)
from .kernels import quantized as _quant
from .kernels import attention as _attn

# Stock implementations, captured at import so patching never recurses.
_orig_matmul = mx.matmul
_orig_addmm = mx.addmm
_orig_rms_norm = mx.fast.rms_norm
_orig_sdpa = mx.fast.scaled_dot_product_attention
_orig_quantized_matmul = mx.quantized_matmul

handled: Counter = Counter()
fallbacks: Counter = Counter()
strict: bool = False


def _fallback(op: str, reason: str):
    fallbacks[f"{op}:{reason}"] += 1
    if strict:
        raise RuntimeError(f"batch-invariant mode: no kernel for {op} ({reason})")


def _is_float(x) -> bool:
    return isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating)


def matmul(a, b, *, stream=None):
    if _is_float(a) and _is_float(b) and b.ndim == 2 and a.ndim >= 1:
        handled["matmul"] += 1
        return batch_invariant_matmul(a, b)
    _fallback("matmul", "non-2d-rhs-or-non-float")
    return _orig_matmul(a, b, stream=stream)


def addmm(c, a, b, /, alpha: float = 1.0, beta: float = 1.0, *, stream=None):
    if _is_float(a) and _is_float(b) and b.ndim == 2 and a.ndim >= 1:
        handled["addmm"] += 1
        prod = batch_invariant_matmul(a, b)
        if alpha != 1.0:
            prod = prod * alpha
        return prod + (c if beta == 1.0 else c * beta)
    _fallback("addmm", "non-2d-rhs-or-non-float")
    return _orig_addmm(c, a, b, alpha, beta, stream=stream)


def quantized_matmul(
    x, w, /, scales, biases=None, transpose: bool = True, group_size=None, bits=None,
    mode: str = "affine", *, stream=None,
):
    gs = 64 if group_size is None else group_size
    b = 4 if bits is None else bits
    if _quant.supports(x, w, scales, biases, transpose, gs, b, mode):
        handled["quantized_matmul"] += 1
        return batch_invariant_quantized_matmul(x, w, scales, biases, group_size=gs, bits=b)
    _fallback("quantized_matmul", f"{mode}-{b}bit-g{gs}-{'T' if transpose else 'N'}")
    return _orig_quantized_matmul(
        x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits, mode=mode, stream=stream
    )


def rms_norm(x, weight, eps: float, *, stream=None):
    if _is_float(x):
        handled["rms_norm"] += 1
        return batch_invariant_rms_norm(x, weight, eps)
    _fallback("rms_norm", "non-float")
    return _orig_rms_norm(x, weight, eps, stream=stream)


def scaled_dot_product_attention(
    q, k, v, *, scale: float, mask=None, sinks=None, force_fused=False, stream=None
):
    if _attn.supports(q, k, v, mask, sinks):
        handled["sdpa_decode" if q.shape[2] == 1 else "sdpa_prefill"] += 1
        return batch_invariant_attention(q, k, v, scale=scale, mask=mask)
    _fallback("sdpa", "array-mask" if isinstance(mask, mx.array) else "sinks-or-shape")
    return _orig_sdpa(q, k, v, scale=scale, mask=mask, sinks=sinks, stream=stream)


def reset_stats():
    handled.clear()
    fallbacks.clear()


def stats() -> dict:
    return {"handled": dict(handled), "fallbacks": dict(fallbacks)}
