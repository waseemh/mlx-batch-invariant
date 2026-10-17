"""The override mechanism.

MLX has no ``torch.Library``; dispatch is resolved by Python attribute lookup
at call time. So the mode swaps the module-level functions that ``mlx.nn``
and model code reach for (``mx.matmul``, ``mx.addmm``, ``mx.fast.rms_norm``,
``mx.fast.scaled_dot_product_attention``) and the ``@`` operator on
``mx.array``, then restores them on exit. ``nn.Linear`` uses ``x @ W.T`` and
``mx.addmm``; ``nn.RMSNorm`` uses ``mx.fast.rms_norm``; ``mlx_lm`` attention
uses ``mx.fast.scaled_dot_product_attention``. All four are covered.

``nn.QuantizedLinear`` uses ``mx.quantized_matmul``; affine 4/8-bit with
group size 32/64/128 is covered. Not covered (falls back, counted): other
quantization modes (mxfp4, nvfp4, ...), batched 3-D matmul, array-mask or
sinks attention.
"""

from __future__ import annotations

import contextlib
import threading

import mlx.core as mx

from . import ops

_lock = threading.Lock()
_depth = 0
_saved: dict = {}


def _patched_array_matmul(self, other):
    return ops.matmul(self, other)


def _apply():
    global _saved
    _saved = {
        "matmul": mx.matmul,
        "addmm": mx.addmm,
        "quantized_matmul": mx.quantized_matmul,
        "rms_norm": mx.fast.rms_norm,
        "sdpa": mx.fast.scaled_dot_product_attention,
        "array_matmul": mx.array.__matmul__,
    }
    mx.matmul = ops.matmul
    mx.addmm = ops.addmm
    mx.quantized_matmul = ops.quantized_matmul
    mx.fast.rms_norm = ops.rms_norm
    mx.fast.scaled_dot_product_attention = ops.scaled_dot_product_attention
    mx.array.__matmul__ = _patched_array_matmul


def _restore():
    mx.matmul = _saved["matmul"]
    mx.addmm = _saved["addmm"]
    mx.quantized_matmul = _saved["quantized_matmul"]
    mx.fast.rms_norm = _saved["rms_norm"]
    mx.fast.scaled_dot_product_attention = _saved["sdpa"]
    mx.array.__matmul__ = _saved["array_matmul"]


def is_batch_invariant_mode_enabled() -> bool:
    return _depth > 0


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True, *, strict: bool = False):
    """Route matmul / addmm / rms_norm / decode-attention through the
    batch-invariant kernels for the duration of the block.

    ``strict=True`` raises instead of silently falling back to a stock kernel
    for unsupported inputs. Re-entrant; thread-global (MLX dispatch is).
    """
    global _depth
    if not enabled:
        yield
        return
    with _lock:
        if _depth == 0:
            _apply()
        _depth += 1
        prev_strict = ops.strict
        ops.strict = strict
    try:
        yield
    finally:
        with _lock:
            ops.strict = prev_strict
            _depth -= 1
            if _depth == 0:
                _restore()
