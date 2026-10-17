"""Batch-invariant kernels for MLX.

    import mlx_batch_invariant as bi
    with bi.set_batch_invariant_mode():
        logits = model(tokens)          # same bits at batch 1 and batch 128
    print(bi.stats())                   # which ops were handled / fell back
"""

from ._mode import is_batch_invariant_mode_enabled, set_batch_invariant_mode
from .ops import reset_stats, stats
from .kernels import (
    batch_invariant_decode_attention,
    batch_invariant_matmul,
    batch_invariant_rms_norm,
)

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "stats",
    "reset_stats",
    "batch_invariant_matmul",
    "batch_invariant_rms_norm",
    "batch_invariant_decode_attention",
]
__version__ = "0.0.1"
