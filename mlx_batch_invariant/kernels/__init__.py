"""Batch-invariant Metal kernels.

Every kernel here obeys one rule: the floating-point reduction order for a
given output element must not depend on how many *other* rows / sequences
share the launch. Concretely:

* no split-K, no shape-dependent tile or instruction selection (matmul,
  quantized matmul with on-the-fly fp32 dequantization)
* one threadgroup per row with a fixed-shape reduction tree (rmsnorm)
* fixed split *size* along the KV axis, combined in a fixed order (attention)

Transcendentals and division use ``metal::precise::`` explicitly. MLX 0.32
compiles custom kernels with fast math, 0.29 did not, and ``metal::exp`` /
``metal::rsqrt`` then resolve to different implementations; the precise ones
are bit-identical across both (measured), so the kernels' bits survive an
MLX upgrade. ``fma``, ``max`` and the simdgroup MMA are exact either way.
"""

from .matmul import batch_invariant_matmul
from .rmsnorm import batch_invariant_rms_norm
from .quantized import batch_invariant_quantized_matmul
from .attention import (
    batch_invariant_attention,
    batch_invariant_decode_attention,
    batch_invariant_prefill_attention,
)

__all__ = [
    "batch_invariant_matmul",
    "batch_invariant_rms_norm",
    "batch_invariant_quantized_matmul",
    "batch_invariant_decode_attention",
    "batch_invariant_prefill_attention",
    "batch_invariant_attention",
]
