"""Batch-invariant Metal kernels.

Every kernel here obeys one rule: the floating-point reduction order for a
given output element must not depend on how many *other* rows / sequences
share the launch. Concretely:

* no split-K, no shape-dependent tile or instruction selection (matmul)
* one threadgroup per row with a fixed-shape reduction tree (rmsnorm)
* fixed split *size* along the KV axis, combined in a fixed order (attention)
"""

from .matmul import batch_invariant_matmul
from .rmsnorm import batch_invariant_rms_norm
from .attention import batch_invariant_decode_attention

__all__ = [
    "batch_invariant_matmul",
    "batch_invariant_rms_norm",
    "batch_invariant_decode_attention",
]
