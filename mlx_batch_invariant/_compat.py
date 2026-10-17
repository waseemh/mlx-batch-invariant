"""Small shims for MLX API differences across the versions this runs on."""

from __future__ import annotations

import mlx.core as mx


def result_type(a: mx.array, b: mx.array):
    """``mx.result_type`` exists from 0.30; older versions promote via an op."""
    if hasattr(mx, "result_type"):
        return mx.result_type(a, b)
    if a.dtype == b.dtype:
        return a.dtype
    return (mx.array(0, a.dtype) + mx.array(0, b.dtype)).dtype


def device_info() -> dict:
    if hasattr(mx, "device_info"):
        return mx.device_info()
    return mx.metal.device_info()
