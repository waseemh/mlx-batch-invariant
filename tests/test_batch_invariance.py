import mlx.core as mx
import pytest

import mlx_batch_invariant as bi
from harness.sweep import DTYPES, OPS, sweep_one

BATCHES = [1, 3, 8, 64, 200]


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("dt_name", list(DTYPES))
def test_invariant_mode_is_bitwise_batch_invariant(op_name, dt_name):
    op = OPS[op_name](DTYPES[dt_name], seed=0)
    r = sweep_one(op, BATCHES, invariant=True, seed=0)
    assert r.passed, f"{op_name}/{dt_name} diverged at B={r.first_fail} (max|diff|={r.max_diff})"
    assert not r.fallbacks, f"unexpected fallbacks: {r.fallbacks}"


@pytest.mark.parametrize("op_name", list(OPS))
def test_invariant_kernels_match_stock_within_tolerance(op_name):
    op = OPS[op_name](mx.float32, seed=1)
    batch = op.make_batch(5, 2)
    ref = op.run(batch)
    with bi.set_batch_invariant_mode(strict=True):
        got = op.run(batch)
    mx.eval(ref, got)
    rel = (mx.abs(ref - got).max() / (mx.abs(ref).max() + 1e-6)).item()
    assert rel < 1e-4, rel


def test_mode_patches_and_restores():
    orig = (mx.matmul, mx.addmm, mx.fast.rms_norm, mx.fast.scaled_dot_product_attention, mx.array.__matmul__)
    assert not bi.is_batch_invariant_mode_enabled()
    with bi.set_batch_invariant_mode():
        assert bi.is_batch_invariant_mode_enabled()
        assert mx.matmul is not orig[0]
        with bi.set_batch_invariant_mode():  # re-entrant
            assert bi.is_batch_invariant_mode_enabled()
        assert bi.is_batch_invariant_mode_enabled()
    assert not bi.is_batch_invariant_mode_enabled()
    assert (mx.matmul, mx.addmm, mx.fast.rms_norm, mx.fast.scaled_dot_product_attention, mx.array.__matmul__) == orig


def test_strict_mode_raises_on_unsupported():
    q = mx.random.normal((1, 4, 7, 32))  # prefill: T_q > 1 has no kernel yet
    with bi.set_batch_invariant_mode(strict=True):
        with pytest.raises(RuntimeError):
            mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, mask="causal")


def test_non_strict_counts_fallbacks():
    q = mx.random.normal((1, 4, 7, 32))
    bi.reset_stats()
    with bi.set_batch_invariant_mode():
        mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, mask="causal")
    assert bi.stats()["fallbacks"].get("sdpa:prefill") == 1
