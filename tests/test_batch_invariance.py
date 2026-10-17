import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_batch_invariant as bi
from harness.sweep import DTYPES, OPS, attention_structure_checks, sweep_one

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


@pytest.mark.parametrize("bits,gs", [(4, 32), (4, 64), (4, 128), (8, 32), (8, 64), (8, 128)])
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_quantized_matmul_matches_stock(bits, gs, dt):
    mx.random.seed(5)
    K, N = 1024, 200
    W = mx.random.normal((N, K)).astype(dt)
    wq, sc, bi_ = mx.quantize(W, group_size=gs, bits=bits)
    x = mx.random.normal((9, K)).astype(dt)
    ref = mx.quantized_matmul(x, wq, sc, bi_, transpose=True, group_size=gs, bits=bits)
    got = bi.batch_invariant_quantized_matmul(x, wq, sc, bi_, group_size=gs, bits=bits)
    mx.eval(ref, got)
    rel = (mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max() / mx.abs(ref.astype(mx.float32)).max()).item()
    # Stock dequantizes in the weight dtype (bf16 rounds every dequantized
    # weight to 8 mantissa bits); we dequantize in fp32. A couple of bf16 ulps
    # of disagreement is the stock kernel's rounding, not ours.
    assert rel < (1e-2 if dt == mx.bfloat16 else 2e-3), rel


def test_quantized_linear_routes_through_mode():
    lin = nn.QuantizedLinear(256, 64, group_size=64, bits=4)
    x = mx.random.normal((3, 256)).astype(mx.float16)
    bi.reset_stats()
    with bi.set_batch_invariant_mode(strict=True):
        y = lin(x)
    mx.eval(y)
    assert bi.stats()["handled"].get("quantized_matmul") == 1


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


@pytest.mark.parametrize("dt_name", list(DTYPES))
def test_attention_chunking_and_decode_equivalence(dt_name):
    r = attention_structure_checks(DTYPES[dt_name], invariant=True)
    assert all(r.values()), r


def test_prefill_matches_stock_causal_with_cache():
    mx.random.seed(3)
    q = mx.random.normal((2, 8, 37, 64)); k = mx.random.normal((2, 2, 150, 64)); v = mx.random.normal((2, 2, 150, 64))
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask="causal")
    got = bi.batch_invariant_prefill_attention(q, k, v, scale=0.125, causal=True)
    rel = (mx.abs(ref - got).max() / mx.abs(ref).max()).item()
    assert rel < 1e-4, rel
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    got = bi.batch_invariant_prefill_attention(q, k, v, scale=0.125, causal=False)
    assert (mx.abs(ref - got).max() / mx.abs(ref).max()).item() < 1e-4


def test_strict_mode_raises_on_unsupported():
    q = mx.random.normal((1, 4, 7, 32))
    m = mx.zeros((7, 7), dtype=mx.bool_)  # array masks have no kernel
    with bi.set_batch_invariant_mode(strict=True):
        with pytest.raises(RuntimeError):
            mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, mask=m)


def test_non_strict_counts_fallbacks():
    q = mx.random.normal((1, 4, 7, 32))
    m = mx.zeros((7, 7), dtype=mx.bool_)
    bi.reset_stats()
    with bi.set_batch_invariant_mode():
        mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, mask=m)
    assert bi.stats()["fallbacks"].get("sdpa:array-mask") == 1


def test_experimental_mma_attention_is_correct_and_invariant():
    """The MMA attention kernel is not wired in, but keep it honest."""
    from mlx_batch_invariant.kernels import attention_mma as am
    from harness.sweep import bits
    mx.random.seed(11)
    Hq, Hkv, D, T = 8, 2, 128, 600
    sc = D ** -0.5
    q = mx.random.normal((2, Hq, T, D)).astype(mx.float16)
    k = mx.random.normal((2, Hkv, T, D)).astype(mx.float16)
    v = mx.random.normal((2, Hkv, T, D)).astype(mx.float16)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal")
    full = am.batch_invariant_prefill_attention(q, k, v, scale=sc, causal=True)
    mx.eval(ref, full)
    rel = (mx.abs(ref.astype(mx.float32) - full.astype(mx.float32)).max() / mx.abs(ref.astype(mx.float32)).max()).item()
    assert rel < 2e-2, rel
    # chunk at 257 (tiles straddle the 512 split boundary) and decode of the last token
    part2 = am.batch_invariant_prefill_attention(q[:, :, 257:], k, v, scale=sc, causal=True)
    dec = am.batch_invariant_decode_attention(q[:, :, -1:], k, v, scale=sc)
    mx.eval(part2, dec)
    assert bits(part2) == bits(full[:, :, 257:])
    assert bits(dec) == bits(full[:, :, -1:])
    # batch invariance: row 0 alone vs in the batch of 2
    alone = am.batch_invariant_prefill_attention(q[:1], k[:1], v[:1], scale=sc, causal=True)
    mx.eval(alone)
    assert bits(alone) == bits(full[:1])


def test_simdgroup_matrix_lane_layout():
    """attention_mma's row rescale assumes this lane -> element layout."""
    src = """
        simdgroup_float8x8 A; simdgroup_load(A, inp, 8);
        uint lane = thread_index_in_simdgroup;
        out[lane * 2] = A.thread_elements()[0]; out[lane * 2 + 1] = A.thread_elements()[1];
    """
    kern = mx.fast.metal_kernel(name="lane_layout_probe", input_names=["inp"], output_names=["out"], source=src)
    a = mx.arange(64, dtype=mx.float32).reshape(8, 8)
    o = kern(inputs=[a], template=[("T", mx.float32)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(64,)], output_dtypes=[mx.float32])[0].tolist()
    for lane in range(32):
        row = ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4
        col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4
        assert o[2 * lane] == row * 8 + col and o[2 * lane + 1] == row * 8 + col + 1, lane
