"""Stock vs batch-invariant kernel timing. Every iteration is evaluated.

    python -m harness.bench                      # matmul: K=N=4096 fp16, M in 1 8 64 512
    python -m harness.bench --ms 1 64 --k 2048 --n 2048
    python -m harness.bench --attention          # decode + prefill attention tables
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

import mlx_batch_invariant as bi
from mlx_batch_invariant._compat import device_info


def bench(f, iters=20, warm=3):
    for _ in range(warm):
        mx.eval(f())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(f())
    mx.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def bench_attention(dt, iters=5):
    Hq, Hkv, D = 32, 8, 128
    sc = D ** -0.5
    sdpa = mx.fast.scaled_dot_product_attention
    print("decode attention (Hq=32, Hkv=8, D=128), ms per call")
    print("| B | KV len | stock | invariant |")
    print("|---|---|---|---|")
    for B, L in [(1, 1024), (1, 8192), (1, 32768), (8, 1024), (8, 8192), (64, 1024), (64, 4096)]:
        q = mx.random.normal((B, Hq, 1, D)).astype(dt)
        k = mx.random.normal((B, Hkv, L, D)).astype(dt)
        v = mx.random.normal((B, Hkv, L, D)).astype(dt)
        mx.eval(q, k, v)
        s_t = bench(lambda: sdpa(q, k, v, scale=sc), iters)
        i_t = bench(lambda: bi.batch_invariant_attention(q, k, v, scale=sc), iters)
        print(f"| {B} | {L} | {s_t:.2f} | {i_t:.2f} ({i_t / s_t:.1f}x) |")
        del k, v
    print("\nprefill attention (B=1, causal), ms per call")
    print("| T | stock | invariant |")
    print("|---|---|---|")
    for T in (128, 1024, 4096):
        q = mx.random.normal((1, Hq, T, D)).astype(dt)
        k = mx.random.normal((1, Hkv, T, D)).astype(dt)
        v = mx.random.normal((1, Hkv, T, D)).astype(dt)
        mx.eval(q, k, v)
        s_t = bench(lambda: sdpa(q, k, v, scale=sc, mask="causal"), iters)
        i_t = bench(lambda: bi.batch_invariant_attention(q, k, v, scale=sc, mask="causal"), iters)
        print(f"| {T} | {s_t:.2f} | {i_t:.2f} ({i_t / s_t:.1f}x) |")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--attention", action="store_true", help="time attention instead of matmul")
    p.add_argument("--ms", nargs="+", type=int, default=[1, 8, 64, 512])
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    a = p.parse_args(argv)
    dt = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[a.dtype]
    K, N = a.k, a.n
    mx.random.seed(0)
    if a.attention:
        info = device_info()
        print(f"MLX {mx.__version__} on {info.get('device_name')} {a.dtype}\n")
        return bench_attention(dt)
    W = mx.random.normal((N, K)).astype(dt)
    wq, sc, b = mx.quantize(W, group_size=64, bits=4)
    info = device_info()
    print(f"MLX {mx.__version__} on {info.get('device_name')}  K=N={K if K == N else (K, N)} {a.dtype}, ms per call\n")
    print("| M | dense stock | dense invariant | 4-bit stock | 4-bit invariant |")
    print("|---|---|---|---|---|")
    for M in a.ms:
        x = mx.random.normal((M, K)).astype(dt)
        mx.eval(x, wq, sc, b, W)
        s_m = bench(lambda: x @ W.T)
        i_m = bench(lambda: bi.batch_invariant_matmul(x, W.T))
        s_q = bench(lambda: mx.quantized_matmul(x, wq, sc, b, transpose=True, group_size=64, bits=4))
        i_q = bench(lambda: bi.batch_invariant_quantized_matmul(x, wq, sc, b, group_size=64, bits=4))
        print(f"| {M} | {s_m:.2f} | {i_m:.2f} ({i_m / s_m:.1f}x) | {s_q:.2f} | {i_q:.2f} ({i_q / s_q:.1f}x) |")


if __name__ == "__main__":
    main()
