"""Batch-sweep conformance harness.

For each (op, dtype) a fixed probe row is embedded in batches of increasing
size and compared *bitwise* against its batch-1 result. Position invariance is
checked too: the probe is placed at row 0 and at the last row.

    python -m harness.sweep                       # default sweep
    python -m harness.sweep --ops matmul sdpa --dtypes fp16
    python -m harness.sweep --strict              # exit 1 if invariant mode fails

Columns: STOCK is unmodified MLX; INVARIANT is the same call inside
``set_batch_invariant_mode``. "first fail" is the smallest batch size whose
probe row differs from batch 1; max|diff| is over the whole sweep.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_batch_invariant as bi
from mlx_batch_invariant._compat import device_info

DTYPES = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}
DEFAULT_BATCHES = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]


_UINT = {2: mx.uint16, 4: mx.uint32, 8: mx.uint64}


def bits(x: mx.array) -> bytes:
    """Raw bytes of ``x``; reinterprets as an unsigned int of the same width so
    bfloat16 (which numpy cannot represent) compares bitwise too."""
    u = mx.contiguous(x).view(_UINT[x.dtype.size]) if hasattr(mx, "contiguous") else x.view(_UINT[x.dtype.size])
    return np.array(u).tobytes()


def maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item())


# ---- op definitions ---------------------------------------------------------
# Each op builds (probe, filler(B)) and a run(batch) -> output; row semantics
# are "axis 0 is the batch" so the harness can slice the probe row.


class Op:
    name: str

    def __init__(self, dt, seed):
        self.dt = dt
        mx.random.seed(seed)
        self.setup()

    def setup(self): ...
    def make_batch(self, B: int, pos: int) -> mx.array: ...
    def run(self, batch: mx.array) -> mx.array: ...


class Matmul(Op):
    name = "matmul"
    K, N = 4096, 4096

    def setup(self):
        self.W = mx.random.normal((self.N, self.K)).astype(self.dt)  # nn.Linear layout
        self.x0 = mx.random.normal((1, self.K)).astype(self.dt)

    def make_batch(self, B, pos):
        rows = [mx.random.normal((1, self.K)).astype(self.dt) for _ in range(B - 1)]
        rows.insert(pos, self.x0)
        return mx.concatenate(rows, axis=0)

    def run(self, batch):
        return batch @ self.W.T


class Addmm(Matmul):
    name = "addmm"

    def setup(self):
        super().setup()
        self.lin = nn.Linear(self.K, self.N, bias=True)
        self.lin.set_dtype(self.dt)

    def run(self, batch):
        return self.lin(batch)


class QuantizedMatmul(Matmul):
    name = "qmatmul_4b"
    bits, gs = 4, 64

    def setup(self):
        super().setup()
        self.wq, self.sc, self.bi = mx.quantize(self.W, group_size=self.gs, bits=self.bits)

    def run(self, batch):
        return mx.quantized_matmul(
            batch, self.wq, self.sc, self.bi, transpose=True, group_size=self.gs, bits=self.bits
        )


class QuantizedMatmul8(QuantizedMatmul):
    name = "qmatmul_8b"
    bits, gs = 8, 32


class RMSNorm(Op):
    name = "rms_norm"
    D = 4096

    def setup(self):
        self.w = mx.random.normal((self.D,)).astype(self.dt)
        self.x0 = mx.random.normal((1, self.D)).astype(self.dt)

    def make_batch(self, B, pos):
        rows = [mx.random.normal((1, self.D)).astype(self.dt) for _ in range(B - 1)]
        rows.insert(pos, self.x0)
        return mx.concatenate(rows, axis=0)

    def run(self, batch):
        return mx.fast.rms_norm(batch, self.w, 1e-5)


class SDPADecode(Op):
    name = "sdpa_decode"
    Hq, Hkv, L, D = 8, 2, 1000, 128

    def setup(self):
        s = (1, self.Hq, 1, self.D)
        kv = (1, self.Hkv, self.L, self.D)
        self.q0 = mx.random.normal(s).astype(self.dt)
        self.k0 = mx.random.normal(kv).astype(self.dt)
        self.v0 = mx.random.normal(kv).astype(self.dt)

    def make_batch(self, B, pos):
        qs, ks, vs = [], [], []
        for _ in range(B - 1):
            qs.append(mx.random.normal(self.q0.shape).astype(self.dt))
            ks.append(mx.random.normal(self.k0.shape).astype(self.dt))
            vs.append(mx.random.normal(self.v0.shape).astype(self.dt))
        qs.insert(pos, self.q0); ks.insert(pos, self.k0); vs.insert(pos, self.v0)
        return (mx.concatenate(qs, 0), mx.concatenate(ks, 0), mx.concatenate(vs, 0))

    def run(self, batch):
        q, k, v = batch
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.D ** -0.5)


class SDPAPrefill(Op):
    name = "sdpa_prefill"
    Hq, Hkv, Tq, L, D = 8, 2, 48, 300, 128  # L > Tq: causal against a cache

    def setup(self):
        self.q0 = mx.random.normal((1, self.Hq, self.Tq, self.D)).astype(self.dt)
        kv = (1, self.Hkv, self.L, self.D)
        self.k0 = mx.random.normal(kv).astype(self.dt)
        self.v0 = mx.random.normal(kv).astype(self.dt)

    def make_batch(self, B, pos):
        qs, ks, vs = [], [], []
        for _ in range(B - 1):
            qs.append(mx.random.normal(self.q0.shape).astype(self.dt))
            ks.append(mx.random.normal(self.k0.shape).astype(self.dt))
            vs.append(mx.random.normal(self.v0.shape).astype(self.dt))
        qs.insert(pos, self.q0); ks.insert(pos, self.k0); vs.insert(pos, self.v0)
        return (mx.concatenate(qs, 0), mx.concatenate(ks, 0), mx.concatenate(vs, 0))

    def run(self, batch):
        q, k, v = batch
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.D ** -0.5, mask="causal")


OPS = {c.name: c for c in (Matmul, Addmm, QuantizedMatmul, QuantizedMatmul8, RMSNorm, SDPADecode, SDPAPrefill)}


# ---- sequence-structure checks (attention only) -----------------------------
# These are the properties the batch sweep cannot see: the same token must get
# the same bits whether its prompt was prefilled in one chunk or several, and
# whether it was produced by the prefill kernel or by a later decode step.


def attention_structure_checks(dt, invariant: bool, seed: int = 0) -> dict[str, bool]:
    Hq, Hkv, D = 8, 2, 128
    T, T1 = 600, 257  # T1 deliberately not a multiple of the 256 split
    mx.random.seed(seed)
    q = mx.random.normal((2, Hq, T, D)).astype(dt)
    k = mx.random.normal((2, Hkv, T, D)).astype(dt)
    v = mx.random.normal((2, Hkv, T, D)).astype(dt)
    sc = D ** -0.5
    sdpa = lambda *a, **kw: mx.fast.scaled_dot_product_attention(*a, scale=sc, **kw)
    with bi.set_batch_invariant_mode(enabled=invariant):
        full = sdpa(q, k, v, mask="causal")
        part1 = sdpa(q[:, :, :T1], k[:, :, :T1], v[:, :, :T1], mask="causal")
        part2 = sdpa(q[:, :, T1:], k, v, mask="causal")
        chunked = mx.concatenate([part1, part2], axis=2)
        dec = sdpa(q[:, :, T - 1 : T], k, v)  # decode step for the last token
        mx.eval(full, chunked, dec)
    return {
        "chunked_prefill == full_prefill": bits(full) == bits(chunked),
        "decode(last) == prefill(last)": bits(full[:, :, T - 1 : T]) == bits(dec),
    }


# ---- sweep ------------------------------------------------------------------


@dataclass
class Result:
    op: str
    dtype: str
    mode: str
    passed: bool = True
    first_fail: int | None = None
    max_diff: float = 0.0
    fallbacks: dict = field(default_factory=dict)


def sweep_one(op: Op, batches: list[int], invariant: bool, seed: int) -> Result:
    res = Result(op.name, str(op.dt).split(".")[-1], "INVARIANT" if invariant else "STOCK")
    bi.reset_stats()
    with bi.set_batch_invariant_mode(enabled=invariant):
        mx.random.seed(seed)
        ref = op.run(op.make_batch(1, 0))[0:1]
        mx.eval(ref)
        ref_bits = bits(ref)
        for B in batches:
            for pos in sorted({0, B - 1}):
                mx.random.seed(seed + B * 7 + pos)
                out = op.run(op.make_batch(B, pos))[pos : pos + 1]
                mx.eval(out)
                if bits(out) != ref_bits:
                    res.passed = False
                    res.first_fail = res.first_fail or B
                    res.max_diff = max(res.max_diff, maxdiff(out, ref))
    res.fallbacks = bi.stats()["fallbacks"]
    return res


def fmt(r: Result) -> str:
    status = "pass" if r.passed else f"FAIL @B={r.first_fail} max|diff|={r.max_diff:.3g}"
    fb = f"  fallbacks={r.fallbacks}" if r.fallbacks else ""
    return f"{r.op:<12} {r.dtype:<9} {r.mode:<10} {status}{fb}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    p.add_argument("--dtypes", nargs="+", default=list(DTYPES), choices=list(DTYPES))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCHES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict", action="store_true", help="exit 1 if INVARIANT mode fails anywhere")
    a = p.parse_args(argv)

    info = device_info()
    print(f"MLX {mx.__version__} on {info.get('device_name')} ({info.get('architecture')})")
    print(f"batch sizes: {a.batch_sizes}\n")
    print(f"{'op':<12} {'dtype':<9} {'mode':<10} result")
    print("-" * 64)
    any_inv_fail = False
    for op_name in a.ops:
        for dt_name in a.dtypes:
            op = OPS[op_name](DTYPES[dt_name], a.seed)
            for invariant in (False, True):
                r = sweep_one(op, a.batch_sizes, invariant, a.seed)
                print(fmt(r), flush=True)
                if invariant and not r.passed:
                    any_inv_fail = True
    print()
    print("attention structure checks (chunked prefill, decode vs prefill):")
    for dt_name in a.dtypes:
        for invariant in (False, True):
            r = attention_structure_checks(DTYPES[dt_name], invariant, a.seed)
            mode = "INVARIANT" if invariant else "STOCK"
            for name, ok in r.items():
                print(f"{'sdpa':<12} {dt_name:<9} {mode:<10} {name}: {'pass' if ok else 'FAIL'}")
                if invariant and not ok:
                    any_inv_fail = True
    print()
    if any_inv_fail:
        print("INVARIANT mode FAILED somewhere above.")
        return 1 if a.strict else 0
    print("INVARIANT mode: all probe rows bitwise identical across the sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
