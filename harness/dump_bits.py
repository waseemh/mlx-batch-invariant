"""Dump bit hashes of the invariant kernels' outputs (and a model's logits) so
two MLX versions, or two machines, can be compared.

    python -m harness.dump_bits --out bits_<version>.json [--model <mlx_lm model>]
    python -m harness.dump_bits --compare bits_a.json bits_b.json

Determinism is only promised per (MLX version, chip, OS); this measures how
far it actually carries. Hashes cover: each op in harness.sweep on its probe
input at batch 1, both stock and invariant; and, with --model, row-0 logits
at batch 1 under both modes in prefill and decode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys

import mlx.core as mx

import mlx_batch_invariant as bi
from mlx_batch_invariant._compat import device_info
from harness.sweep import DTYPES, OPS, bits


def _h(x) -> str:
    return hashlib.blake2b(bits(x), digest_size=8).hexdigest()


def dump(model_name: str | None) -> dict:
    out = {"mlx": mx.__version__, "device": device_info().get("device_name"), "macos": platform.mac_ver()[0], "ops": {}, "model": {}}
    for op_name, cls in OPS.items():
        for dt_name, dt in DTYPES.items():
            op = cls(dt, seed=0)
            for invariant in (False, True):
                with bi.set_batch_invariant_mode(enabled=invariant):
                    mx.random.seed(0)
                    y = op.run(op.make_batch(1, 0))
                    mx.eval(y)
                out["ops"][f"{op_name}/{dt_name}/{'inv' if invariant else 'stock'}"] = _h(y)
    if model_name:
        from harness.model_sweep import forward
        from mlx_lm import load
        model, tok = load(model_name)
        ids = mx.array([tok.encode("The quick brown fox jumps over the lazy dog because")])
        for decode in (False, True):
            for invariant in (False, True):
                with bi.set_batch_invariant_mode(enabled=invariant):
                    logits = forward(model, ids, decode)
                    mx.eval(logits)
                out["model"][f"{model_name}/{'decode' if decode else 'prefill'}/{'inv' if invariant else 'stock'}"] = _h(logits)
    return out


def compare(a: dict, b: dict) -> int:
    print(f"A: mlx {a['mlx']} on {a['device']} macOS {a['macos']}")
    print(f"B: mlx {b['mlx']} on {b['device']} macOS {b['macos']}\n")
    rc = 0
    for section in ("ops", "model"):
        keys = sorted(set(a[section]) & set(b[section]))
        if not keys:
            continue
        same = {"stock": [0, 0], "inv": [0, 0]}
        diff_keys = []
        for k in keys:
            mode = "inv" if k.endswith("/inv") else "stock"
            same[mode][1] += 1
            if a[section][k] == b[section][k]:
                same[mode][0] += 1
            else:
                diff_keys.append(k)
        for mode, (n, tot) in same.items():
            print(f"{section:<6} {mode:<6} identical across versions: {n}/{tot}")
        if diff_keys:
            print("  differing: " + ", ".join(diff_keys))
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out")
    p.add_argument("--model")
    p.add_argument("--compare", nargs=2)
    a = p.parse_args(argv)
    if a.compare:
        return compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])))
    d = dump(a.model)
    json.dump(d, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: mlx {d['mlx']}, {len(d['ops'])} op hashes, {len(d['model'])} model hashes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
