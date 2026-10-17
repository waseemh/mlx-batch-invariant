"""End-to-end sweep on an mlx_lm model: find the first layer whose output for a
fixed prompt changes with batch size.

Requires ``pip install mlx-lm`` and an *unquantized* checkpoint (quantized
linears use ``mx.quantized_matmul``, which has no batch-invariant kernel yet
and is counted as a fallback).

    python -m harness.model_sweep --model mlx-community/Qwen2.5-0.5B-Instruct-bf16
    python -m harness.model_sweep --model ... --decode   # also check one cached decode step

The prompt sits in row 0; the other rows are random token ids of the same
length (so no padding mask is needed). Every transformer block's output for
row 0 is hashed and compared to the batch-1 run, under stock MLX and under
``set_batch_invariant_mode``.

Verified on Qwen/Qwen2.5-0.5B (bf16, mlx_lm 0.32.2): stock diverges at
block 0 from batch 4 on; invariant mode is bitwise clean at B=4/16/64 in
both modes with no fallbacks. Expect to adjust ``_blocks()`` for exotic
architectures.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import mlx.core as mx
import numpy as np

import mlx_batch_invariant as bi
from harness.sweep import bits


def _hash(x: mx.array) -> str:
    return hashlib.blake2b(bits(x), digest_size=8).hexdigest()


def _blocks(model):
    """The transformer blocks: the first list of >= 2 same-class nn.Module
    children found walking the model (``layers`` in most mlx_lm models,
    ``h`` in gpt2, ...)."""
    import mlx.nn as nn

    def walk(mod):
        for _, child in mod.children().items():
            if isinstance(child, list) and len(child) >= 2 and all(isinstance(c, nn.Module) for c in child) \
                    and len({type(c) for c in child}) == 1:
                return child
            if isinstance(child, nn.Module):
                found = walk(child)
                if found:
                    return found
        return None

    blocks = walk(model)
    if not blocks:
        raise RuntimeError("could not find the transformer block list in this model")
    return list(blocks)


class Recorder:
    """Class-level hook on the transformer block so every architecture works."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.index = {id(b): i for i, b in enumerate(blocks)}
        self.hashes: dict[int, str] = {}
        self.cls = type(blocks[0])
        self._orig = self.cls.__call__

    def __enter__(self):
        rec = self

        def wrapped(blk, *args, **kwargs):
            out = rec._orig(blk, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            mx.eval(h)
            rec.hashes[rec.index[id(blk)]] = _hash(h[0:1])
            return out

        self.cls.__call__ = wrapped
        return self

    def __exit__(self, *exc):
        self.cls.__call__ = self._orig


def forward(model, tokens, decode: bool):
    if not decode:
        return model(tokens)
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(tokens, cache=cache)
    nxt = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    return model(nxt, cache=cache)


def run(model, tokens_row0, B, decode, vocab, seed):
    mx.random.seed(seed + B)
    T = tokens_row0.shape[1]
    filler = mx.random.randint(0, vocab, (B - 1, T))
    tokens = mx.concatenate([tokens_row0, filler], axis=0) if B > 1 else tokens_row0
    with Recorder(_blocks(model)) as rec:
        logits = forward(model, tokens, decode)
        mx.eval(logits)
    return rec.hashes, _hash(logits[0:1])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog because")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 64])
    p.add_argument("--decode", action="store_true", help="prefill with cache, then compare one decode step")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    try:
        from mlx_lm import load
    except ImportError:
        print("pip install mlx-lm", file=sys.stderr)
        return 2

    model, tok = load(a.model)
    ids = mx.array([tok.encode(a.prompt)])
    vocab = int(model.args.vocab_size) if hasattr(model, "args") else 32000
    print(f"model={a.model} prompt_tokens={ids.shape[1]} blocks={len(_blocks(model))} decode={a.decode}\n")

    rc = 0
    for invariant in (False, True):
        label = "INVARIANT" if invariant else "STOCK"
        bi.reset_stats()
        with bi.set_batch_invariant_mode(enabled=invariant):
            ref_layers, ref_logits = run(model, ids, 1, a.decode, vocab, a.seed)
            for B in a.batch_sizes:
                if B == 1:
                    continue
                layers, logits = run(model, ids, B, a.decode, vocab, a.seed)
                first = next((i for i in sorted(layers) if layers[i] != ref_layers[i]), None)
                ok = first is None and logits == ref_logits
                where = "" if ok else f" first divergent block={first if first is not None else 'head/logits'}"
                print(f"{label:<10} B={B:<4} {'pass' if ok else 'FAIL'}{where}")
                if invariant and not ok:
                    rc = 1
        fb = bi.stats()["fallbacks"]
        if invariant and fb:
            print(f"           fallbacks (not batch-invariant): {fb}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
