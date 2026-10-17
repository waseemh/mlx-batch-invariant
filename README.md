# mlx-batch-invariant

Batch-invariant (deterministic) kernels for [MLX](https://github.com/ml-explore/mlx)
on Apple Silicon, following Thinking Machines'
[*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
and their PyTorch library
[`batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops).

The claim being enforced: **a row's result must not depend on how many other
rows share the batch.** Stock GPU kernels change their reduction strategy with
batch size (split-K, tile selection, split-KV), so the same prompt yields
different bits at temperature 0 depending on server load. These kernels fix
the reduction order per output element so the bits are identical at batch 1
and batch 256.

Status: **works end to end on `mlx_lm` models, unquantized and 4-bit**
(Qwen2.5-0.5B bf16 and its 4-bit affine quantization): bitwise-identical
logits at batch 1, 4, 16 and 64, in prefill and in cached decode, with no
fallbacks. Matmul kernels run within 1.0–1.4x of stock MLX. See *Results*
and *Roadmap*.

## Install

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Requires MLX ≥ 0.32 and an Apple GPU (uses `mx.fast.metal_kernel`).

## Use

```python
import mlx.core as mx
import mlx_batch_invariant as bi

with bi.set_batch_invariant_mode():
    logits = model(tokens)        # nn.Linear, nn.RMSNorm, decode SDPA now batch-invariant

print(bi.stats())                 # {'handled': {...}, 'fallbacks': {...}}
```

`strict=True` raises instead of silently falling back to a stock kernel for
inputs the batch-invariant kernels don't cover yet. Always check
`bi.stats()["fallbacks"]`: a run with fallbacks is not deterministic.

## What is overridden

MLX has no `torch.Library`, so the mode monkeypatches the module-level
functions that `mlx.nn` and `mlx_lm` reach for, and restores them on exit:

| Patched symbol                            | Routed to                              | Falls back when                          |
|-------------------------------------------|----------------------------------------|------------------------------------------|
| `mx.matmul`, `mx.array.__matmul__`        | `batch_invariant_matmul`               | rhs not 2-D, non-float dtype             |
| `mx.addmm`                                | `batch_invariant_matmul` + add         | same                                     |
| `mx.quantized_matmul`                     | `batch_invariant_quantized_matmul`     | mode ≠ affine, transpose=False, bits ∉ {4,8}, group ∉ {32,64,128} |
| `mx.fast.rms_norm`                        | `batch_invariant_rms_norm`             | non-float                                |
| `mx.fast.scaled_dot_product_attention`    | `batch_invariant_attention`            | array mask, sinks                        |

Not patched: batched 3-D matmul, mxfp4/mxfp8/nvfp4 quantization modes, and
`mx.compile`d graphs that captured the stock ops before the mode was entered.

## Kernel design

- **matmul** (`kernels/matmul.py`): the canonical order is *K cut into fixed
  256-wide groups; inside a group, slabs of 32 fed to the hardware
  `simdgroup_multiply_accumulate` in steps of 8 (operands in the input
  dtype, fp32 accumulator); group partials folded left to right with plain
  fp32 adds.* Every launch shape executes exactly that sequence per output
  element, so the launch can adapt to M without the bits changing:
  - **M ≤ 32** uses 8-row tiles (a decode step no longer pads one row to
    sixty-four) and one threadgroup per (tile, K group), writing fp32 group
    partials that a second kernel folds. Parallelism along K is fine as long
    as the group boundaries never move with M; this is the "fixed split
    size" idea from the post applied to GEMM.
  - **M > 32** uses 64-row tiles and folds all groups inside one
    threadgroup.
  Slabs are staged `[row][k]` with 4-wide vector loads and a padded leading
  dimension; B is read through its strides, so the transposed weight view
  from `nn.Linear` is consumed in place.
- **quantized matmul** (`kernels/quantized.py`): the same tile body with one
  change: the B slab is dequantized to fp32 while it is staged. MLX's affine
  format packs `bits`-wide codes low-to-high in uint32 words with one scale
  and bias per `group_size` codes of a row; each thread unpacks a whole word,
  so global reads are coalesced word loads. Uses the fp32 MMA (an op only has
  to be consistent with itself). Covers 4- and 8-bit, group 32/64/128,
  `transpose=True` (the `nn.QuantizedLinear` layout).
- **rms_norm** (`kernels/rmsnorm.py`): one 256-thread threadgroup per row,
  fixed-stride partial sums, fixed-shape tree reduction.
- **attention** (`kernels/attention.py`): fixed split **size** along KV
  (256 keys), not fixed split count. Key `j` always lands in split `j // 256`
  at the same position, and splits are merged in index order, so neither
  batch size nor cache growth changes an existing row's reduction. Decode
  (`T_q == 1`) runs as per-split partials plus an ordered combine; prefill
  runs one thread per query doing the *same* split loop and merge
  sequentially. The inner loop and merge are literally the same source text
  in all three kernels, which is what makes a token's attention output
  bitwise identical whether it came from prefill or from a later decode
  step. Causal masking follows MLX's convention (query `i` of `T` against a
  cache of `L` sees keys `0..L-T+i`).

## Harness

```bash
.venv/bin/python -m harness.sweep --strict          # op-level batch sweep, bitwise
.venv/bin/python -m pytest -q tests                 # same, as a test suite
.venv/bin/python -m harness.model_sweep --model <unquantized mlx-community repo> [--decode]
```

`harness.sweep` embeds a fixed probe row in batches of 1…256 at row 0 and at
the last row, and compares the probe's output bitwise to its batch-1 result,
for stock MLX and for the invariant mode. It then runs two attention
structure checks the batch sweep cannot see: a 600-token prompt prefilled in
one shot vs. in two chunks (257 + 343, deliberately off the 256 split), and
the last token's output from prefill vs. from a decode step against the same
cache. `harness.model_sweep` does the same
through an `mlx_lm` model: the prompt sits in row 0 with random-token rows
beside it, every transformer block's output for row 0 is hashed, and the
first block that diverges from the batch-1 run is reported.

### Results on this machine (MLX 0.32.4, Apple M4)

```
op           dtype     mode       result
----------------------------------------------------------------
matmul       float16   STOCK      FAIL @B=2 max|diff|=0.0625
matmul       float16   INVARIANT  pass
matmul       bfloat16  STOCK      FAIL @B=2 max|diff|=0.0156
matmul       bfloat16  INVARIANT  pass
matmul       float32   STOCK      FAIL @B=2 max|diff|=0.000427
matmul       float32   INVARIANT  pass
addmm        float16   STOCK      FAIL @B=2 max|diff|=0.000977
addmm        float16   INVARIANT  pass
addmm        bfloat16  STOCK      FAIL @B=2 max|diff|=0.0156
addmm        bfloat16  INVARIANT  pass
addmm        float32   STOCK      FAIL @B=2 max|diff|=5.25e-06
addmm        float32   INVARIANT  pass
qmatmul_4b   float16   STOCK      FAIL @B=2 max|diff|=0.125
qmatmul_4b   bfloat16  STOCK      FAIL @B=16 max|diff|=1
qmatmul_4b   float32   STOCK      FAIL @B=2 max|diff|=0.000534
qmatmul_8b   *         STOCK      FAIL @B=2
qmatmul_*    *         INVARIANT  pass
rms_norm     *         STOCK      pass
rms_norm     *         INVARIANT  pass
sdpa_decode  *         STOCK      pass        (also at KV length 4k, 16k, 64k)
sdpa_decode  *         INVARIANT  pass
sdpa_prefill *         STOCK      pass
sdpa_prefill *         INVARIANT  pass

attention structure checks:
sdpa  fp16  STOCK      chunked_prefill == full_prefill: pass
sdpa  fp16  STOCK      decode(last) == prefill(last):   FAIL
sdpa  bf16  STOCK      decode(last) == prefill(last):   pass
sdpa  fp32  STOCK      decode(last) == prefill(last):   FAIL
sdpa  *     INVARIANT  both checks:                     pass
```

End to end, `Qwen/Qwen2.5-0.5B` (bf16, mlx_lm 0.32.2), 10-token prompt,
24 blocks:

```
$ python -m harness.model_sweep --model Qwen/Qwen2.5-0.5B            [--decode]
STOCK      B=4    FAIL first divergent block=0          (same with --decode)
STOCK      B=16   FAIL first divergent block=0
STOCK      B=64   FAIL first divergent block=0
INVARIANT  B=4    pass                                   (same with --decode)
INVARIANT  B=16   pass
INVARIANT  B=64   pass
```

Same model quantized locally to 4-bit affine, group 64 (`mlx_lm convert -q`,
4.5 bits/weight), both modes: identical table, STOCK diverges at block 0,
INVARIANT passes at B=4/16/64 with no fallbacks.

No fallbacks were recorded in invariant mode, so every matmul, quantized
matmul, norm and attention call in the model went through the
batch-invariant kernels. Block 0 diverging under stock MLX is the GEMM
result above showing up at the first linear layer.

### Speed (K=N=4096, fp16, M4, ms per call, `python -m harness.bench`)

| M   | dense stock | dense invariant | 4-bit stock | 4-bit invariant |
|-----|-------------|-----------------|-------------|-----------------|
| 1   | 0.67        | 0.68 (1.0x)     | 0.33        | 0.46 (1.4x)     |
| 8   | 0.58        | 0.72 (1.2x)     | 0.49        | 0.50 (1.0x)     |
| 64  | 0.99        | 1.26 (1.3x)     | 0.98        | 1.24 (1.3x)     |
| 512 | 5.00        | 6.48 (1.3x)     | 5.36        | 6.69 (1.2x)     |

Before the performance pass the same kernels were 2.1–3.4x (dense) and
2.3–6.8x (4-bit) slower than stock. What moved the numbers, in order of
effect: staging slabs in the input dtype with 4-wide vector loads and the
mixed-precision MMA (about 2x at large M); 8-row tiles for M ≤ 32 (about 2x
at decode); unpacking a whole uint32 per thread in the quantized path
(about 5x at decode for 4-bit). Threadgroup-memory padding and loop
unrolling made no measurable difference here.

Takeaways: on MLX the batch dependence is in the GEMM (stock
`mx.matmul` already differs between batch 1 and batch 2, K=N=4096). Stock
attention is batch-invariant here, but it is **not prefill/decode
invariant**: the same token gets different bits from the prefill kernel and
from a decode step (fp16, fp32). That is the trainer-vs-sampler mismatch the
original post cares about for on-policy RL, and the shared-source kernels
here close it. Stock `rms_norm` was observed invariant, so that kernel turns
an observation into a guarantee rather than fixing a measured failure. That may change with other MLX versions or chips; the
harness is the thing to trust, not this table.

Determinism is per (MLX version, chip, macOS). Nothing here claims identical
bits across machines.

## Roadmap

1. ~~Prefill attention with chunk invariance.~~ Done; see structure checks.
2. ~~Run `model_sweep` on an unquantized checkpoint.~~ Done; clean.
3. ~~`mx.quantized_matmul` (affine 4/8-bit).~~ Done; clean on a 4-bit model.
4. ~~Performance (matmul).~~ Done: 1.0–1.4x of stock. Remaining: the
   attention kernels are still one thread per (batch, head, split) or per
   query with scalar loads; a threadgroup-per-row version with the same
   split/merge order would help long contexts. RMSNorm is already
   bandwidth-bound.
5. **`mx.compile` interplay**: make sure graphs compiled inside the mode keep
   the invariant ops, or document that compilation must happen inside the
   `with` block.
6. Check `mlx_lm` batched generation and its KV cache chunking (256-token
   growth, slicing) under the mode; verify the view handling doesn't
   reintroduce a copy-then-split path.
7. Model-level decode-vs-prefill check: logits for token `t` from a full
   prefill vs. from decoding tokens one at a time. The attention kernels
   guarantee it per op; the harness should assert it per model.
8. Other quantization modes (`mxfp4`, `mxfp8`, `nvfp4`) and `transpose=False`.
9. Try other architectures in `model_sweep` (Llama, Gemma, Mistral) and
   fp16 weights; the Recorder hooks the block class, so most should work.

## Layout

```
mlx_batch_invariant/   package: _mode.py (override), ops.py (fallback accounting), kernels/
harness/               sweep.py (op-level), model_sweep.py (mlx_lm end-to-end)
tests/                 pytest wrapper around the sweep + mode contract tests
```

MIT.
