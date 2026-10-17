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

Status: **works end to end on `mlx_lm` models across three architectures
(Qwen2, Llama, GPT-2), unquantized and 4-bit**: bitwise-identical logits at
batch 1, 4, 16 and 64, in prefill and in cached decode, with no fallbacks,
**on MLX 0.29.4 and 0.32.4, with identical bits across the two versions**
(stock MLX changes its bits between them). Matmul and decode attention run within 1.0–1.5x of stock MLX;
prefill attention is ~4x off; end-to-end generation on a 0.5B model is 1.2x
slower (bf16) and 2.1x slower (4-bit). See *Results* and *Roadmap*.

## Install

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Requires MLX ≥ 0.29 and an Apple GPU (uses `mx.fast.metal_kernel`). Tested
on 0.29.4 and 0.32.4; `_compat.py` covers the API differences.

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
  (256 keys), not fixed split count, so neither batch size nor cache growth
  changes an existing row's reduction. Lane-parallel: one simdgroup per
  unit, each lane owns `D/32` head dims; the score `q·k` is 32 lane partials
  folded by a fixed xor-butterfly tree and taken from lane 0; the online
  softmax then runs key by key and splits merge left to right. Decode units
  are (batch, kv head, head chunk, split) and the grouped query heads share
  each K/V row load (or, when there is too little work to fill the GPU, one
  head per unit); prefill units are (batch, head, block of 4 queries) with
  causal masking applied per query inside the key loop. The per-key update
  and the merge are the same source text in both kernels, which is what
  makes a token's attention output bitwise identical from prefill and from
  a later decode step. Causal convention matches MLX (query `i` of `T`
  against a cache of `L` sees keys `0..L-T+i`).

## Harness

```bash
.venv/bin/python -m harness.sweep --strict          # op-level batch sweep, bitwise
.venv/bin/python -m pytest -q tests                 # same, as a test suite
.venv/bin/python -m harness.model_sweep --model <mlx_lm model> [--decode]
.venv/bin/python -m harness.bench [--attention]     # kernel timings, stock vs invariant
.venv/bin/python -m harness.generate_bench --model <mlx_lm model>   # tokens/s, reproducibility
```

`harness.dump_bits` records bit hashes for cross-version or cross-machine
comparison (see *Across MLX versions*). `harness.sweep` embeds a fixed probe row in batches of 1…256 at row 0 and at
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

Every model with complete weights in the local Hugging Face cache, both
modes (`--decode` and plain), batch 4/16/64 against batch 1:

| model | arch | blocks | notes | stock | invariant |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B         | qwen2 | 24 | bf16, GQA 14/2, RMSNorm       | diverges @ block 0 | clean |
| Qwen2.5-0.5B 4-bit (local) | qwen2 | 24 | affine 4-bit g64              | diverges @ block 0 | clean |
| Qwen/Qwen2.5-Coder-0.5B   | qwen2 | 24 | bf16                          | diverges @ block 0 | clean |
| HuggingFaceTB/SmolLM2-360M | llama | 32 | bf16, GQA 15/5                | diverges @ block 0–1 | clean |
| SmolLM2-360M 4-bit (local) | llama | 32 | affine 4-bit g64              | diverges @ block 0 | clean |
| gpt2                      | gpt2  | 12 | fp32, MHA, LayerNorm, bias linears (`addmm`) | diverges @ block 0 | clean |
| gpt2-large                | gpt2  | 36 | fp32                          | diverges @ block 0 | clean |

GPT-2 uses `mx.fast.layer_norm`, which the mode does not patch; stock
`layer_norm` was probed batch-invariant at widths 768–4096 in fp16/fp32, so
the GPT-2 result rests on that observation (a guaranteed kernel is on the
roadmap). `bigscience/bloom-560m` is cached too but `mlx_lm` has no bloom
model.

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

Attention (Hq=32, Hkv=8, D=128, fp16, `python -m harness.bench --attention`):

| decode B × KV len | stock | invariant |   | prefill T (causal) | stock | invariant |
|---|---|---|---|---|---|---|
| 1 × 1k   | 0.29  | 0.46 (1.6x) |   | 128  | 0.40  | 0.63 (1.5x)   |
| 1 × 32k  | 1.63  | 1.69 (1.0x) |   | 1024 | 3.20  | 12.44 (3.9x)  |
| 8 × 8k   | 3.08  | 3.08 (1.0x) |   | 4096 | 42.68 | 186.84 (4.4x) |
| 64 × 4k  | 12.61 | 11.70 (0.9x) |  |      |       |               |

End to end (`python -m harness.generate_bench`, Qwen2.5-0.5B, 36-token
prompt, 200 greedy tokens; each mode reproduces its own completion run to
run, no fallbacks):

| model | mode | prompt tok/s | generation tok/s |
|---|---|---|---|
| bf16  | stock     | 1093 | 95.4  |
| bf16  | invariant | 680  | 81.0 (0.85x) |
| 4-bit | stock     | 1472 | 275.2 |
| 4-bit | invariant | 847  | 129.6 (0.47x) |
| SmolLM2-360M 4-bit | stock | 728 | 335.1 |
| SmolLM2-360M 4-bit | invariant | 381 | 137.7 (0.41x) |
| gpt2 (fp32) | stock | 737 | 180.1 |
| gpt2 (fp32) | invariant | 436 | 154.0 (0.86x) |

The 4-bit gap is the quantized matmul at M=1. Measured pipelined (100
calls queued per eval) at this model's weight shapes, stock's 4-bit GEMV
takes 7–24 µs per call where the tile kernel takes 25–58 µs, and a 0.5B
model issues ~170 of them per token. Dense fp16 at the same shapes is at
parity. The tile kernel's floor is structural (an 8-row MMA tile staged
through threadgroup memory per 32-wide slab); simdgroup-scope barriers,
vectorized dequantization and dropping extra graph ops did not move it.
Closing it needs a different canonical order for the quantized op (one a
GEMV can execute), traded against large-M speed, or a leaner MMA-GEMV.

Before the performance pass the matmul kernels were 2.1–3.4x (dense) and
2.3–6.8x (4-bit) slower than stock, and attention was 4–8x (decode) and
48–58x (prefill).

### What was tried for the two remaining gaps (and did not help)

Both experiments are kept so nobody repeats them blind.

*4-bit decode (M=1).* Staging a whole 256-wide K group per slab instead of
32 (one memory round trip per group) was 2x *slower*: more threadgroup
memory, fewer resident simdgroups, and 8x more A-staging work. A
one-simdgroup 8-column tile with simdgroup-scope barriers, vectorized
dequantization, half staging and dropping the extra graph ops each changed
nothing measurable. Staging only the live A rows (now in) is neutral at
M=1. The tile kernel's cost at these shapes is the slab loop itself; a
GEMV cannot reproduce the MMA's per-element order (checked: no scalar
summation order matches it bitwise), so closing this gap means choosing a
different canonical order for the quantized op.

*Prefill attention.* `kernels/attention_mma.py` is a full FlashAttention-
style kernel on the simdgroup MMA: 8-query tiles, `S = Q·Kᵀ` and `P·V`
through `simdgroup_multiply_accumulate`, row rescale in registers via the
observed lane layout, decode as the same kernel with the query heads of a
kv group filling the 8 rows. It is correct and bitwise invariant across
batch, chunking and decode-vs-prefill (tests cover it), but slower than the
lane-parallel kernel on every shape (decode 1.2–1.7x of stock, prefill
4.3x), and 16- or 32-key blocks were slower still from register pressure.
Two lessons are baked in: a block whose keys are all masked for a row must
use `corr = 1` (otherwise `exp(-inf - -inf)` poisons the row, which only
shows up when a tile straddles a 256-key split boundary, i.e. on
*misaligned chunked prefill*, never on full prefill); and the per-row
result of the MMA was row-position-independent in every check, so padding
a decode query into an 8-row tile is sound. What moved the numbers, in order of
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

### Across MLX versions

`python -m harness.dump_bits` hashes every op probe and a model's logits;
`--compare` diffs two dumps. MLX 0.29.4 vs 0.32.4, same M4, same macOS:

| | stock MLX | invariant mode |
|---|---|---|
| op probes identical across versions | 16 / 21 | **21 / 21** |
| Qwen2.5-0.5B logits identical (prefill, decode) | 0 / 2 | **2 / 2** |

Stock changes bits between versions in quantized matmul and prefill
attention, so a model's logits move on upgrade. The invariant kernels did
too at first, for one reason: MLX 0.32 compiles custom kernels with fast
math and 0.29 did not, so `metal::exp` and `metal::rsqrt` resolved to
different implementations (measured: 0.32's `exp` is correctly rounded 22%
of the time, 0.29's 63%). `fma`, `max`, division and the simdgroup MMA were
identical either way. The kernels now call `metal::precise::exp`,
`precise::rsqrt` and `precise::divide` explicitly, which are bit-identical
on both versions, at no measurable cost. The stock failures (batch
dependence, decode ≠ prefill) reproduce on 0.29.4 as well.

Determinism is still per chip and macOS: nothing here claims identical bits
across machines, and a Metal compiler change in a future macOS could move
`precise::exp`. The dump/compare tool is how to find out.

## Roadmap

1. ~~Prefill attention with chunk invariance.~~ Done; see structure checks.
2. ~~Run `model_sweep` on an unquantized checkpoint.~~ Done; clean.
3. ~~`mx.quantized_matmul` (affine 4/8-bit).~~ Done; clean on a 4-bit model.
4. ~~Performance (matmul, decode attention).~~ Done: 1.0–1.5x of stock.
   Remaining, with the dead ends above already explored: (a) 4-bit decode
   at M=1, 2.3–3.5x stock GPU time, needs a GEMV-executable canonical order
   for the quantized op (scalar fma in fixed chunks), paid for at large M;
   (b) prefill attention ~4x: the MMA kernel in `attention_mma.py` needs
   its per-block overhead (4 barriers, score round trip through threadgroup
   memory per 8 keys) amortized without the register blow-up that larger
   key blocks caused, e.g. two simdgroups per tile splitting D.
5. **`mx.compile` interplay**: make sure graphs compiled inside the mode keep
   the invariant ops, or document that compilation must happen inside the
   `with` block.
6. Check `mlx_lm` batched generation and its KV cache chunking (256-token
   growth, slicing) under the mode; verify the view handling doesn't
   reintroduce a copy-then-split path.
7. Model-level decode-vs-prefill check: logits for token `t` from a full
   prefill vs. from decoding tokens one at a time. The attention kernels
   guarantee it per op; the harness should assert it per model.
8. A second machine (another Apple chip generation, another macOS) through
   `harness.dump_bits --compare`.
9. Other quantization modes (`mxfp4`, `mxfp8`, `nvfp4`) and `transpose=False`.
10. A batch-invariant `mx.fast.layer_norm` so GPT-2-style models rest on a
   guarantee rather than an observation.
11. ~~Try other architectures~~ Done for Qwen2, Llama, GPT-2 (see the model
   table); Gemma, Mistral, Phi and Qwen3 still untested (no cached weights).
12. Try other architectures in `model_sweep` (Llama, Gemma, Mistral) and
   fp16 weights; the Recorder hooks the block class, so most should work.

## Layout

```
mlx_batch_invariant/   package: _mode.py (override), ops.py (fallback accounting), kernels/
harness/               sweep.py (op-level), model_sweep.py (mlx_lm end-to-end)
tests/                 pytest wrapper around the sweep + mode contract tests
```

## License

MIT. See [LICENSE](LICENSE).
