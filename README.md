# mlx-batch-invariant

Batch-invariant (deterministic) kernels for [MLX](https://github.com/ml-explore/mlx)
on Apple Silicon, following Thinking Machines'
[*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
and their PyTorch library
[`batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops).

**The problem.** At temperature 0 the same prompt still comes back
different, because stock GPU kernels change their reduction strategy with
batch size (split-K, tile selection, split-KV). How many other requests
share the batch changes the order floating-point sums are taken in, and
therefore the bits.

**The fix.** Every kernel here commits to one reduction order per output
element that never depends on batch size. One line switches a model over:

```python
import mlx_batch_invariant as bi

with bi.set_batch_invariant_mode():
    logits = model(tokens)      # same bits at batch 1 and batch 256
```

## Status

- Bitwise-identical outputs across batch sizes 1 to 256 for matmul,
  quantized matmul, RMSNorm and attention, in fp16, bf16 and fp32.
- End to end on `mlx_lm` models across three architectures (Qwen2, Llama,
  GPT-2), unquantized and 4-bit: identical logits at batch 1, 4, 16 and 64,
  in prefill and in cached decode, with no fallbacks. Stock MLX diverges at
  the first transformer block on every one of them.
- A token's attention output is the same from prefill and from a later
  decode step, and a prompt prefilled in chunks matches one-shot prefill.
  Stock MLX fails the first of these.
- Identical bits on MLX 0.29.4 and 0.32.4. Stock MLX changes its bits
  between those versions.
- Speed relative to stock: dense matmul and decode attention 0.9–1.3x,
  4-bit matmul at decode 2.3–3.5x, prefill attention ~4x. Generation on a
  0.5B model runs at 0.85x (bf16) and 0.47x (4-bit) of stock tokens/s.

Tested on one machine (Apple M4, macOS 15.7). Determinism is per chip and
OS; see *Limitations*.

## Install

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Requires MLX ≥ 0.29 and an Apple GPU (the kernels use
`mx.fast.metal_kernel`). `mlx-lm` is optional, for the model-level harness.

## Use

```python
import mlx.core as mx
import mlx_batch_invariant as bi

with bi.set_batch_invariant_mode():
    logits = model(tokens)

print(bi.stats())   # {'handled': {...}, 'fallbacks': {...}}
```

Inputs the kernels do not cover fall back to the stock op and are counted.
A run with fallbacks is not deterministic, so check
`bi.stats()["fallbacks"]`, or pass `strict=True` to raise instead.

### What is overridden

MLX has no `torch.Library`, so the mode swaps the module-level functions
that `mlx.nn` and `mlx_lm` call, and restores them on exit.

| Patched symbol | Routed to | Falls back when |
|---|---|---|
| `mx.matmul`, `mx.array.__matmul__` | `batch_invariant_matmul` | rhs not 2-D, non-float dtype |
| `mx.addmm` | `batch_invariant_matmul` + add | same |
| `mx.quantized_matmul` | `batch_invariant_quantized_matmul` | mode ≠ affine, `transpose=False`, bits ∉ {4, 8}, group ∉ {32, 64, 128} |
| `mx.fast.rms_norm` | `batch_invariant_rms_norm` | non-float |
| `mx.fast.scaled_dot_product_attention` | `batch_invariant_attention` | array mask, sinks |

Not patched: batched 3-D matmul, `mx.fast.layer_norm` (stock was observed
batch-invariant; see *Limitations*), mxfp4/mxfp8/nvfp4 quantization, and
`mx.compile`d graphs that captured the stock ops before the mode was
entered.

## How the kernels stay invariant

The rule is the same everywhere: the arithmetic per output element is fixed,
and only the launch shape adapts to the batch.

**Matmul** (`kernels/matmul.py`). K is cut into fixed 256-wide groups. Inside
a group, 32-wide slabs feed the hardware `simdgroup_multiply_accumulate` in
steps of 8 (operands in the input dtype, fp32 accumulator); group partials
are folded left to right with plain fp32 adds. For M ≤ 32 the launch uses
8-row tiles and one threadgroup per (tile, K group), writing group partials
that a second kernel folds, which is the post's "fixed split size" idea
applied to GEMM; for larger M, 64-row tiles fold all groups in place. Both
execute the same sequence per element.

**Quantized matmul** (`kernels/quantized.py`). The same tile body; the
weight slab is dequantized while it is staged. MLX's affine format packs
codes low-to-high in uint32 words with one scale and bias per group, and
each thread unpacks a whole word.

**RMSNorm** (`kernels/rmsnorm.py`). One threadgroup per row, fixed-stride
partial sums, fixed-shape tree.

**Attention** (`kernels/attention.py`). Keys are processed in fixed splits
of 256, so neither batch size nor cache growth changes an existing row's
reduction. One simdgroup per unit; each lane owns a slice of the head
dimension, the dot product is folded by a fixed shuffle tree, the online
softmax runs key by key, and splits merge left to right. Decode units share
each K/V row across the grouped query heads; prefill units share it across
a block of queries. The per-key update and the merge are the same source
text in both kernels, which is what makes decode equal prefill bitwise.

Transcendentals and division use `metal::precise::` explicitly. MLX 0.32
compiles custom kernels with fast math and 0.29 did not, so the default
`exp` and `rsqrt` differ between versions; the precise variants are
bit-identical on both (measured), which is why the kernels' bits survive an
MLX upgrade.

## Verification

```bash
.venv/bin/python -m harness.sweep --strict                       # op-level, bitwise
.venv/bin/python -m pytest -q tests                              # same, as a test suite
.venv/bin/python -m harness.model_sweep --model <model> [--decode]   # whole model
.venv/bin/python -m harness.bench [--attention]                  # ms per call, stock vs invariant
.venv/bin/python -m harness.generate_bench --model <model>       # tokens/s, reproducibility
.venv/bin/python -m harness.dump_bits --out a.json [--model <model>]  # hashes, for cross-version diffs
```

- `sweep` embeds a fixed probe row in batches of 1…256, at the first and
  last position, and compares its output bitwise to the batch-1 result. It
  also checks a 600-token prompt prefilled in one shot vs. two chunks (257
  + 343, deliberately off the 256 split), and the last token from prefill
  vs. from a decode step.
- `model_sweep` puts the prompt in row 0 beside random-token rows, hashes
  every transformer block's output for row 0, and reports the first block
  that diverges from batch 1.
- `dump_bits --compare a.json b.json` diffs two runs, across MLX versions or
  machines.

## Results

Apple M4, macOS 15.7, MLX 0.32.4 unless stated.

### Op level

| op | stock MLX | invariant mode |
|---|---|---|
| matmul, addmm (fp16, bf16, fp32) | differs at batch 2 | identical through 256 |
| quantized matmul 4-bit, 8-bit | differs at batch 2 (bf16: 16) | identical |
| rms_norm | identical | identical |
| decode attention, up to 64k keys | identical | identical |
| prefill attention | identical | identical |
| chunked prefill == full prefill | passes | passes |
| decode(last) == prefill(last) | **fails** (fp16, fp32) | passes |

On MLX the batch dependence lives in the GEMM, dense and quantized. Stock
attention is batch-invariant but not prefill/decode invariant, which is the
trainer-vs-sampler mismatch the original post cares about for on-policy RL.

### Model level

Batch 4, 16 and 64 against batch 1, prefill and cached decode:

| model | arch | notes | stock | invariant |
|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B | qwen2 | bf16, GQA 14/2 | diverges at block 0 | clean |
| Qwen2.5-0.5B 4-bit (local) | qwen2 | affine 4-bit, group 64 | diverges at block 0 | clean |
| Qwen/Qwen2.5-Coder-0.5B | qwen2 | bf16 | diverges at block 0 | clean |
| HuggingFaceTB/SmolLM2-360M | llama | bf16, GQA 15/5 | diverges at block 0–1 | clean |
| SmolLM2-360M 4-bit (local) | llama | affine 4-bit, group 64 | diverges at block 0 | clean |
| gpt2 | gpt2 | fp32, MHA, LayerNorm, bias linears | diverges at block 0 | clean |
| gpt2-large | gpt2 | fp32, 36 blocks | diverges at block 0 | clean |

No fallbacks were recorded in invariant mode on any of them. The 4-bit
models were quantized locally with `mlx_lm convert -q`.

### Across MLX versions

Same machine, MLX 0.29.4 vs 0.32.4, via `dump_bits --compare`:

| | stock MLX | invariant mode |
|---|---|---|
| op probes identical across versions | 16 / 21 | 21 / 21 |
| Qwen2.5-0.5B logits identical (prefill, decode) | 0 / 2 | 2 / 2 |

Stock changes bits between versions in quantized matmul and prefill
attention. The full sweep, tests and model sweeps pass on both versions.

## Performance

Matmul, K=N=4096, fp16, ms per call:

| M | dense stock | dense invariant | 4-bit stock | 4-bit invariant |
|---|---|---|---|---|
| 1 | 0.67 | 0.68 (1.0x) | 0.33 | 0.46 (1.4x) |
| 8 | 0.58 | 0.72 (1.2x) | 0.49 | 0.50 (1.0x) |
| 64 | 0.99 | 1.26 (1.3x) | 0.98 | 1.24 (1.3x) |
| 512 | 5.00 | 6.48 (1.3x) | 5.36 | 6.69 (1.2x) |

Attention, Hq=32, Hkv=8, D=128, fp16, ms per call:

| decode, batch × KV length | stock | invariant | | prefill, tokens | stock | invariant |
|---|---|---|---|---|---|---|
| 1 × 1k | 0.29 | 0.46 (1.6x) | | 128 | 0.40 | 0.63 (1.5x) |
| 1 × 32k | 1.63 | 1.69 (1.0x) | | 1024 | 3.20 | 12.44 (3.9x) |
| 8 × 8k | 3.08 | 3.08 (1.0x) | | 4096 | 42.68 | 186.84 (4.4x) |
| 64 × 4k | 12.61 | 11.70 (0.9x) | | | | |

Generation, 200 greedy tokens after a short prompt; each mode reproduces its
own completion run to run:

| model | stock tok/s | invariant tok/s |
|---|---|---|
| Qwen2.5-0.5B bf16 | 95.4 | 81.0 (0.85x) |
| Qwen2.5-0.5B 4-bit | 275.2 | 129.6 (0.47x) |
| SmolLM2-360M 4-bit | 335.1 | 137.7 (0.41x) |
| gpt2 fp32 | 180.1 | 154.0 (0.86x) |

The 4-bit gap is the quantized matmul at M=1. Measured pipelined at this
model's weight shapes, stock's 4-bit GEMV takes 7–24 µs per call where the
tile kernel takes 25–58 µs, and a 0.5B model issues about 170 of them per
token. The floor is structural: an 8-row MMA tile staged through
threadgroup memory per slab. Simdgroup-scope barriers, whole-group staging,
vectorized dequantization, half staging and fewer graph ops were each tried
and did not move it, and a scalar GEMV cannot reproduce the MMA's
per-element order (no scalar summation order matches it bitwise). Closing
it means a different canonical order for the quantized op, paid for at
large M.

For prefill attention, `kernels/attention_mma.py` is a complete
FlashAttention-style kernel on the simdgroup MMA, correct and bitwise
invariant (tests cover it), but slower than the lane-parallel kernel on
every shape, so it is not wired in. It records two lessons: a block whose
keys are all masked for a row needs `corr = 1`, or `exp(-inf - -inf)`
poisons the row, which only shows on chunked prefill at an offset that
straddles a 256-key split; and larger key blocks lose to register pressure.

## Limitations

- Determinism is per chip, macOS and MLX version. Two versions agree here;
  no second machine has been tested. `dump_bits` makes that check two
  commands.
- GPT-2 relies on stock `mx.fast.layer_norm`, which was probed
  batch-invariant at widths 768–4096 but is not guaranteed by a kernel here.
- Code that captured the stock ops before entering the mode, including
  `mx.compile`d graphs, bypasses it silently.
- Stock and invariant mode use different arithmetic, so their greedy
  completions can differ from each other. Determinism holds within a mode.

## Roadmap

1. 4-bit decode: a GEMV-executable canonical order for the quantized op.
2. Prefill attention: amortize the MMA kernel's per-block overhead without
   the register blow-up, e.g. two simdgroups per tile splitting D.
3. A batch-invariant `mx.fast.layer_norm`.
4. `mx.compile` interplay, and `mlx_lm` batched generation with its KV cache
   chunking under the mode.
5. A model-level assertion that decoding token by token equals full prefill.
6. A second machine, and architectures without cached weights here (Gemma,
   Mistral, Phi, Qwen3).
7. Other quantization modes and `transpose=False`.

## Layout

```
mlx_batch_invariant/   _mode.py (override), ops.py (fallback accounting), _compat.py, kernels/
harness/               sweep, model_sweep, bench, generate_bench, dump_bits
tests/                 the sweep as pytest, mode contract tests, experimental-kernel tests
```

## License

MIT. See [LICENSE](LICENSE).
