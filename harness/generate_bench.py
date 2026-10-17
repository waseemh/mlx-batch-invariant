"""End-to-end generation throughput, stock vs batch-invariant mode.

    python -m harness.generate_bench --model models/Qwen2.5-0.5B-4bit
    python -m harness.generate_bench --model Qwen/Qwen2.5-0.5B --max-tokens 200

Uses ``mlx_lm.stream_generate`` with greedy sampling, reports prompt and
generation tokens/s from mlx_lm's own counters, checks that invariant mode
reproduces its own completion run to run, and asserts that it recorded no
fallbacks (a run with fallbacks is not deterministic). Stock and invariant
mode use different arithmetic, so their greedy completions may legitimately
differ from each other.
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx

import mlx_batch_invariant as bi


def run(model, tok, prompt, max_tokens):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    text, last = [], None
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
        text.append(r.text)
        last = r
    mx.synchronize()
    return "".join(text), last.prompt_tps, last.generation_tps, last.prompt_tokens, last.generation_tokens


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="Write a short essay about why floating-point addition is not associative, with an example.")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--repeat", type=int, default=2, help="runs per mode; the last one is reported (first warms up)")
    a = p.parse_args(argv)

    from mlx_lm import load

    model, tok = load(a.model)
    msgs = [{"role": "user", "content": a.prompt}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True) if tok.chat_template else a.prompt
    print(f"model={a.model} max_tokens={a.max_tokens}\n")
    print("| mode | prompt tok/s | generation tok/s | tokens |")
    print("|---|---|---|---|")
    outputs = {}
    for invariant in (False, True):
        label = "INVARIANT" if invariant else "STOCK"
        bi.reset_stats()
        texts = []
        with bi.set_batch_invariant_mode(enabled=invariant):
            for _ in range(max(2, a.repeat)):
                text, ptps, gtps, pt, gt = run(model, tok, prompt, a.max_tokens)
                texts.append(text)
        outputs[label] = texts
        fb = bi.stats()["fallbacks"]
        print(f"| {label} | {ptps:.0f} | {gtps:.1f} | {pt}+{gt} |")
        if invariant and fb:
            print(f"\nFALLBACKS in invariant mode (not deterministic): {fb}")
            return 1
    for label, texts in outputs.items():
        print(f"{label}: completion reproduced across {len(texts)} runs: {all(t == texts[0] for t in texts)}")
    print(f"stock and invariant completions agree: {outputs['STOCK'][0] == outputs['INVARIANT'][0]} (not required)")
    print(f"\n--- invariant-mode completion ---\n{outputs['INVARIANT'][0][:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
