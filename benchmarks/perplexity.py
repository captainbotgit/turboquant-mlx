"""Perplexity benchmark for TurboQuant kill switch validation.

Measures perplexity on WikiText-2 with and without TurboQuant compression.
If the delta exceeds the kill switch threshold (3%), the benchmark STOPS
and reports the regression.

Usage:
    python benchmarks/perplexity.py \
        --model mlx-community/gemma-2b-Q4 \
        --ctx-lengths 4096 32768 65536 \
        --quant turboquant --bits 3

    # Baseline only (no quantization):
    python benchmarks/perplexity.py \
        --model mlx-community/gemma-2b-Q4 \
        --ctx-lengths 4096
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KILL_SWITCH_THRESHOLD = 0.03  # 3% perplexity delta


def load_wikitext2(tokenizer, max_tokens: int = 100_000) -> mx.array:
    """Load WikiText-2 test set and tokenize.

    Falls back to a synthetic dataset if WikiText-2 is not available.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    except Exception:
        logger.warning("Could not load WikiText-2 from HuggingFace — using synthetic data")
        # Deterministic synthetic text for reproducible benchmarks
        rng = np.random.RandomState(42)
        vocab_size = getattr(tokenizer, "vocab_size", 32000)
        tokens = rng.randint(1, vocab_size, size=max_tokens)
        return mx.array(tokens[:max_tokens], dtype=mx.int32)

    tokens = tokenizer.encode(text)
    tokens = tokens[:max_tokens]
    return mx.array(tokens, dtype=mx.int32)


def compute_perplexity(
    model: nn.Module,
    tokens: mx.array,
    ctx_length: int,
    batch_size: int = 1,
) -> tuple[float, float]:
    """Compute perplexity over a token sequence using sliding window.

    Args:
        model: MLX-LM model.
        tokens: 1D token array.
        ctx_length: Context window size.
        batch_size: Not used yet (always 1).

    Returns:
        (perplexity, elapsed_seconds)
    """
    from mlx_lm.models.cache import make_prompt_cache

    total_loss = 0.0
    total_tokens = 0
    n_tokens = tokens.shape[0]

    # Process in chunks of ctx_length
    start_time = time.perf_counter()

    stride = ctx_length // 2  # 50% overlap for better estimation
    for begin in range(0, n_tokens - 1, stride):
        end = min(begin + ctx_length, n_tokens - 1)
        input_ids = tokens[begin:end][None, :]  # (1, seq_len)
        target_ids = tokens[begin + 1 : end + 1]  # (seq_len,)

        cache = make_prompt_cache(model)

        # Forward pass
        logits = model(input_ids, cache=cache)  # (1, seq_len, vocab_size)
        logits = logits[0]  # (seq_len, vocab_size)

        # Only count loss for the non-overlapping portion (except first chunk)
        if begin > 0:
            overlap = ctx_length - stride
            logits = logits[overlap:]
            target_ids = target_ids[overlap:]

        # Cross-entropy loss
        log_probs = nn.losses.cross_entropy(logits, target_ids, reduction="sum")
        mx.eval(log_probs)

        total_loss += log_probs.item()
        total_tokens += target_ids.shape[0]

        if total_tokens >= 10_000:
            break

    elapsed = time.perf_counter() - start_time
    perplexity = math.exp(total_loss / max(total_tokens, 1))

    return perplexity, elapsed


def run_benchmark(args):
    """Run the full perplexity benchmark with kill switch validation."""
    from mlx_lm import load

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    logger.info("Loading evaluation dataset")
    max_tokens = max(args.ctx_lengths) * 3
    tokens = load_wikitext2(tokenizer, max_tokens=max_tokens)
    logger.info(f"Dataset: {tokens.shape[0]} tokens")

    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "kill_switch_threshold": KILL_SWITCH_THRESHOLD,
        "measurements": [],
    }

    # --- Baseline (no quantization) ---
    logger.info("=" * 60)
    logger.info("BASELINE (fp16 KV cache)")
    logger.info("=" * 60)

    baseline_ppls = {}
    for ctx_len in args.ctx_lengths:
        logger.info(f"  Context length: {ctx_len}")
        ppl, elapsed = compute_perplexity(model, tokens, ctx_len)
        baseline_ppls[ctx_len] = ppl
        logger.info(f"  Perplexity: {ppl:.4f} ({elapsed:.1f}s)")
        results["measurements"].append({
            "mode": "baseline",
            "ctx_length": ctx_len,
            "perplexity": ppl,
            "elapsed_seconds": elapsed,
        })

    if args.quant is None:
        _save_results(results, args)
        return results

    # --- TurboQuant ---
    from turboquant_mlx import patch_model

    logger.info("=" * 60)
    logger.info(f"TURBOQUANT (bits={args.bits}, mode={args.mode})")
    logger.info("=" * 60)

    model = patch_model(model, bits=args.bits, mode=args.mode)

    for ctx_len in args.ctx_lengths:
        logger.info(f"  Context length: {ctx_len}")
        ppl, elapsed = compute_perplexity(model, tokens, ctx_len)
        logger.info(f"  Perplexity: {ppl:.4f} ({elapsed:.1f}s)")

        baseline_ppl = baseline_ppls.get(ctx_len)
        if baseline_ppl:
            delta = (ppl - baseline_ppl) / baseline_ppl
            logger.info(f"  Delta vs baseline: {delta:+.4%}")

            results["measurements"].append({
                "mode": f"turboquant-{args.bits}bit-{args.mode}",
                "ctx_length": ctx_len,
                "perplexity": ppl,
                "elapsed_seconds": elapsed,
                "baseline_perplexity": baseline_ppl,
                "delta_pct": delta * 100,
            })

            # KILL SWITCH CHECK
            if delta > KILL_SWITCH_THRESHOLD:
                logger.error("=" * 60)
                logger.error("KILL SWITCH TRIGGERED")
                logger.error(f"  Perplexity delta: {delta:+.4%} > {KILL_SWITCH_THRESHOLD:.0%}")
                logger.error(f"  Baseline: {baseline_ppl:.4f}")
                logger.error(f"  TurboQuant: {ppl:.4f}")
                logger.error(f"  Context length: {ctx_len}")
                logger.error(f"  Config: bits={args.bits}, mode={args.mode}")
                logger.error("=" * 60)
                logger.error("STOPPING — do not proceed. See KILL_SWITCH.md for protocol.")

                results["kill_switch_triggered"] = True
                results["kill_switch_details"] = {
                    "ctx_length": ctx_len,
                    "baseline_ppl": baseline_ppl,
                    "turboquant_ppl": ppl,
                    "delta_pct": delta * 100,
                }
                _save_results(results, args)
                sys.exit(1)
        else:
            results["measurements"].append({
                "mode": f"turboquant-{args.bits}bit-{args.mode}",
                "ctx_length": ctx_len,
                "perplexity": ppl,
                "elapsed_seconds": elapsed,
            })

    results["kill_switch_triggered"] = False
    _save_results(results, args)
    logger.info("All benchmarks passed kill switch validation.")
    return results


def _save_results(results: dict, args):
    """Save results to JSON in benchmarks/ directory."""
    out_dir = Path(__file__).parent
    model_name = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_{model_name}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant perplexity benchmark")
    parser.add_argument("--model", required=True, help="MLX model path or HF repo")
    parser.add_argument(
        "--ctx-lengths",
        nargs="+",
        type=int,
        default=[4096],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--quant",
        choices=["turboquant"],
        default=None,
        help="Quantization method (omit for baseline only)",
    )
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits")
    parser.add_argument(
        "--mode",
        choices=["prod", "mse"],
        default="prod",
        help="TurboQuant mode",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
