"""Memory profiling benchmark for TurboQuant KV cache compression.

Measures peak RAM and KV cache sizes with and without TurboQuant.

Usage:
    python benchmarks/memory_profile.py \
        --model mlx-community/gemma-2b-Q4 \
        --ctx-length 32768 --quant turboquant
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_rss_mb() -> float:
    """Get current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def measure_kv_cache_size(cache) -> float:
    """Measure total KV cache size in MB."""
    total_bytes = sum(c.nbytes for c in cache)
    return total_bytes / (1024 * 1024)


def run_profile(args):
    """Run memory profiling with and without TurboQuant."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    logger.info(f"Loading model: {args.model}")
    rss_before_model = get_rss_mb()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    rss_after_model = get_rss_mb()
    logger.info(f"Model loaded: {rss_after_model - rss_before_model:.0f} MB")

    # Generate a dummy prompt of the target context length
    prompt_tokens = mx.ones((1, args.ctx_length), dtype=mx.int32)

    results = {
        "model": args.model,
        "ctx_length": args.ctx_length,
        "timestamp": datetime.now().isoformat(),
        "model_rss_mb": rss_after_model - rss_before_model,
        "measurements": [],
    }

    # --- Baseline ---
    logger.info("=" * 60)
    logger.info(f"BASELINE — ctx_length={args.ctx_length}")

    cache = make_prompt_cache(model)
    rss_before = get_rss_mb()
    start = time.perf_counter()

    # Process in chunks to avoid OOM
    chunk_size = min(2048, args.ctx_length)
    for i in range(0, args.ctx_length, chunk_size):
        end = min(i + chunk_size, args.ctx_length)
        chunk = prompt_tokens[:, i:end]
        model(chunk, cache=cache)
        mx.eval(cache[0].state)

    elapsed = time.perf_counter() - start
    rss_after = get_rss_mb()
    kv_mb = measure_kv_cache_size(cache)

    logger.info(f"  KV cache: {kv_mb:.1f} MB")
    logger.info(f"  RSS delta: {rss_after - rss_before:.1f} MB")
    logger.info(f"  Elapsed: {elapsed:.1f}s")
    logger.info(f"  Tokens/sec: {args.ctx_length / elapsed:.0f}")

    results["measurements"].append({
        "mode": "baseline",
        "kv_cache_mb": kv_mb,
        "rss_delta_mb": rss_after - rss_before,
        "elapsed_seconds": elapsed,
        "tokens_per_sec": args.ctx_length / elapsed,
    })

    if args.quant is None:
        _save_results(results, args)
        return

    # --- TurboQuant ---
    from turboquant_mlx import patch_model, unpatch_model

    # Reset
    del cache
    mx.metal.clear_cache()

    logger.info("=" * 60)
    logger.info(f"TURBOQUANT (bits={args.bits}) — ctx_length={args.ctx_length}")

    model = patch_model(model, bits=args.bits, mode=args.mode)
    cache = make_prompt_cache(model)
    rss_before = get_rss_mb()
    start = time.perf_counter()

    for i in range(0, args.ctx_length, chunk_size):
        end = min(i + chunk_size, args.ctx_length)
        chunk = prompt_tokens[:, i:end]
        model(chunk, cache=cache)

    elapsed = time.perf_counter() - start
    rss_after = get_rss_mb()
    kv_mb = measure_kv_cache_size(cache)

    logger.info(f"  KV cache: {kv_mb:.1f} MB")
    logger.info(f"  RSS delta: {rss_after - rss_before:.1f} MB")
    logger.info(f"  Elapsed: {elapsed:.1f}s")
    logger.info(f"  Tokens/sec: {args.ctx_length / elapsed:.0f}")

    baseline = results["measurements"][0]
    compression = baseline["kv_cache_mb"] / max(kv_mb, 0.01)
    logger.info(f"  Compression ratio: {compression:.1f}x")

    results["measurements"].append({
        "mode": f"turboquant-{args.bits}bit-{args.mode}",
        "kv_cache_mb": kv_mb,
        "rss_delta_mb": rss_after - rss_before,
        "elapsed_seconds": elapsed,
        "tokens_per_sec": args.ctx_length / elapsed,
        "compression_ratio": compression,
    })

    unpatch_model(model)
    _save_results(results, args)


def _save_results(results: dict, args):
    out_dir = Path(__file__).parent
    model_name = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"memory_{model_name}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant memory profiler")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx-length", type=int, default=32768)
    parser.add_argument("--quant", choices=["turboquant"], default=None)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--mode", choices=["prod", "mse"], default="prod")
    args = parser.parse_args()
    run_profile(args)


if __name__ == "__main__":
    main()
