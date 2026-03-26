# TurboQuant-MLX Validation Results

**Date:** 2026-03-26
**Validator:** Claude Code (automated pipeline)
**Hardware:** Apple Silicon (M-series), macOS Darwin 25.2.0

---

## Summary

| Metric | Result | Status |
|--------|--------|--------|
| Perplexity delta (3-bit) | +18.88% | KILL SWITCH TRIGGERED |
| Perplexity delta (4-bit) | +3.59% | KILL SWITCH TRIGGERED (barely) |
| Memory compression | 0.4x (worse than baseline) | FAIL |
| Qwen 27B quality (short ctx) | Excellent | PASS |
| Qwen 27B speed | ~10.7 tok/s | PASS |

**Recommendation: NOT READY for Captain production use.**

---

## Step 2: Baseline Perplexity

- **Model:** mlx-community/gemma-2-2b-it-4bit
- **Dataset:** WikiText-2 (test split)
- **Context length:** 4096
- **Baseline perplexity:** 12.1066

## Step 3: TurboQuant Perplexity (Kill Switch)

### 3-bit (prod mode — MSE + QJL)
- **TurboQuant perplexity:** 14.3918
- **Delta:** +18.88%
- **Kill switch threshold:** 3%
- **Result:** KILL SWITCH TRIGGERED (exit code 1)

### 4-bit (prod mode — MSE + QJL)
- **TurboQuant perplexity:** 12.5418
- **Delta:** +3.59%
- **Result:** KILL SWITCH TRIGGERED (barely over 3% threshold)

### Analysis
The perplexity degradation at 3-bit is severe (18.88%), far exceeding the paper's
claimed <1% delta. At 4-bit it's marginal (+3.59%), suggesting the core algorithm
works but the current implementation has issues at lower bit widths. Possible causes:
- Lloyd-Max codebooks may not be optimal for the actual KV distribution
- Random rotation matrices may not provide sufficient decorrelation
- The Q4 base model quantization may interact poorly with KV quantization

## Step 4: Memory Profile

| Mode | KV Cache Size | Notes |
|------|--------------|-------|
| Baseline (fp16) | 416.0 MB | Standard KVCache |
| TurboQuant (3-bit) | 1044.7 MB | 2.5x WORSE |

**Compression ratio: 0.4x (no savings — actually increases memory)**

### Root Cause
The quantized indices are stored as `int32` (4 bytes per element) instead of
bit-packed representation. For 3-bit quantization:
- **Ideal per token:** ~3 bits × 256 dims = 96 bytes + norms
- **Actual per token:** 4 bytes × 256 dims (indices) + norms + 1 byte × 256 (QJL signs) = ~1,288 bytes
- **fp16 per token:** 2 bytes × 256 dims = 512 bytes

The indices must be bit-packed to achieve actual memory savings.

## Step 5: Qwen 27B Captain Workload Tests

**Model:** Qwen3.5-27B-Claude-Opus-Distilled-MLX-4bit (with TurboQuant 3-bit patch)

### Test Results

| Test | Tokens | Time | Speed | Quality |
|------|--------|------|-------|---------|
| Planning | 300 | 27.9s | 10.7 tok/s | Excellent |
| Reasoning | 300 | 29.0s | 10.4 tok/s | Excellent |
| Delegation | 300 | 28.1s | 10.7 tok/s | Excellent |
| Comprehension | 300 | 39.1s | 7.7 tok/s | Excellent |

### Quality Assessment
All four tests produced coherent, well-structured, relevant responses:
- **Planning:** Proper weekly marketing structure with budget allocation
- **Reasoning:** Accurate dental industry benchmarks, proper CPA analysis
- **Delegation:** Clear slide-by-slide carousel breakdown with CTAs
- **Comprehension:** Correctly identified TurboQuant's purpose and usage from README

### Important Caveat
These tests used short prompts (< 100 tokens) with 300 token generation. At this
context length, most KV entries stay in the fp16 recent window (32 tokens) and
compression only affects a small portion of the cache. **The quality results do not
validate long-context performance** where compression would dominate.

## Bugs Found During Validation

Three critical bugs were discovered and fixed during the validation run:

### Bug 1: Patch Never Activates (CRITICAL)
- **File:** `patch.py:patch_model()`
- **Issue:** `make_cache` was set on `model.model` but `make_prompt_cache()` checks `model` (top-level). TurboQuant never activated — all prior benchmarks used standard fp16 cache.
- **Fix:** Set `make_cache` on both top-level model and inner model.

### Bug 2: Compressed Tokens Lost on Re-compression (CRITICAL)
- **File:** `patch.py:TurboQuantKVCache.update_and_fetch()`
- **Issue:** When compression triggers on subsequent chunks, old compressed state was replaced (not accumulated). Previously compressed tokens were silently discarded, causing offset/KV length mismatch and shape errors.
- **Fix:** Decompress old states, merge with new tokens, re-compress.

### Bug 3: Hybrid Model Support (Qwen3.5)
- **File:** `patch.py:_get_model_config()` and `turboquant_make_cache()`
- **Issue:** Qwen3.5 has SSM (Mamba) layers requiring `ArraysCache`, not `KVCache`. Config extraction failed for multimodal models with nested `text_config`.
- **Fix:** Detect SSM layers from original `make_cache()` and preserve their cache type. Look for config in `language_model.args` for multimodal models.

## Recommendation

**NOT READY for Captain production deployment.** Specific blockers:

1. **Kill switch triggered:** 18.88% perplexity degradation at 3-bit far exceeds the 3% threshold. Even 4-bit barely fails at 3.59%.

2. **No memory savings:** The compressed representation is actually 2.5x larger than fp16 due to int32 index storage. Bit-packing is required.

3. **Short-context quality is misleading:** The Qwen 27B tests show good quality, but compression barely engages at short context. Long-context validation is needed.

4. **Critical bugs found:** Three bugs prevented the system from working at all. While fixed, this indicates the code needs thorough review and testing before deployment.

### Path to Production

1. **Implement bit-packing** for quantized indices (most impactful)
2. **Investigate perplexity regression** — compare against paper's CUDA implementation
3. **Add long-context validation** (32K+ tokens) on Qwen 27B
4. **Comprehensive test suite** covering incremental cache accumulation
5. **Re-validate** after fixes with kill switch at 3% threshold
