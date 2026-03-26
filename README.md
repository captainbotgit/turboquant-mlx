# TurboQuant-MLX

3-bit KV cache compression for MLX on Apple Silicon. Implements [TurboQuant](https://arxiv.org/abs/2504.19874) and [PolarQuant](https://arxiv.org/abs/2502.02617) for efficient long-context LLM inference.

## What it does

TurboQuant compresses the KV cache during LLM inference from 16-bit to ~3 bits per dimension, targeting:
- **6x memory reduction** in KV cache
- **2-8x inference speedup** (attention bottleneck)
- **<3% perplexity degradation** (kill switch enforced)

This enables running models like Qwen 27B with 128K context on M4 Pro 48GB, where the uncompressed KV cache would exceed available memory.

## Installation

```bash
pip install -e ".[dev]"
```

Requires: Python ≥3.10, MLX ≥0.18.0, Apple Silicon Mac.

## Quick Start

```python
from turboquant_mlx import patch_model
from mlx_lm import load, generate

# Load any MLX-LM model
model, tokenizer = load("mlx-community/gemma-2b-Q4")

# Enable TurboQuant — that's it
model = patch_model(model, bits=3)

# Generate as usual — KV cache is compressed transparently
output = generate(model, tokenizer, prompt="The meaning of life is", max_tokens=512)
print(output)

# Restore original behavior
from turboquant_mlx import unpatch_model
unpatch_model(model)
```

## How it works

**Stage 1 — MSE Quantization**: Rotates KV vectors by a random orthogonal matrix, then scalar-quantizes each coordinate using Lloyd-Max codebooks optimized for the resulting Gaussian marginals.

**Stage 2 — QJL Residual**: Computes the residual from Stage 1, projects it through a random Gaussian matrix, and stores only the sign bits (1 bit each). This provides an unbiased inner-product estimator.

**Total**: (b-1) bits for MSE + 1 bit for QJL = b bits per dimension. Default is b=3.

The compression is applied only to cached past tokens — new tokens during generation remain in full precision (float16).

## Architecture

```
turboquant_mlx/
├── codebook.py      # Lloyd-Max codebook computation
├── polar_quant.py   # PolarQuant (polar coordinate transform)
├── qjl.py           # QJL 1-bit residual projection
├── turbo.py         # TurboQuant compressor (MSE + QJL pipeline)
├── patch.py         # MLX-LM monkey-patching (TurboQuantKVCache)
└── __init__.py      # Public API
```

## Kill Switch

**If perplexity increases by >3% on any benchmark, STOP.** See [KILL_SWITCH.md](KILL_SWITCH.md) for the full validation protocol.

```bash
# Run perplexity benchmark with kill switch
python benchmarks/perplexity.py \
  --model mlx-community/gemma-2b-Q4 \
  --ctx-lengths 4096 \
  --quant turboquant --bits 3
```

## Benchmarks

```bash
# Memory profiling
python benchmarks/memory_profile.py \
  --model mlx-community/gemma-2b-Q4 \
  --ctx-length 32768 --quant turboquant

# Perplexity (baseline + TurboQuant)
python benchmarks/perplexity.py \
  --model mlx-community/gemma-2b-Q4 \
  --ctx-lengths 4096 32768 --quant turboquant --bits 3
```

## Tests

```bash
pytest tests/ -v
```

## Test Matrix

| Model | Context Lengths | Purpose |
|-------|----------------|---------|
| gemma-2b-Q4 | 4K, 32K, 64K | Fast iteration, catch regressions |
| Qwen-7B-Q4 | 4K, 32K | Mid-scale validation |
| 70B Q4 | 4K, 32K, 64K | Final target (overnight) |

## References

- Zandieh et al. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Han et al. (2025). *PolarQuant: Quantizing KV Caches with Polar Transformation.* [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)

## License

MIT
