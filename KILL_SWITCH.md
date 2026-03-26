# Kill Switch Protocol

## The Rule

**If perplexity delta exceeds 3% on any benchmark model at any milestone, STOP.**

This is novel quantization work. A >3% regression indicates a fundamental implementation error, not a tuning problem. Do not attempt to fix forward.

## What to do when triggered

1. **Stop all development** on the module that caused the regression
2. **Document** in this file:
   - Which module introduced the regression
   - Which test caught it (model, context length, bit width)
   - Baseline perplexity vs measured perplexity
   - The exact delta percentage
3. **Do not** attempt to tune parameters to make the number go away
4. **Review** the algorithm implementation against the paper
5. **Report** findings before any further changes

## Validation checkpoints

Run the perplexity benchmark at each milestone:

```bash
python benchmarks/perplexity.py \
  --model mlx-community/gemma-2b-Q4 \
  --ctx-lengths 4096 \
  --quant turboquant --bits 3
```

### Milestone checklist

- [ ] PolarQuant module complete → run benchmark
- [ ] QJL module complete → run benchmark
- [ ] Full TurboQuant pipeline → run benchmark
- [ ] MLX-LM patch integration → run benchmark
- [ ] Each bit-width configuration (2, 3, 4) → run benchmark

## Threshold justification

The TurboQuant paper reports <1% perplexity delta at 3.5 bits on standard benchmarks. We use a 3% threshold to account for:
- Implementation differences (MLX vs CUDA)
- Numerical precision (Metal float32 vs CUDA float64)
- Model differences (Q4 base model adds its own quantization noise)

A delta >3% cannot be explained by these factors and indicates a bug.

## Incident log

*(Record kill switch triggers here)*

None yet.
