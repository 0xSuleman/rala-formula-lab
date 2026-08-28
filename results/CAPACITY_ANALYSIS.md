# Architecture scaling and capacity analysis

This artifact records a synthetic forward/backward stress test for the hybrid
attention implementation. It validates shape compatibility, gradient flow, and
basic numerical sanity. It does not establish task accuracy, convergence,
production readiness, or superiority over a baseline.

## Model configurations

| Configuration | Dimension | Heads | Layers | MLP ratio | Parameters | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | 64 | 4 | 2 | 2 | 83.3K | 64 |
| Medium | 128 | 4 | 6 | 4 | 1.30M | 64 |
| High | 256 | 8 | 12 | 4 | 10.31M | 64 |
| Extreme | 512 | 8 | 24 | 4 | 82.07M | 64 |

## Recorded high-configuration smoke test

- Synthetic input: two 32×32 RGB tensors.
- Logit shape: `[2, 10]`.
- Forward pass: 220.6 ms on the recorded CPU environment.
- Backward pass: 171.0 ms on the recorded CPU environment.
- Observed NaNs: false.
- Observed infinities: false.

The timings above are environment-specific smoke-test measurements and are not
a systems benchmark.

## Rank diagnostics

| Layer | Memory rank ratio | Output rank ratio |
|---:|---:|---:|
| 0 | 0.994 | 1.000 |
| 1 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 |
| 3 | 0.998 | 1.000 |
| 4 | 1.000 | 1.000 |
| 5 | 1.000 | 1.000 |
| 6 | 0.998 | 1.000 |
| 7 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 |
| 9 | 0.998 | 1.000 |
| 10 | 1.000 | 1.000 |
| 11 | 1.000 | 1.000 |

The output rank remained full in this single synthetic pass. A matched
multi-seed ablation is required before attributing the result to any gate or
claiming an accuracy benefit.

## What remains unverified

- Training convergence at the high and extreme configurations.
- Multi-seed accuracy and uncertainty intervals.
- Causal contribution of salience, global-memory, and output gates.
- Memory use and throughput under controlled hardware benchmarking.
- Parameter-matched comparison with Softmax and linear attention.

Reproduce the structural test with:

```bash
python scripts/stress_test.py
```
