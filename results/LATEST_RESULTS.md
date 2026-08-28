# Latest verified results

Generated from committed JSON artifacts by `scripts/build_results.py`.
All accuracy results are single-seed (`seed=7`) observations unless stated otherwise.

## Matched CIFAR-10 comparison

D=64, four heads, two layers, 10,000 training samples, 30 epochs, patch size 4.

| Variant | Best validation | Epoch | Final validation | Recorded inference |
|---|---:|---:|---:|---:|
| Hybrid | 52.70% | 29 | 51.95% | 1333.04 ms |
| Softmax | 53.75% | 28 | 52.50% | 636.46 ms |
| Linear | 51.80% | 30 | 51.80% | 661.72 ms |

The hybrid model falls 1.05 percentage points below the best Softmax validation checkpoint and 0.90 points above linear attention. Recorded timing is retained for provenance but is not a controlled hardware benchmark.

## Newer scaling runs

The best committed large-model run reached **46.09%** validation accuracy at epoch 9 (`results/runs/cifar10-scaling/20260612-100954-d512-l7-h8.json`). These runs changed multiple variables and are not an ablation.

## Associative recall

The latest run reached **15.10%** best validation accuracy while ending at **72.68%** training and **14.70%** validation accuracy. This is evidence of memorization without useful held-out generalization.

## Interpretation limits

- No uncertainty interval is available because the recorded runs use one seed.
- Cross-file inference time is not comparable across CPU/GPU and configuration changes.
- Rank observations are diagnostics, not causal evidence of accuracy improvements.
- The next decisive study is a parameter-matched, multi-seed gate ablation.
