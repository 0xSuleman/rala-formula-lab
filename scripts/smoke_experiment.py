"""Small deterministic CPU experiment used by CI and onboarding."""
from __future__ import annotations

import json
from dataclasses import asdict

from rala_lab.training import ExperimentConfig, run_experiment


def main() -> None:
    config = ExperimentConfig(
        dataset="synthetic", seed=123, batch_size=16, epochs=1,
        sample_limit=64, dim=16, heads=4, layers=1, patch_size=4,
        attention_type="hybrid", window_size=4, stats_batches=1,
        warmup_passes=0, device="cpu",
    )
    result = run_experiment(config)
    payload = {
        "config": asdict(config),
        "history": [asdict(epoch) for epoch in result.history],
        "val_accuracy": result.history[-1].val_acc,
        "output_rank_ratio": result.final_stats.output_rank_ratio,
    }
    first = json.dumps(payload, sort_keys=True)
    repeat = run_experiment(config)
    repeat_payload = {
        "history": [asdict(epoch) for epoch in repeat.history],
        "val_accuracy": repeat.history[-1].val_acc,
        "output_rank_ratio": repeat.final_stats.output_rank_ratio,
    }
    if first != json.dumps({**payload, **repeat_payload}, sort_keys=True):
        raise SystemExit("deterministic smoke experiment changed between identical seeded runs")
    print(first)


if __name__ == "__main__":
    main()
