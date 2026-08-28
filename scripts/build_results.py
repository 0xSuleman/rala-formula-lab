"""Validate result JSON files and rebuild portfolio-facing summaries/figures."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
MATCHED_RUN = RUNS_DIR / "cifar10" / "hybrid-d64-10k-30ep-matched-baselines.json"


@dataclass(frozen=True)
class RunRecord:
    artifact: str
    suite: str
    variant: str
    config: dict[str, Any]
    history: list[dict[str, Any]]
    final_stats: dict[str, Any]
    inference_ms: float

    @property
    def best_row(self) -> dict[str, Any]:
        return max(self.history, key=lambda row: float(row["val_acc"]))

    @property
    def final_row(self) -> dict[str, Any]:
        return self.history[-1]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path.relative_to(ROOT)}: {exc}") from exc

    required = {"config", "history", "final_stats", "inference_ms"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path.relative_to(ROOT)} is missing: {sorted(missing)}")
    if not payload["history"]:
        raise ValueError(f"{path.relative_to(ROOT)} has no training history")

    for index, row in enumerate(payload["history"], start=1):
        for key in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc"):
            if key not in row:
                raise ValueError(
                    f"{path.relative_to(ROOT)} history row {index} is missing {key}"
                )
    return payload


def load_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    for path in sorted(RUNS_DIR.rglob("*.json")):
        payload = load_json(path)
        relative = path.relative_to(ROOT).as_posix()
        suite = path.parent.name
        config = payload["config"]
        records.append(
            RunRecord(
                artifact=relative,
                suite=suite,
                variant=str(config.get("attention_type", "unknown")),
                config=config,
                history=payload["history"],
                final_stats=payload["final_stats"],
                inference_ms=float(payload["inference_ms"]),
            )
        )
        for name, baseline in sorted(payload.get("baselines", {}).items()):
            records.append(
                RunRecord(
                    artifact=f"{relative}#baseline={name}",
                    suite=suite,
                    variant=name,
                    config=baseline["config"],
                    history=baseline["history"],
                    final_stats=baseline["final_stats"],
                    inference_ms=float(baseline["inference_ms"]),
                )
            )
    if not records:
        raise ValueError("No JSON result artifacts were found")
    return records


def percent(value: Any) -> float:
    return round(float(value) * 100.0, 4)


def optional_float(*values: Any) -> str:
    for value in values:
        if value is not None:
            return f"{float(value):.6f}"
    return ""


def row_for_csv(record: RunRecord) -> dict[str, Any]:
    config = record.config
    best = record.best_row
    final = record.final_row
    stats = record.final_stats
    return {
        "artifact": record.artifact,
        "suite": record.suite,
        "variant": record.variant,
        "dataset": config.get("dataset", ""),
        "task": config.get("task", config.get("dataset", "")),
        "seed": config.get("seed", ""),
        "dimension": config.get("dim", ""),
        "heads": config.get("heads", ""),
        "layers": config.get("layers", ""),
        "samples": config.get("sample_limit", ""),
        "epochs": config.get("epochs", ""),
        "best_val_accuracy_pct": percent(best["val_acc"]),
        "best_epoch": best["epoch"],
        "final_train_accuracy_pct": percent(final["train_acc"]),
        "final_val_accuracy_pct": percent(final["val_acc"]),
        "final_generalization_gap_pct": round(
            percent(final["train_acc"]) - percent(final["val_acc"]), 4
        ),
        "inference_ms_recorded": round(record.inference_ms, 4),
        "memory_rank_ratio": optional_float(
            stats.get("memory_rank_ratio"), stats.get("kv_rank_ratio")
        ),
        "output_rank_ratio": optional_float(stats.get("output_rank_ratio")),
        "device": config.get("device", ""),
    }


def write_csv(records: list[RunRecord]) -> None:
    rows = [row_for_csv(record) for record in records]
    target = RESULTS_DIR / "summary.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matched_records(records: list[RunRecord]) -> list[RunRecord]:
    prefix = MATCHED_RUN.relative_to(ROOT).as_posix()
    matched = [record for record in records if record.artifact.split("#", 1)[0] == prefix]
    order = {"hybrid": 0, "softmax": 1, "linear": 2}
    return sorted(matched, key=lambda record: order.get(record.variant, 99))


def write_latest_results(records: list[RunRecord]) -> None:
    matched = matched_records(records)
    scaling = [
        record
        for record in records
        if record.suite == "cifar10-scaling" and "#baseline=" not in record.artifact
    ]
    recall = [
        record
        for record in records
        if record.suite == "associative-recall" and "#baseline=" not in record.artifact
    ]
    best_scaling = max(scaling, key=lambda record: record.best_row["val_acc"])
    latest_recall = max(recall, key=lambda record: record.artifact)

    lines = [
        "# Latest verified results",
        "",
        "Generated from committed JSON artifacts by `scripts/build_results.py`.",
        "All accuracy results are single-seed (`seed=7`) observations unless stated otherwise.",
        "",
        "## Matched CIFAR-10 comparison",
        "",
        "D=64, four heads, two layers, 10,000 training samples, 30 epochs, patch size 4.",
        "",
        "| Variant | Best validation | Epoch | Final validation | Recorded inference |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in matched:
        lines.append(
            f"| {record.variant.title()} | {percent(record.best_row['val_acc']):.2f}% "
            f"| {record.best_row['epoch']} | {percent(record.final_row['val_acc']):.2f}% "
            f"| {record.inference_ms:.2f} ms |"
        )
    lines.extend(
        [
            "",
            "The hybrid model falls 1.05 percentage points below the best Softmax validation checkpoint and 0.90 points above linear attention. Recorded timing is retained for provenance but is not a controlled hardware benchmark.",
            "",
            "## Newer scaling runs",
            "",
            f"The best committed large-model run reached **{percent(best_scaling.best_row['val_acc']):.2f}%** validation accuracy at epoch {best_scaling.best_row['epoch']} (`{best_scaling.artifact}`). These runs changed multiple variables and are not an ablation.",
            "",
            "## Associative recall",
            "",
            f"The latest run reached **{percent(latest_recall.best_row['val_acc']):.2f}%** best validation accuracy while ending at **{percent(latest_recall.final_row['train_acc']):.2f}%** training and **{percent(latest_recall.final_row['val_acc']):.2f}%** validation accuracy. This is evidence of memorization without useful held-out generalization.",
            "",
            "## Interpretation limits",
            "",
            "- No uncertainty interval is available because the recorded runs use one seed.",
            "- Cross-file inference time is not comparable across CPU/GPU and configuration changes.",
            "- Rank observations are diagnostics, not causal evidence of accuracy improvements.",
            "- The next decisive study is a parameter-matched, multi-seed gate ablation.",
            "",
        ]
    )
    (RESULTS_DIR / "LATEST_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig: Any, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        target = FIGURES_DIR / f"{name}.{extension}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        if extension == "svg":
            # Matplotlib emits trailing spaces in SVG path data. Normalizing
            # them keeps generated artifacts friendly to git diff --check.
            normalized = "\n".join(
                line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()
            )
            target.write_text(normalized + "\n", encoding="utf-8")


def build_figures(records: list[RunRecord]) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"hybrid": "#2563eb", "softmax": "#7c3aed", "linear": "#0f766e"}

    matched = matched_records(records)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for record in matched:
        ax.plot(
            [row["epoch"] for row in record.history],
            [percent(row["val_acc"]) for row in record.history],
            label=record.variant.title(),
            color=colors.get(record.variant),
            linewidth=2.2,
        )
    ax.set(title="Matched CIFAR-10 validation curves", xlabel="Epoch", ylabel="Validation accuracy (%)")
    ax.legend(frameon=True)
    ax.set_ylim(bottom=15)
    fig.tight_layout()
    save_figure(fig, "matched-cifar10-validation")
    plt.close(fig)

    cifar = [
        record
        for record in records
        if record.suite in {"cifar10", "cifar10-scaling"}
        and "#baseline=" not in record.artifact
    ]
    cifar = sorted(cifar, key=lambda record: record.best_row["val_acc"])
    labels = [Path(record.artifact).stem.replace("20260612-", "") for record in cifar]
    values = [percent(record.best_row["val_acc"]) for record in cifar]
    bar_colors = ["#93c5fd" if record.suite == "cifar10" else "#1d4ed8" for record in cifar]
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels, values, color=bar_colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set(title="Best validation accuracy across committed CIFAR-10 runs", xlabel="Best validation accuracy (%)")
    ax.set_xlim(0, max(values) + 6)
    fig.tight_layout()
    save_figure(fig, "cifar10-run-overview")
    plt.close(fig)

    recall = sorted(
        [record for record in records if record.suite == "associative-recall"],
        key=lambda record: int(record.config.get("sample_limit", 0)),
    )
    labels = [f"D{record.config.get('dim')} / {int(record.config.get('sample_limit', 0)):,}" for record in recall]
    train = [percent(record.final_row["train_acc"]) for record in recall]
    validation = [percent(record.final_row["val_acc"]) for record in recall]
    positions = list(range(len(recall)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([position - width / 2 for position in positions], train, width, label="Final train", color="#dc2626")
    ax.bar([position + width / 2 for position in positions], validation, width, label="Final validation", color="#f59e0b")
    ax.set_xticks(positions, labels)
    ax.set(title="Associative-recall generalization gap", xlabel="Dimension / samples", ylabel="Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "associative-recall-generalization")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate artifacts without writing outputs")
    args = parser.parse_args()

    records = load_records()
    if not MATCHED_RUN.exists():
        raise ValueError(f"Missing primary comparison artifact: {MATCHED_RUN.relative_to(ROOT)}")
    print(f"Validated {len(records)} run variants from {len(list(RUNS_DIR.rglob('*.json')))} JSON files.")
    if args.check:
        return
    write_csv(records)
    write_latest_results(records)
    build_figures(records)
    print("Rebuilt results/summary.csv, results/LATEST_RESULTS.md, and six figure files.")


if __name__ == "__main__":
    main()
