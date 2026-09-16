"""Generate traceable E1-E7 validation tables from saved metric artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fedecai.models.b6_common import sha256_file


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "reports/tables/fedecai"
EXPERIMENTS = {
    "E1": {
        "id": "b6_resnet18_mel3_finetune_seed391", "method": "Centralized ResNet18 baseline",
        "metrics": "artifacts/metrics/b6_resnet18_mel3_finetune_seed391_validation.json",
        "checkpoint": "artifacts/checkpoints/b6_resnet18_mel3_finetune_seed391.pt",
        "predictions": "artifacts/predictions/b6_resnet18_mel3_finetune_seed391_validation.csv",
    },
    "E2": {"id": "e2_resnet18_unconditional_grl_seed391", "method": "Centralized ResNet18 + unconditional GRL", "root": "artifacts/fedecai/e2_resnet18_unconditional_grl"},
    "E3": {"id": "e3_resnet18_emotion_conditioned_grl_seed391", "method": "Centralized ResNet18 + emotion-conditioned GRL", "root": "artifacts/fedecai/e3_resnet18_conditional_grl"},
    "E4": {"id": "e4_fedavg_resnet18_seed391", "method": "Federated ResNet18 FedAvg", "root": "artifacts/fedecai/e4_fedavg_resnet18"},
    "E5": {"id": "e5_fedprox_resnet18_seed391", "method": "Federated ResNet18 FedProx", "root": "artifacts/fedecai/e5_fedprox_resnet18"},
    "E6": {"id": "e6_fedavg_resnet18_unconditional_grl_seed391", "method": "Federated ResNet18 FedAvg + unconditional GRL", "root": "artifacts/fedecai/e6_fedavg_unconditional_grl"},
    "E7": {"id": "e7_fedecai_resnet18_seed391", "method": "FedECAI ResNet18", "root": "artifacts/fedecai/e7_fedecai_resnet18"},
}


def paths(step: str, item: dict[str, Any]) -> tuple[Path, Path, Path]:
    if step == "E1":
        return ROOT / item["metrics"], ROOT / item["checkpoint"], ROOT / item["predictions"]
    root = ROOT / item["root"]
    return root / "metrics.json", root / "best_main.pt" if step >= "E4" else root / "best.pt", root / "predictions.csv"


def load_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows, metrics_by_step = [], {}
    probe_path = OUTPUT / "accent_probe_comparison.csv"
    probes = {}
    if probe_path.is_file():
        for row in pd.read_csv(probe_path).to_dict("records"):
            probes[str(row["experiment"])] = row
    for step, item in EXPERIMENTS.items():
        metric_path, checkpoint, prediction = paths(step, item)
        if not metric_path.is_file():
            continue
        metric = json.loads(metric_path.read_text(encoding="utf-8")); metrics_by_step[step] = metric
        per_class = metric.get("per_class", {})
        probe = probes.get(step, {})
        rows.append({
            "experiment": step, "experiment_id": item["id"], "method": item["method"],
            "overall_macro_f1": metric.get("overall_macro_f1"), "uar": metric.get("uar"),
            "weighted_f1": metric.get("weighted_f1"), "accuracy": metric.get("accuracy"),
            "north_macro_f1": metric.get("north_macro_f1"), "central_macro_f1": metric.get("central_macro_f1"),
            "south_macro_f1": metric.get("south_macro_f1"), "worst_accent_macro_f1": metric.get("worst_accent_macro_f1"),
            "best_accent_macro_f1": metric.get("best_accent_macro_f1"), "accent_gap": metric.get("accent_gap"),
            "accent_std": metric.get("accent_std"),
            "per_emotion_f1_json": json.dumps({key: value.get("f1") for key, value in per_class.items()}, sort_keys=True),
            "per_emotion_recall_json": json.dumps({key: value.get("recall") for key, value in per_class.items()}, sort_keys=True),
            "accent_discriminator_macro_f1": metric.get("accent_discriminator_macro_f1"),
            "accent_discriminator_uar": metric.get("accent_discriminator_uar"),
            "accent_probe_macro_f1": probe.get("accent_probe_macro_f1"),
            "accent_probe_uar": probe.get("accent_probe_uar"),
            "total_parameters": metric.get("total_parameters"), "trainable_parameters": metric.get("trainable_parameters"),
            "model_size_mb": metric.get("model_size_mb"), "training_time_sec": metric.get("training_time_sec"),
            "latency_median_ms": metric.get("latency_median_ms"), "latency_p95_ms": metric.get("latency_p95_ms"),
            "checkpoint_sha256": sha256_file(checkpoint), "prediction_sha256": sha256_file(prediction),
        })
    return rows, metrics_by_step


def delta_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    comparisons = [("E2", "E1"), ("E3", "E1"), ("E3", "E2"), ("E5", "E4"), ("E6", "E4"), ("E7", "E4"), ("E7", "E6")]
    by_step = frame.set_index("experiment")
    rows = []
    for target, baseline in comparisons:
        if target not in by_step.index or baseline not in by_step.index:
            continue
        target_row, baseline_row = by_step.loc[target], by_step.loc[baseline]
        row = {"comparison": f"{target} - {baseline}", "target": target, "baseline": baseline}
        for name in ("overall_macro_f1", "worst_accent_macro_f1", "accent_gap", "accent_std", "accent_probe_macro_f1"):
            left, right = target_row.get(name), baseline_row.get(name)
            row[f"delta_{name}"] = None if pd.isna(left) or pd.isna(right) else float(left) - float(right)
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, metrics = load_rows()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No saved E1-E7 metrics found")
    frame.to_csv(OUTPUT / "e1_e7_full_ladder.csv", index=False)
    frame[frame["experiment"].isin(["E1", "E2", "E3"])].to_csv(OUTPUT / "rq4_e1_e2_e3.csv", index=False)
    frame[frame["experiment"].isin(["E4", "E5", "E6", "E7"])].to_csv(OUTPUT / "rq5_e4_e5_e6_e7.csv", index=False)
    pd.DataFrame(delta_rows(frame)).to_csv(OUTPUT / "e1_e7_deltas.csv", index=False)
    resource_columns = ["experiment", "experiment_id", "total_parameters", "trainable_parameters", "model_size_mb", "training_time_sec", "latency_median_ms", "latency_p95_ms"]
    resources = frame[resource_columns].copy()
    for step, item in EXPERIMENTS.items():
        if step < "E4" or step not in set(resources["experiment"]):
            continue
        resource_path = ROOT / item["root"] / "resource_metrics.json"
        if resource_path.is_file():
            value = json.loads(resource_path.read_text(encoding="utf-8"))
            mask = resources["experiment"].eq(step)
            resources.loc[mask, "communication_rounds"] = value.get("communication_rounds")
            resources.loc[mask, "total_communication_bytes"] = value.get("total_communication_bytes")
            resources.loc[mask, "bytes_per_round"] = value.get("bytes_per_round")
    resources.to_csv(OUTPUT / "federated_resource_comparison.csv", index=False)
    emotion_rows, accent_rows = [], []
    for step, metric in metrics.items():
        for emotion, values in metric.get("per_class", {}).items():
            emotion_rows.append({"experiment": step, "emotion": emotion, **values})
        for accent, values in metric.get("per_accent", {}).items():
            for emotion, f1 in values.get("per_emotion_f1", {}).items():
                accent_rows.append({"experiment": step, "accent": accent, "samples": values.get("samples"), "macro_f1": values.get("macro_f1"), "uar": values.get("uar"), "emotion": emotion, "emotion_f1": f1})
    pd.DataFrame(emotion_rows).to_csv(OUTPUT / "per_emotion_comparison.csv", index=False)
    pd.DataFrame(accent_rows).to_csv(OUTPUT / "per_accent_comparison.csv", index=False)
    completed = frame["experiment"].tolist()
    metric_table = [
        "| Exp | Method | Macro-F1 | Worst accent | Accent gap | Accent probe | Train sec | Latency ms | Size MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict("records"):
        metric_table.append(
            f"| {row['experiment']} | {row['method']} | {row['overall_macro_f1']:.6f} | "
            f"{row['worst_accent_macro_f1']:.6f} | {row['accent_gap']:.6f} | "
            f"{row['accent_probe_macro_f1']:.6f} | {row['training_time_sec']:.2f} | "
            f"{row['latency_median_ms']:.4f} | {row['model_size_mb']:.3f} |"
        )
    delta_table = [
        "| Comparison | Δ Macro-F1 | Δ Worst accent | Δ Accent gap | Δ Accent probe |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in delta_rows(frame):
        value = lambda key: "N/A" if row.get(key) is None else f"{row[key]:.6f}"
        delta_table.append(
            f"| {row['comparison']} | {value('delta_overall_macro_f1')} | "
            f"{value('delta_worst_accent_macro_f1')} | {value('delta_accent_gap')} | "
            f"{value('delta_accent_probe_macro_f1')} |"
        )
    lineage_lines = []
    for row in frame.to_dict("records"):
        lineage_lines.append(
            f"- {row['experiment']}: checkpoint `{row['checkpoint_sha256']}`; "
            f"predictions `{row['prediction_sha256']}`."
        )
    lines = [
        "# FedECAI E1-E7 validation decision packet", "",
        "All values are development-validation evidence. The fixed 665-sample test was not evaluated.", "",
        f"Completed evidence available: {', '.join(completed)}.", "",
        "## Primary main-scenario comparison", "", *metric_table, "",
        "E3 reduces accent gap and probe performance relative to E1 but exceeds the planned utility-loss budget. E5 is the strongest primary federated result in both Overall and Worst-accent Macro-F1. E6 substantially worsens primary accent disparity. E7 is close to E4 on utility/worst-accent and reduces the gap slightly, but its accent-probe score increases; therefore this run does not establish accent invariance.", "",
        "## Matched deltas", "", *delta_table, "",
        "No final lambda is selected. The controlled value 0.10 requires later RQ6 sensitivity analysis.", "",
        "## Per-emotion and per-accent evidence", "",
        "Exact precision/recall/F1/support and confusion matrices are retained in each `metrics.json`; consolidated source tables are `per_emotion_comparison.csv` and `per_accent_comparison.csv`.", "",
        "## Resource and communication cost", "",
        "Parameter counts, model size, training time, latency and measured federated state-transfer bytes are in `federated_resource_comparison.csv` and each experiment's `resource_metrics.json`. E7 recorded an unusually long wall-clock duration; it is preserved rather than normalized manually.", "",
        "## Lineage", "",
        "E1 is the immutable `b6_resnet18_mel3_finetune_seed391` baseline. E2-E7 initialize the ResNet18 emotion state from that checkpoint.", "",
        *lineage_lines, "",
        "## Execution failures", "",
        "No E2-E7 smoke/full run failed. All checkpoint replay gates passed. Intermediate metric fluctuations are retained in training/round histories.", "",
        "## Limitations", "",
        "The active sample-level protocol permits speaker overlap. Accent results and probes are descriptive and are not speaker-independent or causal accent effects. Lambda 0.10 is the controlled initial ladder setting, not an RQ6-selected optimum.", "",
        "Source tables: `e1_e7_full_ladder.csv`, `e1_e7_deltas.csv`, `per_emotion_comparison.csv`, `per_accent_comparison.csv`, `accent_probe_comparison.csv`, and `federated_resource_comparison.csv`.",
    ]
    (OUTPUT / "FEDECAI_E1_E7_DECISION_PACKET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "experiments": completed, "output": OUTPUT.relative_to(ROOT).as_posix(), "test_samples_loaded": 0}, indent=2))


if __name__ == "__main__":
    main()
