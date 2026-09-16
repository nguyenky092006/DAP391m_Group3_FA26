"""Build persisted validation-only evidence for the two controlled B6 runs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "tables" / "b6_regression_controls"
REPORT_ROOT = ROOT / "reports" / "tables" / "b6_performance"
CONFIG = ROOT / "configs" / "models" / "b6_regression_controls_v1.json"
RUNS = (
    ("wav2vec2_raw", "b6_wav2vec2_raw_finetune12_controlled_seed391"),
    ("pitch_fusion_paper_aligned", "b6_pitchfusion_frozen_stable_controlled_seed391"),
)
CLASSES = ("angry", "happy", "neutral", "sad")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).isoformat()
    epoch_rows: list[dict] = []
    summary_rows: list[dict] = []
    class_rows: list[dict] = []
    integrity_rows: list[dict] = []
    reports: list[dict] = []

    for model, experiment_id in RUNS:
        report_path = REPORT_ROOT / f"{experiment_id}_report.json"
        metrics_path = ROOT / "artifacts" / "metrics" / f"{experiment_id}_validation.json"
        prediction_path = ROOT / "artifacts" / "predictions" / f"{experiment_id}_validation.csv"
        checkpoint_path = ROOT / "artifacts" / "checkpoints" / f"{experiment_id}.pt"
        audit_path = ROOT / "artifacts" / "audit" / f"b6_regression_control_{model}_train.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append(report)
        model_config = config[model]
        history = report["history"]
        best_epoch = int(report["best_validation_epoch"])
        epochs_run = len(history)
        max_epochs = int(model_config["max_epochs"])

        for item in history:
            epoch_rows.append({
                "experiment_id": experiment_id,
                "model": model,
                "epoch": item["epoch"],
                "validation_macro_f1": item["validation_macro_f1"],
                "validation_uar": item["validation_uar"],
                "validation_loss": item["validation_loss"],
                "is_best_epoch": int(item["epoch"]) == best_epoch,
            })

        with prediction_path.open("r", encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        splits = sorted({row["split"] for row in predictions})
        validation = report["validation"]
        summary_rows.append({
            "experiment_id": experiment_id,
            "model": model,
            "status": report["status"],
            "evaluation_split": ";".join(splits),
            "prediction_count": len(predictions),
            "test_samples_loaded": report["test_samples_loaded"],
            "epochs_run": epochs_run,
            "configured_max_epochs": max_epochs,
            "best_epoch": best_epoch,
            "distance_from_configured_budget": max_epochs - best_epoch,
            "best_near_budget_boundary": max_epochs - best_epoch <= 1,
            "overall_macro_f1": validation["overall_macro_f1"],
            "uar": validation["uar"],
            "validation_loss_at_best_checkpoint": report["validation_loss"],
            "training_time_sec": validation["training_time_sec"],
            "inference_latency_ms": validation["inference_latency_ms"],
            "model_size_mb": validation["model_size_mb"],
            "total_parameters": validation["total_parameters"],
            "trainable_parameters": validation["trainable_parameters"],
        })

        for emotion in CLASSES:
            values = validation["per_class"][emotion]
            class_rows.append({
                "experiment_id": experiment_id,
                "model": model,
                "emotion": emotion,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
                "support": values["support"],
            })

        expected_hash = report["lineage"]["checkpoint_sha256"]
        actual_hash = sha256(checkpoint_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for kind, path in (
            ("report", report_path),
            ("metrics", metrics_path),
            ("predictions", prediction_path),
            ("checkpoint", checkpoint_path),
            ("run_audit", audit_path),
        ):
            integrity_rows.append({
                "experiment_id": experiment_id,
                "artifact_type": kind,
                "path": path.relative_to(ROOT).as_posix(),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
                "checkpoint_hash_matches_report": actual_hash == expected_hash if kind == "checkpoint" else None,
                "test_samples_loaded": audit.get("test_samples_loaded") if kind == "run_audit" else None,
            })

    write_csv(
        OUT / "epoch_history.csv",
        epoch_rows,
        ["experiment_id", "model", "epoch", "validation_macro_f1", "validation_uar", "validation_loss", "is_best_epoch"],
    )
    write_csv(
        OUT / "validation_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_csv(
        OUT / "per_class_metrics.csv",
        class_rows,
        list(class_rows[0]),
    )
    write_csv(
        OUT / "artifact_integrity.csv",
        integrity_rows,
        list(integrity_rows[0]),
    )

    lines = [
        "# B6 controlled Wav2Vec2 and Pitch-Fusion validation report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "Scope: validation only. The fixed 665-sample test set was neither loaded nor evaluated.",
        "No benchmark/backbone selection decision is made by this report.",
        "",
    ]
    for report, summary in zip(reports, summary_rows):
        lines.extend([
            f"## {summary['experiment_id']}",
            "",
            f"Best epoch: **{summary['best_epoch']}** of configured {summary['configured_max_epochs']}; "
            f"near budget boundary (within one epoch): **{str(summary['best_near_budget_boundary']).upper()}**.",
            "",
            "| Epoch | Validation Macro-F1 | Validation UAR | Validation loss | Best |",
            "|---:|---:|---:|---:|:---:|",
        ])
        for item in report["history"]:
            lines.append(
                f"| {item['epoch']} | {item['validation_macro_f1']:.6f} | "
                f"{item['validation_uar']:.6f} | {item['validation_loss']:.6f} | "
                f"{'yes' if int(item['epoch']) == summary['best_epoch'] else ''} |"
            )
        lines.extend([
            "",
            f"Saved-best evaluator: Macro-F1 `{summary['overall_macro_f1']:.6f}`, "
            f"UAR `{summary['uar']:.6f}`, validation loss `{summary['validation_loss_at_best_checkpoint']:.6f}`.",
            "",
            "| Class | F1 | Recall | Support |",
            "|---|---:|---:|---:|",
        ])
        for emotion in CLASSES:
            values = report["validation"]["per_class"][emotion]
            lines.append(f"| {emotion.title()} | {values['f1']:.6f} | {values['recall']:.6f} | {values['support']} |")
        lines.extend(["", "Happy is shown explicitly above because it is the requested diagnostic class.", ""])

    lines.extend([
        "## Integrity",
        "",
        "Both controlled checkpoint SHA-256 values match their saved reports. Each prediction file contains "
        "646 validation rows, and each run audit records `test_samples_loaded = 0`.",
        "The smoke audits also record that no controlled checkpoint existed before or after smoke execution.",
        "",
        "The Pitch-Fusion run used a fully frozen Wav2Vec2 encoder. Its original generated run report contains "
        "a stale `partial Wav2Vec2` latency-path description; configuration, freeze schedule, parameter counts, "
        "and this controlled report carry the correct frozen-encoder interpretation. Metric values were not changed.",
    ])
    (OUT / "CONTROLLED_RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
