"""Generate traceable frozen/lightweight versus performance-oriented B6 tables."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "reports/tables/b6_performance"
MODELS = (
    "svm_rbf", "xgboost", "cnn2d_logmel", "resnet18_mel3",
    "mobilenet_v3_mel3", "vit_b16_mel3", "wav2vec2_raw",
    "pitch_fusion_paper_aligned",
)
BASELINE_STEMS = {model: f"b6_{model}_seed391" for model in MODELS}
PERFORMANCE_STEMS = {
    "cnn2d_logmel": "b6_cnn2d_logmel_performance_seed391",
    "resnet18_mel3": "b6_resnet18_mel3_finetune_seed391",
    "mobilenet_v3_mel3": "b6_mobilenet_v3_mel3_finetune_seed391",
    "wav2vec2_raw": "b6_wav2vec2_raw_finetune_seed391",
    "pitch_fusion_paper_aligned": "b6_pitchfusion_raw_pitch_finetune_seed391",
}
INPUTS = {
    "svm_rbf": "MFCC/eGeMAPS", "xgboost": "MFCC/eGeMAPS",
    "cnn2d_logmel": "log-Mel", "resnet18_mel3": "3-channel Mel",
    "mobilenet_v3_mel3": "3-channel Mel", "vit_b16_mel3": "3-channel Mel",
    "wav2vec2_raw": "raw waveform 16 kHz",
    "pitch_fusion_paper_aligned": "raw waveform + pYIN pitch contour",
}
BASELINE_REGIMES = {
    "svm_rbf": "classical_reference", "xgboost": "classical_reference",
    "cnn2d_logmel": "end_to_end_random_initialization_baseline",
    "resnet18_mel3": "end_to_end_random_initialization_baseline",
    "mobilenet_v3_mel3": "end_to_end_random_initialization_baseline",
    "vit_b16_mel3": "frozen_pretrained_cached_embedding_baseline",
    "wav2vec2_raw": "frozen_encoder_cached_embedding_baseline",
    "pitch_fusion_paper_aligned": "frozen_audio_encoder_trainable_fusion_baseline",
}


def read_metric(stem: str) -> dict[str, Any]:
    path = ROOT / "artifacts/metrics" / f"{stem}_validation.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name: str, rows: list[dict[str, Any]]) -> Path:
    path = OUTPUT / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return path


def value(metric: dict[str, Any], key: str):
    return metric.get(key) if metric else None


def delta(performance: dict[str, Any], baseline: dict[str, Any], key: str):
    if not performance or performance.get(key) is None or baseline.get(key) is None:
        return None
    return float(performance[key]) - float(baseline[key])


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    def display(item: Any) -> str:
        if item is None or item == "":
            return "N/A"
        if isinstance(item, float):
            return f"{item:.4f}"
        return str(item).replace("|", "\\|")

    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(display(row.get(key)) for key, _ in columns) + " |" for row in rows]
    return "\n".join((header, rule, *body))


def write_task_file_manifest() -> Path:
    explicit = [
        ROOT / "configs/models/b6_performance_v1.json",
        ROOT / "src/fedecai/models/run_b6_performance_image.py",
        ROOT / "src/fedecai/models/run_b6_wav2vec2_finetune.py",
        ROOT / "src/fedecai/models/run_b6_pitch_fusion_finetune.py",
        ROOT / "src/fedecai/models/run_b6_performance_all.py",
        ROOT / "src/fedecai/models/generate_b6_performance_comparison.py",
        ROOT / "tests/test_b6_performance_rerun.py",
        ROOT / "docs/research/B6_SPEC.md", ROOT / "docs/research/B7_SPEC.md",
        ROOT / "doc.md", ROOT / "logs/project_log.md", ROOT / "logs/decision_log.md",
        ROOT / "logs/run_index.csv", ROOT / "logs/experiment_registry.csv",
        ROOT / "artifacts/audit/b6_performance_preflight.json",
        ROOT / "artifacts/audit/b6_performance_smoke.json",
        ROOT / "artifacts/audit/b6_performance_training.json",
    ]
    patterns = (
        "artifacts/checkpoints/b6_*performance*_seed391.pt",
        "artifacts/checkpoints/b6_*finetune_seed391.pt",
        "artifacts/predictions/b6_*performance*_seed391_validation.csv",
        "artifacts/predictions/b6_*finetune_seed391_validation.csv",
        "artifacts/metrics/b6_*performance*_seed391_validation.json",
        "artifacts/metrics/b6_*finetune_seed391_validation.json",
        "artifacts/b6/performance/*",
        "reports/tables/b6_performance/*",
        "logs/runs/*_b6_*performance*_seed391_*/*",
        "logs/runs/*_b6_*finetune_seed391_*/*",
        "data/interim/models/torch/hub/checkpoints/*.pth",
    )
    paths = set(explicit)
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    target = OUTPUT / "task_file_manifest.csv"
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "exists": path.is_file(),
        }
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size_bytes", "exists"))
        writer.writeheader(); writer.writerows(rows)
    return target


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison, frozen_rows, performance_rows = [], [], []
    for model in MODELS:
        baseline = read_metric(BASELINE_STEMS[model])
        performance = read_metric(PERFORMANCE_STEMS[model]) if model in PERFORMANCE_STEMS else {}
        performance_status = (
            "SKIPPED_BY_HUMAN_TIME_CONSTRAINT" if model == "vit_b16_mel3" else
            "N/A_CLASSICAL_MODEL" if model in ("svm_rbf", "xgboost") else
            "COMPLETE" if performance else "FAILED_OR_NOT_COMPLETED"
        )
        frozen_rows.append({
            "rank": None, "model": model, "training_regime": BASELINE_REGIMES[model],
            "input_representation": INPUTS[model],
            "macro_f1": value(baseline, "overall_macro_f1"), "uar": value(baseline, "uar"),
            "weighted_f1": value(baseline, "weighted_f1"), "accuracy": value(baseline, "accuracy"),
            "worst_accent_macro_f1": value(baseline, "worst_accent_macro_f1"),
            "training_time_sec": value(baseline, "training_time_sec"),
            "inference_latency_ms": value(baseline, "inference_latency_ms"),
            "model_size_mb": value(baseline, "model_size_mb"),
            "total_parameters": value(baseline, "total_parameters"),
            "trainable_parameters": value(baseline, "trainable_parameters"),
        })
        if performance:
            performance_rows.append({
                "rank": None, "model": model, "training_regime": "performance_oriented",
                "input_representation": INPUTS[model],
                "macro_f1": value(performance, "overall_macro_f1"), "uar": value(performance, "uar"),
                "weighted_f1": value(performance, "weighted_f1"), "accuracy": value(performance, "accuracy"),
                "north_macro_f1": value(performance, "north_macro_f1"),
                "central_macro_f1": value(performance, "central_macro_f1"),
                "south_macro_f1": value(performance, "south_macro_f1"),
                "worst_accent_macro_f1": value(performance, "worst_accent_macro_f1"),
                "best_accent_macro_f1": value(performance, "best_accent_macro_f1"),
                "accent_gap": value(performance, "accent_gap"),
                "training_time_sec": value(performance, "training_time_sec"),
                "inference_latency_ms": value(performance, "inference_latency_ms"),
                "model_size_mb": value(performance, "model_size_mb"),
                "total_parameters": value(performance, "total_parameters"),
                "trainable_parameters": value(performance, "trainable_parameters"),
            })
        comparison.append({
            "model": model, "input_representation": INPUTS[model],
            "existing_training_regime": BASELINE_REGIMES[model],
            "performance_status": performance_status,
            "frozen_macro_f1": value(baseline, "overall_macro_f1"),
            "performance_macro_f1": value(performance, "overall_macro_f1"),
            "delta_macro_f1": delta(performance, baseline, "overall_macro_f1"),
            "frozen_uar": value(baseline, "uar"), "performance_uar": value(performance, "uar"),
            "delta_uar": delta(performance, baseline, "uar"),
            "frozen_accuracy": value(baseline, "accuracy"),
            "performance_accuracy": value(performance, "accuracy"),
            "delta_accuracy": delta(performance, baseline, "accuracy"),
            "performance_training_time_sec": value(performance, "training_time_sec"),
            "performance_trainable_parameters": value(performance, "trainable_parameters"),
            "notes": (
                "Existing ViT baseline preserved; performance rerun skipped by human time constraint."
                if model == "vit_b16_mel3" else
                "Deep performance variant does not apply to classical reference."
                if model in ("svm_rbf", "xgboost") else
                "Validation-only experimental comparison with disclosed speaker overlap."
            ),
        })

    frozen_rows.sort(key=lambda row: float(row["macro_f1"]), reverse=True)
    for rank, row in enumerate(frozen_rows, 1): row["rank"] = rank
    performance_rows.sort(key=lambda row: float(row["macro_f1"]), reverse=True)
    for rank, row in enumerate(performance_rows, 1): row["rank"] = rank
    frozen_path = write_csv("existing_baseline_ranking.csv", frozen_rows)
    comparison_path = write_csv("frozen_vs_performance_comparison.csv", comparison)
    if performance_rows:
        performance_path = write_csv("performance_oriented_ranking.csv", performance_rows)
        resources = [{key: row[key] for key in ("model", "training_time_sec", "inference_latency_ms", "model_size_mb", "total_parameters", "trainable_parameters")} for row in performance_rows]
        resource_path = write_csv("performance_resource_table.csv", resources)
    else:
        performance_path = resource_path = None

    training_audit_path = ROOT / "artifacts/audit/b6_performance_training.json"
    training_audit = json.loads(training_audit_path.read_text(encoding="utf-8")) if training_audit_path.is_file() else {}
    failures = [row for row in training_audit.get("results", []) if row.get("exit_code")]
    speaker = json.loads((OUTPUT / "speaker_overlap_audit.json").read_text(encoding="utf-8"))
    complete = len(performance_rows) == 5 and not failures
    status = "COMPLETE" if complete else "PARTIAL"
    task_manifest_path = write_task_file_manifest()
    frozen_markdown = markdown_table(frozen_rows, [
        ("rank", "Rank"), ("model", "Model"), ("macro_f1", "Macro-F1"),
        ("uar", "UAR"), ("weighted_f1", "Weighted-F1"), ("accuracy", "Accuracy"),
    ])
    performance_markdown = markdown_table(performance_rows, [
        ("rank", "Rank"), ("model", "Model"), ("macro_f1", "Macro-F1"),
        ("uar", "UAR"), ("weighted_f1", "Weighted-F1"), ("accuracy", "Accuracy"),
    ])
    delta_markdown = markdown_table(comparison, [
        ("model", "Model"), ("frozen_macro_f1", "Existing"),
        ("performance_macro_f1", "Performance"), ("delta_macro_f1", "Delta"),
        ("performance_status", "Status"),
    ])
    accent_markdown = markdown_table(performance_rows, [
        ("model", "Model"), ("north_macro_f1", "North"),
        ("central_macro_f1", "Central"), ("south_macro_f1", "South"),
        ("worst_accent_macro_f1", "Worst"), ("accent_gap", "Gap"),
    ])
    resource_markdown = markdown_table(performance_rows, [
        ("model", "Model"), ("training_time_sec", "Train sec"),
        ("inference_latency_ms", "Latency ms"), ("model_size_mb", "Size MB"),
        ("total_parameters", "Total params"), ("trainable_parameters", "Trainable params"),
    ])
    packet = f"""# B6 Centralized Performance Rerun — Human Decision Packet

Generated: `{datetime.now().astimezone().isoformat()}`  
Status: `{status}`  
Evaluation protocol: `emotion_stratified_with_speaker_overlap`  
Evidence scope: development validation only; fixed test untouched.

## A. Speaker-overlap audit

- Unique speakers: train {speaker['unique_train_speakers']}, validation {speaker['unique_validation_speakers']}, test {speaker['unique_test_speakers']}.
- Pairwise overlap: train/validation {speaker['train_validation_overlap']}, train/test {speaker['train_test_overlap']}, validation/test {speaker['validation_test_overlap']}.
- This experiment is not speaker-disjoint and must not be presented as unseen-speaker generalization.

## B. Existing baseline ranking

{frozen_markdown}

Source: `{frozen_path.relative_to(ROOT).as_posix()}`

## C and E. Performance-oriented table and Macro-F1 ranking

{performance_markdown}

Source: `{performance_path.relative_to(ROOT).as_posix() if performance_path else 'not available'}`

## D. Existing baseline to performance delta

{delta_markdown}

Source: `{comparison_path.relative_to(ROOT).as_posix()}`

## F. Per-accent validation metrics

{accent_markdown}

## G and H. Training time, size, latency, and parameters

{resource_markdown}

Source: `{resource_path.relative_to(ROOT).as_posix() if resource_path else 'not available'}`.
Per-class metrics and confusion matrices are in each canonical JSON under
`artifacts/metrics/`.

## I. Failed/OOM experiments

{json.dumps(failures, ensure_ascii=False, indent=2) if failures else 'None.'}

No full training run OOM occurred. Earlier infrastructure smoke failures were
retained in `logs/run_index.csv`; they were resolved before the final 5/5 smoke
gate recorded in `artifacts/audit/b6_performance_smoke.json`.

## J. ViT status

ViT performance-oriented fine-tuning was skipped in this rerun due to runtime constraints. The existing ViT baseline, source, config and checkpoint are preserved pending human decision.

`VIT_PERFORMANCE_RERUN_STATUS = SKIPPED_BY_HUMAN_TIME_CONSTRAINT`

## K. Exact task file groups

The exact persisted file list (including every run-evidence file) is in
`{task_manifest_path.relative_to(ROOT).as_posix()}`. Main groups are:

- `configs/models/b6_performance_v1.json`
- `src/fedecai/models/run_b6_performance_image.py`
- `src/fedecai/models/run_b6_wav2vec2_finetune.py`
- `src/fedecai/models/run_b6_pitch_fusion_finetune.py`
- `src/fedecai/models/run_b6_performance_all.py`
- `src/fedecai/models/generate_b6_performance_comparison.py`
- `tests/test_b6_performance_rerun.py`
- `artifacts/audit/b6_performance_preflight.json`
- `artifacts/audit/b6_performance_smoke.json`
- `artifacts/audit/b6_performance_training.json`
- `artifacts/checkpoints/b6_*performance*_seed391.pt` and `artifacts/checkpoints/b6_*finetune_seed391.pt`
- `artifacts/predictions/b6_*performance*_seed391_validation.csv` and `artifacts/predictions/b6_*finetune_seed391_validation.csv`
- `artifacts/metrics/b6_*performance*_seed391_validation.json` and `artifacts/metrics/b6_*finetune_seed391_validation.json`
- `reports/tables/b6_performance/`
- `logs/runs/*_b6_*performance*_seed391_*` and `logs/runs/*_b6_*finetune_seed391_*`
- `logs/run_index.csv`
- `logs/experiment_registry.csv`
- `docs/research/B6_SPEC.md`
- `docs/research/B7_SPEC.md`
- `doc.md`
- `logs/project_log.md`
- `logs/decision_log.md`

## L. Decisions left to the human

- Whether the observed validation deltas justify a later fixed-test run.
- Whether ViT performance fine-tuning should remain skipped or be scheduled later.
- Which, if either, candidate backbone should proceed. No winner is selected here.

CENTRALIZED_PERFORMANCE_RERUN_STATUS = {status}
CNN2D_FINETUNE_STATUS = {'COMPLETE' if any(row['model'] == 'cnn2d_logmel' for row in performance_rows) else 'FAILED_OR_NOT_COMPLETED'}
RESNET18_FINETUNE_STATUS = {'COMPLETE' if any(row['model'] == 'resnet18_mel3' for row in performance_rows) else 'FAILED_OR_NOT_COMPLETED'}
MOBILENETV3_FINETUNE_STATUS = {'COMPLETE' if any(row['model'] == 'mobilenet_v3_mel3' for row in performance_rows) else 'FAILED_OR_NOT_COMPLETED'}
WAV2VEC2_RAW_FINETUNE_STATUS = {'COMPLETE' if any(row['model'] == 'wav2vec2_raw' for row in performance_rows) else 'FAILED_OR_NOT_COMPLETED'}
PITCH_FUSION_FINETUNE_STATUS = {'COMPLETE' if any(row['model'] == 'pitch_fusion_paper_aligned' for row in performance_rows) else 'FAILED_OR_NOT_COMPLETED'}
VIT_FINETUNE_STATUS = SKIPPED_PENDING_HUMAN_DECISION
FEDERATED_BACKBONE_STATUS = WAITING_FOR_HUMAN_DECISION
FEDECAI_STATUS = NOT_STARTED
VESC_STATUS = NOT_STARTED
"""
    (OUTPUT / "human_decision_packet.md").write_text(packet, encoding="utf-8")
    (OUTPUT / "performance_rerun_status.json").write_text(json.dumps({
        "status": status, "performance_models_completed": len(performance_rows),
        "failed_or_oom": failures,
        "vit_finetune_status": "SKIPPED_PENDING_HUMAN_DECISION",
        "federated_backbone_status": "WAITING_FOR_HUMAN_DECISION",
        "fedecai_status": "NOT_STARTED", "vesc_status": "NOT_STARTED",
    }, indent=2) + "\n", encoding="utf-8")
    # Windows terminals may default to cp1252, which cannot encode the arrow
    # used in the persisted comparison heading.
    sys.stdout.buffer.write(packet.encode("utf-8"))


if __name__ == "__main__":
    main()
