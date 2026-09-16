"""Run and verify the required eight-model B6 development benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fedecai.models.b6_common import enforce_b6_training_lock

REPORT_DIR = ROOT / "reports/tables/b6"
DISPLAY_NAMES = {
    "svm_rbf": "SVM RBF",
    "xgboost": "XGBoost",
    "cnn2d_logmel": "CNN2D Log-Mel Baseline",
    "resnet18_mel3": "ResNet-18 3-channel Mel-Spectrogram",
    "mobilenet_v3_mel3": "MobileNetV3-Small 3-channel Mel-Spectrogram",
    "vit_b16_mel3": "ViT-B/16 3-channel Mel-Spectrogram",
    "wav2vec2_raw": "Wav2Vec2 Raw Waveform (Frozen Encoder)",
    "pitch_fusion_paper_aligned": "Pitch-Fusion Paper-Aligned Reproduction",
}

METRIC_MODELS = tuple(DISPLAY_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Validate existing eight-model development artifacts and rebuild the summary without running trainers",
    )
    parser.add_argument("--force", action="store_true", help="Retrain all eight models")
    return parser.parse_args()


def run(command: list[str], label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_report(filename: str) -> dict[str, Any]:
    path = REPORT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing expected B6 report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary() -> dict[str, Any]:
    specifications = (
        ("svm_rbf", "svm_rbf_validation_report.json", "svm_rbf_validation_predictions.csv", True),
        ("xgboost", "xgboost_validation_report.json", "xgboost_validation_predictions.csv", True),
        ("cnn2d_logmel", "cnn2d_logmel_report.json", "cnn2d_logmel_validation_predictions.csv", False),
        ("resnet18_mel3", "resnet18_mel3_report.json", "resnet18_mel3_validation_predictions.csv", False),
        ("mobilenet_v3_mel3", "mobilenet_v3_mel3_report.json", "mobilenet_v3_mel3_validation_predictions.csv", False),
        ("vit_b16_mel3", "vit_b16_mel3_report.json", "vit_b16_mel3_validation_predictions.csv", False),
        ("wav2vec2_raw", "wav2vec2_raw_report.json", "wav2vec2_raw_validation_predictions.csv", False),
        ("pitch_fusion_paper_aligned", "pitch_fusion_paper_aligned_report.json", "pitch_fusion_paper_aligned_validation_predictions.csv", False),
    )
    results = []
    for model, report_name, prediction_name, classical in specifications:
        report = read_report(report_name)
        prediction_path = REPORT_DIR / prediction_name
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing validation predictions: {prediction_path}")
        with prediction_path.open(encoding="utf-8") as stream:
            prediction_rows = sum(1 for _ in stream) - 1
        if prediction_rows != 646:
            raise ValueError(f"{model} has {prediction_rows} validation predictions, expected 646")
        if classical:
            metrics = report["validation_metrics"]
            cv_macro_f1 = report["best_cv_macro_f1"]
            test_untouched = int(report["test_samples_reserved"]) == 665
        else:
            metrics = report["validation"]
            cv_macro_f1 = report.get("selection", {}).get("mean_cv_macro_f1")
            test_untouched = int(report["test_samples_loaded"]) == 0
        results.append({
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "cv_macro_f1": None if cv_macro_f1 is None else float(cv_macro_f1),
            "validation_accuracy": float(metrics["overall"]["accuracy"]),
            "validation_macro_f1": float(metrics["overall"]["macro_f1"]),
            "validation_uar": float(metrics["overall"]["uar"]),
            "validation_prediction_rows": prediction_rows,
            "test_untouched": test_untouched,
        })
    if not all(row["test_untouched"] for row in results):
        raise RuntimeError("A B6 report violates the reserved-test contract")
    return {
        "schema_version": "b6.all_models.v1",
        "status": "PASS",
        "models_completed": len(results),
        "test_untouched": True,
        "results": sorted(results, key=lambda row: row["validation_macro_f1"], reverse=True),
    }


def write_validation_metrics_table() -> Path:
    """Publish a reader-friendly table derived only from canonical metrics JSON."""
    fields = [
        "model", "overall_macro_f1", "uar", "weighted_f1", "accuracy",
        "north_macro_f1", "central_macro_f1", "south_macro_f1",
        "worst_accent_macro_f1", "best_accent_macro_f1", "accent_gap",
        "accent_std", "inference_latency_ms", "model_size_mb",
        "total_parameters", "trainable_parameters", "training_time_sec",
        "feature_extraction_time_sec",
    ]
    rows = []
    for model in METRIC_MODELS:
        source = ROOT / "artifacts/metrics" / f"b6_{model}_seed391_validation.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing canonical validation metrics: {source}")
        metrics = json.loads(source.read_text(encoding="utf-8"))
        rows.append({"model": model, **{field: metrics.get(field) for field in fields[1:]}})
    destination = ROOT / "artifacts/metrics/b6_centralized_validation_summary.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def main() -> None:
    args = parse_args()
    enforce_b6_training_lock(ROOT, "eight-model B6 orchestration")
    dap301m_python = (ROOT / "DAP301m/Scripts/python.exe").resolve()
    if not dap301m_python.is_file():
        raise FileNotFoundError(f"Required DAP301m interpreter is absent: {dap301m_python}")
    if args.summary_only:
        summary = build_summary()
        metrics_table = write_validation_metrics_table()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "b6_all_models_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({**summary, "metrics_table": str(metrics_table.relative_to(ROOT)).replace("\\", "/")}, ensure_ascii=False, indent=2), flush=True)
        return
    python = str(dap301m_python)
    classical = [
        python, str(ROOT / "src/fedecai/models/run_b6_classical_pipeline.py"),
        "--config", "configs/models/b6_classical_centralized_v1.json",
        "--jobs", str(args.jobs),
    ]
    deep = [
        python, str(ROOT / "src/fedecai/models/run_b6_deep_pipeline.py"),
        "--device", args.device, "--models", "cnn2d_logmel",
    ]
    images = [
        python, str(ROOT / "src/fedecai/models/run_b6_deep_pipeline.py"),
        "--config", "configs/models/b6_image_models_v1.json", "--device", args.device,
        "--models", "resnet18_mel3", "mobilenet_v3_mel3",
    ]
    vit = [
        python, str(ROOT / "src/fedecai/models/run_b6_vit_cached.py"),
        "--device", args.device,
    ]
    pitch = [python, str(ROOT / "src/fedecai/models/run_b6_pitch_fusion_paper.py"), "--device", args.device]
    wav2vec2 = [
        python, str(ROOT / "src/fedecai/models/run_b6_wav2vec2_raw.py"),
        "--device", args.device,
    ]
    if args.preflight_only:
        classical.append("--preflight-only")
        deep.append("--preflight-only")
        images.append("--preflight-only")
        vit = [
            python, str(ROOT / "src/fedecai/models/run_b6_vit_cached.py"),
            "--device", args.device, "--smoke-test",
        ]
        wav2vec2.append("--preflight-only")
        pitch.append("--preflight-only")
    if args.force:
        classical.append("--force")
        deep.append("--force")
        images.append("--force")
        pitch.append("--force")
        vit.append("--force-head")
        wav2vec2.append("--force-head")
    run(classical, "B6 phase 1/6: SVM RBF + XGBoost")
    run(deep, "B6 phase 2/6: CNN2D log-Mel")
    run(images, "B6 phase 3/6: ResNet-18 + MobileNetV3 on shared Mel")
    run(vit, "B6 phase 4/6: frozen pretrained ViT + cached embeddings")
    run(wav2vec2, "B6 phase 5/6: raw-waveform frozen Wav2Vec2 + cached embeddings")
    run(pitch, "B6 phase 6/6: paper-aligned Pitch-Fusion")
    if args.preflight_only:
        print("\nB6 all-model preflight: PASS", flush=True)
        return
    summary = build_summary()
    write_validation_metrics_table()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "b6_all_models_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
