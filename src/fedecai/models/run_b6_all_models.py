"""Run and verify all five B6 models with one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports/tables/b6"
DISPLAY_NAMES = {
    "svm_rbf": "SVM RBF",
    "xgboost": "XGBoost",
    "cnn2d_logmel": "CNN2D Log-Mel Baseline",
    "wav2vec2_embedding_mlp": "Wav2Vec2 Frozen Embedding MLP",
    "pitch_fusion_paper_aligned": "Pitch-Fusion Paper-Aligned Reproduction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain all five models")
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
        ("wav2vec2_embedding_mlp", "wav2vec2_embedding_mlp_report.json", "wav2vec2_embedding_mlp_validation_predictions.csv", False),
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


def main() -> None:
    args = parse_args()
    python = sys.executable
    classical = [python, str(ROOT / "src/fedecai/models/run_b6_classical_pipeline.py"), "--jobs", str(args.jobs)]
    deep = [python, str(ROOT / "src/fedecai/models/run_b6_deep_pipeline.py"), "--device", args.device]
    pitch = [python, str(ROOT / "src/fedecai/models/run_b6_pitch_fusion_paper.py"), "--device", args.device]
    if args.preflight_only:
        classical.append("--preflight-only")
        deep.append("--preflight-only")
        pitch.append("--preflight-only")
    if args.force:
        classical.append("--force")
        deep.append("--force")
        pitch.append("--force")
    run(classical, "B6 phase 1/3: SVM RBF + XGBoost")
    run(deep, "B6 phase 2/3: CNN2D + Wav2Vec2 MLP + fusion ablation")
    run(pitch, "B6 phase 3/3: paper-aligned Pitch-Fusion")
    if args.preflight_only:
        print("\nB6 all-model preflight: PASS", flush=True)
        return
    summary = build_summary()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "b6_all_models_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
