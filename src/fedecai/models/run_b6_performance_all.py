"""Audit, smoke-test and run the five-model performance-oriented B6 rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pandas as pd
import torch

from fedecai.models.b6_common import enforce_b6_training_lock, git_state, load_config, sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = (
    "cnn2d_logmel", "mobilenet_v3_mel3", "resnet18_mel3",
    "wav2vec2_raw", "pitch_fusion_paper_aligned",
)
PERFORMANCE_EXPERIMENT_IDS = {
    "cnn2d_logmel": "b6_cnn2d_logmel_performance_seed391",
    "mobilenet_v3_mel3": "b6_mobilenet_v3_mel3_finetune_seed391",
    "resnet18_mel3": "b6_resnet18_mel3_finetune_seed391",
    "wav2vec2_raw": "b6_wav2vec2_raw_finetune_seed391",
    "pitch_fusion_paper_aligned": "b6_pitchfusion_raw_pitch_finetune_seed391",
}
BASELINES = {
    "svm_rbf": ("b6_svm_rbf_seed391", "b6_svm_rbf_seed391.joblib", "classical_reference"),
    "xgboost": ("b6_xgboost_seed391", "b6_xgboost_seed391.joblib", "classical_reference"),
    "cnn2d_logmel": ("b6_cnn2d_logmel_seed391", "b6_cnn2d_logmel_seed391.pt", "end_to_end_random_initialization_baseline"),
    "resnet18_mel3": ("b6_resnet18_mel3_seed391", "b6_resnet18_mel3_seed391.pt", "end_to_end_random_initialization_baseline"),
    "mobilenet_v3_mel3": ("b6_mobilenet_v3_mel3_seed391", "b6_mobilenet_v3_mel3_seed391.pt", "end_to_end_random_initialization_baseline"),
    "vit_b16_mel3": ("b6_vit_b16_mel3_seed391", "b6_vit_b16_mel3_seed391.pt", "frozen_pretrained_cached_embedding_baseline"),
    "wav2vec2_raw": ("b6_wav2vec2_raw_seed391", "b6_wav2vec2_raw_seed391.pt", "frozen_encoder_cached_embedding_baseline"),
    "pitch_fusion_paper_aligned": ("b6_pitch_fusion_paper_aligned_seed391", "b6_pitch_fusion_paper_aligned_seed391.pt", "frozen_audio_encoder_trainable_fusion_baseline"),
}
BASELINE_CONFIGS = {
    "svm_rbf": "configs/models/b6_classical_centralized_v1.json",
    "xgboost": "configs/models/b6_classical_centralized_v1.json",
    "cnn2d_logmel": "configs/models/b6_deep_v1.json",
    "resnet18_mel3": "configs/models/b6_image_models_v1.json",
    "mobilenet_v3_mel3": "configs/models/b6_image_models_v1.json",
    "vit_b16_mel3": "configs/models/b6_image_models_v1.json",
    "wav2vec2_raw": "configs/models/b6_wav2vec2_raw_v1.json",
    "pitch_fusion_paper_aligned": "configs/models/b6_pitch_fusion_paper_v1.json",
}


def environment_record() -> dict[str, Any]:
    import torchvision
    import transformers
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "environment_name": "DAP301m",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "amp_available": hasattr(torch, "autocast") and hasattr(torch.amp, "GradScaler"),
        "platform": platform.platform(),
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream); fields = list(reader.fieldnames or []); rows = list(reader)
    rows.append({field: row.get(field, "") for field in fields})
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def merge_audit_results(path: Path, new_results: list[dict[str, Any]], *, smoke: bool) -> dict[str, Any]:
    """Merge independently executed model runs without discarding earlier evidence."""
    previous: dict[str, Any] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        previous = {row["model"]: row for row in payload.get("results", [])}
    previous.update({row["model"]: row for row in new_results})
    if not smoke:
        for model, experiment_id in PERFORMANCE_EXPERIMENT_IDS.items():
            if model in previous:
                continue
            metrics = ROOT / "artifacts/metrics" / f"{experiment_id}_validation.json"
            predictions = ROOT / "artifacts/predictions" / f"{experiment_id}_validation.csv"
            checkpoint = ROOT / "artifacts/checkpoints" / f"{experiment_id}.pt"
            matching_runs = sorted((ROOT / "logs/runs").glob(f"*_{experiment_id}_train"))
            completed_runs = [
                run for run in matching_runs
                if (run / "run_summary.md").is_file()
                and "Status: `COMPLETED`" in (run / "run_summary.md").read_text(encoding="utf-8")
            ]
            if all(item.is_file() for item in (metrics, predictions, checkpoint)) and completed_runs:
                previous[model] = {
                    "model": model,
                    "experiment_id": experiment_id,
                    "phase": "train",
                    "status": "COMPLETED",
                    "exit_code": 0,
                    "run_dir": str(completed_runs[-1].relative_to(ROOT)).replace("\\", "/"),
                    "reconciled_from_persisted_evidence": True,
                }
    ordered = [previous[model] for model in MODEL_ORDER if model in previous]
    complete = len(ordered) == len(MODEL_ORDER) and not any(row["exit_code"] for row in ordered)
    status = "PASS" if smoke and complete else "COMPLETE" if complete else "BLOCKED" if smoke else "PARTIAL_FAILURE"
    payload = {"status": status, "results": ordered}
    if smoke:
        payload["evidence_class"] = "SMOKE_TEST_ONLY"
    write_json_atomic(payload, path)
    return payload


def audit(config_path: Path) -> dict[str, Any]:
    output = ROOT / "reports/tables/b6_performance"
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    manifest_path = ROOT / "data/manifests/emotion_stratified_v1.csv"
    frame = pd.read_csv(manifest_path, dtype={"sample_id": "string", "speaker_id": "string"})
    frame = frame[frame["split"].isin(("train", "validation", "test"))].copy()
    groups = {split: set(frame.loc[frame["split"].eq(split), "speaker_id"].astype(str)) for split in ("train", "validation", "test")}
    speaker = {
        "status": "PASS_WITH_DISCLOSED_SPEAKER_OVERLAP",
        "evaluation_protocol": config["evaluation_protocol"],
        "split_version": config["split_version"],
        "unique_train_speakers": len(groups["train"]),
        "unique_validation_speakers": len(groups["validation"]),
        "unique_test_speakers": len(groups["test"]),
        "train_validation_overlap": len(groups["train"] & groups["validation"]),
        "train_test_overlap": len(groups["train"] & groups["test"]),
        "validation_test_overlap": len(groups["validation"] & groups["test"]),
        "speaker_disjoint": False,
        "interpretation": "Performance-oriented upper-bound comparison; do not claim unseen-speaker generalization.",
        "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json_atomic(speaker, output / "speaker_overlap_audit.json")

    frozen_rows, failures = [], []
    for model, (stem, checkpoint_name, regime) in BASELINES.items():
        metrics = ROOT / "artifacts/metrics" / f"{stem}_validation.json"
        prediction = ROOT / "artifacts/predictions" / f"{stem}_validation.csv"
        checkpoint = ROOT / "artifacts/checkpoints" / checkpoint_name
        baseline_config = ROOT / BASELINE_CONFIGS[model]
        exists = all(path.is_file() for path in (metrics, prediction, checkpoint, baseline_config))
        row_count = sum(1 for _ in prediction.open(encoding="utf-8")) - 1 if prediction.is_file() else None
        metric = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
        valid = exists and row_count == 646 and metric.get("split_version") == config["split_version"] and metric.get("seed") == config["seed"]
        row = {
            "model": model, "training_regime": regime, "status": "PASS" if valid else "FAIL",
            "comparison_class": "existing_frozen_or_lightweight_baseline",
            "config_path": str(baseline_config.relative_to(ROOT)).replace("\\", "/"),
            "metrics_path": str(metrics.relative_to(ROOT)).replace("\\", "/"),
            "prediction_path": str(prediction.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "prediction_rows": row_count, "split_version": metric.get("split_version"),
            "seed": metric.get("seed"), "overall_macro_f1": metric.get("overall_macro_f1"),
            "metrics_sha256": sha256_file(metrics) if metrics.is_file() else None,
            "prediction_sha256": sha256_file(prediction) if prediction.is_file() else None,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            "config_sha256": sha256_file(baseline_config) if baseline_config.is_file() else None,
        }
        frozen_rows.append(row)
        if not valid:
            failures.append(model)
    frozen_path = output / "frozen_baseline_audit.csv"
    with frozen_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frozen_rows[0])); writer.writeheader(); writer.writerows(frozen_rows)

    execution_plan = {
        "status": "READY" if not failures and torch.cuda.is_available() else "BLOCKED",
        "models_to_run": list(MODEL_ORDER),
        "models_reused": ["svm_rbf", "xgboost"],
        "vit_performance_rerun_status": "SKIPPED_BY_HUMAN_TIME_CONSTRAINT",
        "vit_source_config_checkpoint_and_results_preserved": True,
        "model_order": list(MODEL_ORDER),
        "selection_metric": "validation_macro_f1",
        "test_used_for_selection": False,
        "environment": environment_record(),
        "frozen_baseline_failures": failures,
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
    }
    write_json_atomic(execution_plan, output / "execution_plan.json")
    write_json_atomic({
        "centralized_performance_rerun_preflight": execution_plan["status"],
        "speaker_audit": speaker, "frozen_baselines_valid": not failures,
        "vit_performance_rerun_status": execution_plan["vit_performance_rerun_status"],
    }, ROOT / "artifacts/audit/b6_performance_preflight.json")
    return execution_plan


def command_for(model: str, config_path: Path, smoke: bool, force: bool) -> list[str]:
    common = ["--config", str(config_path), "--device", "cuda"]
    if model in ("cnn2d_logmel", "resnet18_mel3", "mobilenet_v3_mel3"):
        command = [sys.executable, str(ROOT / "src/fedecai/models/run_b6_performance_image.py"), "--model", model, *common]
    elif model == "wav2vec2_raw":
        command = [sys.executable, str(ROOT / "src/fedecai/models/run_b6_wav2vec2_finetune.py"), *common]
    else:
        command = [sys.executable, str(ROOT / "src/fedecai/models/run_b6_pitch_fusion_finetune.py"), *common]
    if smoke:
        command.append("--smoke-test")
        if model in ("wav2vec2_raw", "pitch_fusion_paper_aligned"):
            command.extend(("--max-audio-samples", "160000"))
    if force and not smoke:
        command.append("--force")
    return command


def run_one(command: list[str], model: str, experiment_id: str, phase: str, config_path: Path) -> dict[str, Any]:
    started = datetime.now().astimezone()
    run_id = started.strftime(f"%Y%m%d_%H%M%S_{experiment_id}_{phase}")
    run_dir = ROOT / "logs/runs" / run_id; run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "commands.log").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment_record(), indent=2) + "\n", encoding="utf-8")
    child_environment = os.environ.copy(); child_environment["PYTHONPATH"] = str(ROOT / "src")
    child_environment["TORCH_HOME"] = str(ROOT / "data/interim/models/torch")
    stdout_path, stderr_path = run_dir / "stdout.log", run_dir / "stderr.log"
    before = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            stdout.write(line); stdout.flush(); print(line, end="", flush=True)
        return_code = process.wait()
    elapsed = time.perf_counter() - before
    status = "SMOKE_TEST_ONLY" if phase == "smoke" and return_code == 0 else "COMPLETED" if return_code == 0 else "FAILED"
    metrics = ROOT / "artifacts/metrics" / f"{experiment_id}_validation.json"
    checkpoint = ROOT / "artifacts/checkpoints" / f"{experiment_id}.pt"
    prediction = ROOT / "artifacts/predictions" / f"{experiment_id}_validation.csv"
    summary = f"""# Run Summary

Run ID: `{run_id}`  
Experiment ID: `{experiment_id}`  
Phase: `{phase}`  
Status: `{status}`  
Exit code: `{return_code}`  
Start: `{started.isoformat()}`  
Elapsed seconds: `{elapsed:.3f}`  
Evaluation protocol: `emotion_stratified_with_speaker_overlap`  
Fixed test used for selection: `NO`
"""
    (run_dir / "run_summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        metrics.read_text(encoding="utf-8") if return_code == 0 and phase == "train" and metrics.is_file()
        else json.dumps({"experiment_id": experiment_id, "phase": phase, "status": status}, indent=2) + "\n",
        encoding="utf-8",
    )
    append_csv(ROOT / "logs/run_index.csv", {
        "run_id": run_id, "experiment_id": experiment_id, "timestamp": started.isoformat(),
        "step": "B6", "model": model, "variant": f"performance_{phase}", "seed": 391,
        "status": status, "dataset_version": "visec_clean_eligible_v1",
        "split_version": "emotion_stratified_v1", "git_commit": git_state(ROOT)["git_commit"],
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "stdout_path": str(stdout_path.relative_to(ROOT)).replace("\\", "/"),
        "stderr_path": str(stderr_path.relative_to(ROOT)).replace("\\", "/"),
        "metrics_path": str(metrics.relative_to(ROOT)).replace("\\", "/") if metrics.is_file() and phase == "train" else str((run_dir / "metrics.json").relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/") if checkpoint.is_file() and phase == "train" else "",
        "prediction_path": str(prediction.relative_to(ROOT)).replace("\\", "/") if prediction.is_file() and phase == "train" else "",
        "training_time_sec": elapsed,
        "notes": "Performance-oriented upper-bound comparison; speaker overlap disclosed; test untouched.",
    })
    return {"model": model, "experiment_id": experiment_id, "phase": phase, "status": status, "exit_code": return_code, "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "smoke", "train", "all"), default="all")
    parser.add_argument("--config", default="configs/models/b6_performance_v1.json")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_b6_training_lock(ROOT, "performance-oriented B6 orchestration")
    if "DAP301m" not in str(Path(sys.executable).resolve()):
        raise RuntimeError(f"Performance rerun requires DAP301m: {sys.executable}")
    config_path = Path(args.config).resolve() if Path(args.config).is_absolute() else ROOT / args.config
    config = load_config(config_path)
    plan = audit(config_path)
    if (ROOT / "artifacts/audit/b6_performance_training.json").is_file():
        merge_audit_results(ROOT / "artifacts/audit/b6_performance_training.json", [], smoke=False)
    print(json.dumps(plan, indent=2), flush=True)
    if plan["status"] != "READY" or args.phase == "audit":
        if plan["status"] != "READY":
            raise RuntimeError("Performance rerun preflight BLOCKED")
        return
    selected = args.models or list(MODEL_ORDER)
    results = []
    if args.phase in ("smoke", "all"):
        for model in selected:
            experiment_id = config[model]["experiment_id"]
            results.append(run_one(command_for(model, config_path, True, False), model, experiment_id, "smoke", config_path))
        if any(row["exit_code"] for row in results):
            merge_audit_results(ROOT / "artifacts/audit/b6_performance_smoke.json", results, smoke=True)
            raise RuntimeError("At least one performance smoke test failed; full training not started")
        merge_audit_results(ROOT / "artifacts/audit/b6_performance_smoke.json", results, smoke=True)
        if args.phase == "smoke":
            return
    if args.phase in ("train", "all"):
        train_results = []
        for model in selected:
            experiment_id = config[model]["experiment_id"]
            train_results.append(run_one(command_for(model, config_path, False, args.force), model, experiment_id, "train", config_path))
        payload = merge_audit_results(ROOT / "artifacts/audit/b6_performance_training.json", train_results, smoke=False)
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
