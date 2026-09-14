"""Blocking data/protocol preflight for the eight-model centralized benchmark."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
REQUIRED_MODELS = {
    "svm_rbf": "mfcc_egemaps",
    "xgboost": "mfcc_egemaps",
    "cnn2d_logmel": "log_mel",
    "resnet18_mel3": "mel_3ch",
    "mobilenet_v3_mel3": "mel_3ch",
    "vit_b16_mel3": "mel_3ch",
    "wav2vec2_raw": "raw_waveform_16khz",
    "pitch_fusion_paper_aligned": "raw_waveform_16khz_plus_pitch_contour",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, passed: bool, evidence: Any, blocking: bool = True) -> dict[str, Any]:
    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "blocking": bool(blocking and not passed),
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/audit/benchmark_preflight.json")
    args = parser.parse_args()

    manifest_path = ROOT / "data/manifests/split_manifest_v2_enriched.csv"
    manifest = pd.read_csv(
        manifest_path, dtype={"sample_id": "string", "speaker_id": "string"}
    )
    eligible = manifest[manifest["split"] != "excluded"].copy()
    partitions = {
        name: eligible[eligible["split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    speaker_sets = {
        name: set(frame["speaker_id"].astype(str))
        for name, frame in partitions.items()
    }
    overlaps = {
        "train_validation": len(speaker_sets["train"] & speaker_sets["validation"]),
        "train_test": len(speaker_sets["train"] & speaker_sets["test"]),
        "validation_test": len(speaker_sets["validation"] & speaker_sets["test"]),
    }
    duplicate_cross_split = int(
        eligible.groupby("audio_sha256")["split"].nunique().gt(1).sum()
    )

    classical = read_json(ROOT / "configs/models/b6_classical_centralized_v1.json")
    image = read_json(ROOT / "configs/models/b6_image_models_v1.json")
    deep = read_json(ROOT / "configs/models/b6_deep_v1.json")
    all_runner = (ROOT / "src/fedecai/models/run_b6_all_models.py").read_text(encoding="utf-8")
    deep_runner = (ROOT / "src/fedecai/models/run_b6_deep_pipeline.py").read_text(encoding="utf-8")
    pitch_runner = (ROOT / "src/fedecai/models/run_b6_pitch_fusion_paper.py").read_text(encoding="utf-8")
    classical_runner = (ROOT / "src/fedecai/models/run_b6_classical_pipeline.py").read_text(encoding="utf-8")
    common = (ROOT / "src/fedecai/models/b6_common.py").read_text(encoding="utf-8")
    raw_runner_path = ROOT / "src/fedecai/models/run_b6_wav2vec2_raw.py"
    raw_runner = raw_runner_path.read_text(encoding="utf-8") if raw_runner_path.is_file() else ""
    evaluator_path = ROOT / "src/fedecai/models/centralized_evaluator.py"
    evaluator = evaluator_path.read_text(encoding="utf-8") if evaluator_path.is_file() else ""
    costs_path = ROOT / "src/fedecai/models/benchmark_costs.py"
    costs = costs_path.read_text(encoding="utf-8") if costs_path.is_file() else ""
    output_contract_path = ROOT / "src/fedecai/models/centralized_output_contract.py"
    output_contract = output_contract_path.read_text(encoding="utf-8") if output_contract_path.is_file() else ""
    report_generator_path = ROOT / "src/fedecai/models/generate_centralized_report.py"
    report_generator = report_generator_path.read_text(encoding="utf-8") if report_generator_path.is_file() else ""
    smoke_path = ROOT / "artifacts/audit/centralized_smoke_tests.json"
    smoke = read_json(smoke_path) if smoke_path.is_file() else {}

    released_mapping = (
        eligible[["emotion", "emotion_id"]]
        .drop_duplicates()
        .sort_values("emotion_id")
        .to_dict("records")
    )
    configured_orders = {
        "classical": classical["emotion_order"],
        "deep": deep["emotion_order"],
        "image": image["emotion_order"],
    }
    mapping_consistent = len({tuple(value) for value in configured_orders.values()}) == 1
    model_code_uses_released_id = "emotion_id" in deep_runner

    parsed_runner = ast.parse(all_runner)
    declared_registry: set[str] = set()
    for node in parsed_runner.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DISPLAY_NAMES"
            for target in node.targets
        ):
            declared_registry = set(ast.literal_eval(node.value))
    metric_tokens = {
        "weighted_f1": 'average="weighted"',
        "per_class_precision": "precision_score",
        "per_class_recall": "recall_score",
        "confusion_matrix": "confusion_matrix",
        "accent_std": "accent_macro_f1_std",
        "per_emotion_within_accent": "per_emotion_f1",
    }
    missing_metric_contract = [
        name for name, token in metric_tokens.items() if token not in common
    ]
    required_prediction_fields = {
        "experiment_id", "model", "seed", "sample_id", "speaker_id", "accent",
        "y_true", "y_pred", "probability_angry",
        "probability_happy", "probability_neutral", "probability_sad", "split",
    }
    prediction_function = common[common.index("def prediction_rows"):]
    missing_prediction_fields = []
    for field in sorted(required_prediction_fields):
        if field.startswith("probability_"):
            present = 'f"probability_{label}"' in prediction_function
        else:
            present = f'"{field}"' in prediction_function
        if not present:
            missing_prediction_fields.append(field)

    augmented_columns = [name for name in eligible if "augment" in name.lower()]
    checks = [
        check("eligible_population", len(eligible) == 4992, {"rows": len(eligible)}),
        check("labels", set(eligible["emotion"]) == {"angry", "happy", "neutral", "sad"}, sorted(eligible["emotion"].unique())),
        check("accents", set(eligible["accent_clean"]) == {"north", "central", "south"}, sorted(eligible["accent_clean"].unique())),
        check("speaker_count", eligible["speaker_id"].nunique() == 147, {"speakers": int(eligible["speaker_id"].nunique())}),
        check("speaker_disjoint", all(value == 0 for value in overlaps.values()), overlaps),
        check("duplicate_waveforms_cross_splits", duplicate_cross_split == 0, {"cross_split_hashes": duplicate_cross_split}),
        check("augmentation_train_only", not augmented_columns, {"augmentation_columns": augmented_columns, "interpretation": "No augmentation is currently materialized."}),
        check("fixed_split_version", set(eligible["split_version"]) == {"speaker_disjoint_v2"}, sorted(eligible["split_version"].unique())),
        check("fixed_test_membership", len(partitions["test"]) == 665 and partitions["test"]["speaker_id"].nunique() == 22, {"samples": len(partitions["test"]), "speakers": int(partitions["test"]["speaker_id"].nunique()), "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/")}),
        check("emotion_mapping_consistency", mapping_consistent and not model_code_uses_released_id, {"released_emotion_id_mapping": released_mapping, "model_emotion_orders": configured_orders, "model_code_consumes_released_emotion_id": model_code_uses_released_id}),
        check("exact_eight_model_registry", set(REQUIRED_MODELS) == declared_registry, {"required": list(REQUIRED_MODELS), "declared": sorted(declared_registry), "mismatch": sorted(set(REQUIRED_MODELS) ^ declared_registry)}),
        check("classical_input_policy", classical["feature_contract"]["numeric_prefixes"] == ["mfcc_", "egemaps_"] and classical["feature_contract"]["expected_features"] == 168, classical["feature_contract"]),
        check("image_input_policy", set(image["models"]) == {"resnet18_mel3", "mobilenet_v3_mel3", "vit_b16_mel3"} and image["feature_contract"].get("mel_3ch_mode") == "repeat_normalized_log_mel", image["feature_contract"]),
        check("cnn_input_policy", "cnn2d_logmel" in deep["models"], {"model": "cnn2d_logmel", "input": "log_mel"}),
        check("wav2vec2_raw_waveform_runner", all(token in raw_runner for token in ("CachedWav2Vec2Classifier", "validate_embedding_cache", "fully_frozen", "16000")) and "wav2vec2_raw" in all_runner, {"runner": str(raw_runner_path.relative_to(ROOT)).replace("\\", "/"), "official_model": "raw waveform -> frozen Wav2Vec2 -> masked pooling -> cached embedding -> classifier"}),
        check("pitch_fusion_waveform_pitch_runner", "decode_wav" in pitch_runner and "interpolate_pitch" in pitch_runner, {"runner": "src/fedecai/models/run_b6_pitch_fusion_paper.py"}),
        check("train_only_preprocessing", "StandardScaler().fit(train" in deep_runner and "PredefinedSplit" in (ROOT / "src/fedecai/models/run_b6_classical_pipeline.py").read_text(encoding="utf-8"), {"classical": "Pipeline scaler + predefined training folds", "deep": "fit(train) / fold-train loaders"}),
        check("complete_metric_contract", not missing_metric_contract, {"missing": missing_metric_contract}),
        check("complete_prediction_contract", not missing_prediction_fields, {"missing": missing_prediction_fields}),
        check("shared_evaluator", all(token in evaluator for token in ("validate_predictions", "evaluate_prediction_file", "fixed_test_ids", "validate_common_test_membership")), {"path": str(evaluator_path.relative_to(ROOT)).replace("\\", "/")}),
        check("fixed_test_evaluation_pipeline", all(token in evaluator for token in ("expected_sample_ids", "Fixed-test membership mismatch", "expected_split")), {"required_output": "sample-level predictions on the same 665-row test membership for all eight models"}),
        check("latency_benchmark", all(token in costs for token in ("benchmark_inference_latency", "warmup_runs", "timed_runs", "torch.cuda.synchronize", "latency_p95_ms", "primary_latency_statistic")), {"path": str(costs_path.relative_to(ROOT)).replace("\\", "/"), "primary": "median"}),
        check("parameter_count", all(token in costs for token in ("total_parameters", "trainable_parameters", "classical_parameter_description")), {"path": str(costs_path.relative_to(ROOT)).replace("\\", "/")}),
        check("model_size_protocol", all(token in costs for token in ("checkpoint_size_mb", "effective_deployment_size_mb", "model_size_mb", "MEBIBYTE")), {"definition": "serialized deployable artifacts; bytes / 1024^2"}),
        check("training_time_protocol", all(token in costs for token in ("TrainingTimer", "training_start_timestamp", "training_end_timestamp", "training_time_sec")), {"feature_cache_fields": ["feature_extraction_time_sec", "model_training_time_sec", "end_to_end_training_pipeline_time_sec"]}),
        check("training_output_contract", all(token in output_contract for token in ("artifacts/predictions", "artifacts/checkpoints", "artifacts/metrics", "logs/runs", "run_summary.md", "environment.json")), {"path": str(output_contract_path.relative_to(ROOT)).replace("\\", "/")}),
        check("trainer_output_integration", all(all(token in source for token in ("artifacts/checkpoints", "artifacts/predictions", "artifacts/metrics", "evaluate_prediction_file", "benchmark_inference_latency", "model_size", "TrainingTimer")) for source in (classical_runner, deep_runner, pitch_runner, raw_runner)), {"runners": ["classical", "deep/image", "wav2vec2_raw", "pitch_fusion"], "contract": "checkpoint + normalized predictions + prediction-derived metrics + cost evidence"}),
        check("report_generation_contract", all(token in report_generator for token in ("centralized_benchmark_table", "pareto_performance_cost", "per_accent_model_comparison", "model_efficiency_comparison", "report_figure_manifest.csv", "REQUIRED_MODELS")), {"path": str(report_generator_path.relative_to(ROOT)).replace("\\", "/"), "generation_policy": "refuse until exactly eight real metrics files are supplied"}),
        check("eight_model_smoke_tests", smoke.get("status") == "PASS" and len(smoke.get("models", [])) == 8 and all(item.get("passed") for item in smoke.get("models", [])), {"path": str(smoke_path.relative_to(ROOT)).replace("\\", "/"), "status": smoke.get("status", "MISSING"), "models": {item.get("model"): item.get("passed") for item in smoke.get("models", [])}}),
        check("execution_logging_ready", (ROOT / "logs/run_index.csv").is_file(), {"required": "logs/run_index.csv and per-run evidence directories"}),
        check("runtime_dap301m", Path(sys.executable).resolve() == (ROOT / "DAP301m/Scripts/python.exe").resolve(), {"python_executable": sys.executable, "python": sys.version.split()[0]}),
        check("runtime_torch", importlib.util.find_spec("torch") is not None, {"version": importlib.metadata.version("torch") if importlib.util.find_spec("torch") else None}),
        check("runtime_torchvision", importlib.util.find_spec("torchvision") is not None, {"version": importlib.metadata.version("torchvision") if importlib.util.find_spec("torchvision") else None}),
        check("runtime_torchaudio_optional", importlib.util.find_spec("torchaudio") is not None, {"version": importlib.metadata.version("torchaudio") if importlib.util.find_spec("torchaudio") else None, "required_by_current_paths": False}, blocking=False),
        check("runtime_transformers", importlib.util.find_spec("transformers") is not None, {"version": importlib.metadata.version("transformers") if importlib.util.find_spec("transformers") else None}),
    ]

    blocking_failures = [item for item in checks if item["blocking"]]
    result = {
        "schema_version": "fedecai.centralized_benchmark_preflight.v1",
        "status": "PASS" if not blocking_failures else "BLOCKED",
        "training_authorized": not blocking_failures,
        "dataset_version": "visec_clean_eligible_v1",
        "split_version": "speaker_disjoint_v2",
        "seed": 391,
        "centralized_infrastructure_status": "READY" if not blocking_failures else "BLOCKED",
        "checks": checks,
        "blocking_failures": [item["check"] for item in blocking_failures],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["training_authorized"] else 2)


if __name__ == "__main__":
    main()
