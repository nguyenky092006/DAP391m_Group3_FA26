"""Train leakage-safe SVM and XGBoost B6 baselines in one resumable run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from fedecai.models.benchmark_costs import (
    TrainingTimer, benchmark_inference_latency, classical_parameter_description,
    model_size,
)
from fedecai.models.centralized_evaluator import evaluate_prediction_file

try:
    from .b6_common import (
        classification_metrics,
        enforce_b6_training_lock,
        git_state,
        load_b6_table,
        load_config,
        prediction_rows,
        register_experiment,
        sha256_file,
        write_csv_atomic,
        write_json_atomic,
    )
except ImportError:
    from b6_common import (  # type: ignore
        classification_metrics,
        enforce_b6_training_lock,
        git_state,
        load_b6_table,
        load_config,
        prediction_rows,
        register_experiment,
        sha256_file,
        write_csv_atomic,
        write_json_atomic,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/models/b6_classical_centralized_v1.json")
    parser.add_argument("--features", type=Path, default=root / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", type=Path, default=root / "data/manifests/emotion_stratified_v1.csv")
    parser.add_argument("--models", nargs="+", choices=("svm_rbf", "xgboost"), default=["svm_rbf", "xgboost"])
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def make_estimators(config: dict[str, Any], seed: int, jobs: int) -> dict[str, tuple[Any, dict[str, list[Any]], dict[str, Any]]]:
    svm = config["models"]["svm_rbf"]
    estimators: dict[str, tuple[Any, dict[str, list[Any]], dict[str, Any]]] = {
        "svm_rbf": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
            ]),
            {"model__C": svm["C"], "model__gamma": svm["gamma"]},
            {},
        )
    }
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return estimators
    xgb = config["models"]["xgboost"]
    estimators["xgboost"] = (
        XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=jobs,
            subsample=xgb["subsample"],
            colsample_bytree=xgb["colsample_bytree"],
        ),
        {
            "n_estimators": xgb["n_estimators"],
            "max_depth": xgb["max_depth"],
            "learning_rate": xgb["learning_rate"],
        },
        {"sample_weight": "balanced"},
    )
    return estimators


def cv_rows(
    search: GridSearchCV, model_name: str, folds: list[str]
) -> list[dict[str, Any]]:
    results = search.cv_results_
    rows = []
    for index, params in enumerate(results["params"]):
        row = {
            "model": model_name,
            "rank_macro_f1": int(results["rank_test_macro_f1"][index]),
            "mean_macro_f1": float(results["mean_test_macro_f1"][index]),
            "std_macro_f1": float(results["std_test_macro_f1"][index]),
            "mean_uar": float(results["mean_test_uar"][index]),
            "mean_fit_seconds": float(results["mean_fit_time"][index]),
            "parameters": json.dumps(params, sort_keys=True),
        }
        for fold_index, fold in enumerate(folds):
            row[f"{fold}_macro_f1"] = float(
                results[f"split{fold_index}_test_macro_f1"][index]
            )
            row[f"{fold}_uar"] = float(results[f"split{fold_index}_test_uar"][index])
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    enforce_b6_training_lock(root, "classical training/preflight")
    config = load_config(args.config.resolve())
    print("[1/4] Validating B4 features and frozen B5 emotion-stratified split")
    frame, feature_columns = load_b6_table(args.features.resolve(), args.manifest.resolve(), config)
    counts = frame["split"].value_counts().to_dict()
    folds = sorted(frame.loc[frame["split"] == "train", "cv_fold"].unique())
    print(json.dumps({"rows": len(frame), "features": len(feature_columns), "splits": counts, "cv_folds": folds}, indent=2))
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "mode": "preflight_only"}, indent=2))
        return

    estimators = make_estimators(config, int(config["seed"]), args.jobs)
    missing = [name for name in args.models if name not in estimators]
    if missing:
        raise RuntimeError(
            f"Missing model dependencies for {missing}. Run: "
            r".\DAP301m\Scripts\python.exe -m pip install -r requirements.txt"
        )

    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    labels = list(config["emotion_order"])
    encoder = LabelEncoder().fit(labels)
    x_train = train[feature_columns].to_numpy(dtype=np.float32)
    y_train = encoder.transform(train["emotion"])
    x_validation = validation[feature_columns].to_numpy(dtype=np.float32)
    y_validation_text = validation["emotion"].to_numpy()
    fold_to_index = {fold: index for index, fold in enumerate(folds)}
    predefined = PredefinedSplit(np.array([fold_to_index[value] for value in train["cv_fold"]], dtype=int))
    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "uar": make_scorer(recall_score, average="macro", zero_division=0),
    }
    artifact_dir = root / "artifacts/checkpoints"
    report_dir = root / "reports/tables/b6"
    reports: dict[str, Any] = {}

    for number, model_name in enumerate(args.models, start=2):
        report_path = report_dir / f"{model_name}_validation_report.json"
        model_path = artifact_dir / f"b6_{model_name}_seed{config['seed']}.joblib"
        canonical_prediction = root / "artifacts/predictions" / f"b6_{model_name}_seed{config['seed']}_validation.csv"
        canonical_metrics = root / "artifacts/metrics" / f"b6_{model_name}_seed{config['seed']}_validation.json"
        if all(path.exists() for path in (report_path, model_path, canonical_prediction, canonical_metrics)) and not args.force:
            reports[model_name] = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"[{number}/4] {model_name}: verified full artifact set exists, resume skip")
            continue
        print(f"[{number}/4] Training and grouped-CV tuning: {model_name}")
        estimator, parameter_grid, fit_policy = estimators[model_name]
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=parameter_grid,
            scoring=scoring,
            refit="macro_f1",
            cv=predefined,
            n_jobs=args.jobs if model_name == "svm_rbf" else 1,
            return_train_score=False,
            error_score="raise",
        )
        fit_kwargs = {}
        if fit_policy.get("sample_weight") == "balanced":
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
        with TrainingTimer() as training_timer:
            started = time.perf_counter()
            search.fit(x_train, y_train, **fit_kwargs)
            search_seconds = time.perf_counter() - started
            calibration_seconds = 0.0
            decision_estimator = search.best_estimator_
            probability_estimator = search.best_estimator_
            calibration = None
            if model_name == "svm_rbf":
                # Calibrate probabilities using the same emotion-stratified folds.
                calibration = "sigmoid CalibratedClassifierCV using B5 grouped folds"
                calibration_started = time.perf_counter()
                probability_estimator = CalibratedClassifierCV(
                    estimator=clone(search.best_estimator_), method="sigmoid",
                    cv=predefined, n_jobs=args.jobs, ensemble=False,
                )
                probability_estimator.fit(x_train, y_train)
                calibration_seconds = time.perf_counter() - calibration_started
        elapsed = search_seconds + calibration_seconds
        # The calibrated probability argmax may differ from the SVM maximum-
        # margin decision and was observed to collapse the rare Sad class.
        # Preserve the selected classifier's decision rule for labels while
        # using the grouped calibrator only for confidence values.
        predicted_ids = decision_estimator.predict(x_validation)
        predicted = encoder.inverse_transform(predicted_ids)
        probabilities_raw = probability_estimator.predict_proba(x_validation)
        probability_order = encoder.inverse_transform(probability_estimator.classes_)
        reorder = [list(probability_order).index(label) for label in labels]
        probabilities = probabilities_raw[:, reorder]
        metrics = classification_metrics(y_validation_text, predicted, validation["accent_clean"].to_numpy(), labels)
        repository_state = git_state(root)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": decision_estimator,
                "probability_estimator": probability_estimator,
                "prediction_rule": "estimator.predict; calibrated estimator supplies confidence only",
                "labels": labels,
                "features": feature_columns,
            },
            model_path,
        )
        parameter_evidence = classical_parameter_description(decision_estimator)
        size_evidence = model_size(model_path)
        latency_probe = x_validation[:1]

        def inference_path() -> None:
            decision_estimator.predict(latency_probe)
            probability_estimator.predict_proba(latency_probe)

        latency_evidence = benchmark_inference_latency(
            inference_path, device="cpu", batch_size=1, precision="float32",
            warmup_runs=10, timed_runs=50, preprocessing_included=False,
        )
        costs = {
            **parameter_evidence, **size_evidence, **training_timer.as_dict(),
            **latency_evidence, "feature_extraction_time_sec": None,
            "model_training_time_sec": training_timer.training_time_sec,
            "end_to_end_training_pipeline_time_sec": training_timer.training_time_sec,
            "cost_protocol": {
                "feature_extraction": "B4 MFCC/eGeMAPS precomputation excluded",
                "latency_path": "model-ready 168-feature vector -> fitted estimator",
            },
        }
        report = {
            "status": "PASS",
            "model": model_name,
            "scope": "B6 model selection on grouped train CV and one validation report; test untouched",
            "config_schema": config["schema_version"],
            "seed": config["seed"],
            "train_samples": len(train),
            "validation_samples": len(validation),
            "test_samples_reserved": int(counts.get("test", 0)),
            "features": len(feature_columns),
            "cv_folds": folds,
            "best_parameters": search.best_params_,
            "best_cv_macro_f1": float(search.best_score_),
            "grid_search_seconds": search_seconds,
            "probability_calibration": calibration,
            "prediction_rule": (
                "selected estimator decision rule; grouped calibrator supplies confidence only"
                if model_name == "svm_rbf"
                else "predict_proba argmax"
            ),
            "calibration_seconds": calibration_seconds,
            "total_tuning_seconds": elapsed,
            "costs": costs,
            "validation_metrics": metrics,
            "model_artifact": str(model_path.relative_to(root)).replace("\\", "/"),
            "model_artifact_sha256": sha256_file(model_path),
            "lineage": {
                "config_sha256": sha256_file(args.config.resolve()),
                "features_sha256": sha256_file(args.features.resolve()),
                "manifest_sha256": sha256_file(args.manifest.resolve()),
                **repository_state,
            },
        }
        write_csv_atomic(
            cv_rows(search, model_name, folds),
            report_dir / f"{model_name}_cv_results.csv",
        )
        experiment_id = f"b6_{model_name}_seed{config['seed']}"
        rows = prediction_rows(
            validation, predicted, probabilities, labels,
            experiment_id=experiment_id, model=model_name, seed=int(config["seed"]),
        )
        write_csv_atomic(rows, report_dir / f"{model_name}_validation_predictions.csv")
        canonical_prediction = root / "artifacts/predictions" / f"{experiment_id}_validation.csv"
        canonical_metrics = root / "artifacts/metrics" / f"{experiment_id}_validation.json"
        write_csv_atomic(rows, canonical_prediction)
        evaluated = evaluate_prediction_file(
            canonical_prediction, canonical_metrics,
            expected_sample_ids=validation["sample_id"].astype(str),
            expected_split="validation", dataset_version="visec_clean_eligible_v1",
            split_version=config["split_version"], costs=costs,
            require_argmax_consistency=model_name != "svm_rbf",
        )
        write_json_atomic(report, report_path)
        register_experiment(root, {
            "experiment_id": f"b6_{model_name}_dev_v1_{int(time.time())}", "step": "B6",
            "rq": "RQ3", "model": model_name,
            "variant": "speaker_grouped_cv_centralized",
            "dataset_version": "visec_clean_eligible_v1",
            "status": "completed",
            "model_config": str(args.config.resolve().relative_to(root)).replace("\\", "/"),
            "feature_config": "configs/features/visec_features_v1.json",
            "seed": config["seed"], "split_version": config["split_version"],
            "git_commit": repository_state["git_commit"], "hardware": costs["device"],
            "train_time_sec": costs["training_time_sec"],
            "inference_latency_ms": costs["inference_latency_ms"],
            "model_size_mb": costs["model_size_mb"],
            "overall_macro_f1": evaluated["overall_macro_f1"],
            "worst_accent_macro_f1": evaluated["worst_accent_macro_f1"],
            "accent_gap": evaluated["accent_gap"],
            "checkpoint_path": str(model_path.relative_to(root)).replace("\\", "/"),
            "prediction_path": str(canonical_prediction.relative_to(root)).replace("\\", "/"),
            "notes": "Development model selection only; final test remained unopened.",
        })
        reports[model_name] = report

    # A targeted --models rerun must retain other already verified model
    # reports in the consolidated summary.
    for known_model in ("svm_rbf", "xgboost"):
        existing = report_dir / f"{known_model}_validation_report.json"
        if known_model not in reports and existing.exists():
            reports[known_model] = json.loads(existing.read_text(encoding="utf-8"))

    print("[4/4] Publishing consolidated B6 classical summary")
    summary = {
        "status": "PASS",
        "schema_version": config["schema_version"],
        "completed_models": list(reports),
        "test_partition_touched": False,
        "models": reports,
    }
    write_json_atomic(summary, report_dir / "b6_classical_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
