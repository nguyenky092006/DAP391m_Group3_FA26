"""Train leakage-safe SVM and XGBoost B6 baselines in one resumable run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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

try:
    from .b6_common import (
        classification_metrics,
        load_b6_table,
        load_config,
        prediction_rows,
        sha256_file,
        write_csv_atomic,
        write_json_atomic,
    )
except ImportError:
    from b6_common import (  # type: ignore
        classification_metrics,
        load_b6_table,
        load_config,
        prediction_rows,
        sha256_file,
        write_csv_atomic,
        write_json_atomic,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/models/b6_classical_v1.json")
    parser.add_argument("--features", type=Path, default=root / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", type=Path, default=root / "data/manifests/split_manifest_v2_enriched.csv")
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
    config = load_config(args.config.resolve())
    print("[1/4] Validating B4 features and frozen B5 split v2")
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
            r".\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt"
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
    artifact_dir = root / "artifacts/b6/classical"
    report_dir = root / "reports/tables/b6"
    reports: dict[str, Any] = {}

    for number, model_name in enumerate(args.models, start=2):
        report_path = report_dir / f"{model_name}_validation_report.json"
        if report_path.exists() and not args.force:
            reports[model_name] = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"[{number}/4] {model_name}: verified report exists, resume skip")
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
        started = time.perf_counter()
        search.fit(x_train, y_train, **fit_kwargs)
        search_seconds = time.perf_counter() - started
        calibration_seconds = 0.0
        decision_estimator = search.best_estimator_
        probability_estimator = search.best_estimator_
        calibration = None
        if model_name == "svm_rbf":
            # Calibrate probabilities using the same speaker-grouped folds. This
            # replaces deprecated SVC(probability=True) and prevents row-level
            # calibration folds from mixing held-out speakers.
            calibration = "sigmoid CalibratedClassifierCV using B5 grouped folds"
            calibration_started = time.perf_counter()
            probability_estimator = CalibratedClassifierCV(
                estimator=clone(search.best_estimator_),
                method="sigmoid",
                cv=predefined,
                n_jobs=args.jobs,
                ensemble=False,
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
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / f"{model_name}.joblib"
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
            "validation_metrics": metrics,
            "model_artifact": str(model_path.relative_to(root)).replace("\\", "/"),
            "model_artifact_sha256": sha256_file(model_path),
            "lineage": {
                "config_sha256": sha256_file(args.config.resolve()),
                "features_sha256": sha256_file(args.features.resolve()),
                "manifest_sha256": sha256_file(args.manifest.resolve()),
            },
        }
        write_csv_atomic(
            cv_rows(search, model_name, folds),
            report_dir / f"{model_name}_cv_results.csv",
        )
        write_csv_atomic(prediction_rows(validation, predicted, probabilities, labels), report_dir / f"{model_name}_validation_predictions.csv")
        write_json_atomic(report, report_path)
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
