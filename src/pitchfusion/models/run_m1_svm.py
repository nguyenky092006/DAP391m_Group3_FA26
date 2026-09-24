"""Run the approved fixed M1 SVM-RBF baseline without model selection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from pitchfusion.models.training import (
    classification_metrics,
    load_m1_table,
    load_config,
    prediction_rows,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[3]


def save_checkpoint(value: dict[str, object], path: Path) -> None:
    """Atomically save resumable M1 progress after each completed stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/models/m1_svm_rbf_v1.json",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data/interim/features/m1_features_v1.parquet",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifests/split_manifest_v2_enriched.csv",
    )
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    frame, feature_columns = load_m1_table(
        args.features.resolve(), args.manifest.resolve(), config
    )
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    if set(train["speaker_id"].astype(str)) & set(validation["speaker_id"].astype(str)):
        raise ValueError("Speaker leakage detected between train and validation")
    folds = sorted(train["cv_fold"].astype(str).unique())
    if len(folds) < 2:
        raise ValueError("M1 requires at least two grouped training folds")
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": "M1 SVM-RBF",
                "train": len(train),
                "validation": len(validation),
                "features": len(feature_columns),
                "grouped_folds": folds,
                "test_samples_used": 0,
            },
            indent=2,
        )
    )
    if args.preflight_only:
        return

    artifact_dir = ROOT / "artifacts/m1_m4/m1"
    report_dir = ROOT / "reports/m1_m4"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "m1_svm_rbf.joblib"
    report_path = report_dir / "m1_svm_rbf_report.json"
    checkpoint_path = artifact_dir / "m1_svm_rbf_checkpoint.joblib"
    if model_path.exists() and report_path.exists() and not args.force:
        print("M1 completed, skipping. Use --force to run it again.", flush=True)
        return

    lineage = {
        "config_sha256": sha256_file(args.config.resolve()),
        "features_sha256": sha256_file(args.features.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
    }
    checkpoint: dict[str, object] = {"lineage": lineage, "cv_rows": []}
    if checkpoint_path.exists() and not args.force:
        candidate = joblib.load(checkpoint_path)
        if candidate.get("lineage") != lineage:
            raise ValueError(
                "M1 checkpoint inputs changed; use --force to start a clean run"
            )
        checkpoint = candidate
        print(f"Resuming M1 from {checkpoint_path}", flush=True)

    model_config = config["models"]["m1_svm_rbf"]
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=float(model_config["C"]),
                    gamma=model_config["gamma"],
                    class_weight=model_config["class_weight"],
                    random_state=int(config["seed"]),
                ),
            ),
        ]
    )
    encoder = LabelEncoder().fit(config["emotion_order"])
    x_train = train[feature_columns].to_numpy(dtype=np.float32)
    y_train = encoder.transform(train["emotion"])
    x_validation = validation[feature_columns].to_numpy(dtype=np.float32)
    fold_to_index = {name: index for index, name in enumerate(folds)}
    predefined = PredefinedSplit(
        np.asarray([fold_to_index[str(value)] for value in train["cv_fold"]])
    )
    started = time.perf_counter()
    cv_rows = list(checkpoint.get("cv_rows", []))
    completed_folds = {str(row["fold"]) for row in cv_rows}
    for index, (fit_indices, held_out_indices) in enumerate(predefined.split(), start=0):
        fold = folds[index]
        if fold in completed_folds:
            print(f"M1 CV fold {fold}: restored from checkpoint", flush=True)
            continue
        fold_started = time.perf_counter()
        fold_estimator = clone(estimator)
        fold_estimator.fit(x_train[fit_indices], y_train[fit_indices])
        fold_prediction = fold_estimator.predict(x_train[held_out_indices])
        row = {
            "fold": fold,
            "macro_f1": float(f1_score(y_train[held_out_indices], fold_prediction, average="macro", zero_division=0)),
            "uar": float(recall_score(y_train[held_out_indices], fold_prediction, average="macro", zero_division=0)),
            "fit_seconds": time.perf_counter() - fold_started,
        }
        cv_rows.append(row)
        checkpoint["cv_rows"] = cv_rows
        checkpoint["stage"] = f"cv_fold_{fold}_complete"
        save_checkpoint(checkpoint, checkpoint_path)
        print(f"M1 CV fold {fold}: checkpoint saved", flush=True)

    if "estimator" in checkpoint:
        estimator = checkpoint["estimator"]  # type: ignore[assignment]
        print("M1 final estimator restored from checkpoint", flush=True)
    else:
        estimator.fit(x_train, y_train)
        checkpoint["estimator"] = estimator
        checkpoint["stage"] = "final_estimator_complete"
        save_checkpoint(checkpoint, checkpoint_path)
        print("M1 final estimator checkpoint saved", flush=True)

    if "calibrator" in checkpoint:
        calibrator = checkpoint["calibrator"]
        print("M1 probability calibrator restored from checkpoint", flush=True)
    else:
        calibrator = CalibratedClassifierCV(
            estimator=clone(estimator),
            method="sigmoid",
            cv=predefined,
            n_jobs=args.jobs,
            ensemble=False,
        )
        calibrator.fit(x_train, y_train)
        checkpoint["calibrator"] = calibrator
        checkpoint["stage"] = "calibration_complete"
        save_checkpoint(checkpoint, checkpoint_path)
        print("M1 calibration checkpoint saved", flush=True)
    predicted_ids = estimator.predict(x_validation)
    predicted = encoder.inverse_transform(predicted_ids)
    raw_probabilities = calibrator.predict_proba(x_validation)
    probability_labels = encoder.inverse_transform(calibrator.classes_)
    reorder = [list(probability_labels).index(label) for label in config["emotion_order"]]
    probabilities = raw_probabilities[:, reorder]
    metrics = classification_metrics(
        validation["emotion"].to_numpy(),
        predicted,
        validation["accent_clean"].to_numpy(),
        config["emotion_order"],
    )
    joblib.dump(
        {
            "estimator": estimator,
            "probability_estimator": calibrator,
            "features": feature_columns,
            "classes": list(encoder.classes_),
            "config": config,
        },
        model_path,
    )
    write_csv_atomic(cv_rows, report_dir / "m1_svm_rbf_grouped_cv.csv")
    write_csv_atomic(
        prediction_rows(
            validation,
            predicted,
            probabilities,
            config["emotion_order"],
        ),
        report_dir / "m1_svm_rbf_validation_predictions.csv",
    )
    write_json_atomic(
        {
            "schema_version": config["schema_version"],
            "model": "m1_svm_rbf",
            "configuration_policy": "fixed M1 baseline; no hyperparameter tuning",
            "grouped_train_cv": {
                "folds": folds,
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in cv_rows])),
                "std_macro_f1": float(np.std([row["macro_f1"] for row in cv_rows])),
                "mean_uar": float(np.mean([row["uar"] for row in cv_rows])),
            },
            "validation": metrics,
            "runtime_seconds": time.perf_counter() - started,
            "test_samples_used": 0,
            "model_artifact": model_path.relative_to(ROOT).as_posix(),
            "model_artifact_sha256": sha256_file(model_path),
            "lineage": lineage,
        },
        report_path,
    )
    checkpoint["stage"] = "complete"
    checkpoint["model_path"] = model_path.relative_to(ROOT).as_posix()
    checkpoint["report_path"] = report_path.relative_to(ROOT).as_posix()
    save_checkpoint(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()
