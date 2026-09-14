"""Run reproducible speaker-disjoint B4 accent probes on frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "b4.accent_probe.v1":
        raise ValueError("Unsupported accent-probe schema")
    if config.get("fit_split") != "train" or config.get("evaluation_split") != "validation":
        raise ValueError("B4 probe must use train-to-validation evaluation")
    if config.get("forbidden_split") != "test":
        raise ValueError("B4 probe must explicitly forbid the test split")
    if config.get("probe", {}).get("scaler_fit_scope") != "train_only":
        raise ValueError("Accent-probe scaler must fit on training only")
    if config.get("speaker_bootstrap", {}).get("unit") != "validation_speaker":
        raise ValueError("Accent-probe bootstrap unit must be validation speaker")
    return config


def load_development_data(
    repo_root: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(
        repo_root / config["split_manifest"],
        dtype={"sample_id": "string", "speaker_id": "string"},
    )
    eligible = manifest.loc[manifest["is_eligible_for_split"].astype(str).str.lower().eq("true")].copy()
    if set(eligible["split_version"].unique()) != {config["split_version"]}:
        raise ValueError("Manifest split version does not match accent-probe config")
    development = eligible.loc[
        eligible["split"].isin([config["fit_split"], config["evaluation_split"]]),
        ["sample_id", "speaker_id", "accent", "split"],
    ].copy()
    if development["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in development manifest")
    features = pd.read_parquet(repo_root / config["feature_table"])
    frame = development.merge(features, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_feature"))
    if len(frame) != len(development) or frame.isna().any().any():
        raise ValueError("Feature join is incomplete or contains missing values")
    train = frame.loc[frame["split"].eq(config["fit_split"])].copy()
    validation = frame.loc[frame["split"].eq(config["evaluation_split"])].copy()
    overlap = set(train["speaker_id"]) & set(validation["speaker_id"])
    if overlap:
        raise ValueError(f"Speaker leakage between train and validation: {sorted(overlap)}")
    if train.empty or validation.empty:
        raise ValueError("Train and validation data are both required")
    audit = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_speakers": train["speaker_id"].nunique(),
        "validation_speakers": validation["speaker_id"].nunique(),
        "train_validation_speaker_overlap": 0,
        "test_samples_loaded": 0,
    }
    return train, validation, audit


def representation_columns(frame: pd.DataFrame, prefixes: list[str]) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in prefixes)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise ValueError(f"No numeric feature columns matched prefixes: {prefixes}")
    return columns


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "uar": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def build_probe(config: dict[str, Any], random_state: int) -> Pipeline:
    probe = config["probe"]
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver=probe["solver"],
                    C=float(probe["C"]),
                    class_weight=probe["class_weight"],
                    max_iter=int(probe["max_iter"]),
                    tol=float(probe["tolerance"]),
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_checked(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> int:
    """Fit a probe and reject evidence from a non-converged optimization."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, labels)
    return int(np.max(model.named_steps["classifier"].n_iter_))


def speaker_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speakers: np.ndarray,
    labels: list[str],
    resamples: int,
    seed: int,
) -> list[dict[str, float | int]]:
    unique_speakers = np.unique(speakers)
    indices = {speaker: np.flatnonzero(speakers == speaker) for speaker in unique_speakers}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(resamples):
        sampled = rng.choice(unique_speakers, size=len(unique_speakers), replace=True)
        selected = np.concatenate([indices[speaker] for speaker in sampled])
        result = metrics(y_true[selected], y_pred[selected], labels)
        rows.append({"replicate": replicate, **result})
    return rows


def run_probe(repo_root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    train, validation, split_audit = load_development_data(repo_root, config)
    labels = list(config["accent_order"])
    seed = int(config["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    report_representations: dict[str, Any] = {}
    y_train = train["accent"].to_numpy(dtype=str)
    y_validation = validation["accent"].to_numpy(dtype=str)
    validation_speakers = validation["speaker_id"].to_numpy(dtype=str)
    majority_label = sorted(Counter(y_train).items(), key=lambda item: (-item[1], item[0]))[0][0]
    majority_prediction = np.full(len(validation), majority_label, dtype=object)
    majority_metrics = metrics(y_validation, majority_prediction, labels)

    for representation_index, representation in enumerate(config["representations"]):
        name = representation["name"]
        columns = representation_columns(train, representation["column_prefixes"])
        x_train = train[columns].to_numpy(dtype=np.float64)
        x_validation = validation[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
            raise ValueError(f"Non-finite values in representation {name}")
        model = build_probe(config, seed + representation_index)
        observed_iterations = fit_checked(model, x_train, y_train)
        prediction = model.predict(x_validation)
        observed = metrics(y_validation, prediction, labels)

        permutation_rng = np.random.default_rng(seed + 10_000 * (representation_index + 1))
        representation_permutations = []
        permutation_iterations = []
        for permutation in range(int(config["permutation_baseline"]["permutations"])):
            permuted_model = build_probe(config, seed + representation_index * 1000 + permutation + 1)
            permutation_iterations.append(
                fit_checked(permuted_model, x_train, permutation_rng.permutation(y_train))
            )
            permuted_prediction = permuted_model.predict(x_validation)
            value = metrics(y_validation, permuted_prediction, labels)
            row = {"representation": name, "permutation": permutation, **value}
            permutation_rows.append(row)
            representation_permutations.append(value)

        representation_bootstrap = speaker_bootstrap(
            y_validation,
            prediction,
            validation_speakers,
            labels,
            int(config["speaker_bootstrap"]["resamples"]),
            seed + 20_000 * (representation_index + 1),
        )
        for row in representation_bootstrap:
            bootstrap_rows.append({"representation": name, **row})

        permutation_macro = np.asarray([row["macro_f1"] for row in representation_permutations])
        permutation_uar = np.asarray([row["uar"] for row in representation_permutations])
        bootstrap_macro = np.asarray([row["macro_f1"] for row in representation_bootstrap])
        bootstrap_uar = np.asarray([row["uar"] for row in representation_bootstrap])
        summary = {
            "dimension": len(columns),
            "observed_fit_iterations": observed_iterations,
            "observed": observed,
            "majority_baseline": {"label": majority_label, **majority_metrics},
            "permutation_baseline": {
                "resamples": len(permutation_macro),
                "maximum_fit_iterations": max(permutation_iterations),
                "macro_f1_mean": float(permutation_macro.mean()),
                "macro_f1_std": float(permutation_macro.std(ddof=1)),
                "uar_mean": float(permutation_uar.mean()),
                "uar_std": float(permutation_uar.std(ddof=1)),
                "macro_f1_empirical_p": float((1 + np.sum(permutation_macro >= observed["macro_f1"])) / (len(permutation_macro) + 1)),
                "uar_empirical_p": float((1 + np.sum(permutation_uar >= observed["uar"])) / (len(permutation_uar) + 1)),
            },
            "speaker_bootstrap_95_ci": {
                "resamples": len(bootstrap_macro),
                "macro_f1_low": float(np.quantile(bootstrap_macro, 0.025)),
                "macro_f1_high": float(np.quantile(bootstrap_macro, 0.975)),
                "uar_low": float(np.quantile(bootstrap_uar, 0.025)),
                "uar_high": float(np.quantile(bootstrap_uar, 0.975)),
            },
        }
        report_representations[name] = summary
        metric_rows.append(
            {
                "representation": name,
                "dimension": len(columns),
                "validation_macro_f1": observed["macro_f1"],
                "validation_uar": observed["uar"],
                "majority_macro_f1": majority_metrics["macro_f1"],
                "majority_uar": majority_metrics["uar"],
                "permutation_resamples": summary["permutation_baseline"]["resamples"],
                "permutation_macro_f1_mean": summary["permutation_baseline"]["macro_f1_mean"],
                "permutation_macro_f1_std": summary["permutation_baseline"]["macro_f1_std"],
                "permutation_uar_mean": summary["permutation_baseline"]["uar_mean"],
                "permutation_uar_std": summary["permutation_baseline"]["uar_std"],
                "permutation_macro_f1_empirical_p": summary["permutation_baseline"]["macro_f1_empirical_p"],
                "permutation_uar_empirical_p": summary["permutation_baseline"]["uar_empirical_p"],
                "bootstrap_resamples": summary["speaker_bootstrap_95_ci"]["resamples"],
                "bootstrap_macro_f1_low": summary["speaker_bootstrap_95_ci"]["macro_f1_low"],
                "bootstrap_macro_f1_high": summary["speaker_bootstrap_95_ci"]["macro_f1_high"],
                "bootstrap_uar_low": summary["speaker_bootstrap_95_ci"]["uar_low"],
                "bootstrap_uar_high": summary["speaker_bootstrap_95_ci"]["uar_high"],
            }
        )
        for (_, row), predicted in zip(validation.iterrows(), prediction):
            prediction_rows.append(
                {
                    "representation": name,
                    "sample_id": row["sample_id"],
                    "speaker_id": row["speaker_id"],
                    "split": row["split"],
                    "true_accent": row["accent"],
                    "predicted_accent": predicted,
                }
            )

    paths = {
        "metrics": output_dir / "b4_accent_probe_metrics.csv",
        "predictions": output_dir / "b4_accent_probe_validation_predictions.csv",
        "permutations": output_dir / "b4_accent_probe_permutations.csv",
        "bootstrap": output_dir / "b4_accent_probe_speaker_bootstrap.csv",
    }
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(prediction_rows).to_csv(paths["predictions"], index=False)
    pd.DataFrame(permutation_rows).to_csv(paths["permutations"], index=False)
    pd.DataFrame(bootstrap_rows).to_csv(paths["bootstrap"], index=False)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    )
    report = {
        "status": "PASS",
        "scope": config["interpretation"],
        "config_path": config_path.relative_to(repo_root).as_posix(),
        "config_sha256": file_sha256(config_path),
        "seed": seed,
        "split_version": config["split_version"],
        "split_manifest_sha256": file_sha256(repo_root / config["split_manifest"]),
        "feature_table_sha256": file_sha256(repo_root / config["feature_table"]),
        "git_commit": commit,
        "worktree_dirty": dirty,
        "split_audit": split_audit,
        "representations": report_representations,
        "outputs": {key: value.relative_to(repo_root).as_posix() for key, value in paths.items()},
    }
    report_path = output_dir / "b4_accent_probe_report.json"
    report["outputs"]["report"] = report_path.relative_to(repo_root).as_posix()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/features/b4_accent_probe_v1.json")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/tables/b4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    report = run_probe(root, args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
