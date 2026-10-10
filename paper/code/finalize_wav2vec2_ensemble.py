"""Aggregate independently fine-tuned Wav2Vec2 checkpoints without test tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from cf_ser.config import ARTIFACTS, LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()
    checkpoint_dirs = [Path(path).resolve() for path in args.checkpoints]
    if len(checkpoint_dirs) < 2:
        raise ValueError("Ensemble aggregation requires at least two checkpoints")

    summaries = []
    frames = []
    for directory in checkpoint_dirs:
        summary_path = directory / "training_summary.json"
        prediction_path = directory / "test_predictions.csv"
        if not summary_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(f"Incomplete checkpoint outputs: {directory}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["checkpoint_dir"] = str(directory)
        summaries.append(summary)
        frame = pd.read_csv(prediction_path)
        frame["seed"] = int(summary["seed"])
        frames.append(frame)

    probability_columns = [f"p_{label}" for label in LABELS]
    reference = frames[0][["path", "true_label"]].sort_values("path").reset_index(drop=True)
    for frame in frames[1:]:
        current = frame[["path", "true_label"]].sort_values("path").reset_index(drop=True)
        if not reference.equals(current):
            raise AssertionError("Checkpoint test predictions do not cover the same samples and labels")

    stacked = pd.concat([frame[["path", "true_label", *probability_columns]] for frame in frames], ignore_index=True)
    ensemble = stacked.groupby(["path", "true_label"], as_index=False)[probability_columns].mean()
    probability = ensemble[probability_columns].to_numpy()
    ensemble["prediction"] = [LABELS[index] for index in probability.argmax(axis=1)]
    y_true = [LABELS.index(label) for label in ensemble.true_label]
    y_pred = [LABELS.index(label) for label in ensemble.prediction]
    per_class = f1_score(y_true, y_pred, labels=range(len(LABELS)), average=None)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    accuracy = float(accuracy_score(y_true, y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))

    member_rows = []
    for summary in summaries:
        member_rows.append({
            "seed": summary["seed"], "checkpoint_dir": summary["checkpoint_dir"],
            "best_val_macro_f1": summary["best_val_macro_f1"],
            "test_macro_f1": summary["test_macro_f1"], "test_accuracy": summary["test_accuracy"],
            **{f"f1_{label}": summary["per_class_f1"][label] for label in LABELS},
        })
    member_frame = pd.DataFrame(member_rows).sort_values("seed")
    member_frame.to_csv(ARTIFACTS / "WAV2VEC2_ENSEMBLE_MEMBERS.csv", index=False)
    result = pd.DataFrame([{
        "model": "Wav2Vec2 probability ensemble", "n_members": len(checkpoint_dirs),
        "seeds": ";".join(str(int(value)) for value in member_frame.seed),
        "test_macro_f1": macro_f1, "test_accuracy": accuracy,
        **{f"f1_{label}": float(value) for label, value in zip(LABELS, per_class)},
        "member_val_macro_f1_mean": member_frame.best_val_macro_f1.mean(),
        "member_val_macro_f1_sd": member_frame.best_val_macro_f1.std(ddof=1),
        "member_test_macro_f1_mean": member_frame.test_macro_f1.mean(),
        "member_test_macro_f1_sd": member_frame.test_macro_f1.std(ddof=1),
    }])
    result.to_csv(ARTIFACTS / "WAV2VEC2_ENSEMBLE_BASELINE_RESULTS.csv", index=False)
    pd.DataFrame(matrix, index=LABELS, columns=[f"pred_{label}" for label in LABELS]).rename_axis("true_label").to_csv(
        ARTIFACTS / "WAV2VEC2_ENSEMBLE_CONFUSION_MATRIX.csv"
    )
    ensemble.to_csv(ARTIFACTS / "WAV2VEC2_ENSEMBLE_TEST_PREDICTIONS.csv", index=False)
    config = {
        "selection_rule": "learning rate selected by validation Macro-F1 only; checkpoints use identical protocol",
        "pretrained_checkpoint": summaries[0]["model_name"],
        "labels": LABELS,
        "checkpoint_dirs": [str(path) for path in checkpoint_dirs],
        "seeds": [int(summary["seed"]) for summary in summaries],
        "learning_rates": [summary["training_args"]["learning_rate"] for summary in summaries],
        "ensemble_method": "unweighted arithmetic mean of class probabilities",
    }
    (ARTIFACTS / "wav2vec2_ensemble_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(member_frame.to_string(index=False))


if __name__ == "__main__":
    main()
