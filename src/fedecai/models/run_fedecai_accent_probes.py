"""Apply the existing B4 train-to-validation accent probe to E1-E7 features."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from fedecai.models.b6_common import write_json_atomic
from fedecai.models.run_fedecai_ladder import (
    ROOT, FedECAIMelDataset, build_model, load_config, load_development_frame,
    load_e1_state, target_frames,
)


EXPERIMENTS = {
    "E1": ("e1_locked_b6", "artifacts/checkpoints/b6_resnet18_mel3_finetune_seed391.pt"),
    "E2": ("e2_resnet18_unconditional_grl", "artifacts/fedecai/e2_resnet18_unconditional_grl/best.pt"),
    "E3": ("e3_resnet18_conditional_grl", "artifacts/fedecai/e3_resnet18_conditional_grl/best.pt"),
    "E4": ("e4_fedavg_resnet18", "artifacts/fedecai/e4_fedavg_resnet18/best_main.pt"),
    "E5": ("e5_fedprox_resnet18", "artifacts/fedecai/e5_fedprox_resnet18/best_main.pt"),
    "E6": ("e6_fedavg_unconditional_grl", "artifacts/fedecai/e6_fedavg_unconditional_grl/best_main.pt"),
    "E7": ("e7_fedecai_resnet18", "artifacts/fedecai/e7_fedecai_resnet18/best_main.pt"),
}


def probe_model(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="lbfgs", C=1.0, class_weight="balanced", max_iter=2000,
            tol=0.001, random_state=seed,
        )),
    ])


def fit(model: Pipeline, x: np.ndarray, y: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)


def extract(model, frame: pd.DataFrame, frames: int, config: dict, device: torch.device) -> np.ndarray:
    dataset = FedECAIMelDataset(frame, frames, config["emotion_classes"], config["accent_classes"])
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    values = []
    model.eval()
    with torch.inference_mode():
        for mel, _, _ in loader:
            values.append(model.forward_features(mel.to(device)).float().cpu().numpy())
    return np.concatenate(values)


def load_step_model(step: str, device: torch.device):
    _, base = load_config("E2" if step == "E1" else step)
    config = dict(base)
    if step == "E1":
        config["adversary"] = "none"
    model = build_model(config)
    checkpoint_path = ROOT / EXPERIMENTS[step][1]
    if step == "E1":
        state, _ = load_e1_state()
        model.emotion_model.network.load_state_dict(state, strict=True)
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model_state"]
        model.load_state_dict(state, strict=True)
    return model.to(device), config, checkpoint_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in DAP301m")
    device = torch.device("cuda")
    train, validation = load_development_frame()
    _, base = load_config("E2")
    frames = target_frames(train, int(base["max_frames"]))
    y_train = train["accent_clean"].astype(str).to_numpy()
    y_validation = validation["accent_clean"].astype(str).to_numpy()
    labels = base["accent_classes"]
    majority = Counter(y_train).most_common(1)[0][0]
    majority_prediction = np.full(len(validation), majority, dtype=object)
    majority_macro = float(f1_score(y_validation, majority_prediction, labels=labels, average="macro", zero_division=0))
    majority_uar = float(recall_score(y_validation, majority_prediction, labels=labels, average="macro", zero_division=0))
    rows = []
    for index, step in enumerate(EXPERIMENTS):
        model, config, checkpoint_path = load_step_model(step, device)
        x_train = extract(model, train, frames, config, device)
        x_validation = extract(model, validation, frames, config, device)
        observed_model = probe_model(391 + index)
        fit(observed_model, x_train, y_train)
        predicted = observed_model.predict(x_validation)
        observed_macro = float(f1_score(y_validation, predicted, labels=labels, average="macro", zero_division=0))
        observed_uar = float(recall_score(y_validation, predicted, labels=labels, average="macro", zero_division=0))
        rng = np.random.default_rng(391 + 10_000 * (index + 1))
        permutation_macro, permutation_uar = [], []
        for permutation in range(50):
            permuted_model = probe_model(391 + index * 1000 + permutation + 1)
            fit(permuted_model, x_train, rng.permutation(y_train))
            permuted = permuted_model.predict(x_validation)
            permutation_macro.append(f1_score(y_validation, permuted, labels=labels, average="macro", zero_division=0))
            permutation_uar.append(recall_score(y_validation, permuted, labels=labels, average="macro", zero_division=0))
        row = {
            "experiment": step, "experiment_id": (
                "b6_resnet18_mel3_finetune_seed391" if step == "E1" else config["experiment_id"]
            ),
            "representation": "resnet18_pooled_pre_fc_512",
            "accent_probe_macro_f1": observed_macro, "accent_probe_uar": observed_uar,
            "majority_baseline_macro_f1": majority_macro, "majority_baseline_uar": majority_uar,
            "permutation_resamples": 50,
            "permutation_macro_f1_mean": float(np.mean(permutation_macro)),
            "permutation_macro_f1_std": float(np.std(permutation_macro, ddof=1)),
            "permutation_uar_mean": float(np.mean(permutation_uar)),
            "permutation_uar_std": float(np.std(permutation_uar, ddof=1)),
            "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
            "train_samples": len(train), "validation_samples": len(validation),
            "speaker_overlap_limitation": True, "test_samples_loaded": 0,
        }
        rows.append(row)
        artifact_root = ROOT / ("artifacts/fedecai/e1_locked_b6" if step == "E1" else config["artifact_root"])
        write_json_atomic(row, artifact_root / "accent_probe.json")
        metric_path = None if step == "E1" else artifact_root / "metrics.json"
        if metric_path and metric_path.is_file():
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            metric["accent_probe_macro_f1"] = observed_macro
            metric["accent_probe_uar"] = observed_uar
            metric["accent_probe_majority_macro_f1"] = majority_macro
            metric["accent_probe_permutation_macro_f1_mean"] = row["permutation_macro_f1_mean"]
            write_json_atomic(metric, metric_path)
        print(f"{step} accent_probe_macro_f1={observed_macro:.6f} uar={observed_uar:.6f}", flush=True)
        del model
        torch.cuda.empty_cache()
    output = ROOT / "reports/tables/fedecai/accent_probe_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"status": "PASS", "experiments": len(rows), "test_samples_loaded": 0}, indent=2))


if __name__ == "__main__":
    main()
