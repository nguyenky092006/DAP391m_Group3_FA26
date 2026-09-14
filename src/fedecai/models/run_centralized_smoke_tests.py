"""Run tiny implementation-only smoke tests for the eight centralized paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from fedecai.features.mel_3ch import build_mel_3ch, pad_or_crop_mel_3ch
from fedecai.models.b6_common import load_b6_table, load_config, write_json_atomic
from fedecai.models.b6_deep_models import LogMelCNN2D, MelImageClassifier


ROOT = Path(__file__).resolve().parents[3]
MODELS = (
    "svm_rbf", "xgboost", "cnn2d_logmel", "resnet18_mel3",
    "mobilenet_v3_mel3", "vit_b16_mel3", "wav2vec2_raw",
    "pitch_fusion_paper_aligned",
)


def actual_training_subset(frame, feature_columns):
    train = frame[frame["split"] == "train"].copy()
    subset = train.groupby("emotion", sort=True).head(3).reset_index(drop=True)
    encoder = LabelEncoder().fit(["angry", "happy", "neutral", "sad"])
    values = subset[feature_columns].to_numpy(dtype=np.float32)
    labels = encoder.transform(subset["emotion"])
    return subset, values, labels


def smoke_svm(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    scaled = StandardScaler().fit_transform(values)
    model = SVC(kernel="rbf", C=1.0, gamma="scale").fit(scaled, labels)
    predicted = model.predict(scaled[:2])
    return {"samples": len(labels), "prediction_shape": list(predicted.shape)}


def smoke_xgboost(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    model = XGBClassifier(
        objective="multi:softprob", num_class=4, n_estimators=2,
        max_depth=2, learning_rate=0.1, tree_method="hist", n_jobs=1,
        random_state=391,
    ).fit(values, labels)
    probabilities = model.predict_proba(values[:2])
    return {"samples": len(labels), "probability_shape": list(probabilities.shape)}


def smoke_neural(name: str, log_mel_path: Path, device: torch.device) -> dict[str, Any]:
    mel = np.load(log_mel_path).astype(np.float32)
    if name == "cnn2d_logmel":
        normalized = (mel - mel.mean()) / max(float(mel.std()), 1e-6)
        width = min(normalized.shape[1], 128)
        value = np.zeros((1, 80, 128), dtype=np.float32)
        value[0, :, :width] = normalized[:, :width]
        model = LogMelCNN2D()
    else:
        value = pad_or_crop_mel_3ch(build_mel_3ch(mel), 128)
        backbone = {
            "resnet18_mel3": "resnet18",
            "mobilenet_v3_mel3": "mobilenet_v3_small",
            "vit_b16_mel3": "vit_b_16",
        }[name]
        model = MelImageClassifier(backbone)
    model = model.to(device).train()
    batch = torch.from_numpy(value[None]).to(device)
    target = torch.tensor([0], device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = torch.nn.CrossEntropyLoss()(logits, target)
    loss.backward(); optimizer.step()
    if tuple(logits.shape) != (1, 4) or not torch.isfinite(logits).all():
        raise RuntimeError(f"Invalid {name} smoke output")
    result = {"input_shape": list(batch.shape), "logits_shape": list(logits.shape)}
    del optimizer, model, batch, target, logits, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def smoke_subprocess(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return {"command": command, "stdout_tail": completed.stdout.strip().splitlines()[-1:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", default="artifacts/audit/centralized_smoke_tests.json")
    args = parser.parse_args()
    if "DAP301m" not in str(Path(sys.executable).resolve()):
        raise RuntimeError(f"Smoke tests must use DAP301m, got {sys.executable}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    config = load_config(ROOT / "configs/models/b6_classical_centralized_v1.json")
    frame, feature_columns = load_b6_table(
        ROOT / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet",
        ROOT / "data/manifests/split_manifest_v2_enriched.csv", config,
    )
    subset, values, labels = actual_training_subset(frame, feature_columns)
    log_mel = ROOT / str(subset.iloc[0]["log_mel_artifact_path"])
    tasks: dict[str, Callable[[], dict[str, Any]]] = {
        "svm_rbf": lambda: smoke_svm(values, labels),
        "xgboost": lambda: smoke_xgboost(values, labels),
        "cnn2d_logmel": lambda: smoke_neural("cnn2d_logmel", log_mel, device),
        "resnet18_mel3": lambda: smoke_neural("resnet18_mel3", log_mel, device),
        "mobilenet_v3_mel3": lambda: smoke_neural("mobilenet_v3_mel3", log_mel, device),
        "vit_b16_mel3": lambda: smoke_subprocess([
            sys.executable, str(ROOT / "src/fedecai/models/run_b6_vit_cached.py"),
            "--smoke-test", "--device", args.device,
        ]),
        "wav2vec2_raw": lambda: smoke_subprocess([
            sys.executable, str(ROOT / "src/fedecai/models/run_b6_wav2vec2_raw.py"),
            "--smoke-test", "--device", args.device,
        ]),
        "pitch_fusion_paper_aligned": lambda: smoke_subprocess([
            sys.executable, str(ROOT / "src/fedecai/models/run_b6_pitch_fusion_paper.py"),
            "--smoke-test", "--device", args.device,
        ]),
    }
    results = []
    for name in MODELS:
        try:
            evidence = tasks[name]()
            row = {"model": name, "status": "SMOKE_TEST_ONLY", "passed": True, **evidence}
        except Exception as error:  # preserve all path-level failures in one run
            row = {
                "model": name, "status": "SMOKE_TEST_ONLY", "passed": False,
                "error": repr(error), "traceback": traceback.format_exc(),
            }
        results.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "traceback"}), flush=True)
    output = ROOT / args.output
    report = {
        "schema_version": "fedecai.centralized_smoke.v1",
        "status": "PASS" if all(row["passed"] for row in results) else "FAIL",
        "evidence_class": "SMOKE_TEST_ONLY",
        "python_executable": sys.executable, "device": str(device),
        "models": results,
        "benchmark_metrics_accepted": False,
        "test_split_loaded": False,
    }
    write_json_atomic(report, output)
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
