"""Train a Prosody-Fusion ablation with no access to F0 mean or F0 SD.

The architecture, split, optimizer, waveform branch, and four-dimensional
prosody interface remain unchanged.  The two F0 coordinates are forced to zero
during train, validation, test, and later intervention inference.  This avoids
the distribution-shift flaw of masking F0 only at test time.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from cf_ser.config import ARTIFACTS, CHECKPOINTS, LABELS, SPLIT_CSV
from cf_ser.data import AudioDataset
from cf_ser.models import ProsodyFusion
from cf_ser.train import fit_norm, prosody_tensor, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = CHECKPOINTS / "prosody_no_f0_ablation"
OUT.mkdir(parents=True, exist_ok=True)


def no_f0_tensor(paths, feature_map, norm):
    values = prosody_tensor(paths, feature_map, norm, DEVICE)
    values[:, :2] = 0.0
    return values


@torch.no_grad()
def evaluate(model, loader, feature_map, norm):
    model.eval()
    labels, predictions = [], []
    for audio, target, paths in loader:
        logits = model(audio.to(DEVICE), no_f0_tensor(paths, feature_map, norm))
        labels.extend(target.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
        "per_class_f1": f1_score(labels, predictions, average=None, labels=range(len(LABELS))),
        "confusion": confusion_matrix(labels, predictions, labels=range(len(LABELS))),
    }


def train_one(seed, frame, features):
    set_seed(seed)
    model = ProsodyFusion().to(DEVICE)
    feature_map = {str(row.path): row._asdict() for row in features.itertuples(index=False)}
    train = frame[frame.split == "train"]
    val = frame[frame.split == "val"]
    test = frame[frame.split == "test"]
    norm = fit_norm(train, features)
    loaders = {
        "train": DataLoader(AudioDataset(train, True), batch_size=32, shuffle=True, num_workers=0, pin_memory=True),
        "val": DataLoader(AudioDataset(val), batch_size=32, num_workers=0, pin_memory=True),
        "test": DataLoader(AudioDataset(test), batch_size=32, num_workers=0, pin_memory=True),
    }
    counts = train.emotion.value_counts()
    weights = torch.tensor([len(train) / (4 * counts[label]) for label in LABELS], dtype=torch.float32, device=DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    best = None
    bad_epochs = 0
    history = []
    started = time.time()
    for epoch in range(20):
        model.train()
        loaders["train"].dataset.epoch = epoch
        losses = []
        for audio, target, paths in loaders["train"]:
            audio = audio.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(audio, no_f0_tensor(paths, feature_map, norm)), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = evaluate(model, loaders["val"], feature_map, norm)
        history.append({"epoch": epoch + 1, "train_loss": np.mean(losses), **{f"val_{k}": v for k, v in validation.items() if np.isscalar(v)}})
        print(f"prosody_no_f0 seed={seed} epoch={epoch + 1} val_f1={validation['macro_f1']:.4f}", flush=True)
        if best is None or validation["macro_f1"] > best[0] + 1e-4:
            best = (validation["macro_f1"], epoch + 1, {key: value.detach().cpu() for key, value in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= 6:
            break

    model.load_state_dict(best[2])
    test_result = evaluate(model, loaders["test"], feature_map, norm)
    checkpoint = {
        "model_key": "prosody_fusion_no_f0", "state_dict": model.state_dict(),
        "labels": LABELS, "prosody_norm": norm, "f0_mask_indices": [0, 1],
        "seed": seed, "lr": 8e-4, "best_epoch": best[1],
        "val_macro_f1": best[0], "test_macro_f1": test_result["macro_f1"],
        "test_accuracy": test_result["accuracy"], "seconds": time.time() - started,
    }
    path = OUT / f"prosody_fusion_no_f0_seed{seed}.pt"
    torch.save(checkpoint, path)
    pd.DataFrame(history).to_csv(OUT / f"training_history_seed{seed}.csv", index=False)
    pd.DataFrame(test_result["confusion"], index=LABELS, columns=LABELS).to_csv(OUT / f"confusion_seed{seed}.csv")
    return {
        "model": "Prosody-Fusion-noF0", "seed": seed, "checkpoint": str(path),
        "best_epoch": best[1], "val_macro_f1": best[0],
        "test_macro_f1": test_result["macro_f1"], "test_accuracy": test_result["accuracy"],
        **{f"f1_{label}": float(value) for label, value in zip(LABELS, test_result["per_class_f1"])},
        "seconds": checkpoint["seconds"],
    }


def main():
    frame = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    features = pd.read_csv(ARTIFACTS / "audio_features.csv")
    rows = [train_one(seed, frame, features) for seed in (2025, 2026, 2027)]
    results = pd.DataFrame(rows)
    results.to_csv(ARTIFACTS / "PROSODY_NO_F0_BASELINE_RESULTS.csv", index=False)
    aggregate = {
        "n_seeds": len(results),
        "mean_test_macro_f1": float(results.test_macro_f1.mean()),
        "sd_test_macro_f1": float(results.test_macro_f1.std(ddof=1)),
        "mean_test_accuracy": float(results.test_accuracy.mean()),
    }
    (ARTIFACTS / "PROSODY_NO_F0_BASELINE_SUMMARY.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
