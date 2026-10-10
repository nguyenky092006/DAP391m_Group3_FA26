"""Fine-tune one Vietnamese Wav2Vec2 baseline on the frozen ViSEC split.

This script deliberately reads ``data/splits.csv`` directly.  It never
regenerates the split and it does not touch any RQ1 artifact.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from cf_ser.config import ARTIFACTS, LABELS, LABEL_TO_ID, NUM_SAMPLES, SAMPLE_RATE
from cf_ser.data import fix_length, load_audio


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ViSECDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, training: bool, seed: int):
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        waveform = load_audio(row.path)
        rng = np.random.default_rng(self.seed + self.epoch * 100_003 + index)
        waveform = fix_length(waveform, training=self.training, rng=rng)
        return waveform, LABEL_TO_ID[row.emotion], str(row.path)


class Collator:
    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, batch):
        waveforms, labels, paths = zip(*batch)
        encoded = self.extractor(
            list(waveforms), sampling_rate=SAMPLE_RATE, padding=True,
            truncation=True, max_length=NUM_SAMPLES, return_tensors="pt",
        )
        return encoded, torch.tensor(labels, dtype=torch.long), list(paths)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    labels, predictions, probabilities, paths = [], [], [], []
    total_loss = 0.0
    batches = 0
    for encoded, target, batch_paths in loader:
        inputs = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
        target = target.to(device, non_blocking=True)
        logits = model(**inputs).logits
        if criterion is not None:
            total_loss += float(criterion(logits, target).item())
            batches += 1
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        labels.extend(target.cpu().tolist())
        predictions.extend(probs.argmax(axis=1).tolist())
        probabilities.append(probs)
        paths.extend(batch_paths)
    return {
        "loss": total_loss / max(1, batches),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
        "labels": np.asarray(labels),
        "predictions": np.asarray(predictions),
        "probabilities": np.concatenate(probabilities),
        "paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--publish-artifacts", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Wav2Vec2 fine-tuning run")
    device = torch.device("cuda")

    split_path = ROOT / "data" / "splits.csv"
    split = pd.read_csv(split_path, dtype={"speaker": str})
    required = {"path", "emotion", "emotion_id", "speaker", "accent", "split"}
    if required - set(split.columns):
        raise ValueError(f"Frozen split is missing columns: {sorted(required-set(split.columns))}")
    if set(split.split) != {"train", "val", "test"}:
        raise ValueError("Frozen split must contain train, val, and test")
    if split.path.duplicated().any():
        raise ValueError("Frozen split contains duplicate paths")
    expected_map = split.groupby("emotion").emotion_id.first().to_dict()
    if expected_map != LABEL_TO_ID:
        raise ValueError(f"Label mapping mismatch: {expected_map} != {LABEL_TO_ID}")
    for path in split.path:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    id2label = {i: label for i, label in enumerate(LABELS)}
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    elif hasattr(model, "wav2vec2") and hasattr(model.wav2vec2, "feature_extractor"):
        model.wav2vec2.feature_extractor._freeze_parameters()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    datasets = {
        name: ViSECDataset(split[split.split == name], name == "train", args.seed)
        for name in ("train", "val", "test")
    }
    collator = Collator(extractor)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size if name == "train" else max(args.batch_size, 4),
            shuffle=name == "train",
            num_workers=0,
            pin_memory=True,
            collate_fn=collator,
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }

    counts = split[split.split == "train"].emotion.value_counts()
    class_weights = torch.tensor(
        [len(datasets["train"]) / (len(LABELS) * counts[label]) for label in LABELS],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    checkpoint_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "checkpoints" / f"wav2vec2_seed{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -np.inf
    bad_epochs = 0
    started = time.time()

    for epoch in range(args.epochs):
        datasets["train"].epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, (encoded, target, _) in enumerate(loaders["train"]):
            inputs = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs).logits
                loss = criterion(logits, target) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item()) * args.grad_accum

        val = evaluate(model, loaders["val"], device, criterion)
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": running / len(loaders["train"]),
            "val_loss": val["loss"],
            "val_macro_f1": val["macro_f1"],
            "val_accuracy": val["accuracy"],
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        if val["macro_f1"] > best_f1 + 1e-4:
            best_f1 = val["macro_f1"]
            bad_epochs = 0
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            extractor.save_pretrained(checkpoint_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    best_model = AutoModelForAudioClassification.from_pretrained(checkpoint_dir).to(device)
    test = evaluate(best_model, loaders["test"], device, criterion)
    per_class = f1_score(test["labels"], test["predictions"], average=None, labels=range(len(LABELS)))
    confusion = confusion_matrix(test["labels"], test["predictions"], labels=range(len(LABELS)))

    metrics = {
        "model_name": args.model_name,
        "architecture_name": "Wav2Vec2",
        "seed": args.seed,
        "split_file": str(split_path),
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "train_n": len(datasets["train"]),
        "val_n": len(datasets["val"]),
        "test_n": len(datasets["test"]),
        "best_val_macro_f1": best_f1,
        "test_macro_f1": test["macro_f1"],
        "test_accuracy": test["accuracy"],
        "per_class_f1": {label: float(value) for label, value in zip(LABELS, per_class)},
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - started,
        "training_args": vars(args),
    }
    (checkpoint_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(checkpoint_dir / "training_history.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).to_csv(checkpoint_dir / "confusion_matrix.csv")
    pd.DataFrame({"label": LABELS, "f1": per_class}).to_csv(checkpoint_dir / "per_class_f1.csv", index=False)
    if args.publish_artifacts:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "model": "Wav2Vec2", "pretrained_checkpoint": args.model_name,
            "seed": args.seed, "best_epoch": int(np.argmax([row["val_macro_f1"] for row in history])) + 1,
            "best_val_macro_f1": best_f1, "test_macro_f1": test["macro_f1"],
            "test_accuracy": test["accuracy"],
            **{f"f1_{label}": float(value) for label, value in zip(LABELS, per_class)},
        }]).to_csv(ARTIFACTS / "WAV2VEC2_BASELINE_RESULTS.csv", index=False)
        pd.DataFrame(confusion, index=LABELS, columns=[f"pred_{label}" for label in LABELS]).rename_axis("true_label").to_csv(
            ARTIFACTS / "WAV2VEC2_CONFUSION_MATRIX.csv"
        )
    prediction_rows = []
    for path, true_id, pred_id, probs in zip(test["paths"], test["labels"], test["predictions"], test["probabilities"]):
        row = {"path": path, "true_label": LABELS[int(true_id)], "prediction": LABELS[int(pred_id)]}
        row.update({f"p_{label}": float(probs[i]) for i, label in enumerate(LABELS)})
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(checkpoint_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
