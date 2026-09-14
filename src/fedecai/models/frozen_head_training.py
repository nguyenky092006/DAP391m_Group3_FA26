"""Shared lightweight classifier training for frozen cached representations."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedecai.models.b6_deep_models import FrozenEmbeddingHead


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loaders(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_labels: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_set = TensorDataset(
        torch.from_numpy(np.asarray(train_embeddings, dtype=np.float32)),
        torch.from_numpy(np.asarray(train_labels, dtype=np.int64)),
    )
    validation_set = TensorDataset(
        torch.from_numpy(np.asarray(validation_embeddings, dtype=np.float32)),
        torch.from_numpy(np.asarray(validation_labels, dtype=np.int64)),
    )
    options = {
        "batch_size": int(batch_size), "num_workers": int(num_workers),
        "pin_memory": True,
    }
    return (
        DataLoader(train_set, shuffle=True, **options),
        DataLoader(validation_set, shuffle=False, **options),
    )


def evaluate_head(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions, probabilities, loss_sum = [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for embeddings, labels in loader:
            logits = model(embeddings.to(device, non_blocking=True))
            loss_sum += float(criterion(logits, labels.to(device)).item()) * len(labels)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
    return (
        np.concatenate(predictions), np.concatenate(probabilities),
        loss_sum / len(loader.dataset),
    )


def train_frozen_head(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_labels: np.ndarray,
    *,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    prefix: str,
) -> tuple[FrozenEmbeddingHead, list[dict[str, Any]], int, np.ndarray, np.ndarray]:
    seed_all(seed)
    train_loader, validation_loader = loaders(
        train_embeddings, train_labels, validation_embeddings, validation_labels,
        batch_size=int(config["batch_size"]),
        num_workers=int(config.get("num_workers", 0)),
    )
    model = FrozenEmbeddingHead(
        dimension=int(train_embeddings.shape[1]), classes=4,
        dropout=float(config["dropout"]),
    ).to(device)
    counts = np.bincount(train_labels, minlength=4).astype(np.float32)
    weights = len(train_labels) / (4 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        for embeddings, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
        predicted, _, validation_loss = evaluate_head(model, validation_loader, device)
        score = float(f1_score(validation_labels, predicted, average="macro", zero_division=0))
        history.append({
            "epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset),
            "validation_loss": validation_loss, "validation_macro_f1": score,
        })
        print(
            f"{prefix} epoch {epoch}/{config['max_epochs']} "
            f"train_loss={history[-1]['train_loss']:.4f} val_macro_f1={score:.4f}",
            flush=True,
        )
        if score > best_score + float(config["min_delta"]):
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= int(config["patience"]):
                print(f"{prefix} early stopping at epoch {epoch}; best={best_epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("Frozen-head training produced no checkpoint")
    model.load_state_dict(best_state)
    predicted, probabilities, _ = evaluate_head(model, validation_loader, device)
    return model, history, best_epoch, predicted, probabilities
