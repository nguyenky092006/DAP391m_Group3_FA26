"""Run the B6 E1-E7 centralized adversarial and federated experiment ladder."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedecai.models.b6_common import classification_metrics, prediction_rows, write_csv_atomic, write_json_atomic
from fedecai.models.b6_deep_models import AccentInvariantCNN

ROOT = Path(__file__).resolve().parents[3]


def torch_save_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class MultiTaskLogMelDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, emotion: np.ndarray, accent: np.ndarray, frames: int) -> None:
        self.paths = [ROOT / str(value) for value in frame["log_mel_artifact_path"]]
        self.emotion, self.accent, self.frames = emotion, accent, frames

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        value = np.load(self.paths[index]).astype(np.float32)
        value = (value - value.mean()) / max(float(value.std()), 1e-6)
        output = np.zeros((80, self.frames), dtype=np.float32)
        width = min(value.shape[1], self.frames); output[:, :width] = value[:, :width]
        return torch.from_numpy(output[None]), torch.tensor(self.emotion[index]), torch.tensor(self.accent[index])


def loader(frame: pd.DataFrame, emotions: LabelEncoder, accents: LabelEncoder, frames: int, batch: int, shuffle: bool) -> DataLoader:
    dataset = MultiTaskLogMelDataset(frame, emotions.transform(frame["emotion"]), accents.transform(frame["accent_clean"]), frames)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)


def model_for(kind: str, dropout: float) -> AccentInvariantCNN:
    return AccentInvariantCNN(dropout=dropout, adversarial=kind != "baseline", conditional=kind == "conditional")


def weights(values: np.ndarray, classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(values, minlength=classes).astype(np.float32)
    return torch.tensor(len(values) / (classes * np.maximum(counts, 1)), device=device)


def evaluate(model: AccentInvariantCNN, data: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); predictions, probabilities = [], []
    with torch.inference_mode():
        for values, _, _ in data:
            logits, _ = model(values.to(device))
            predictions.append(logits.argmax(1).cpu().numpy()); probabilities.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities)


def train_epoch(
    model: AccentInvariantCNN, data: DataLoader, optimizer: torch.optim.Optimizer,
    device: torch.device, strength: float, emotion_loss: nn.Module, accent_loss: nn.Module,
    clip: float, proximal: dict[str, torch.Tensor] | None = None, mu: float = 0.0,
) -> dict[str, float]:
    model.train(); total = emotion_total = accent_total = 0.0; count = 0
    for values, emotion, accent in data:
        values, emotion, accent = values.to(device), emotion.to(device), accent.to(device)
        condition = torch.nn.functional.one_hot(emotion, 4).float() if model.conditional else None
        optimizer.zero_grad(set_to_none=True)
        emotion_logits, accent_logits = model(values, strength, condition)
        e_loss = emotion_loss(emotion_logits, emotion)
        a_loss = accent_loss(accent_logits, accent) if accent_logits is not None else torch.zeros((), device=device)
        loss = e_loss + a_loss
        if proximal is not None and mu:
            loss = loss + 0.5 * mu * sum((parameter - proximal[name]).pow(2).sum() for name, parameter in model.named_parameters())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
        size = len(emotion); count += size; total += float(loss.item()) * size
        emotion_total += float(e_loss.item()) * size; accent_total += float(a_loss.item()) * size
    return {"loss": total / count, "emotion_loss": emotion_total / count, "accent_loss": accent_total / count}


def metric_bundle(model: AccentInvariantCNN, data: DataLoader, frame: pd.DataFrame, emotions: LabelEncoder, device: torch.device):
    predicted, probabilities = evaluate(model, data, device)
    predicted_text = emotions.inverse_transform(predicted)
    metrics = classification_metrics(frame["emotion"].to_numpy(), predicted_text, frame["accent_clean"].to_numpy(), list(emotions.classes_))
    return metrics, predicted_text, probabilities


def grl_strength(step: int, total: int, warmup: int, maximum: float) -> float:
    if step <= warmup: return 0.0
    return maximum * min(1.0, (step - warmup) / max(1, total - warmup))


def train_centralized(name: str, kind: str, train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any], emotions: LabelEncoder, accents: LabelEncoder, frames: int, device: torch.device, force: bool) -> None:
    artifact_dir = ROOT / "artifacts/b6/fedecai"; report_dir = ROOT / "reports/tables/b6/fedecai"
    artifact_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    model_path, report_path = artifact_dir / f"{name}.pt", report_dir / f"{name}.json"
    progress_path = artifact_dir / f"{name}_progress.pt"
    if model_path.exists() and report_path.exists() and not force:
        print(f"{name}: complete, skipping", flush=True); return
    cfg = config["training"]; batch = int(cfg["batch_size"])
    train_loader = loader(train, emotions, accents, frames, batch, True); valid_loader = loader(validation, emotions, accents, frames, batch, False)
    seed_all(int(config["seed"])); model = model_for(kind, float(cfg["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    e_loss = nn.CrossEntropyLoss(weight=weights(emotions.transform(train["emotion"]), 4, device))
    a_loss = nn.CrossEntropyLoss(weight=weights(accents.transform(train["accent_clean"]), 3, device))
    best, best_state, history, start_epoch = -1.0, None, [], 1
    if progress_path.exists() and not force:
        saved = torch.load(progress_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer_state"])
        best, best_state, history = saved["best"], saved["best_state"], saved["history"]
        start_epoch = saved["epoch"] + 1
        print(f"{name}: resuming at epoch {start_epoch}", flush=True)
    elif force:
        torch_save_atomic({"epoch": 0, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best": best, "best_state": best_state, "history": history}, progress_path)
    started = time.perf_counter()
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        strength = grl_strength(epoch, int(cfg["epochs"]), int(cfg["warmup_epochs"]), float(cfg["adversarial_lambda"])) if kind != "baseline" else 0.0
        losses = train_epoch(model, train_loader, optimizer, device, strength, e_loss, a_loss, float(cfg["gradient_clip"]))
        metrics, _, _ = metric_bundle(model, valid_loader, validation, emotions, device); score = metrics["overall"]["macro_f1"]
        history.append({"epoch": epoch, "grl_strength": strength, **losses, "validation_macro_f1": score})
        if score > best: best, best_state = score, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch_save_atomic({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best": best, "best_state": best_state, "history": history}, progress_path)
        print(f"{name} epoch {epoch}/{cfg['epochs']} macro_f1={score:.4f} grl={strength:.4f}", flush=True)
    model.load_state_dict(best_state); metrics, predicted, probabilities = metric_bundle(model, valid_loader, validation, emotions, device)
    torch_save_atomic({"model_state": model.state_dict(), "kind": kind, "config": cfg}, model_path)
    write_csv_atomic(prediction_rows(validation, predicted, probabilities, list(emotions.classes_)), report_dir / f"{name}_validation_predictions.csv")
    write_json_atomic({"status": "PASS", "experiment": name, "kind": kind, "history": history, "validation": metrics, "runtime_seconds": time.perf_counter()-started, "test_samples_loaded": 0}, report_path)


def average_states(states: list[dict[str, torch.Tensor]], sizes: list[int]) -> dict[str, torch.Tensor]:
    total = sum(sizes); output = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            output[key] = sum(state[key] * (size / total) for state, size in zip(states, sizes))
        else:
            output[key] = states[0][key]
    return output


def train_federated(name: str, kind: str, scenario: str, train: pd.DataFrame, validation: pd.DataFrame, assignments: pd.DataFrame, config: dict[str, Any], emotions: LabelEncoder, accents: LabelEncoder, frames: int, device: torch.device, force: bool) -> None:
    full_name = f"{name}_{scenario}"; artifact_dir = ROOT / "artifacts/b6/fedecai"; report_dir = ROOT / "reports/tables/b6/fedecai"
    artifact_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    model_path, report_path, progress_path = artifact_dir/f"{full_name}.pt", report_dir/f"{full_name}.json", artifact_dir/f"{full_name}_progress.pt"
    if model_path.exists() and report_path.exists() and not force:
        print(f"{full_name}: complete, skipping", flush=True); return
    cfg, fed = config["training"], config["federated"]; client_column = f"{scenario}_client_id"
    mapping = dict(zip(assignments["speaker_id"].astype(str), assignments[client_column]))
    working = train.copy(); working["client_id"] = working["speaker_id"].astype(str).map(mapping)
    if working["client_id"].isna().any(): raise ValueError(f"Missing {scenario} client assignment")
    valid_loader = loader(validation, emotions, accents, frames, int(cfg["batch_size"]), False)
    seed_all(int(config["seed"])); global_model = model_for(kind, float(cfg["dropout"])).to(device)
    start_round, history, best, best_state = 1, [], -1.0, None
    if progress_path.exists() and not force:
        saved = torch.load(progress_path, map_location=device, weights_only=False); global_model.load_state_dict(saved["model_state"])
        start_round, history, best, best_state = saved["round"] + 1, saved["history"], saved["best"], saved["best_state"]
    elif force:
        torch_save_atomic({"round": 0, "model_state": global_model.state_dict(), "best": best, "best_state": best_state, "history": history}, progress_path)
    started = time.perf_counter()
    for round_index in range(start_round, int(fed["rounds"]) + 1):
        global_state = {key: value.detach().clone() for key, value in global_model.state_dict().items()}; states, sizes, client_losses = [], [], []
        strength = grl_strength(round_index, int(fed["rounds"]), int(cfg["warmup_epochs"]), float(cfg["adversarial_lambda"])) if kind != "baseline" else 0.0
        for client_id in sorted(working["client_id"].unique()):
            local_frame = working[working["client_id"] == client_id].reset_index(drop=True); local = model_for(kind, float(cfg["dropout"])).to(device); local.load_state_dict(global_state)
            data = loader(local_frame, emotions, accents, frames, int(cfg["batch_size"]), True)
            optimizer = torch.optim.AdamW(local.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
            e_loss = nn.CrossEntropyLoss(weight=weights(emotions.transform(local_frame["emotion"]), 4, device)); a_loss = nn.CrossEntropyLoss(weight=weights(accents.transform(train["accent_clean"]), 3, device))
            losses = None
            proximal = {key: value.detach() for key, value in global_model.named_parameters()} if name == "E5_fedprox_cnn" else None
            for _ in range(int(fed["local_epochs"])):
                losses = train_epoch(local, data, optimizer, device, strength, e_loss, a_loss, float(cfg["gradient_clip"]), proximal, float(fed["fedprox_mu"]))
            states.append({key: value.detach().clone() for key, value in local.state_dict().items()}); sizes.append(len(local_frame)); client_losses.append({"client": client_id, **losses})
        global_model.load_state_dict(average_states(states, sizes)); metrics, _, _ = metric_bundle(global_model, valid_loader, validation, emotions, device); score = metrics["overall"]["macro_f1"]
        history.append({"round": round_index, "grl_strength": strength, "validation_macro_f1": score, "clients": client_losses})
        if score > best: best, best_state = score, {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
        torch_save_atomic({"round": round_index, "model_state": global_model.state_dict(), "best": best, "best_state": best_state, "history": history}, progress_path)
        print(f"{full_name} round {round_index}/{fed['rounds']} macro_f1={score:.4f} grl={strength:.4f}", flush=True)
    global_model.load_state_dict(best_state); metrics, predicted, probabilities = metric_bundle(global_model, valid_loader, validation, emotions, device)
    torch_save_atomic({"model_state": global_model.state_dict(), "kind": kind, "scenario": scenario, "config": config}, model_path)
    write_csv_atomic(prediction_rows(validation, predicted, probabilities, list(emotions.classes_)), report_dir/f"{full_name}_validation_predictions.csv")
    write_json_atomic({"status":"PASS","experiment":name,"scenario":scenario,"kind":kind,"rounds":history,"validation":metrics,"runtime_seconds":time.perf_counter()-started,"test_samples_loaded":0}, report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--force", action="store_true"); parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--device", choices=("cuda","cpu"), default="cuda"); return parser.parse_args()


def main() -> None:
    args = parse_args(); config=json.loads((ROOT/"configs/models/b6_fedecai_v1.json").read_text()); frame=pd.read_parquet(ROOT/"data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    manifest=pd.read_csv(ROOT/"data/manifests/split_manifest_v2_enriched.csv",dtype={"sample_id":"string","speaker_id":"string"}); data=manifest.merge(frame[["sample_id","log_mel_artifact_path"]],on="sample_id",validate="one_to_one")
    train=data[data["split"]=="train"].reset_index(drop=True); validation=data[data["split"]=="validation"].reset_index(drop=True); assignments=pd.read_csv(ROOT/"data/manifests/federated_speaker_partitions_v1.csv",dtype={"speaker_id":"string"})
    device=torch.device(args.device); 
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; use .venv-gpu")
    paths=[ROOT/str(value) for value in train["log_mel_artifact_path"]]; missing=[path for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(missing[0])
    frames=min(int(np.percentile([np.load(path,mmap_mode="r").shape[1] for path in paths],95)),int(config["training"]["max_frames"])); emotions=LabelEncoder().fit(config["emotion_order"]); accents=LabelEncoder().fit(config["accent_order"])
    print(json.dumps({"status":"PASS","device":str(device),"train":len(train),"validation":len(validation),"frames":frames,"test_loaded":False}),flush=True)
    if args.preflight_only:return
    for name,kind in (("E1_cnn","baseline"),("E2_unconditional_grl","unconditional"),("E3_emotion_conditioned_grl","conditional")):
        train_centralized(name,kind,train,validation,config,emotions,accents,frames,device,args.force)
    for name,kind in (("E4_fedavg_cnn","baseline"),("E5_fedprox_cnn","baseline"),("E6_fedavg_unconditional_grl","unconditional"),("E7_fedecai","conditional")):
        for scenario in config["federated"]["scenarios"]: train_federated(name,kind,scenario,train,validation,assignments,config,emotions,accents,frames,device,args.force)
    print(json.dumps({"status":"PASS","centralized_experiments":3,"federated_experiments":8,"test_untouched":True},indent=2))


if __name__ == "__main__": main()
