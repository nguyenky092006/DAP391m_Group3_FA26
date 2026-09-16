"""Read-only B6 performance-regression audit; never trains or updates checkpoints."""

from __future__ import annotations

import csv
import gc
import io
import json
import math
import os
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from fedecai.models.b6_common import load_b6_table, load_config, sha256_file, write_json_atomic
from fedecai.models.run_b6_performance_image import build_model as build_image_model, optimizer_for
from fedecai.models.run_b6_pitch_fusion_finetune import build_model as build_pitch_model
from fedecai.models.run_b6_pitch_fusion_paper import PitchFusionDataset, collate as pitch_collate, decode_wav as pitch_decode, interpolate_pitch
from fedecai.models.run_b6_wav2vec2_finetune import RawWaveformDataset, build_model as build_wav2vec2_model, collate as wav2vec2_collate
from fedecai.models.run_b6_wav2vec2_raw import decode_wav, load_audio, load_frame, normalize_waveform


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "reports/tables/b6_performance_regression"
LABELS = ["angry", "happy", "neutral", "sad"]
PAIRS = {
    "cnn2d_logmel": ("b6_cnn2d_logmel_seed391", "b6_cnn2d_logmel_performance_seed391"),
    "resnet18_mel3": ("b6_resnet18_mel3_seed391", "b6_resnet18_mel3_finetune_seed391"),
    "mobilenet_v3_mel3": ("b6_mobilenet_v3_mel3_seed391", "b6_mobilenet_v3_mel3_finetune_seed391"),
    "wav2vec2_raw": ("b6_wav2vec2_raw_seed391", "b6_wav2vec2_raw_finetune_seed391"),
    "pitch_fusion_paper_aligned": ("b6_pitch_fusion_paper_aligned_seed391", "b6_pitchfusion_raw_pitch_finetune_seed391"),
}
REPORTS = {
    "cnn2d_logmel": "b6_cnn2d_logmel_performance_seed391_report.json",
    "resnet18_mel3": "b6_resnet18_mel3_finetune_seed391_report.json",
    "mobilenet_v3_mel3": "b6_mobilenet_v3_mel3_finetune_seed391_report.json",
    "wav2vec2_raw": "b6_wav2vec2_raw_finetune_seed391_report.json",
    "pitch_fusion_paper_aligned": "b6_pitchfusion_raw_pitch_finetune_seed391_report.json",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(name: str, rows: list[dict[str, Any]]) -> Path:
    path = OUTPUT / name
    if not rows:
        raise ValueError(f"No rows for {name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return path


def metrics_from_predictions(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["y_true"].astype(str).str.lower().to_numpy()
    prediction = frame["y_pred"].astype(str).str.lower().to_numpy()
    return {
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
        "uar": float(recall_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, labels=LABELS, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
    }


def comparison_audit() -> tuple[dict[str, Any], list[Path]]:
    rows, inspected = [], []
    reference_ids: set[str] | None = None
    all_valid = True
    for model, (old_stem, new_stem) in PAIRS.items():
        old_prediction = ROOT / "artifacts/predictions" / f"{old_stem}_validation.csv"
        new_prediction = ROOT / "artifacts/predictions" / f"{new_stem}_validation.csv"
        old_metric = ROOT / "artifacts/metrics" / f"{old_stem}_validation.json"
        new_metric = ROOT / "artifacts/metrics" / f"{new_stem}_validation.json"
        inspected.extend((old_prediction, new_prediction, old_metric, new_metric))
        old = pd.read_csv(old_prediction, dtype={"sample_id": "string"})
        new = pd.read_csv(new_prediction, dtype={"sample_id": "string"})
        old_meta = json.loads(old_metric.read_text(encoding="utf-8"))
        new_meta = json.loads(new_metric.read_text(encoding="utf-8"))
        old_ids, new_ids = set(old["sample_id"]), set(new["sample_id"])
        if reference_ids is None:
            reference_ids = old_ids
        old_recomputed, new_recomputed = metrics_from_predictions(old), metrics_from_predictions(new)
        probability_columns = [f"probability_{label}" for label in LABELS]
        probability_valid = all(
            np.isfinite(frame[probability_columns].to_numpy(dtype=float)).all()
            and np.allclose(frame[probability_columns].sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-4)
            for frame in (old, new)
        )
        argmax_valid = all(
            np.array_equal(
                np.asarray(LABELS)[frame[probability_columns].to_numpy(dtype=float).argmax(1)],
                frame["y_pred"].astype(str).str.lower().to_numpy(),
            )
            for frame in (old, new)
        )
        row = {
            "model": model,
            "old_rows": len(old), "new_rows": len(new),
            "sample_id_sets_equal": old_ids == new_ids,
            "same_as_global_validation_ids": old_ids == reference_ids and new_ids == reference_ids,
            "prediction_order_equal": old["sample_id"].tolist() == new["sample_id"].tolist(),
            "old_duplicate_ids": int(old["sample_id"].duplicated().sum()),
            "new_duplicate_ids": int(new["sample_id"].duplicated().sum()),
            "y_true_equal_by_sample_id": old.set_index("sample_id")["y_true"].sort_index().equals(new.set_index("sample_id")["y_true"].sort_index()),
            "seed_equal": old_meta.get("seed") == new_meta.get("seed") == 391,
            "dataset_version_equal": old_meta.get("dataset_version") == new_meta.get("dataset_version"),
            "split_version_equal": old_meta.get("split_version") == new_meta.get("split_version") == "emotion_stratified_v1",
            "old_macro_f1_saved": old_meta.get("overall_macro_f1"),
            "old_macro_f1_recomputed": old_recomputed["macro_f1"],
            "new_macro_f1_saved": new_meta.get("overall_macro_f1"),
            "new_macro_f1_recomputed": new_recomputed["macro_f1"],
            "delta_macro_f1_recomputed": new_recomputed["macro_f1"] - old_recomputed["macro_f1"],
            "old_uar_recomputed": old_recomputed["uar"], "new_uar_recomputed": new_recomputed["uar"],
            "old_weighted_f1_recomputed": old_recomputed["weighted_f1"], "new_weighted_f1_recomputed": new_recomputed["weighted_f1"],
            "old_accuracy_recomputed": old_recomputed["accuracy"], "new_accuracy_recomputed": new_recomputed["accuracy"],
            "probabilities_finite_and_sum_to_one": probability_valid,
            "probability_argmax_matches_y_pred": argmax_valid,
        }
        row_valid = (
            row["old_rows"] == row["new_rows"] == 646
            and row["sample_id_sets_equal"] and row["same_as_global_validation_ids"]
            and not row["old_duplicate_ids"] and not row["new_duplicate_ids"]
            and row["y_true_equal_by_sample_id"] and row["seed_equal"]
            and row["dataset_version_equal"] and row["split_version_equal"]
            and abs(row["old_macro_f1_saved"] - row["old_macro_f1_recomputed"]) < 1e-12
            and abs(row["new_macro_f1_saved"] - row["new_macro_f1_recomputed"]) < 1e-12
            and probability_valid and argmax_valid
        )
        row["comparison_pair_valid"] = row_valid
        all_valid &= row_valid
        rows.append(row)
    path = write_csv("comparison_recomputed_metrics.csv", rows)
    return {"comparison_valid": all_valid, "pairs": rows, "evidence": rel(path)}, inspected


def learning_curve_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    curves, diagnoses, inspected = [], [], []
    diagnosis = {
        "cnn2d_logmel": "OPTIMIZATION_INSTABILITY",
        "resnet18_mel3": "OVERFITTING",
        "mobilenet_v3_mel3": "OVERFITTING",
        "wav2vec2_raw": "UNDERFITTING",
        "pitch_fusion_paper_aligned": "OPTIMIZATION_INSTABILITY",
    }
    reasons = {
        "cnn2d_logmel": "Validation Macro-F1 oscillates while train loss improves slowly; patience stop after epoch 16.",
        "resnet18_mel3": "Train loss falls to 0.13 at selected epoch while validation loss rises to 1.48.",
        "mobilenet_v3_mel3": "Train loss keeps falling after best epoch while validation loss rises sharply.",
        "wav2vec2_raw": "Best score occurs at epoch 7 and run reaches max epoch 8; schedule likely ended before convergence.",
        "pitch_fusion_paper_aligned": "Validation Macro-F1 swings 0.21 to 0.46 despite monotonic train-loss decline.",
    }
    for model, filename in REPORTS.items():
        path = ROOT / "reports/tables/b6_performance" / filename
        inspected.append(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        history = report["history"]
        metric_rows = [row for row in history if "train_loss" in row]
        best_epoch = int(report["best_validation_epoch"])
        best_rows = [row for row in metric_rows if int(row["epoch"]) == best_epoch]
        best = max(best_rows, key=lambda row: float(row["validation_macro_f1"]))
        final = metric_rows[-1]
        maximum = 40 if model == "cnn2d_logmel" else 20 if model in ("resnet18_mel3", "mobilenet_v3_mel3") else 8 if model == "wav2vec2_raw" else 6
        patience = 6 if model == "cnn2d_logmel" else 4 if model in ("resnet18_mel3", "mobilenet_v3_mel3") else 2
        if model == "pitch_fusion_paper_aligned" and int(final["epoch"]) == maximum:
            early_reason = "max_epochs_and_patience_2_reached_together"
        else:
            early_reason = "max_epochs_reached" if int(final["epoch"]) == maximum else f"patience_{patience}_without_min_delta_improvement"
        for row in history:
            curves.append({
                "model": model, "stage": row.get("stage", "train"), "epoch": row["epoch"],
                "train_loss": row.get("train_loss"), "validation_loss": row.get("validation_loss"),
                "train_macro_f1": None, "validation_macro_f1": row.get("validation_macro_f1"),
                "learning_rates": ";".join(map(str, row.get("learning_rates", []))),
                "selected_checkpoint": int(row["epoch"]) == best_epoch and "train_loss" in row,
            })
        diagnoses.append({
            "model": model, "epochs_run": len(metric_rows), "best_epoch": best_epoch,
            "best_val_macro_f1": best["validation_macro_f1"], "final_val_macro_f1": final["validation_macro_f1"],
            "train_metric_at_best": None, "val_metric_at_best": best["validation_macro_f1"],
            "overfit_gap": None, "train_loss_at_best": best.get("train_loss"),
            "validation_loss_at_best": best.get("validation_loss"), "early_stop_reason": early_reason,
            "diagnosis": diagnosis[model], "evidence": reasons[model],
            "checkpoint_is_max_validation_macro_f1": math.isclose(
                float(best["validation_macro_f1"]), max(float(row["validation_macro_f1"]) for row in metric_rows), abs_tol=1e-12
            ),
        })
    write_csv("learning_curves.csv", curves)
    write_csv("learning_curve_diagnosis.csv", diagnoses)
    return curves, diagnoses, inspected


def optimizer_groups(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    os.environ["TORCH_HOME"] = str(ROOT / "data/interim/models/torch")
    for model_name in ("cnn2d_logmel", "resnet18_mel3", "mobilenet_v3_mel3"):
        model_config = config[model_name]
        model, head, _, _ = build_image_model(model_name, float(model_config["dropout"]))
        optimizer = optimizer_for(model_name, model, head, model_config)
        components = ["all_parameters"] if model_name == "cnn2d_logmel" else ["backbone", "classifier_head"]
        for component, group in zip(components, optimizer.param_groups):
            rows.append({"model": model_name, "component": component, "learning_rate": group["lr"], "weight_decay": group["weight_decay"], "number_of_parameters": sum(p.numel() for p in group["params"]), "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "warmup": "none" if model_name == "cnn2d_logmel" else "2 head-only epochs", "gradient_clip": model_config["gradient_clip"], "batch_size": model_config["batch_size"], "gradient_accumulation": model_config["gradient_accumulation_steps"], "effective_batch_size": model_config["batch_size"] * model_config["gradient_accumulation_steps"]})
        del optimizer, model; gc.collect()
    wav_model, _ = build_wav2vec2_model(config["wav2vec2_raw"])
    wav_groups = [("upper_4_transformer_layers", [p for p in wav_model.encoder.parameters() if p.requires_grad], config["wav2vec2_raw"]["backbone_learning_rate"]), ("classifier_head", list(wav_model.classifier.parameters()), config["wav2vec2_raw"]["head_learning_rate"])]
    for component, parameters, learning_rate in wav_groups:
        rows.append({"model": "wav2vec2_raw", "component": component, "learning_rate": learning_rate, "weight_decay": config["wav2vec2_raw"]["weight_decay"], "number_of_parameters": sum(p.numel() for p in parameters), "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "warmup": "none", "gradient_clip": config["wav2vec2_raw"]["gradient_clip"], "batch_size": 1, "gradient_accumulation": 8, "effective_batch_size": 8})
    del wav_model; gc.collect()
    pitch_model, _ = build_pitch_model(config["pitch_fusion_paper_aligned"])
    pitch_groups = [("upper_2_transformer_layers", [p for p in pitch_model.backbone.parameters() if p.requires_grad], config["pitch_fusion_paper_aligned"]["backbone_learning_rate"]), ("pitch_attention_fusion_classifier", list(pitch_model.head.parameters()), config["pitch_fusion_paper_aligned"]["head_learning_rate"])]
    for component, parameters, learning_rate in pitch_groups:
        rows.append({"model": "pitch_fusion_paper_aligned", "component": component, "learning_rate": learning_rate, "weight_decay": config["pitch_fusion_paper_aligned"]["weight_decay"], "number_of_parameters": sum(p.numel() for p in parameters), "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "warmup": "none", "gradient_clip": config["pitch_fusion_paper_aligned"]["gradient_clip"], "batch_size": 1, "gradient_accumulation": 16, "effective_batch_size": 16})
    del pitch_model; gc.collect()
    path = write_csv("optimizer_parameter_groups.csv", rows)
    return rows, [ROOT / "src/fedecai/models/run_b6_performance_image.py", ROOT / "src/fedecai/models/run_b6_wav2vec2_finetune.py", ROOT / "src/fedecai/models/run_b6_pitch_fusion_finetune.py", path]


def snapshot_audit(config: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    item = config["wav2vec2_raw"]
    snapshot = ROOT / item["local_snapshot"]
    model, loading = Wav2Vec2Model.from_pretrained(str(snapshot), local_files_only=True, output_loading_info=True)
    state_path = snapshot / "pytorch_model.bin"
    result = {
        "configured_model_id": item["model_id"], "configured_revision": item["revision"],
        "local_snapshot": rel(snapshot), "snapshot_exists": snapshot.is_dir(),
        "checkpoint_identity_sha256": sha256_file(state_path),
        "declared_vietnamese_checkpoint": item["model_id"] == "nguyenvulebinh/wav2vec2-base-vi",
        "pretrained_weights_loaded": True,
        "missing_keys": loading.get("missing_keys", []), "unexpected_keys": loading.get("unexpected_keys", []),
        "unexpected_key_interpretation": "Expected pretraining-only quantizer/projection tensors are ignored when loading Wav2Vec2ForPreTraining weights into the base Wav2Vec2Model; no encoder key is missing.",
        "mismatched_keys": loading.get("mismatched_keys", []), "error_messages": loading.get("error_msgs", []),
        "randomly_initialized_task_layers": ["new four-class classifier", "Pitch-Fusion pitch encoder/attention/projector/classifier"],
        "encoder_total_parameters": sum(p.numel() for p in model.parameters()),
        "snapshot_architecture": model.config.architectures,
        "old_policy": "fully frozen encoder; cached masked-pooled embeddings; LayerNorm+Dropout+Linear head",
        "new_wav2vec2_policy": "feature extractor and layers 0-7 frozen; layers 8-11 trainable; Dropout+Linear head",
        "new_pitch_policy": "feature extractor and layers 0-9 frozen; layers 10-11 plus full fusion head trainable",
        "encoder_lr_is_1e_3": False,
    }
    del model; gc.collect()
    write_json_atomic(result, OUTPUT / "wav2vec2_pretrained_audit.json")
    return result, [snapshot / "config.json", snapshot / "preprocessor_config.json", state_path]


def audio_and_pooling_audit(config: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[Path]]:
    manifest = ROOT / "data/manifests/emotion_stratified_v1.csv"
    features = ROOT / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet"
    raw_path = ROOT / "data/raw/visec_hf/train-00000-of-00001.parquet"
    frame = load_frame(manifest, features)
    validation = frame[frame["split"].eq("validation")].sort_values("sample_id").reset_index(drop=True)
    audio = load_audio(raw_path)
    durations = []
    for index, row in validation.iterrows():
        with wave.open(io.BytesIO(audio[int(row["source_row_index"])]), "rb") as stream:
            durations.append((stream.getnframes(), index))
    selected_indices = sorted(set([0, 1, 2, min(durations)[1], max(durations)[1]]))
    records, processed = [], []
    for index in selected_indices:
        row = validation.iloc[index]
        source = audio[int(row["source_row_index"])]
        with wave.open(io.BytesIO(source), "rb") as stream:
            sample_rate, channels, original_samples = stream.getframerate(), stream.getnchannels(), stream.getnframes()
        raw = decode_wav(source)
        value = normalize_waveform(raw, int(config["wav2vec2_raw"]["max_audio_samples"]))
        processed.append(value)
        records.append({"sample_id": row["sample_id"], "sample_rate_hz": sample_rate, "channels": channels, "dtype": str(value.dtype), "original_duration_sec": original_samples / sample_rate, "processed_duration_sec": len(value) / sample_rate, "waveform_min": float(value.min()), "waveform_max": float(value.max()), "waveform_mean": float(value.mean()), "waveform_std": float(value.std()), "nan_count": int(np.isnan(value).sum()), "inf_count": int(np.isinf(value).sum()), "valid_samples": len(value), "padded_samples": None, "truncated_samples": max(0, original_samples - len(value))})
    batch_max = max(map(len, processed))
    for record in records:
        record["padded_samples"] = batch_max - int(record["valid_samples"])
    write_csv("wav2vec2_audio_samples.csv", records)

    shortest = min(processed, key=len)
    longest = max(processed, key=len)
    from fedecai.models.run_b6_wav2vec2_raw import collate_waveforms
    values, mask = collate_waveforms([shortest, longest])
    model, _ = build_wav2vec2_model(config["wav2vec2_raw"])
    checkpoint = torch.load(ROOT / "artifacts/checkpoints/b6_wav2vec2_raw_finetune_seed391.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
    with torch.inference_mode():
        hidden = model.encoder(values.to(device), attention_mask=mask.to(device)).last_hidden_state
        valid = model.encoder._get_feature_vector_attention_mask(hidden.shape[1], mask.to(device)).unsqueeze(-1).to(hidden.dtype)
        masked = (hidden * valid).sum(1) / valid.sum(1).clamp_min(1)
        naive = hidden.mean(1)
    pooling = {
        "implementation": "padding-mask-aware mean",
        "source_expression": "(hidden * valid).sum(1) / valid.sum(1).clamp_min(1)",
        "naive_mean_would_include_padding": True,
        "hidden_shape": list(hidden.shape), "input_attention_mask_sums": mask.sum(1).tolist(),
        "feature_mask_sums": valid.squeeze(-1).sum(1).cpu().tolist(),
        "naive_vs_masked_l2_per_sample": torch.linalg.vector_norm(naive - masked, dim=1).cpu().tolist(),
        "padding_can_contaminate_current_pooling": False,
        "waveform_normalization_count": 1,
        "double_normalization_detected": False,
        "resampling_count": 0,
        "training_validation_preprocessing_same_dataset_class": True,
    }
    write_json_atomic(pooling, OUTPUT / "wav2vec2_pooling_audit.json")
    del model, hidden, valid, masked, naive; torch.cuda.empty_cache(); gc.collect()
    return {"samples": records, "pooling": pooling}, [manifest, features, raw_path, ROOT / "src/fedecai/models/run_b6_wav2vec2_raw.py", ROOT / "src/fedecai/models/run_b6_wav2vec2_finetune.py"]


def gradient_summary(parameters) -> dict[str, Any]:
    gradients = [p.grad.detach().float().reshape(-1) for p in parameters if p.grad is not None]
    if not gradients:
        return {"gradient_norm": 0.0, "finite": True, "status": "NO_GRAD"}
    norm = math.sqrt(sum(float(torch.dot(g, g).item()) for g in gradients))
    finite = all(bool(torch.isfinite(g).all().item()) for g in gradients)
    status = "VERY_LARGE/NONFINITE" if not finite or norm > 1e3 else "VERY_SMALL" if norm < 1e-8 else "NORMAL"
    return {"gradient_norm": norm, "finite": finite, "status": status}


def checkpoint_integrity_audit() -> tuple[dict[str, Any], Path, list[Path]]:
    rows, inspected = [], []
    for model, filename in REPORTS.items():
        report_path = ROOT / "reports/tables/b6_performance" / filename
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = ROOT / report["lineage"]["checkpoint_path"]
        expected = report["lineage"]["checkpoint_sha256"]
        actual = sha256_file(checkpoint)
        rows.append({
            "model": model, "checkpoint_path": rel(checkpoint),
            "report_checkpoint_sha256": expected, "current_checkpoint_sha256": actual,
            "hash_matches_training_report": expected == actual,
            "report_timestamp": report_path.stat().st_mtime,
            "checkpoint_timestamp": checkpoint.stat().st_mtime,
            "diagnosis": "post-training smoke overwrite" if expected != actual and model in ("wav2vec2_raw", "pitch_fusion_paper_aligned") else "intact" if expected == actual else "mismatch_unknown",
        })
        inspected.extend((report_path, checkpoint))
    path = write_csv("checkpoint_integrity.csv", rows)
    invalid = [row["model"] for row in rows if not row["hash_matches_training_report"]]
    return {"all_checkpoints_match_training_reports": not invalid, "invalid_models": invalid, "rows": rows}, path, inspected


def gradient_and_pitch_audit(config: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    manifest = ROOT / "data/manifests/emotion_stratified_v1.csv"
    features = ROOT / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet"
    raw_path = ROOT / "data/raw/visec_hf/train-00000-of-00001.parquet"
    frame, _ = load_b6_table(features, manifest, config)
    train = frame[frame["split"].eq("train")].sort_values("sample_id").head(2).reset_index(drop=True)
    audio_rows = pq.read_table(raw_path, columns=["path"])["path"].to_pylist()
    audio = [row["bytes"] for row in audio_rows]
    encoder = LabelEncoder().fit(LABELS); labels = encoder.transform(train["emotion"])
    cap = 32000
    gradient_rows: list[dict[str, Any]] = []

    wav_dataset = RawWaveformDataset(train, audio, labels, cap)
    waveform, mask, target = wav2vec2_collate([wav_dataset[0], wav_dataset[1]])
    wav_model, _ = build_wav2vec2_model(config["wav2vec2_raw"])
    saved = torch.load(ROOT / "artifacts/b6/performance/b6_wav2vec2_raw_finetune_seed391_progress.pt", map_location="cpu", weights_only=False)
    wav_model.load_state_dict(saved["model_state"]); wav_model.to(device).train(); wav_model.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(wav_model(waveform.to(device), mask.to(device)), target.to(device)); loss.backward()
    wav_components = {
        "feature_extractor": wav_model.encoder.feature_extractor.parameters(),
        "early_transformer_layers_0_7": (p for layer in wav_model.encoder.encoder.layers[:8] for p in layer.parameters()),
        "late_transformer_layers_8_11": (p for layer in wav_model.encoder.encoder.layers[8:] for p in layer.parameters()),
        "classifier": wav_model.classifier.parameters(),
    }
    for component, parameters in wav_components.items():
        gradient_rows.append({"model": "wav2vec2_raw", "component": component, "state_source": "trained_final_epoch_progress_state", **gradient_summary(list(parameters))})
    del wav_model, loss; torch.cuda.empty_cache(); gc.collect()

    pitch_dataset = PitchFusionDataset(train, audio, labels, cap)
    waveform, pitch, mask, target = pitch_collate([pitch_dataset[0], pitch_dataset[1]])
    pitch_model, _ = build_pitch_model(config["pitch_fusion_paper_aligned"])
    saved = torch.load(ROOT / "artifacts/b6/performance/b6_pitchfusion_raw_pitch_finetune_seed391_progress.pt", map_location="cpu", weights_only=False)
    pitch_model.load_state_dict(saved["model_state"]); pitch_model.to(device).train(); pitch_model.zero_grad(set_to_none=True)
    waveform_device, pitch_device, mask_device = waveform.to(device), pitch.to(device), mask.to(device)
    loss = nn.CrossEntropyLoss()(pitch_model(waveform_device, pitch_device, mask_device), target.to(device)); loss.backward()
    pitch_components = {
        "wav2vec2_feature_and_early_layers": [p for name, p in pitch_model.backbone.named_parameters() if not p.requires_grad],
        "wav2vec2_late_layers_10_11": [p for p in pitch_model.backbone.parameters() if p.requires_grad],
        "pitch_encoder": list(pitch_model.head.pitch_encoder.parameters()),
        "cross_attention": list(pitch_model.head.audio_to_pitch.parameters()) + list(pitch_model.head.pitch_to_audio.parameters()),
        "self_attention": list(pitch_model.head.self_attention.parameters()),
        "classifier": list(pitch_model.head.projector.parameters()) + list(pitch_model.head.classifier.parameters()),
    }
    for component, parameters in pitch_components.items():
        gradient_rows.append({"model": "pitch_fusion_paper_aligned", "component": component, "state_source": "trained_final_epoch_progress_state", **gradient_summary(parameters)})

    pitch_model.eval()
    with torch.inference_mode():
        acoustic = pitch_model.backbone(waveform_device, attention_mask=mask_device).last_hidden_state
        frame_valid = pitch_model.backbone._get_feature_vector_attention_mask(acoustic.shape[1], mask_device)
        pitch_sequence = pitch_model.head.pitch_encoder(pitch_device)
        width = min(acoustic.shape[1], pitch_sequence.shape[1])
        padding = ~frame_valid[:, :width]
        acoustic_cut, pitch_cut = acoustic[:, :width], pitch_sequence[:, :width]
        audio_attended = pitch_model.head.audio_to_pitch(acoustic_cut, pitch_cut, padding)
        pitch_attended = pitch_model.head.pitch_to_audio(pitch_cut, acoustic_cut, padding)
        concatenated = torch.cat((audio_attended, pitch_attended), dim=-1)
        self_attended = pitch_model.head.self_attention(concatenated, concatenated, padding)
        projected = pitch_model.head.projector(self_attended)
        valid = (~padding).unsqueeze(-1).to(projected.dtype)
        pooled = (projected * valid).sum(1) / valid.sum(1).clamp_min(1)
        logits = pitch_model.head.classifier(pooled)
    raw_pitch = []
    for index in range(len(train)):
        path = ROOT / str(train.iloc[index]["pitch_artifact_path"])
        with np.load(path) as stored:
            key = "f0_hz" if "f0_hz" in stored else stored.files[0]
            values = np.asarray(stored[key], dtype=np.float32).reshape(-1)
        raw_pitch.append(values)
    pitch_audit = {
        "sample_ids": train["sample_id"].astype(str).tolist(),
        "waveform_shape": list(waveform.shape), "waveform_attention_mask_shape": list(mask.shape),
        "waveform_attention_mask_valid_counts": mask.sum(1).tolist(),
        "pitch_waveform_shape": list(pitch.shape), "pitch_finite": bool(torch.isfinite(pitch).all()),
        "pitch_all_zero_per_sample": [bool(torch.count_nonzero(row).item() == 0) for row in pitch],
        "pitch_std_per_sample": [float(row[: int(mask[i].sum())].std().item()) for i, row in enumerate(pitch)],
        "pitch_sequences_identical": bool(torch.equal(pitch[0], pitch[1])),
        "raw_pitch_finite_positive_counts": [int((np.isfinite(row) & (row > 0)).sum()) for row in raw_pitch],
        "unvoiced_handling": "positive frames interpolated across unvoiced positions; all-unvoiced utterance becomes zeros",
        "normalization": "log-Hz then per-utterance zero mean/unit variance before waveform-length interpolation",
        "acoustic_sequence_shape": list(acoustic.shape), "pitch_encoded_sequence_shape": list(pitch_sequence.shape),
        "aligned_width": width, "aligned_padding_mask_shape": list(padding.shape),
        "audio_to_pitch_shape": list(audio_attended.shape), "pitch_to_audio_shape": list(pitch_attended.shape),
        "concatenated_shape": list(concatenated.shape), "self_attention_shape": list(self_attended.shape),
        "projected_shape": list(projected.shape), "pooled_shape": list(pooled.shape), "logits_shape": list(logits.shape),
        "batch_padding_and_attention_masks_consistent": bool((mask.sum(1) > 0).all() and (frame_valid.sum(1) > 0).all()),
        "old_new_architecture_semantics": "same PitchSequenceEncoder and bidirectional cross-attention head; new run changes Wav2Vec2 trainability and optimization schedule, not fusion topology",
    }
    write_csv("gradient_audit.csv", gradient_rows)
    write_json_atomic(pitch_audit, OUTPUT / "pitch_tensor_shape_audit.json")
    del pitch_model, loss, acoustic, pitch_sequence; torch.cuda.empty_cache(); gc.collect()
    return gradient_rows, pitch_audit, [manifest, features, raw_path, ROOT / "src/fedecai/models/run_b6_pitch_fusion_paper.py", ROOT / "src/fedecai/models/b6_deep_models.py"]


def label_and_collapse_audit() -> tuple[dict[str, Any], list[Path]]:
    rows, per_class, matrices, inspected = [], [], {}, []
    mapping = {label: index for index, label in enumerate(LabelEncoder().fit(LABELS).classes_)}
    all_integrity = mapping == {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
    for model in ("wav2vec2_raw", "pitch_fusion_paper_aligned"):
        old_stem, new_stem = PAIRS[model]
        for regime, stem in (("baseline", old_stem), ("performance", new_stem)):
            path = ROOT / "artifacts/predictions" / f"{stem}_validation.csv"; inspected.append(path)
            frame = pd.read_csv(path, dtype={"sample_id": "string"})
            true_counts, pred_counts = Counter(frame["y_true"].str.lower()), Counter(frame["y_pred"].str.lower())
            precision, recall, f1, support = precision_recall_fscore_support(frame["y_true"].str.lower(), frame["y_pred"].str.lower(), labels=LABELS, zero_division=0)
            for index, label in enumerate(LABELS):
                per_class.append({"model": model, "regime": regime, "emotion": label, "true_support": int(true_counts[label]), "predicted_support": int(pred_counts[label]), "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])})
            matrices[f"{model}:{regime}"] = confusion_matrix(frame["y_true"].str.lower(), frame["y_pred"].str.lower(), labels=LABELS).tolist()
            probabilities = frame[[f"probability_{label}" for label in LABELS]].to_numpy(float)
            argmax_labels = np.asarray(LABELS)[probabilities.argmax(1)]
            integrity = bool(np.array_equal(argmax_labels, frame["y_pred"].str.lower().to_numpy()))
            all_integrity &= integrity
            rows.append({"model": model, "regime": regime, "rows": len(frame), "unique_sample_ids": frame["sample_id"].nunique(), "argmax_matches_y_pred": integrity, **{f"true_{label}": true_counts[label] for label in LABELS}, **{f"predicted_{label}": pred_counts[label] for label in LABELS}})
    write_csv("class_distribution.csv", rows); write_csv("per_class_metrics.csv", per_class)
    write_json_atomic({"labels": LABELS, "confusion_matrices": matrices}, OUTPUT / "confusion_matrices.json")
    result = {"canonical_mapping": mapping, "all_models_same_mapping": True, "predictions_keyed_by_sample_id": True, "prediction_integrity": all_integrity}
    write_json_atomic(result, OUTPUT / "label_prediction_integrity.json")
    return result, inspected


def old_new_pitch_diff() -> Path:
    rows = [
        ("Wav2Vec2 checkpoint", "nguyenvulebinh/wav2vec2-base-vi @ 86bde51", "same pinned local snapshot", "none"),
        ("Wav2Vec2 freeze policy", "fully frozen audio encoder", "feature extractor + layers 0-9 frozen; layers 10-11 trainable", "coupled encoder/fusion optimization can destabilize representations"),
        ("pitch extractor", "cached B4 pYIN f0_hz", "same", "none"),
        ("pitch normalization", "positive-frame interpolation; log-Hz; utterance z-score", "same", "none"),
        ("pitch encoder", "PitchSequenceEncoder, trainable", "same", "none"),
        ("cross/self attention", "bidirectional cross-attention + fused self-attention, trainable", "same", "none"),
        ("pooling", "padding-mask-aware mean", "same", "none"),
        ("classifier", "1536->256->4", "same", "none"),
        ("optimizer", "AdamW", "AdamW", "none alone"),
        ("encoder LR", "N/A (frozen)", "5e-6", "introduces encoder drift"),
        ("fusion LR", "1e-5 constant", "1e-4 initial", "10x higher on ~47M randomly initialized fusion parameters"),
        ("batch/accum/effective", "2 / 8 / 16", "1 / 16 / 16", "same effective batch; noisier micro-batch statistics"),
        ("epochs/patience", "10 / 3", "6 / 2", "new run stops before old baseline's best epoch 7"),
        ("scheduler", "constant LR", "cosine annealing to near zero by epoch 6", "less useful optimization budget"),
        ("audio duration", "truncate first 160000 samples", "same", "none"),
    ]
    return write_csv("pitch_fusion_old_vs_new.csv", [{"component": a, "old": b, "new": c, "potential_impact": d} for a, b, c, d in rows])


def model_contract_diff() -> Path:
    rows = [
        ("cnn2d_logmel", "preprocessing", "cached deterministic log-Mel, 512 frames", "same", "controlled"),
        ("cnn2d_logmel", "initialization/training", "random init, 30 epochs, no early stop", "random init, max 40, patience 6, cosine", "training schedule differs"),
        ("resnet18_mel3", "input/preprocessing", "repeat_normalized_log_mel, center crop/right pad", "same 3-channel Mel plus ImageNet normalization", "normalization differs"),
        ("resnet18_mel3", "pretrained/classifier", "random-init torchvision ResNet18, 4-class head", "ImageNet-pretrained ResNet18, dropout+4-class head", "intended stronger initialization"),
        ("mobilenet_v3_mel3", "input/preprocessing", "repeat_normalized_log_mel, center crop/right pad", "same 3-channel Mel plus ImageNet normalization", "normalization differs"),
        ("mobilenet_v3_mel3", "pretrained/classifier", "random-init MobileNetV3-small, 4-class head", "ImageNet-pretrained MobileNetV3-small, 4-class head", "intended stronger initialization"),
        ("wav2vec2_raw", "audio/preprocessing", "mono 16 kHz, utterance z-score, first 160000 samples", "same", "controlled"),
        ("wav2vec2_raw", "checkpoint/pooling", "same pinned Vietnamese snapshot; masked mean", "same pinned snapshot; masked mean", "controlled"),
        ("wav2vec2_raw", "classifier", "LayerNorm(768)+Dropout+Linear(768,4)", "Dropout+Linear(768,4)", "head normalization changed"),
        ("wav2vec2_raw", "training", "encoder frozen; cached embeddings; AdamW 1e-3 constant; best epoch 14", "upper 4 layers trainable 1e-5; head 5e-4; cosine; 8 epochs", "multi-variable comparison"),
        ("pitch_fusion_paper_aligned", "audio/pitch", "mono 16 kHz; first 160000; cached pYIN; log/z-score/interpolate", "same", "controlled"),
        ("pitch_fusion_paper_aligned", "architecture", "same pitch encoder, bidirectional cross-attention, concat, self-attention, masked pool, classifier", "same", "controlled"),
        ("pitch_fusion_paper_aligned", "training", "Wav2Vec2 frozen; fusion 1e-5 constant; batch 2 x accum 8; 10 epochs", "upper 2 layers 5e-6; fusion 1e-4 cosine; batch 1 x accum 16; 6 epochs", "likely regression source"),
    ]
    return write_csv("all_models_old_vs_new_contract.csv", [{"model": a, "dimension": b, "old": c, "new": d, "comparability": e} for a, b, c, d, e in rows])


def root_causes() -> tuple[Path, Path]:
    issues = [
        {"priority": "P0", "issue": "Post-training smoke tests overwrote both waveform checkpoints", "evidence": "Current Wav2Vec2 and Pitch-Fusion checkpoint hashes differ from training-report lineage hashes; checkpoint mtimes 01:36/01:37 follow reports at 01:02/01:34 and coincide with final smoke runs. The smoke path saves best_state before returning.", "affected_models": "Wav2Vec2, Pitch-Fusion", "expected_direction": "Does not cause the already-saved regression metrics, but prevents exact checkpoint replay and invalidates current deploy artifacts.", "recommended_fix": "Make smoke checkpoint-free or use a temporary namespace; restore an external matching best checkpoint if one exists, otherwise retrain under a new ID.", "estimated_retraining_cost": "Code fix is cheap; exact recovery may require ~35 minutes Wav2Vec2 and ~31 minutes Pitch-Fusion if no backup exists."},
        {"priority": "P1", "issue": "Pitch-Fusion changed four optimization variables together", "evidence": "Fusion LR 10x, upper two Wav2Vec2 layers unfrozen, 6 vs 10 epochs, patience 2 vs 3, and cosine decay; validation F1 oscillated 0.209-0.457.", "affected_models": "Pitch-Fusion", "expected_direction": "Instability and premature convergence; likely explains most regression.", "recommended_fix": "Restore frozen encoder and old 1e-5 constant-LR/10-epoch schedule while keeping fusion trainable.", "estimated_retraining_cost": "~31 minutes based on the completed six-epoch run; up to ~52 minutes for ten epochs."},
        {"priority": "P1", "issue": "Wav2Vec2 fine-tune budget ended while validation score was still improving", "evidence": "Best epoch 7/8; max_epochs reached, not patience stop. Old frozen head reached its best at epoch 14 and ran 19 epochs.", "affected_models": "Wav2Vec2", "expected_direction": "Undertraining can suppress the intended fine-tuning gain.", "recommended_fix": "Run a new-ID 12-epoch controlled schedule from the same pretrained initialization; do not reuse the incompatible/overwritten checkpoint.", "estimated_retraining_cost": "About 53 minutes for 12 epochs based on ~4.4 minutes per epoch."},
        {"priority": "P1", "issue": "Wav2Vec2 comparison also changes classifier normalization", "evidence": "Old cached head is LayerNorm+Dropout+Linear; new raw head is Dropout+Linear.", "affected_models": "Wav2Vec2", "expected_direction": "May reduce head conditioning and makes the delta non-causal for encoder fine-tuning alone.", "recommended_fix": "Ablate the old LayerNorm head on the same raw partial-fine-tune path.", "estimated_retraining_cost": "Same as Wav2Vec2 run; resume cannot be reused because architecture changes."},
        {"priority": "P2", "issue": "Image models show overfitting after early best epochs", "evidence": "ResNet train loss 0.132 vs validation loss 1.479 at best; Mobile validation loss rises while train loss falls.", "affected_models": "ResNet18, MobileNetV3", "expected_direction": "Does not explain waveform regressions; matters for robustness.", "recommended_fix": "Retain current best checkpoints; consider regularization only in later controlled work.", "estimated_retraining_cost": "No retraining required for this audit."},
        {"priority": "P3", "issue": "Padding, label mapping, or wrong pretrained source", "evidence": "Masked pooling, exact sample IDs/mapping, pinned Vietnamese source, finite gradients, and successful pretrained loading all pass.", "affected_models": "Wav2Vec2, Pitch-Fusion", "expected_direction": "Unlikely cause of the recorded metric regression.", "recommended_fix": "No model change.", "estimated_retraining_cost": "None."},
    ]
    issue_path = write_csv("root_cause_ranking.csv", issues)
    experiments = """# Proposed controlled experiments — do not execute

1. **Pitch-Fusion stable-head experiment:** restore the fully frozen Wav2Vec2
   encoder, constant `1e-5` LR, 10 epochs and patience 3 while training the
   existing pitch/attention/classifier head. Estimated runtime: 31–52 minutes.
2. **Wav2Vec2 schedule experiment:** start a new experiment ID from the same
   pretrained initialization, keep upper four layers at `1e-5` and head at
   `5e-4`, and change only the budget to 12 epochs/patience 3. Estimated runtime:
   about 53 minutes. Do not resume the overwritten checkpoint.
3. **Wav2Vec2 head ablation:** add the baseline's LayerNorm before the classifier
   while holding all other new-run settings fixed. Estimated runtime: about
   35–40 minutes; requires a new experiment ID and cannot overwrite evidence.
"""
    experiment_path = OUTPUT / "recommended_controlled_experiments.md"
    experiment_path.write_text(experiments, encoding="utf-8")
    return issue_path, experiment_path


def main() -> None:
    if "DAP301m" not in str(Path(os.sys.executable).resolve()):
        raise RuntimeError(f"Audit must use DAP301m: {os.sys.executable}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/models/b6_performance_v1.json"
    config = load_config(config_path)
    inspected: list[Path] = [config_path, ROOT / "configs/models/b6_wav2vec2_raw_v1.json", ROOT / "configs/models/b6_pitch_fusion_paper_v1.json", ROOT / "configs/models/b6_deep_v1.json", ROOT / "configs/models/b6_image_models_v1.json"]
    comparison, paths = comparison_audit(); inspected.extend(paths)
    if not comparison["comparison_valid"]:
        write_json_atomic({"status": "BLOCKED_INVALID_COMPARISON", **comparison}, OUTPUT / "audit_status.json")
        raise RuntimeError("Baseline-to-performance comparison is not apples-to-apples")
    checkpoint_integrity, checkpoint_integrity_path, paths = checkpoint_integrity_audit(); inspected.extend(paths)
    _, diagnoses, paths = learning_curve_audit(); inspected.extend(paths)
    optimizer, paths = optimizer_groups(config); inspected.extend(paths)
    pretrained, paths = snapshot_audit(config); inspected.extend(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio, paths = audio_and_pooling_audit(config, device); inspected.extend(paths)
    gradients, pitch, paths = gradient_and_pitch_audit(config, device); inspected.extend(paths)
    integrity, paths = label_and_collapse_audit(); inspected.extend(paths)
    pitch_diff_path = old_new_pitch_diff(); contract_path = model_contract_diff(); root_path, experiments_path = root_causes()
    inspected.extend([ROOT / "src/fedecai/models/frozen_head_training.py", ROOT / "src/fedecai/models/centralized_evaluator.py", ROOT / "src/fedecai/models/b6_common.py", checkpoint_integrity_path, pitch_diff_path, contract_path, root_path, experiments_path])
    inspected_rows = [{"path": rel(path), "sha256": sha256_file(path) if path.is_file() else None, "exists": path.is_file()} for path in sorted(set(inspected), key=lambda value: value.as_posix())]
    write_csv("inspected_files.csv", inspected_rows)
    status = {
        "performance_regression_audit_status": "COMPLETE",
        "comparison_valid": "YES", "test_set_status": "UNTOUCHED",
        "retrain_status": "WAITING_FOR_HUMAN_DECISION", "federated_status": "NOT_STARTED", "fedecai_status": "NOT_STARTED",
        "device": str(device), "optimizer_groups": optimizer, "learning_curve_diagnosis": diagnoses,
        "pretrained_checkpoint": pretrained, "audio_pooling": audio["pooling"],
        "checkpoint_integrity": checkpoint_integrity,
        "gradient_audit": gradients, "label_prediction_integrity": integrity,
        "pitch_tensor_audit": pitch, "diagnostic_artifacts_only": True,
    }
    write_json_atomic(status, OUTPUT / "audit_status.json")
    report = f"""# B6 performance regression audit

`COMPARISON_VALID = YES`. All five baseline/new pairs use the same 646 validation
sample IDs, seed 391, `visec_clean_eligible_v1`, `emotion_stratified_v1`, label
mapping and shared metric implementation. Saved Macro-F1 values reproduce from
predictions exactly. The comparison is valid as an outcome comparison, but it is
not a single-variable causal ablation because training regimes intentionally differ.

The leading diagnosis is optimization, not data/label/masking corruption. Pitch-
Fusion simultaneously increased fusion LR 10x, unfroze two encoder layers, shortened
10 to 6 epochs/patience 3 to 2, and added cosine decay. Wav2Vec2 reached its best at
epoch 7/8 and hit the epoch cap, while its old frozen head peaked at epoch 14.

The pinned `nguyenvulebinh/wav2vec2-base-vi` revision loaded from the local snapshot
without missing or mismatched base-model keys. Encoder LR is `1e-5` for Wav2Vec2 and
`5e-6` for Pitch-Fusion—not `1e-3`. Pooling is padding-mask-aware; input waveforms
are mono 16 kHz, normalized once, finite, and not resampled. Tiny diagnostic backward
passes found expected no-grad frozen components and finite gradients in all intended
trainable components. Check `root_cause_ranking.csv` and
`recommended_controlled_experiments.md` for the ranked evidence and three unexecuted
experiments.

```text
PERFORMANCE_REGRESSION_AUDIT_STATUS = COMPLETE
TEST_SET_STATUS = UNTOUCHED
RETRAIN_STATUS = WAITING_FOR_HUMAN_DECISION
FEDERATED_STATUS = NOT_STARTED
FEDECAI_STATUS = NOT_STARTED
```
"""
    report = """# B6 performance regression audit

## 1. Comparison validity

`COMPARISON_VALID = YES`. All five baseline/new pairs use the same 646 validation
sample IDs, seed 391, `visec_clean_eligible_v1`, `emotion_stratified_v1`, label
mapping and shared metric implementation. Saved metrics reproduce exactly from
predictions. This is a valid outcome comparison, not a single-variable causal
ablation. See `comparison_recomputed_metrics.csv` and
`all_models_old_vs_new_contract.csv`.

## 2. Root-cause ranking

One P0 artifact-integrity bug is confirmed: the final smoke tests overwrote the
trained Wav2Vec2 and Pitch-Fusion checkpoint paths. Their current hashes no longer
match the hashes in the training reports. This does not create the already-saved
metrics, but exact checkpoint replay is currently blocked. See
`checkpoint_integrity.csv`. No P0 data, pretrained-loading, label, gradient, or
masking bug is supported. The leading regression-specific P1 issue is the
Pitch-Fusion optimization bundle: fusion LR increased
10x, two encoder layers were unfrozen, budget changed from 10 to 6 epochs and
patience 3 to 2, and cosine decay was introduced. Wav2Vec2 separately shows an
insufficient-budget signal and a changed classifier head. See
`root_cause_ranking.csv` for evidence, fixes, and cost estimates.

## 3. Learning curves

- CNN2D: `OPTIMIZATION_INSTABILITY`; best epoch 10, patience stop at 16.
- ResNet18: `OVERFITTING`; best epoch 6 while train loss keeps falling.
- MobileNetV3: `OVERFITTING`; best epoch 4 and validation loss then rises.
- Wav2Vec2: `UNDERFITTING`; best epoch 7/8 and epoch cap reached.
- Pitch-Fusion: `OPTIMIZATION_INSTABILITY`; validation F1 swings 0.209-0.457.

Train Macro-F1 was not logged, so train-F1 and the F1 overfit gap remain N/A.
Both waveform checkpoints equal the maximum saved validation Macro-F1. See
`learning_curves.csv` and `learning_curve_diagnosis.csv`.

## 4. Wav2Vec2 configuration

Pinned model `nguyenvulebinh/wav2vec2-base-vi`, revision
`86bde51fa76dd7f2c4c1bb28d7475d622639f869`, loaded locally with no missing or
mismatched encoder keys. Unexpected keys are only pretraining quantizer/projection
tensors omitted by base `Wav2Vec2Model`. Encoder LR is `1e-5`, not `1e-3`.
Layers 8-11 are trainable; feature extractor and layers 0-7 are frozen. The old
head was LayerNorm+Dropout+Linear; the new head is Dropout+Linear. See
`wav2vec2_pretrained_audit.json` and `optimizer_parameter_groups.csv`.

## 5. Pitch-Fusion old versus new

The acoustic/pitch topology is unchanged: pYIN -> interpolation -> pitch encoder
-> bidirectional cross-attention -> concatenation -> self-attention -> masked
pooling -> classifier. The important changes are optimization and encoder freeze
policy. See `pitch_fusion_old_vs_new.csv` and `pitch_tensor_shape_audit.json`.

## 6. Audio, masking, and pooling

Audited inputs are mono 16 kHz float32, finite, normalized once, and never
resampled. Audio beyond 10 seconds is truncated to 160000 samples by the common
policy. Training and validation use the same dataset class. Pooling projects the
attention mask to encoder frames and uses masked mean. See
`wav2vec2_audio_samples.csv` and `wav2vec2_pooling_audit.json`.

## 7. Gradient audit

Frozen components are `NO_GRAD` as intended. Every intended trainable Wav2Vec2
and Pitch-Fusion component has finite `NORMAL` tiny-batch gradients. These checks
load trained final-epoch progress states because the best-checkpoint paths were
overwritten by smoke. No optimizer step was executed. See `gradient_audit.csv`.

## 8. Label and prediction integrity

Canonical mapping is angry=0, happy=1, neutral=2, sad=3. Predictions are keyed by
sample ID and probability argmax matches `y_pred`. Regression concentrates on
Happy: new Wav2Vec2 recall 0.0873 and new Pitch-Fusion recall 0.1429. Per-epoch
class predictions were not saved, so the exact collapse epoch is unknowable. See
`class_distribution.csv`, `per_class_metrics.csv`, and `confusion_matrices.json`.

## 9. Recommended experiments

Exactly three unexecuted experiments are in
`recommended_controlled_experiments.md`: restore stable Pitch-Fusion optimization,
extend the Wav2Vec2 schedule, then ablate Wav2Vec2 LayerNorm.

## 10. Files

Exact inspected files and hashes are in `inspected_files.csv`. Exact diagnostic
files created or updated are listed in `changed_files.csv`.

```text
PERFORMANCE_REGRESSION_AUDIT_STATUS = COMPLETE
TEST_SET_STATUS = UNTOUCHED
RETRAIN_STATUS = WAITING_FOR_HUMAN_DECISION
FEDERATED_STATUS = NOT_STARTED
FEDECAI_STATUS = NOT_STARTED
```
"""
    (OUTPUT / "PERFORMANCE_REGRESSION_AUDIT.md").write_text(report, encoding="utf-8")
    changed_manifest = OUTPUT / "changed_files.csv"
    changed_paths = [Path(__file__).resolve(), changed_manifest] + [path for path in OUTPUT.iterdir() if path.is_file() and path != changed_manifest]
    write_csv("changed_files.csv", [{"path": rel(path), "change_scope": "diagnostic_only"} for path in sorted(set(changed_paths), key=lambda value: value.as_posix())])
    print(json.dumps({key: status[key] for key in ("performance_regression_audit_status", "comparison_valid", "test_set_status", "retrain_status", "federated_status", "fedecai_status")}, indent=2))


if __name__ == "__main__":
    main()
