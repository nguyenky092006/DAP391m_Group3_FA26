"""Audit whether waveform interventions survive each SER input pipeline.

This script does not retrain or alter any checkpoint.  It measures the same
transformed waveform at the signal level and immediately after the actual
preprocessing used by Mel-CNN, RawWave-CNN, Wav2Vec2, and Prosody-Fusion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor

from cf_ser.acoustics import acoustic_features, apply_transform
from cf_ser.config import ARTIFACTS, CHECKPOINTS, SAMPLE_RATE, SPLIT_CSV, TRANSFORMS
from cf_ser.data import fix_length, load_audio
from cf_ser.models import MelFrontEnd


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12


def relative_l2(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), EPS))


def rms_db(wave):
    wave = np.asarray(wave, dtype=np.float64)
    return float(20 * np.log10(max(np.sqrt(np.mean(wave * wave)), EPS)))


def standardized_prosody(features, norm):
    values = np.asarray([features[column] for column in norm["cols"]], np.float64)
    median = np.asarray(norm["median"], np.float64)
    values = np.where(np.isfinite(values), values, median)
    return (values - np.asarray(norm["mean"], np.float64)) / np.asarray(norm["std"], np.float64)


@torch.no_grad()
def main():
    split = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    neutral = split[(split.split == "test") & (split.emotion == "neutral")].copy()
    params = json.loads((ARTIFACTS / "transformation_params.json").read_text(encoding="utf-8"))

    checkpoint = torch.load(
        CHECKPOINTS / "ensemble" / "prosody_fusion_seed2025.pt",
        map_location="cpu", weights_only=False,
    )
    prosody_norm = checkpoint["prosody_norm"]
    wav_dir = CHECKPOINTS / "wav2vec2_ensemble" / "wav2vec2_seed2025"
    extractor = AutoFeatureExtractor.from_pretrained(wav_dir)
    mel_front = MelFrontEnd().to(DEVICE).eval()

    config_rows = [
        {"pipeline": "Mel-CNN", "normalization": "per-utterance log-Mel mean/SD"},
        {"pipeline": "RawWave-CNN", "normalization": "none after fixed-length waveform"},
        {"pipeline": "Wav2Vec2", "normalization": f"feature extractor do_normalize={extractor.do_normalize}"},
        {"pipeline": "Prosody-Fusion", "normalization": "normalized log-Mel plus standardized explicit prosody"},
    ]
    pd.DataFrame(config_rows).to_csv(ARTIFACTS / "PREPROCESSING_CONFIG_AUDIT.csv", index=False)

    rows = []
    transform_items = list(TRANSFORMS.items())
    for number, row in enumerate(neutral.itertuples(index=False), start=1):
        original = load_audio(row.path)
        waves = {name: apply_transform(original, components, params)[0] for name, components in transform_items}
        fixed = {name: fix_length(wave) for name, wave in waves.items()}
        base = fixed["T0_original"]

        audio_tensor = torch.tensor(np.stack(list(fixed.values())), dtype=torch.float32, device=DEVICE)
        mel_values = mel_front(audio_tensor).cpu().numpy()
        mel_map = dict(zip(fixed, mel_values))
        encoded = extractor(
            list(fixed.values()), sampling_rate=SAMPLE_RATE, padding=True,
            truncation=True, max_length=len(base), return_tensors="np",
        )["input_values"]
        wav2vec_map = dict(zip(fixed, encoded))
        feature_map = {name: acoustic_features(wave) for name, wave in waves.items()}
        prosody_map = {name: standardized_prosody(features, prosody_norm) for name, features in feature_map.items()}

        for name, _ in transform_items:
            signal_features = feature_map[name]
            base_features = feature_map["T0_original"]
            p_delta = prosody_map[name] - prosody_map["T0_original"]
            rows.append({
                "sample_id": Path(row.path).stem,
                "transformation": name,
                "waveform_rel_l2": relative_l2(base, fixed[name]),
                "waveform_rms_db_delta": rms_db(fixed[name]) - rms_db(base),
                "achieved_logf0_mean_delta": signal_features["logf0_mean"] - base_features["logf0_mean"],
                "achieved_duration_delta_seconds": signal_features["duration"] - base_features["duration"],
                "raw_input_rel_l2": relative_l2(base, fixed[name]),
                "mel_input_rel_l2": relative_l2(mel_map["T0_original"], mel_map[name]),
                "wav2vec_input_rel_l2": relative_l2(wav2vec_map["T0_original"], wav2vec_map[name]),
                "prosody_all_l2_delta": float(np.linalg.norm(p_delta)),
                "prosody_f0_l2_delta": float(np.linalg.norm(p_delta[:2])),
                "prosody_energy_abs_delta": float(abs(p_delta[2])),
                "prosody_duration_abs_delta": float(abs(p_delta[3])),
            })
        if number % 25 == 0 or number == len(neutral):
            print(f"preprocessing audit {number}/{len(neutral)}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(ARTIFACTS / "PREPROCESSING_INTERVENTION_AUDIT_DETAIL.csv", index=False)
    metrics = [column for column in detail if column not in {"sample_id", "transformation"}]
    summary = detail.groupby("transformation")[metrics].agg(["mean", "median", "max"])
    summary.columns = [f"{left}_{right}" for left, right in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(ARTIFACTS / "PREPROCESSING_INTERVENTION_AUDIT_SUMMARY.csv", index=False)

    energy = summary[summary.transformation == "T2_energy"].iloc[0]
    findings = [
        "# Preprocessing-aware intervention audit", "",
        f"Audited all {len(neutral)} ground-truth Neutral test utterances. The audit measures intervention observability before model inference.", "",
        "## Energy-only diagnostic", "",
        f"- Mean waveform RMS change: {energy.waveform_rms_db_delta_mean:.6f} dB.",
        f"- RawWave input relative L2 change: {energy.raw_input_rel_l2_mean:.6g}.",
        f"- Mel-CNN normalized log-Mel relative L2 change: {energy.mel_input_rel_l2_mean:.6g}.",
        f"- Wav2Vec2 normalized input relative L2 change: {energy.wav2vec_input_rel_l2_mean:.6g}.",
        f"- Prosody-Fusion explicit standardized log-RMS change: {energy.prosody_energy_abs_delta_mean:.6g}.", "",
        "## Interpretation rule", "",
        "Differences are interpreted as end-to-end **pipeline sensitivity**. A near-zero post-preprocessing distance means that an intervention is not observable to that input branch; it must not be interpreted as evidence that the learned representation ignored the cue.", "",
        f"Wav2Vec2 feature extractor setting: `do_normalize={extractor.do_normalize}`.",
    ]
    (ARTIFACTS / "PREPROCESSING_INTERVENTION_AUDIT_FINDINGS.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
