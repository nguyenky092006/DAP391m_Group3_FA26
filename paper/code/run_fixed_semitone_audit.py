"""Run fixed-semitone F0-baseline interventions on all Neutral test items."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from cf_ser.acoustics import acoustic_features, pitch_transform
from cf_ser.config import ARTIFACTS, CHECKPOINTS, LABELS, SPLIT_CSV
from cf_ser.data import load_audio
from cf_ser.models import ProsodyFusion
from run_rq2_4models import (
    DEVICE, compact_probs, load_compact_families, prosody_batch,
    wav2vec_ensemble_probs,
)


SEED = 2025
CONDITIONS = [
    ("original", None), ("sham", 0.0), ("minus_2st", -2.0),
    ("plus_1st", 1.0), ("plus_2st", 2.0), ("plus_3st", 3.0),
    ("plus_4st", 4.0), ("plus_6st", 6.0),
]
PRIMARY_MODELS = ["Mel-CNN", "RawWave-CNN", "Wav2Vec2", "Prosody-Fusion"]


def ci(values, salt=0):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED + salt)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(5000)]
    return np.percentile(means, [2.5, 97.5])


@torch.no_grad()
def no_f0_probs(members, waves, feature_rows):
    from cf_ser.data import fix_length
    audio = torch.tensor(np.asarray([fix_length(wave) for wave in waves]), device=DEVICE)
    probabilities = []
    for model, checkpoint in members:
        prosody = prosody_batch(feature_rows, checkpoint["prosody_norm"])
        prosody[:, :2] = 0.0
        probabilities.append(torch.softmax(model(audio, prosody), dim=1).cpu().numpy())
    return np.mean(probabilities, axis=0)


def load_no_f0_members():
    members = []
    directory = CHECKPOINTS / "prosody_no_f0_ablation"
    for seed in (2025, 2026, 2027):
        path = directory / f"prosody_fusion_no_f0_seed{seed}.pt"
        if not path.exists():
            return []
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        model = ProsodyFusion().to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        members.append((model, checkpoint))
    return members


def main():
    split = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    neutral = split[(split.split == "test") & (split.emotion == "neutral")].copy()
    params = json.loads((ARTIFACTS / "transformation_params.json").read_text(encoding="utf-8"))
    corpus_delta = params["mu_logf0_angry"] - params["mu_logf0_neutral"]
    compact = load_compact_families()
    wav_dirs = [CHECKPOINTS / "wav2vec2_ensemble" / f"wav2vec2_seed{seed}" for seed in (2025, 2026, 2027)]
    extractor = AutoFeatureExtractor.from_pretrained(wav_dirs[0])
    wav_models = [AutoModelForAudioClassification.from_pretrained(path).to(DEVICE).eval() for path in wav_dirs]
    no_f0 = load_no_f0_members()
    model_names = PRIMARY_MODELS + (["Prosody-Fusion-noF0"] if no_f0 else [])

    rows = []
    for number, row in enumerate(neutral.itertuples(index=False), start=1):
        original = load_audio(row.path)
        waves = []
        achieved = []
        for _, semitones in CONDITIONS:
            if semitones is None:
                waves.append(original)
                achieved.append(0.0)
            else:
                log_shift = semitones * np.log(2) / 12
                strength = log_shift / corpus_delta
                transformed, applied = pitch_transform(original, params, strength=strength, mode="mean")
                waves.append(transformed)
                achieved.append(float(12 * applied / np.log(2)))
        features = [acoustic_features(wave) for wave in waves]
        probabilities = {name: compact_probs(members, waves, features) for name, members in compact.items()}
        probabilities["Wav2Vec2"] = wav2vec_ensemble_probs(wav_models, extractor, waves)
        if no_f0:
            probabilities["Prosody-Fusion-noF0"] = no_f0_probs(no_f0, waves, features)
        sample_id = Path(row.path).stem
        for model_name in model_names:
            values = probabilities[model_name]
            original_angry = float(values[0, LABELS.index("angry")])
            sham_angry = float(values[1, LABELS.index("angry")])
            original_prediction = LABELS[int(values[0].argmax())]
            for index, ((condition, requested), applied) in enumerate(zip(CONDITIONS, achieved)):
                probability = values[index]
                rows.append({
                    "sample_id": sample_id, "speaker_id": row.speaker, "accent": row.accent,
                    "model": model_name, "condition": condition,
                    "requested_semitones": requested, "achieved_mean_semitones": applied,
                    "original_prediction": original_prediction,
                    "transformed_prediction": LABELS[int(probability.argmax())],
                    "p_angry": float(probability[LABELS.index("angry")]),
                    "delta_p_angry_from_original": float(probability[LABELS.index("angry")] - original_angry),
                    "delta_p_angry_from_sham": float(probability[LABELS.index("angry")] - sham_angry),
                })
        if number % 20 == 0 or number == len(neutral):
            print(f"fixed-semitone inference {number}/{len(neutral)}", flush=True)

    predictions = pd.DataFrame(rows)
    predictions.to_csv(ARTIFACTS / "FIXED_SEMITONE_PREDICTIONS.csv", index=False)
    base = predictions[predictions.condition == "original"]
    correct = {
        model: set(base[(base.model == model) & (base.original_prediction == "neutral")].sample_id)
        for model in PRIMARY_MODELS
    }
    common = set.intersection(*correct.values())

    summaries = []
    salt = 0
    for analysis_set in ("all_true_neutral", "model_baseline_correct", "four_model_common_correct"):
        for model_name in model_names:
            for condition, requested in CONDITIONS[1:]:
                q = predictions[(predictions.model == model_name) & (predictions.condition == condition)]
                if analysis_set == "model_baseline_correct":
                    valid = correct.get(model_name, set(base[(base.model == model_name) & (base.original_prediction == "neutral")].sample_id))
                    q = q[q.sample_id.isin(valid)]
                elif analysis_set == "four_model_common_correct":
                    q = q[q.sample_id.isin(common)]
                low, high = ci(q.delta_p_angry_from_original, salt)
                salt += 1
                sham_low, sham_high = ci(q.delta_p_angry_from_sham, salt)
                salt += 1
                summaries.append({
                    "analysis_set": analysis_set, "model": model_name, "condition": condition,
                    "requested_semitones": requested, "n": len(q),
                    "mean_achieved_semitones": q.achieved_mean_semitones.mean(),
                    "mean_delta_p_angry": q.delta_p_angry_from_original.mean(),
                    "ci95_low": low, "ci95_high": high,
                    "mean_sham_adjusted_delta_p_angry": q.delta_p_angry_from_sham.mean(),
                    "sham_adjusted_ci95_low": sham_low, "sham_adjusted_ci95_high": sham_high,
                    "angry_flip_rate": np.mean(q.transformed_prediction == "angry"),
                })
    summary = pd.DataFrame(summaries)
    summary.to_csv(ARTIFACTS / "FIXED_SEMITONE_RESULTS.csv", index=False)
    manifest = base[(base.model == "Mel-CNN") & base.sample_id.isin(common)][["sample_id", "speaker_id", "accent"]]
    manifest.to_csv(ARTIFACTS / "FIXED_SEMITONE_COMMON_MANIFEST.csv", index=False)

    main_rows = summary[(summary.analysis_set == "all_true_neutral") & summary.condition.isin(["plus_2st", "plus_4st", "plus_6st"])]
    lines = [
        "# Fixed-semitone F0-baseline audit", "",
        f"Primary population: all {len(neutral)} ground-truth Neutral test utterances. Four-model common-correct N={len(common)}.", "",
        "The fixed semitone grid is primary; the pooled corpus-derived shift is secondary. Sham-adjusted effects separate resynthesis from requested F0 displacement.", "",
        "| Model | Dose | N | Mean delta P(Angry) | 95% CI | Sham-adjusted mean | Angry output rate |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows.itertuples(index=False):
        lines.append(f"| {row.model} | {row.requested_semitones:+.0f} st | {row.n} | {row.mean_delta_p_angry:.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] | {row.mean_sham_adjusted_delta_p_angry:.4f} | {100*row.angry_flip_rate:.2f}% |")
    (ARTIFACTS / "FIXED_SEMITONE_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(main_rows.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
