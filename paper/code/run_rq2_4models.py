"""Run and analyze RQ2 with the frozen three families plus Wav2Vec2.

Every architecture receives the same in-memory transformed float32 waveform.
This script creates only four-model RQ2 artifacts and never writes RQ1 files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from cf_ser.acoustics import acoustic_features, apply_transform
from cf_ser.config import ARTIFACTS, CHECKPOINTS, FIGURES, LABELS, SAMPLE_RATE, SPLIT_CSV, TRANSFORMS
from cf_ser.data import fix_length, load_audio
from cf_ser.models import MODEL_CLASSES


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["Mel-CNN", "RawWave-CNN", "Wav2Vec2", "Prosody-Fusion"]
INTERVENTIONS = {
    "Pitch": "T1_pitch", "Energy": "T2_energy", "Duration": "T3_duration",
    "P+E": "T4_pitch_energy", "P+D": "T5_pitch_duration",
    "E+D": "T6_energy_duration", "All": "T7_all",
}
SEED = 2025
BOOTSTRAPS = 5000


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial_paired(differences):
    differences = np.asarray(differences, float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if len(differences) == 0:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def bootstrap_mean_ci(values, salt=0):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED + salt)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(BOOTSTRAPS)])
    return np.percentile(means, [2.5, 97.5])


def load_compact_families():
    families = {}
    for key, cls in MODEL_CLASSES.items():
        members = []
        for seed in (2025, 2026, 2027):
            path = CHECKPOINTS / "ensemble" / f"{key}_seed{seed}.pt"
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            model = cls().to(DEVICE)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            members.append((model, checkpoint))
        families[cls.name] = members
    return families


def prosody_batch(feature_rows, norm):
    rows = []
    for features in feature_rows:
        values = np.asarray([features[column] for column in norm["cols"]], np.float32)
        values = np.where(np.isfinite(values), values, np.asarray(norm["median"], np.float32))
        rows.append((values - np.asarray(norm["mean"])) / np.asarray(norm["std"]))
    return torch.tensor(np.asarray(rows), dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def compact_probs(members, waves, feature_rows=None):
    audio = torch.tensor(np.asarray([fix_length(waveform) for waveform in waves]), device=DEVICE)
    member_probabilities = []
    for model, checkpoint in members:
        prosody = None
        if model.__class__.__name__ == "ProsodyFusion":
            if feature_rows is None:
                feature_rows = [acoustic_features(waveform) for waveform in waves]
            prosody = prosody_batch(feature_rows, checkpoint["prosody_norm"])
        member_probabilities.append(torch.softmax(model(audio, prosody), dim=1).cpu().numpy())
    return np.mean(member_probabilities, axis=0)


@torch.no_grad()
def wav2vec_probs(model, extractor, waves):
    fixed = [fix_length(waveform) for waveform in waves]
    encoded = extractor(
        fixed, sampling_rate=SAMPLE_RATE, padding=True, truncation=True,
        max_length=len(fixed[0]), return_tensors="pt",
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    return torch.softmax(model(**encoded).logits, dim=1).cpu().numpy()


@torch.no_grad()
def wav2vec_ensemble_probs(models, extractor, waves):
    return np.mean([wav2vec_probs(model, extractor, waves) for model in models], axis=0)


def slug(model):
    return model.lower().replace("-", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--wav2vec-checkpoints", nargs="+", default=None)
    args = parser.parse_args()
    checkpoint_dirs = [Path(path).resolve() for path in args.wav2vec_checkpoints] if args.wav2vec_checkpoints else [CHECKPOINTS / f"wav2vec2_seed{args.seed}"]
    for checkpoint_dir in checkpoint_dirs:
        if not (checkpoint_dir / "training_summary.json").exists():
            raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint_dir}")

    split = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    neutral = split[(split.split == "test") & (split.emotion == "neutral")].copy()
    params = json.loads((ARTIFACTS / "transformation_params.json").read_text(encoding="utf-8"))
    compact = load_compact_families()
    extractor = AutoFeatureExtractor.from_pretrained(checkpoint_dirs[0])
    wav2vec_models = [AutoModelForAudioClassification.from_pretrained(path).to(DEVICE).eval() for path in checkpoint_dirs]

    rows = []
    transform_names = list(TRANSFORMS)
    for number, row in enumerate(neutral.itertuples(index=False), start=1):
        original = load_audio(row.path)
        wave_map = {name: apply_transform(original, components, params)[0] for name, components in TRANSFORMS.items()}
        waves = [wave_map[name] for name in transform_names]
        hashes = [hashlib.sha256(np.asarray(waveform, np.float32).tobytes()).hexdigest() for waveform in waves]
        feature_rows = [acoustic_features(waveform) for waveform in waves]
        family_probabilities = {name: compact_probs(members, waves, feature_rows) for name, members in compact.items()}
        family_probabilities["Wav2Vec2"] = wav2vec_ensemble_probs(wav2vec_models, extractor, waves)
        sample_id = Path(row.path).stem
        for model_name in MODELS:
            probabilities = family_probabilities[model_name]
            original_probability = probabilities[0]
            original_prediction = LABELS[int(original_probability.argmax())]
            for index, transformation in enumerate(transform_names):
                probability = probabilities[index]
                rows.append({
                    "sample_id": sample_id,
                    "speaker_id": row.speaker,
                    "accent": row.accent,
                    "model": model_name,
                    "transformation": transformation,
                    "waveform_sha256": hashes[index],
                    "original_label": "neutral",
                    "original_prediction": original_prediction,
                    "transformed_prediction": LABELS[int(probability.argmax())],
                    **{f"p_{label}_original": float(original_probability[i]) for i, label in enumerate(LABELS)},
                    **{f"p_{label}_transformed": float(probability[i]) for i, label in enumerate(LABELS)},
                    "delta_p_angry": float(probability[LABELS.index("angry")] - original_probability[LABELS.index("angry")]),
                })
        if number % 20 == 0 or number == len(neutral):
            print(f"RQ2 inference {number}/{len(neutral)}", flush=True)

    predictions = pd.DataFrame(rows)
    if predictions.groupby(["sample_id", "transformation"]).waveform_sha256.nunique().max() != 1:
        raise AssertionError("Models did not receive identical transformed waveforms")
    prediction_path = ARTIFACTS / "rq2_4model_predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    base = predictions[predictions.transformation == "T0_original"]
    correct_sets = {
        model: set(base[(base.model == model) & (base.original_prediction == "neutral")].sample_id)
        for model in MODELS
    }
    common = set.intersection(*correct_sets.values())
    metadata = base[base.sample_id.isin(common)][["sample_id", "speaker_id", "accent"]].drop_duplicates("sample_id")
    for model in MODELS:
        q = base[(base.model == model) & base.sample_id.isin(common)].set_index("sample_id")
        metadata[f"{slug(model)}_original_prediction"] = metadata.sample_id.map(q.original_prediction)
        metadata[f"{slug(model)}_p_angry_original"] = metadata.sample_id.map(q.p_angry_original)
        metadata[f"{slug(model)}_p_neutral_original"] = metadata.sample_id.map(q.p_neutral_original)
    metadata = metadata.sort_values("sample_id").reset_index(drop=True)
    metadata.to_csv(ARTIFACTS / "rq2_common_sample_manifest_4models.csv", index=False)

    result_rows = []
    value_map = {}
    salt = 0
    for model in MODELS:
        for label, code in INTERVENTIONS.items():
            q = predictions[(predictions.model == model) & (predictions.transformation == code) & predictions.sample_id.isin(common)]
            q = q.drop_duplicates("sample_id").set_index("sample_id").sort_index()
            if set(q.index) != common:
                raise AssertionError(f"Incomplete common data: {model}, {label}")
            values = q.delta_p_angry
            value_map[(model, label)] = values
            low, high = bootstrap_mean_ci(values, salt)
            salt += 1
            result_rows.append({
                "model": model, "intervention": label, "n": len(values),
                "mean_delta_p_angry": values.mean(), "median_delta_p_angry": values.median(),
                "std_delta_p_angry": values.std(ddof=0), "ci95_low": low, "ci95_high": high,
                "angry_flip_rate": np.mean(q.transformed_prediction == "angry"),
            })
    results = pd.DataFrame(result_rows)
    results["sensitivity_rank"] = results.groupby("model").mean_delta_p_angry.rank(ascending=False, method="min").astype(int)
    results.to_csv(ARTIFACTS / "RQ2_4MODEL_FINAL_RESULTS.csv", index=False)

    tests = []
    for intervention in ("Pitch", "Energy", "Duration"):
        arrays = [value_map[(model, intervention)].sort_index().to_numpy() for model in MODELS]
        statistic, p_value = friedmanchisquare(*arrays)
        tests.append({
            "test_type": "Friedman", "intervention": intervention,
            "comparison": " vs ".join(MODELS), "n": len(common),
            "statistic": statistic, "p_value_raw": p_value, "p_value_holm": p_value,
            "effect_size": statistic / (len(common) * (len(MODELS) - 1)),
            "effect_size_name": "Kendall_W", "direction": "omnibus",
        })
        if p_value < 0.05:
            posthoc_indices = []
            for left, right in combinations(range(len(MODELS)), 2):
                difference = arrays[left] - arrays[right]
                try:
                    stat, raw_p = wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
                except ValueError:
                    stat, raw_p = 0.0, 1.0
                low, high = bootstrap_mean_ci(difference, salt)
                salt += 1
                tests.append({
                    "test_type": "Wilcoxon post-hoc", "intervention": intervention,
                    "comparison": f"{MODELS[left]} vs {MODELS[right]}", "n": len(common),
                    "statistic": stat, "p_value_raw": raw_p, "p_value_holm": np.nan,
                    "effect_size": rank_biserial_paired(difference),
                    "effect_size_name": "matched_rank_biserial",
                    "direction": "A > B" if difference.mean() > 0 else "A < B",
                    "mean_paired_difference": difference.mean(),
                    "median_paired_difference": np.median(difference),
                    "ci95_low": low, "ci95_high": high,
                })
                posthoc_indices.append(len(tests) - 1)
            adjusted = holm([tests[index]["p_value_raw"] for index in posthoc_indices])
            for index, value in zip(posthoc_indices, adjusted):
                tests[index]["p_value_holm"] = value

    mean_matrix = results.pivot(index="model", columns="intervention", values="mean_delta_p_angry").reindex(index=MODELS, columns=list(INTERVENTIONS))
    flip_matrix = results.pivot(index="model", columns="intervention", values="angry_flip_rate").reindex(index=MODELS, columns=list(INTERVENTIONS))
    for left, right in combinations(MODELS, 2):
        rho, raw_p = spearmanr(mean_matrix.loc[left], mean_matrix.loc[right])
        tests.append({
            "test_type": "Spearman rank correlation", "intervention": "all seven",
            "comparison": f"{left} vs {right}", "n": len(INTERVENTIONS),
            "statistic": rho, "p_value_raw": raw_p, "p_value_holm": np.nan,
            "effect_size": rho, "effect_size_name": "Spearman_rho",
            "direction": "ranking agreement",
        })
    test_frame = pd.DataFrame(tests)
    correlation_indices = test_frame.index[test_frame.test_type == "Spearman rank correlation"]
    test_frame.loc[correlation_indices, "p_value_holm"] = holm(test_frame.loc[correlation_indices, "p_value_raw"])
    test_frame.to_csv(ARTIFACTS / "RQ2_4MODEL_STATISTICAL_TESTS.csv", index=False)
    mean_matrix.to_csv(ARTIFACTS / "rq2_4model_delta_matrix.csv")
    flip_matrix.to_csv(ARTIFACTS / "rq2_4model_flip_matrix.csv")

    FIGURES.mkdir(parents=True, exist_ok=True)
    for matrix, filename, title, fmt in [
        (mean_matrix, "rq2_4model_intervention_heatmap.png", "Mean delta P(Angry) on four-model common subset", ".3f"),
        (flip_matrix, "rq2_4model_flip_heatmap.png", "Angry flip rate on four-model common subset", ".1%"),
    ]:
        fig, ax = plt.subplots(figsize=(9.4, 4.5))
        values = matrix.to_numpy()
        if filename == "rq2_4model_intervention_heatmap.png":
            limit = max(abs(values.min()), abs(values.max()), 1e-6)
            image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=max(values.max(), 1e-6))
        ax.set_xticks(range(matrix.shape[1]), matrix.columns)
        ax.set_yticks(range(matrix.shape[0]), matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=9)
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=240)
        plt.close(fig)

    rankings = {
        model: list(mean_matrix.loc[model].sort_values(ascending=False).index)
        for model in MODELS
    }
    wav_summaries = [json.loads((path / "training_summary.json").read_text(encoding="utf-8")) for path in checkpoint_dirs]
    ensemble_baseline_path = ARTIFACTS / "WAV2VEC2_ENSEMBLE_BASELINE_RESULTS.csv"
    if ensemble_baseline_path.exists() and len(checkpoint_dirs) > 1:
        wav_test_macro_f1 = float(pd.read_csv(ensemble_baseline_path).iloc[0].test_macro_f1)
    else:
        wav_test_macro_f1 = float(np.mean([summary["test_macro_f1"] for summary in wav_summaries]))
    friedman_rows = test_frame[test_frame.test_type == "Friedman"].set_index("intervention")
    pitch_posthoc = test_frame[(test_frame.test_type == "Wilcoxon post-hoc") & (test_frame.intervention == "Pitch")]
    correlations = test_frame[test_frame.test_type == "Spearman rank correlation"]
    pitch_significant = int((pitch_posthoc.p_value_holm < 0.05).sum())
    correlation_significant = int((correlations.p_value_holm < 0.05).sum())
    lines = [
        "# RQ2 four-model findings", "",
        f"The four-model common subset contains **N={len(common)}** Neutral test utterances correctly classified as Neutral by all four models before intervention.", "",
        f"Wav2Vec2 baseline test Macro-F1: **{wav_test_macro_f1:.4f}** ({len(checkpoint_dirs)} checkpoint(s)).", "",
        "## Mean delta P(Angry)", "", mean_matrix.to_markdown(), "",
        "## Angry flip rate", "", flip_matrix.to_markdown(), "",
        "## Intervention rankings", "",
    ]
    lines.extend([f"- {model}: {' > '.join(order)}" for model, order in rankings.items()])
    lines.extend(["", "## Statistical results", ""])
    for intervention in ("Pitch", "Energy", "Duration"):
        row = friedman_rows.loc[intervention]
        lines.append(f"- {intervention}: Friedman chi-square={row.statistic:.3f}, p={row.p_value_raw:.3g}, Kendall's W={row.effect_size:.3f}.")
    lines.extend(["", f"{pitch_significant}/6 Pitch model-pair comparisons were significant after Holm correction:", ""])
    for row in pitch_posthoc.itertuples(index=False):
        lines.append(f"- {row.comparison}: mean paired difference={row.mean_paired_difference:.4f}, adjusted p={row.p_value_holm:.3g}, matched rank-biserial={row.effect_size:.3f}.")
    lines.extend(["", "## Interpretation", "",
        f"The pretrained baseline was the strongest classifier (test Macro-F1 {wav_test_macro_f1:.4f}) but had a Pitch response of {mean_matrix.loc['Wav2Vec2','Pitch']:.4f} on the common subset. Higher baseline classification performance therefore did not imply greater sensitivity to the manipulated F0 cue.", "",
        f"Pairwise seven-intervention Spearman correlations ranged from {correlations.statistic.min():.3f} to {correlations.statistic.max():.3f}; {correlation_significant}/6 survived Holm correction. These rankings are supporting descriptive evidence only.", "",
        "Overall, the four architecture families show different intervention-sensitivity profiles. Statistical claims are based on paired observations from the same common utterances. The results do not establish different internal representations.",
    ])
    (ARTIFACTS / "RQ2_4MODEL_FINAL_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Four-model RQ2 complete. Common subset N={len(common)}")
    print(mean_matrix.to_string())
    print(flip_matrix.to_string())


if __name__ == "__main__":
    main()
