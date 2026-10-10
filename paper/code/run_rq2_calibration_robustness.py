"""Validation-only temperature scaling and all-Neutral RQ2 robustness.

This script never trains a model and never fits a parameter on test data.  A
single scalar temperature is fitted to each finalized family ensemble using
the frozen validation split, then applied unchanged to the saved RQ2 test
probabilities.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from cf_ser.acoustics import acoustic_features
from cf_ser.config import ARTIFACTS, CHECKPOINTS, LABELS, SPLIT_CSV
from cf_ser.data import load_audio
from run_rq2_4models import (
    DEVICE, INTERVENTIONS, MODELS, compact_probs, load_compact_families,
    wav2vec_ensemble_probs,
)


SEED = 2025
BOOTSTRAPS = 5000
PROB_ORIGINAL = [f"p_{label}_original" for label in LABELS]
PROB_TRANSFORMED = [f"p_{label}_transformed" for label in LABELS]


def temperature_scale(probabilities, temperature):
    logp = np.log(np.clip(np.asarray(probabilities, float), 1e-12, 1.0)) / float(temperature)
    logp -= logp.max(axis=1, keepdims=True)
    scaled = np.exp(logp)
    return scaled / scaled.sum(axis=1, keepdims=True)


def nll(probabilities, labels):
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def brier(probabilities, labels):
    target = np.eye(len(LABELS))[labels]
    return float(np.square(probabilities - target).sum(axis=1).mean())


def ece(probabilities, labels, bins=15):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            total += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(total)


def fit_temperature(probabilities, labels):
    result = minimize_scalar(
        lambda log_temperature: nll(temperature_scale(probabilities, np.exp(log_temperature)), labels),
        bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def build_validation_predictions(batch_compact=64, batch_wav2vec=8):
    cache = ARTIFACTS / "RQ2_VALIDATION_PREDICTIONS_FOR_CALIBRATION.csv"
    required = {"sample_id", "model", "true_id", *[f"p_{label}" for label in LABELS]}
    split = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    validation = split[split.split.eq("val")].copy()
    validation["sample_id"] = validation.path.map(lambda value: Path(value).stem)
    if cache.exists():
        saved = pd.read_csv(cache, dtype={"sample_id": str})
        if required.issubset(saved.columns) and len(saved) == len(validation) * len(MODELS):
            return saved

    compact = load_compact_families()
    checkpoint_dirs = [CHECKPOINTS / "wav2vec2_ensemble" / f"wav2vec2_seed{seed}" for seed in (2025, 2026, 2027)]
    extractor = AutoFeatureExtractor.from_pretrained(checkpoint_dirs[0])
    wav2vec_models = [AutoModelForAudioClassification.from_pretrained(path).to(DEVICE).eval() for path in checkpoint_dirs]
    waves = [load_audio(path) for path in validation.path]
    rows = []
    for model, members in compact.items():
        parts = []
        for start in range(0, len(waves), batch_compact):
            batch = waves[start:start + batch_compact]
            features = [acoustic_features(waveform) for waveform in batch]
            parts.append(compact_probs(members, batch, features))
        probabilities = np.vstack(parts)
        for row, probability in zip(validation.itertuples(index=False), probabilities):
            record = {"sample_id": row.sample_id, "path": row.path, "model": model, "true_id": int(row.emotion_id), "true_label": row.emotion}
            record.update({f"p_{label}": float(probability[index]) for index, label in enumerate(LABELS)})
            rows.append(record)
        print(f"Validation probabilities: {model}", flush=True)
    parts = []
    for start in range(0, len(waves), batch_wav2vec):
        stop = min(len(waves), start + batch_wav2vec)
        parts.append(wav2vec_ensemble_probs(wav2vec_models, extractor, waves[start:stop]))
        if stop % 80 == 0 or stop == len(waves):
            print(f"Validation Wav2Vec2 {stop}/{len(waves)}", flush=True)
    probabilities = np.vstack(parts)
    for row, probability in zip(validation.itertuples(index=False), probabilities):
        record = {"sample_id": row.sample_id, "path": row.path, "model": "Wav2Vec2", "true_id": int(row.emotion_id), "true_label": row.emotion}
        record.update({f"p_{label}": float(probability[index]) for index, label in enumerate(LABELS)})
        rows.append(record)
    result = pd.DataFrame(rows)
    if result.duplicated(["sample_id", "model"]).any():
        raise AssertionError("Duplicate validation prediction")
    result.to_csv(cache, index=False)
    return result


def fit_calibrators(validation):
    rows, temperatures = [], {}
    columns = [f"p_{label}" for label in LABELS]
    for model in MODELS:
        group = validation[validation.model.eq(model)]
        probabilities = group[columns].to_numpy(float)
        labels = group.true_id.to_numpy(int)
        temperature = fit_temperature(probabilities, labels)
        calibrated = temperature_scale(probabilities, temperature)
        temperatures[model] = temperature
        rows.append({
            "model": model, "fitting_split": "validation", "n_validation": len(group),
            "temperature": temperature, "nll_before": nll(probabilities, labels),
            "nll_after": nll(calibrated, labels), "brier_before": brier(probabilities, labels),
            "brier_after": brier(calibrated, labels), "ece_15bin_before": ece(probabilities, labels),
            "ece_15bin_after": ece(calibrated, labels),
            "test_used_for_fit": False,
        })
    calibration = pd.DataFrame(rows)
    calibration.to_csv(ARTIFACTS / "RQ2_CALIBRATION_FINAL.csv", index=False)
    return temperatures, calibration


def calibrate_rq2_predictions(temperatures):
    frame = pd.read_csv(ARTIFACTS / "rq2_4model_predictions.csv", dtype={"sample_id": str, "speaker_id": str})
    output = []
    for model, group in frame.groupby("model", sort=False):
        group = group.copy()
        original = temperature_scale(group[PROB_ORIGINAL].to_numpy(float), temperatures[model])
        transformed = temperature_scale(group[PROB_TRANSFORMED].to_numpy(float), temperatures[model])
        for index, label in enumerate(LABELS):
            group[f"p_{label}_original_calibrated"] = original[:, index]
            group[f"p_{label}_transformed_calibrated"] = transformed[:, index]
        group["delta_p_angry_calibrated"] = transformed[:, LABELS.index("angry")] - original[:, LABELS.index("angry")]
        output.append(group)
    calibrated = pd.concat(output, ignore_index=True)
    if calibrated.duplicated(["sample_id", "model", "transformation"]).any():
        raise AssertionError("Duplicate RQ2 calibrated prediction")
    return calibrated


def bootstrap_ci(values, salt):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED + salt)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    return tuple(np.percentile(values[indices].mean(axis=1), [2.5, 97.5]))


def summarize(frame, sample_ids, analysis_set):
    rows, salt = [], 0
    selected = frame[frame.sample_id.isin(sample_ids) & frame.transformation.isin(INTERVENTIONS.values())].copy()
    expected = len(sample_ids) * len(MODELS) * len(INTERVENTIONS)
    if len(selected) != expected:
        raise AssertionError(f"Incomplete {analysis_set}: {len(selected)} != {expected}")
    reverse = {value: key for key, value in INTERVENTIONS.items()}
    for model in MODELS:
        for code in INTERVENTIONS.values():
            group = selected[selected.model.eq(model) & selected.transformation.eq(code)]
            raw = group.delta_p_angry.to_numpy(float)
            calibrated = group.delta_p_angry_calibrated.to_numpy(float)
            raw_lo, raw_hi = bootstrap_ci(raw, salt); salt += 1
            cal_lo, cal_hi = bootstrap_ci(calibrated, salt); salt += 1
            rows.append({
                "analysis_set": analysis_set, "model": model, "intervention": reverse[code],
                "transformation": code, "n": len(group),
                "raw_mean_delta_p_angry": raw.mean(), "raw_median_delta_p_angry": np.median(raw),
                "raw_ci95_low": raw_lo, "raw_ci95_high": raw_hi,
                "calibrated_mean_delta_p_angry": calibrated.mean(),
                "calibrated_median_delta_p_angry": np.median(calibrated),
                "calibrated_ci95_low": cal_lo, "calibrated_ci95_high": cal_hi,
            })
    return pd.DataFrame(rows)


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial(differences):
    differences = np.asarray(differences, float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def calibrated_statistics(calibrated, common_ids, common_summary):
    selected = calibrated[calibrated.sample_id.isin(common_ids) & calibrated.transformation.isin(INTERVENTIONS.values())].copy()
    pivot = selected.pivot(index="sample_id", columns=["model", "transformation"], values="delta_p_angry_calibrated")
    rows = []
    for intervention in ["Pitch", "Energy", "Duration"]:
        code = INTERVENTIONS[intervention]
        arrays = [pivot[(model, code)].to_numpy() for model in MODELS]
        statistic, p_value = friedmanchisquare(*arrays)
        rows.append({
            "analysis": "Friedman", "intervention": intervention, "model_a": "All", "model_b": "All",
            "n": len(pivot), "statistic": statistic, "p_value_raw": p_value,
            "p_value_holm": p_value, "mean_difference": np.nan,
            "effect_size": statistic / (len(pivot) * (len(MODELS) - 1)), "effect_size_name": "Kendall_W",
            "notes": "calibrated common subset",
        })
        pair_rows = []
        if p_value < 0.05:
            for model_a, model_b in combinations(MODELS, 2):
                difference = pivot[(model_a, code)] - pivot[(model_b, code)]
                try:
                    stat, raw_p = wilcoxon(difference, zero_method="wilcox")
                except ValueError:
                    stat, raw_p = 0.0, 1.0
                pair_rows.append({
                    "analysis": "paired_Wilcoxon", "intervention": intervention,
                    "model_a": model_a, "model_b": model_b, "n": len(difference),
                    "statistic": stat, "p_value_raw": raw_p, "p_value_holm": np.nan,
                    "mean_difference": difference.mean(), "effect_size": rank_biserial(difference),
                    "effect_size_name": "matched_rank_biserial", "notes": "Holm within intervention",
                })
            adjusted = holm([row["p_value_raw"] for row in pair_rows])
            for row, value in zip(pair_rows, adjusted):
                row["p_value_holm"] = value
            rows.extend(pair_rows)

    for model in MODELS:
        group = common_summary[common_summary.model.eq(model)].set_index("intervention")
        raw = group.loc[list(INTERVENTIONS), "raw_mean_delta_p_angry"]
        calibrated_values = group.loc[list(INTERVENTIONS), "calibrated_mean_delta_p_angry"]
        rho, p_value = spearmanr(raw, calibrated_values)
        rows.append({
            "analysis": "raw_vs_calibrated_profile_rank", "intervention": "All seven",
            "model_a": model, "model_b": model, "n": len(INTERVENTIONS), "statistic": rho,
            "p_value_raw": p_value, "p_value_holm": np.nan, "mean_difference": np.nan,
            "effect_size": rho, "effect_size_name": "Spearman_rho",
            "notes": "raw versus calibrated intervention ranking",
        })
    return pd.DataFrame(rows)


def ranking(summary, column):
    return {
        model: list(group.sort_values(column, ascending=False).intervention)
        for model, group in summary.groupby("model", sort=False)
    }


def write_all_neutral_findings(common, all_neutral):
    raw_common = ranking(common, "raw_mean_delta_p_angry")
    raw_all = ranking(all_neutral, "raw_mean_delta_p_angry")
    cal_common = ranking(common, "calibrated_mean_delta_p_angry")
    cal_all = ranking(all_neutral, "calibrated_mean_delta_p_angry")
    lines = [
        "# RQ2 all-Neutral robustness", "",
        "This is a secondary robustness analysis. The strict four-model common subset remains primary.", "",
        f"All **{int(all_neutral.n.iloc[0])}** ground-truth Neutral test utterances with complete four-model predictions were included. No baseline-correctness or intervention-response filter was applied.", "",
        "| Model | Raw ranking, common N=36 | Raw ranking, all Neutral | Calibrated ranking, common N=36 | Calibrated ranking, all Neutral |",
        "|---|---|---|---|---|",
    ]
    raw_rhos, cal_rhos = [], []
    for model in MODELS:
        positions_a = [raw_common[model].index(name) for name in INTERVENTIONS]
        positions_b = [raw_all[model].index(name) for name in INTERVENTIONS]
        raw_rho = spearmanr(positions_a, positions_b).statistic
        positions_a = [cal_common[model].index(name) for name in INTERVENTIONS]
        positions_b = [cal_all[model].index(name) for name in INTERVENTIONS]
        cal_rho = spearmanr(positions_a, positions_b).statistic
        raw_rhos.append(raw_rho); cal_rhos.append(cal_rho)
        lines.append(
            f"| {model} | {' > '.join(raw_common[model])} | {' > '.join(raw_all[model])} | "
            f"{' > '.join(cal_common[model])} | {' > '.join(cal_all[model])} |"
        )
    pitch = all_neutral[all_neutral.intervention.eq("Pitch")].sort_values("calibrated_mean_delta_p_angry", ascending=False)
    lines.extend([
        "", "## Assessment", "",
        f"- Raw common-vs-all-Neutral ranking correlations by model: {', '.join(f'{m}={v:.3f}' for m, v in zip(MODELS, raw_rhos))}.",
        f"- Calibrated common-vs-all-Neutral ranking correlations by model: {', '.join(f'{m}={v:.3f}' for m, v in zip(MODELS, cal_rhos))}.",
        f"- Calibrated all-Neutral Pitch ordering: {' > '.join(pitch.model)}.",
        "- Differences from the strict common subset are reported as robustness variation, not used to replace the primary controlled analysis.",
    ])
    (ARTIFACTS / "RQ2_ALL_NEUTRAL_ROBUSTNESS_FINDINGS_FINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    validation = build_validation_predictions()
    temperatures, calibration = fit_calibrators(validation)
    calibrated = calibrate_rq2_predictions(temperatures)
    common_manifest = pd.read_csv(ARTIFACTS / "rq2_common_sample_manifest_4models.csv", dtype={"sample_id": str})
    common_ids = set(common_manifest.sample_id)
    all_ids = set(calibrated.sample_id)
    if len(common_ids) != 36 or len(all_ids) != 227:
        raise AssertionError(f"Unexpected RQ2 sample counts: common={len(common_ids)}, all={len(all_ids)}")
    common = summarize(calibrated, common_ids, "primary_common_baseline_correct")
    common.to_csv(ARTIFACTS / "RQ2_CALIBRATED_RESULTS_FINAL.csv", index=False)
    statistics = calibrated_statistics(calibrated, common_ids, common)
    statistics.to_csv(ARTIFACTS / "RQ2_CALIBRATED_STATS_FINAL.csv", index=False)
    all_neutral = summarize(calibrated, all_ids, "secondary_all_true_neutral")
    all_neutral.to_csv(ARTIFACTS / "RQ2_ALL_NEUTRAL_ROBUSTNESS_FINAL.csv", index=False)
    write_all_neutral_findings(common, all_neutral)

    raw_pitch = common[common.intervention.eq("Pitch")].sort_values("raw_mean_delta_p_angry", ascending=False).model.tolist()
    calibrated_pitch = common[common.intervention.eq("Pitch")].sort_values("calibrated_mean_delta_p_angry", ascending=False).model.tolist()
    profile_rhos = statistics[statistics.analysis.eq("raw_vs_calibrated_profile_rank")].set_index("model_a").effect_size
    assessment = {
        "pitch_order_raw": raw_pitch,
        "pitch_order_calibrated": calibrated_pitch,
        "pitch_order_qualitatively_similar": bool(spearmanr(
            [raw_pitch.index(model) for model in MODELS], [calibrated_pitch.index(model) for model in MODELS]
        ).statistic >= 0.8),
        "prosody_fusion_remains_most_pitch_sensitive": bool(calibrated_pitch[0] == "Prosody-Fusion"),
        "profiles_qualitatively_consistent": bool((profile_rhos >= 0.75).all()),
        "profile_spearman_raw_vs_calibrated": {key: float(value) for key, value in profile_rhos.items()},
    }
    (ARTIFACTS / "RQ2_CALIBRATION_ASSESSMENT_FINAL.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    print(calibration.to_string(index=False))
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
