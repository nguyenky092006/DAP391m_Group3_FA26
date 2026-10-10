"""Analyze whether the fixed F0-baseline intervention is stable by ViSEC accent.

RQ1 and RQ2 artifacts are read-only inputs.  The script does not train models,
change the intervention, or use human-perception data.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
import statsmodels.formula.api as smf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from cf_ser.acoustics import pitch_transform
from cf_ser.config import ARTIFACTS, CHECKPOINTS, FIGURES, LABELS, SAMPLE_RATE, SPLIT_CSV
from cf_ser.data import load_audio
from run_rq2_4models import DEVICE, wav2vec_ensemble_probs


SEED = 2025
BOOTSTRAPS = 5000
MODELS = ["Mel-CNN", "RawWave-CNN", "Wav2Vec2", "Prosody-Fusion"]
ACCENT_ORDER = ["north", "mid", "south"]
ACCENT_LABELS = {"north": "North", "mid": "Central", "south": "South"}


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_mean_ci(values, salt=0):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED + salt)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    return tuple(np.percentile(values[indices].mean(axis=1), [2.5, 97.5]))


def independent_rank_biserial(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.nan
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def load_or_infer_wav2vec(neutral, params, checkpoint_dirs, cache_path, batch_size):
    """Return Wav2Vec2 original/F0-mean probabilities, using a valid cache."""
    required = {
        "sample_id", "speaker_id", "accent", "model", "original_prediction",
        "p_angry_original", "p_angry_transformed", "delta_p_angry",
    }
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"sample_id": str, "speaker_id": str})
        if required.issubset(cached.columns) and set(cached.sample_id) == set(neutral.sample_id):
            return cached

    extractor = AutoFeatureExtractor.from_pretrained(checkpoint_dirs[0])
    models = [AutoModelForAudioClassification.from_pretrained(path).to(DEVICE).eval() for path in checkpoint_dirs]
    originals, transformed = [], []
    for number, row in enumerate(neutral.itertuples(index=False), start=1):
        wave = load_audio(row.path)
        changed, _ = pitch_transform(wave, params, strength=1.0, mode="mean")
        originals.append(wave)
        transformed.append(changed)
        if number % 25 == 0:
            print(f"Prepared fixed F0-mean waveforms {number}/{len(neutral)}", flush=True)

    p0_parts, p1_parts = [], []
    for start in range(0, len(neutral), batch_size):
        stop = min(len(neutral), start + batch_size)
        p0_parts.append(wav2vec_ensemble_probs(models, extractor, originals[start:stop]))
        p1_parts.append(wav2vec_ensemble_probs(models, extractor, transformed[start:stop]))
        print(f"Wav2Vec2 inference {stop}/{len(neutral)}", flush=True)
    p0, p1 = np.vstack(p0_parts), np.vstack(p1_parts)
    out = neutral[["sample_id", "speaker", "accent"]].rename(columns={"speaker": "speaker_id"}).copy()
    out["model"] = "Wav2Vec2"
    out["original_prediction"] = [LABELS[int(x)] for x in p0.argmax(axis=1)]
    out["transformed_prediction"] = [LABELS[int(x)] for x in p1.argmax(axis=1)]
    out["p_angry_original"] = p0[:, LABELS.index("angry")]
    out["p_angry_transformed"] = p1[:, LABELS.index("angry")]
    out["delta_p_angry"] = out.p_angry_transformed - out.p_angry_original
    out.to_csv(cache_path, index=False)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def build_prediction_frame(neutral, wav2vec):
    dose = pd.read_csv(ARTIFACTS / "pitch_dose_predictions.csv", dtype={"sample_id": str, "speaker_id": str})
    compact = dose[(dose.condition == "pitch_mean_100") & dose.model.isin([m for m in MODELS if m != "Wav2Vec2"])].copy()
    compact = compact[[
        "sample_id", "speaker_id", "accent", "model", "original_prediction",
        "transformed_prediction", "p_angry_original", "p_angry_transformed", "delta_p_angry",
    ]]
    prediction = pd.concat([compact, wav2vec[compact.columns]], ignore_index=True)
    prediction["sample_id"] = prediction.sample_id.str.zfill(5)
    prediction["speaker_id"] = prediction.speaker_id.astype(str)
    expected = pd.MultiIndex.from_product([neutral.sample_id, MODELS], names=["sample_id", "model"])
    observed = pd.MultiIndex.from_frame(prediction[["sample_id", "model"]])
    if prediction.duplicated(["sample_id", "model"]).any() or set(expected) != set(observed):
        raise AssertionError("RQ3 requires one complete row per Neutral utterance and model")
    prediction["accent"] = pd.Categorical(prediction.accent, ACCENT_ORDER, ordered=True)
    prediction["model"] = pd.Categorical(prediction.model, MODELS, ordered=True)
    prediction["baseline_correct"] = prediction.original_prediction.eq("neutral")
    return prediction


def count_known_speakers(frame):
    return int(frame.loc[frame.speaker_id.ne("0"), "speaker_id"].nunique())


def summarize(frame, analysis_set, pooled=False, salt_start=0):
    rows, salt = [], salt_start
    source = frame.copy()
    if pooled:
        source = source.groupby(["sample_id", "speaker_id", "accent"], observed=True, as_index=False).delta_p_angry.mean()
        source["model"] = "Pooled (utterance mean across models)"
    for (model, accent), group in source.groupby(["model", "accent"], observed=True):
        values = group.delta_p_angry.to_numpy()
        lo, hi = bootstrap_mean_ci(values, salt); salt += 1
        rows.append({
            "analysis_set": analysis_set,
            "scope": "pooled" if pooled else "architecture",
            "model": str(model),
            "accent": ACCENT_LABELS[str(accent)],
            "n_utterances": int(group.sample_id.nunique()),
            "n_known_speakers": count_known_speakers(group),
            "n_missing_speaker_id_utterances": int(group.loc[group.speaker_id.eq("0"), "sample_id"].nunique()),
            "mean_delta_p_angry": float(values.mean()),
            "median_delta_p_angry": float(np.median(values)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "positive_direction_rate": float(np.mean(values > 0)),
        })
    return rows, salt


def gee_tests(primary):
    data = primary.copy()
    data["accent"] = data.accent.astype(str)
    data["model"] = data.model.astype(str)
    formula = (
        'delta_p_angry ~ C(accent, Treatment(reference="south")) * '
        'C(model, Treatment(reference="Mel-CNN"))'
    )
    fit = smf.gee(
        formula, groups="sample_id", data=data, family=Gaussian(), cov_struct=Exchangeable()
    ).fit()
    rows = []
    for term in fit.params.index:
        rows.append({
            "analysis_set": "primary_all_true_neutral", "model": "All",
            "test_family": "GEE_repeated_utterance", "comparison": term,
            "n": len(data), "statistic": float(fit.tvalues[term]),
            "degrees_of_freedom": 1, "p_value_raw": float(fit.pvalues[term]),
            "p_value_holm": np.nan, "effect_size": float(fit.params[term]),
            "effect_size_name": "GEE coefficient", "notes": "Exchangeable correlation within utterance",
        })
    terms = fit.wald_test_terms(skip_single=False).table
    for name in terms.index:
        value = terms.loc[name]
        rows.append({
            "analysis_set": "primary_all_true_neutral", "model": "All",
            "test_family": "GEE_joint_Wald", "comparison": str(name),
            "n": len(data), "statistic": float(np.asarray(value["statistic"]).squeeze()),
            "degrees_of_freedom": int(value["df_constraint"]),
            "p_value_raw": float(value["pvalue"]), "p_value_holm": np.nan,
            "effect_size": np.nan, "effect_size_name": "not_applicable",
            "notes": "Joint repeated-measure test; utterance is the clustering unit",
        })
    return rows


def nonparametric_tests(frame, analysis_set):
    rows = []
    for model, group in frame.groupby("model", observed=True):
        arrays = [group.loc[group.accent == accent, "delta_p_angry"].to_numpy() for accent in ACCENT_ORDER]
        h, p = kruskal(*arrays)
        n, k = sum(map(len, arrays)), len(arrays)
        epsilon = max(0.0, float((h - k + 1) / (n - k))) if n > k else np.nan
        rows.append({
            "analysis_set": analysis_set, "model": str(model), "test_family": "Kruskal-Wallis",
            "comparison": "accent omnibus", "n": n, "statistic": float(h),
            "degrees_of_freedom": k - 1, "p_value_raw": float(p), "p_value_holm": float(p),
            "effect_size": epsilon, "effect_size_name": "epsilon_squared",
            "notes": "Independent accent groups within architecture",
        })
        pairs = []
        for a, b in combinations(ACCENT_ORDER, 2):
            x = group.loc[group.accent == a, "delta_p_angry"].to_numpy()
            y = group.loc[group.accent == b, "delta_p_angry"].to_numpy()
            u, raw = mannwhitneyu(x, y, alternative="two-sided")
            pairs.append({
                "analysis_set": analysis_set, "model": str(model), "test_family": "Mann-Whitney post-hoc",
                "comparison": f"{ACCENT_LABELS[a]} vs {ACCENT_LABELS[b]}", "n": len(x) + len(y),
                "statistic": float(u), "degrees_of_freedom": np.nan, "p_value_raw": float(raw),
                "p_value_holm": np.nan, "effect_size": independent_rank_biserial(x, y),
                "effect_size_name": "rank_biserial", "notes": "Holm correction within architecture",
            })
        adjusted = holm([row["p_value_raw"] for row in pairs])
        for row, corrected in zip(pairs, adjusted):
            row["p_value_holm"] = float(corrected)
        rows.extend(pairs)
    return rows


def make_figures(results):
    primary = results[results.analysis_set.eq("primary_all_true_neutral")]
    pooled = primary[primary.scope.eq("pooled")].copy()
    order_labels = [ACCENT_LABELS[x] for x in ACCENT_ORDER]
    pooled["accent"] = pd.Categorical(pooled.accent, order_labels, ordered=True)
    pooled = pooled.sort_values("accent")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    yerr = np.vstack([pooled.mean_delta_p_angry - pooled.ci95_low, pooled.ci95_high - pooled.mean_delta_p_angry])
    ax.bar(pooled.accent, pooled.mean_delta_p_angry, color=["#4C78A8", "#F2CF5B", "#E45756"], yerr=yerr, capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean ΔP(Angry), model-balanced")
    ax.set_xlabel("ViSEC accent metadata")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_f0_effect_by_accent.png", dpi=240)
    plt.close(fig)

    architecture = primary[primary.scope.eq("architecture")]
    heat = architecture.pivot(index="model", columns="accent", values="mean_delta_p_angry").reindex(index=MODELS, columns=order_labels)
    limit = max(abs(float(heat.min().min())), abs(float(heat.max().max())), 1e-6)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-limit, vmax=limit, ax=ax, cbar_kws={"label": "Mean ΔP(Angry)"})
    ax.set_xlabel("ViSEC accent metadata")
    ax.set_ylabel("Architecture family")
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_model_accent_heatmap.png", dpi=240)
    plt.close(fig)


def write_findings(results, tests, manifest):
    primary = results[results.analysis_set.eq("primary_all_true_neutral")]
    pooled = primary[primary.scope.eq("pooled")].set_index("accent")
    gee = tests[tests.test_family.eq("GEE_joint_Wald")]
    accent_row = gee[gee.comparison.str.startswith("C(accent") & ~gee.comparison.str.contains(":")]
    interaction_row = gee[gee.comparison.str.contains(":")]
    accent_p = float(accent_row.p_value_raw.iloc[0]) if len(accent_row) else np.nan
    interaction_p = float(interaction_row.p_value_raw.iloc[0]) if len(interaction_row) else np.nan
    directions = {accent: float(pooled.loc[accent, "mean_delta_p_angry"]) for accent in pooled.index}
    all_positive = all(value > 0 for value in directions.values())
    architecture = primary[primary.scope.eq("architecture")]
    architecture_all_positive = bool((architecture.mean_delta_p_angry > 0).all())
    nonpositive_cells = architecture.loc[
        architecture.mean_delta_p_angry <= 0, ["model", "accent", "mean_delta_p_angry"]
    ]
    interaction = np.isfinite(interaction_p) and interaction_p < 0.05
    magnitude = np.isfinite(accent_p) and accent_p < 0.05
    if all_positive and architecture_all_positive and not magnitude and not interaction:
        conclusion = "The F0-baseline sensitivity observed in RQ1 was broadly consistent across the three Vietnamese regional accent groups available in ViSEC."
    elif all_positive and architecture_all_positive:
        conclusion = "The direction of the F0-baseline effect was preserved across accents, although its magnitude varied by accent and/or architecture."
    elif all_positive:
        conclusion = "The pooled F0-baseline effect was positive across the three accent groups represented in ViSEC, but its magnitude and direction were not uniform across architectures."
    else:
        conclusion = "F0-baseline sensitivity was accent-dependent in the evaluated SER models."
    counts = manifest.groupby("accent", observed=True).agg(
        n_utterances=("sample_id", "nunique"),
        n_known_speakers=("known_speaker", "sum"),
        n_missing_speaker_id_utterances=("speaker_missing", "sum"),
    )
    # known_speaker is one row per utterance; replace its sum with unique known IDs.
    for accent in counts.index:
        q = manifest[manifest.accent == accent]
        counts.loc[accent, "n_known_speakers"] = q.loc[~q.speaker_missing, "speaker_id"].nunique()
    lines = [
        "# RQ3 final findings", "",
        "## Exploratory accent analysis", "",
        "**RQ3:** How does F0-baseline sensitivity vary across the Vietnamese accent groups represented in ViSEC?", "",
        "This analysis evaluates SER model sensitivity only. It does not test how speakers express anger or how humans perceive the intervention.", "",
        "## Primary sample", "",
        "All true Neutral test utterances with valid ViSEC accent metadata were included. Samples were not selected using intervention outcomes.", "",
        "| Accent | N utterances | Known speaker IDs | Missing-ID utterances | Pooled mean ΔP(Angry) | Median | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for raw in ACCENT_ORDER:
        label = ACCENT_LABELS[raw]
        row = pooled.loc[label]
        c = counts.loc[raw]
        lines.append(f"| {label} | {int(c.n_utterances)} | {int(c.n_known_speakers)} | {int(c.n_missing_speaker_id_utterances)} | {row.mean_delta_p_angry:.4f} | {row.median_delta_p_angry:.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] |")
    lines.extend([
        "", "## Architecture-specific primary means", "",
        "| Architecture | North | Central | South |",
        "|---|---:|---:|---:|",
    ])
    for model in MODELS:
        model_rows = architecture[architecture.model.astype(str).eq(model)].set_index("accent")
        lines.append(
            f"| {model} | {model_rows.loc['North', 'mean_delta_p_angry']:.4f} | "
            f"{model_rows.loc['Central', 'mean_delta_p_angry']:.4f} | "
            f"{model_rows.loc['South', 'mean_delta_p_angry']:.4f} |"
        )
    lines.extend([
        "", "Pooled values first average the four architecture responses within each utterance, so each utterance contributes once.", "",
        "## Statistical assessment", "",
        f"- The model-balanced direction was {'positive for all three accent groups' if all_positive else 'not consistent across all three accent groups'}.",
        f"- Direction across every architecture × accent cell was {'positive' if architecture_all_positive else 'not fully consistent'}.",
        f"- In the interaction model, the GEE accent term at the Mel-CNN reference architecture had p={accent_p:.4g}.",
        f"- The architecture × accent interaction test had p={interaction_p:.4g}; this is {'evidence of architecture-dependent accent variation' if interaction else 'not significant evidence of architecture-dependent accent variation'} at α=0.05.",
        "- Architecture-specific Kruskal–Wallis and Holm-corrected pairwise tests are reported in `RQ3_ACCENT_STATISTICAL_TESTS.csv`.",
        "- Speaker ID 0 is a heterogeneous missing-ID bucket and was not counted as a known speaker. The repeated-measure model therefore clusters by utterance, not speaker.",
        "", "## Secondary analysis", "",
        "Baseline-correct analyses are explicitly secondary. Architecture rows use each model's own baseline-correct Neutral subset; the pooled row uses the four-model intersection.",
        "", "## Conservative paper-ready conclusion", "",
        conclusion, "",
        "This conclusion is limited to the fixed ViSEC split, available accent labels, existing SER checkpoints, and the fixed 100% F0-mean intervention.",
    ])
    if len(nonpositive_cells):
        lines.insert(lines.index("## Secondary analysis") - 1, "- Non-positive architecture-specific cells: " + "; ".join(
            f"{row.model} / {row.accent} ({row.mean_delta_p_angry:.4f})" for row in nonpositive_cells.itertuples(index=False)
        ))
    (ARTIFACTS / "RQ3_FINAL_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav2vec-checkpoints", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    checkpoint_dirs = [Path(p).resolve() for p in args.wav2vec_checkpoints] if args.wav2vec_checkpoints else [
        CHECKPOINTS / "wav2vec2_ensemble" / f"wav2vec2_seed{seed}" for seed in (2025, 2026, 2027)
    ]
    for path in checkpoint_dirs:
        if not (path / "training_summary.json").exists():
            raise FileNotFoundError(path)
    split = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    neutral = split[(split.split == "test") & (split.emotion == "neutral") & split.accent.isin(ACCENT_ORDER)].copy()
    neutral["sample_id"] = neutral.path.map(lambda x: Path(x).stem)
    params = json.loads((ARTIFACTS / "transformation_params.json").read_text(encoding="utf-8"))
    cache = ARTIFACTS / "rq3_wav2vec_f0_mean_predictions.csv"
    wav2vec = load_or_infer_wav2vec(neutral, params, checkpoint_dirs, cache, args.batch_size)
    prediction = build_prediction_frame(neutral, wav2vec)

    manifest = neutral[["sample_id", "path", "speaker", "accent", "emotion", "split"]].rename(columns={"speaker": "speaker_id"})
    manifest["speaker_id"] = manifest.speaker_id.astype(str)
    manifest["speaker_missing"] = manifest.speaker_id.eq("0")
    manifest["known_speaker"] = ~manifest.speaker_missing
    correct = prediction.pivot(index="sample_id", columns="model", values="baseline_correct").reindex(columns=MODELS)
    for model in MODELS:
        manifest[f"baseline_correct_{model.lower().replace('-', '_')}"] = manifest.sample_id.map(correct[model]).astype(bool)
    manifest["baseline_correct_all_4models"] = manifest[[c for c in manifest if c.startswith("baseline_correct_") and c != "baseline_correct_all_4models"]].all(axis=1)
    manifest.to_csv(ARTIFACTS / "RQ3_ACCENT_SAMPLE_MANIFEST.csv", index=False)

    result_rows, salt = summarize(prediction, "primary_all_true_neutral", pooled=True)
    extra, salt = summarize(prediction, "primary_all_true_neutral", pooled=False, salt_start=salt); result_rows.extend(extra)
    common_ids = set(manifest.loc[manifest.baseline_correct_all_4models, "sample_id"])
    common = prediction[prediction.sample_id.isin(common_ids)]
    extra, salt = summarize(common, "secondary_common_baseline_correct", pooled=True, salt_start=salt); result_rows.extend(extra)
    per_model = prediction[prediction.baseline_correct]
    extra, salt = summarize(per_model, "secondary_model_specific_baseline_correct", pooled=False, salt_start=salt); result_rows.extend(extra)
    results = pd.DataFrame(result_rows)
    results.to_csv(ARTIFACTS / "RQ3_ACCENT_RESULTS.csv", index=False)

    test_rows = gee_tests(prediction)
    test_rows.extend(nonparametric_tests(prediction, "primary_all_true_neutral"))
    test_rows.extend(nonparametric_tests(per_model, "secondary_model_specific_baseline_correct"))
    tests = pd.DataFrame(test_rows)
    tests.to_csv(ARTIFACTS / "RQ3_ACCENT_STATISTICAL_TESTS.csv", index=False)
    make_figures(results)
    write_findings(results, tests, manifest)
    print(f"RQ3 accent analysis complete: {len(neutral)} utterances, {len(prediction)} model rows")


if __name__ == "__main__":
    main()
