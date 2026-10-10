"""Consolidate preprocessing, fixed-dose, and no-F0 ablation evidence."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from cf_ser.config import ARTIFACTS


SEED = 2025
POSITIVE_DOSES = [1.0, 2.0, 3.0, 4.0, 6.0]


def bootstrap_ci(values, salt=0, iterations=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED + salt)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(iterations)]
    return np.percentile(means, [2.5, 97.5])


def rank_biserial(differences):
    differences = np.asarray(differences, float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences))
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / ranks.sum())


def main():
    predictions = pd.read_csv(ARTIFACTS / "FIXED_SEMITONE_PREDICTIONS.csv", dtype={"sample_id": str, "speaker_id": str})
    results = pd.read_csv(ARTIFACTS / "FIXED_SEMITONE_RESULTS.csv")
    preprocessing = pd.read_csv(ARTIFACTS / "PREPROCESSING_INTERVENTION_AUDIT_SUMMARY.csv")
    f0 = pd.read_csv(ARTIFACTS / "F0_CONFOUND_AUDIT.csv")
    ablation_baseline = pd.read_csv(ARTIFACTS / "PROSODY_NO_F0_BASELINE_RESULTS.csv")
    full_baseline = pd.read_csv(ARTIFACTS / "model_optimization.csv")
    full_baseline = full_baseline[full_baseline.key == "prosody_fusion"]

    sham = predictions[predictions.condition == "sham"].groupby("model").agg(
        n=("sample_id", "nunique"),
        mean_sham_delta_p_angry=("delta_p_angry_from_original", "mean"),
        median_sham_delta_p_angry=("delta_p_angry_from_original", "median"),
    ).reset_index()
    sham.to_csv(ARTIFACTS / "FIXED_SEMITONE_SHAM_EFFECTS.csv", index=False)

    slopes = []
    for (model, sample_id), frame in predictions[predictions.requested_semitones.isin(POSITIVE_DOSES)].groupby(["model", "sample_id"]):
        frame = frame.sort_values("requested_semitones")
        slope = np.polyfit(frame.requested_semitones, frame.delta_p_angry_from_sham, 1)[0]
        monotonic = bool(np.all(np.diff(frame.delta_p_angry_from_sham.to_numpy()) >= -1e-8))
        slopes.append({"model": model, "sample_id": sample_id, "slope_per_semitone": slope, "monotonic_non_decreasing": monotonic})
    slopes = pd.DataFrame(slopes)
    slope_summary = slopes.groupby("model").agg(
        n=("sample_id", "nunique"),
        mean_slope_per_semitone=("slope_per_semitone", "mean"),
        median_slope_per_semitone=("slope_per_semitone", "median"),
        monotonic_fraction=("monotonic_non_decreasing", "mean"),
    ).reset_index()
    slope_summary.to_csv(ARTIFACTS / "FIXED_SEMITONE_DOSE_SLOPES.csv", index=False)

    tests = []
    salt = 0
    for dose in POSITIVE_DOSES:
        condition = f"plus_{int(dose)}st"
        full = predictions[(predictions.model == "Prosody-Fusion") & (predictions.condition == condition)].set_index("sample_id")
        ablated = predictions[(predictions.model == "Prosody-Fusion-noF0") & (predictions.condition == condition)].set_index("sample_id")
        ids = full.index.intersection(ablated.index)
        difference = full.loc[ids, "delta_p_angry_from_sham"] - ablated.loc[ids, "delta_p_angry_from_sham"]
        low, high = bootstrap_ci(difference, salt)
        salt += 1
        try:
            statistic, p_value = wilcoxon(difference, zero_method="wilcox", method="auto")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        tests.append({
            "comparison": "Prosody-Fusion minus Prosody-Fusion-noF0",
            "dose_semitones": dose, "n": len(difference),
            "mean_difference_sham_adjusted_delta_p_angry": difference.mean(),
            "ci95_low": low, "ci95_high": high,
            "wilcoxon_statistic": statistic, "p_value": p_value,
            "matched_rank_biserial": rank_biserial(difference),
        })
    tests = pd.DataFrame(tests)
    tests.to_csv(ARTIFACTS / "PROSODY_F0_ABLATION_TESTS.csv", index=False)

    energy = preprocessing[preprocessing.transformation == "T2_energy"].iloc[0]
    pooled = f0[(f0.scope == "overall")].iloc[0]
    within = f0[(f0.scope == "within_speaker")].iloc[0]
    dose4 = results[(results.analysis_set == "all_true_neutral") & (results.condition == "plus_4st")]
    dose6 = results[(results.analysis_set == "all_true_neutral") & (results.condition == "plus_6st")]
    full4 = dose4[dose4.model == "Prosody-Fusion"].iloc[0]
    ablated4 = dose4[dose4.model == "Prosody-Fusion-noF0"].iloc[0]
    test4 = tests[tests.dose_semitones == 4].iloc[0]

    lines = [
        "# Reviewer-hardening v2: consolidated findings", "",
        "## Decision", "",
        "The revised evidence supports a **preprocessing-aware end-to-end pipeline sensitivity** paper. It does not support attributing every cross-model difference to architecture or learned internal representations.", "",
        "## 1. Preprocessing confound is confirmed", "",
        f"The energy-only intervention changes the fixed waveform by {energy.waveform_rel_l2_mean:.6f} relative L2 and {energy.waveform_rms_db_delta_mean:.3f} dB. After preprocessing, mean relative L2 is {energy.mel_input_rel_l2_mean:.6f} for normalized log-Mel and {energy.wav2vec_input_rel_l2_mean:.6f} for Wav2Vec2, while RawWave receives {energy.raw_input_rel_l2_mean:.6f}. Prosody-Fusion additionally receives a standardized explicit log-RMS displacement of {energy.prosody_energy_abs_delta_mean:.6f}.", "",
        "Therefore, Wav2Vec2's near-zero energy response is a preprocessing result, not evidence that its learned representation ignores energy.", "",
        "## 2. The pooled F0 target was inflated relative to within-speaker evidence", "",
        f"The pooled train contrast is {pooled.semitones:.3f} semitones, whereas the mean contrast among {int(within.n_neutral)} identified speakers represented in both emotions is {within.semitones:.3f} semitones. Fixed semitone doses are now primary; the pooled approximately six-semitone intervention is secondary.", "",
        "## 3. Fixed-dose result on all Neutral test utterances", "",
        "| Model | +4 st mean delta P(Angry) | 95% CI | +6 st mean delta P(Angry) |", "|---|---:|---:|---:|",
    ]
    for model in dose4.model:
        r4 = dose4[dose4.model == model].iloc[0]
        r6 = dose6[dose6.model == model].iloc[0]
        lines.append(f"| {model} | {r4.mean_delta_p_angry:.4f} | [{r4.ci95_low:.4f}, {r4.ci95_high:.4f}] | {r6.mean_delta_p_angry:.4f} |")
    lines += [
        "", "Wav2Vec2 is not clearly positive at +4 semitones because its confidence interval includes zero; it becomes positive after sham adjustment at +6 semitones. The remaining three principal pipelines show positive +4-semitone mean effects.", "",
        "## 4. Explicit-F0 ablation", "",
        f"Full Prosody-Fusion baseline Macro-F1 across seeds: {full_baseline.test_macro_f1.mean():.4f}; without F0: {ablation_baseline.test_macro_f1.mean():.4f}.",
        f"At +4 semitones, full Prosody-Fusion has mean delta P(Angry)={full4.mean_delta_p_angry:.4f}; no-F0 has {ablated4.mean_delta_p_angry:.4f}. The paired sham-adjusted difference is {test4.mean_difference_sham_adjusted_delta_p_angry:.4f}, 95% CI [{test4.ci95_low:.4f}, {test4.ci95_high:.4f}], Wilcoxon p={test4.p_value:.3g}, rank-biserial={test4.matched_rank_biserial:.3f}.", "",
        "The ablated model remains pitch-sensitive through its waveform/log-Mel branch, but the explicit F0 pathway accounts for a large part of the amplified response. This is the main mechanistic insight supported by the revised experiment.", "",
        "## 5. Claim boundary", "",
        "Use: `Different end-to-end SER pipelines exhibit different intervention-sensitivity profiles, and these profiles are partly explained by preprocessing observability and explicit cue pathways.`", "",
        "Do not use: `The four architectures intrinsically learned different emotional representations.`", "",
        "RQ3 remains exploratory because accent and speaker effects cannot be separated with the available metadata. Human perceptual preservation of lexical tone remains unestablished until independent ratings are collected.",
    ]
    (ARTIFACTS / "REVIEWER_HARDENING_V2_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
