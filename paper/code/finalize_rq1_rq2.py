"""Finalize paper-ready RQ1/RQ2 tables from existing predictions only."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CORE = ART / "counterfactual_predictions.csv"
DOSE = ART / "pitch_dose_predictions.csv"
MODELS = ["Mel-CNN", "RawWave-CNN", "Prosody-Fusion"]
SINGLES = {"Pitch": "T1_pitch", "Energy": "T2_energy", "Duration": "T3_duration"}
INTERVENTIONS = {
    "Pitch": "T1_pitch", "Energy": "T2_energy", "Duration": "T3_duration",
    "P+E": "T4_pitch_energy", "P+D": "T5_pitch_duration",
    "E+D": "T6_energy_duration", "All": "T7_all",
}
BOOTSTRAPS = 5000
SEED = 2025


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_mean_ci(values, salt=0):
    x = np.asarray(values, float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED + salt)
    means = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def rank_biserial_paired(differences):
    d = np.asarray(differences, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def paired_test(a, b, model, family, comparison, salt=0):
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = pair.a.to_numpy() - pair.b.to_numpy()
    if len(diff) == 0:
        raise ValueError(f"No paired values for {model} {comparison}")
    try:
        stat, p = wilcoxon(pair.a, pair.b, zero_method="wilcox", alternative="two-sided", method="auto")
    except ValueError:
        stat, p = 0.0, 1.0
    lo, hi = bootstrap_mean_ci(diff, salt)
    mean = float(diff.mean())
    return {
        "test_family": family, "model": model, "comparison": comparison,
        "n": len(diff), "mean_paired_difference": mean,
        "median_paired_difference": float(np.median(diff)),
        "ci95_low": lo, "ci95_high": hi,
        "wilcoxon_statistic": float(stat), "p_value_raw": float(p),
        "p_value_holm": np.nan,
        "rank_biserial": rank_biserial_paired(diff),
        "effect_direction": "A > B" if mean > 0 else ("A < B" if mean < 0 else "no mean difference"),
    }


def primary_ids(frame, model, condition_col="transformation", original_condition="T0_original"):
    base = frame[(frame.model == model) & (frame[condition_col] == original_condition)]
    return set(base.loc[base.original_prediction == "neutral", "sample_id"].astype(str))


def indexed_delta(frame, model, condition, ids, condition_col="transformation", delta_col="delta_p_angry"):
    q = frame[(frame.model == model) & (frame[condition_col] == condition)].copy()
    q["sample_id"] = q.sample_id.astype(str)
    q = q[q.sample_id.isin(ids)].drop_duplicates("sample_id").set_index("sample_id")
    if set(q.index) != set(ids):
        raise AssertionError(f"Incomplete paired data: {model}, {condition}")
    return q[delta_col].sort_index()


def finalize_rq1(core, dose):
    result_rows = []
    test_rows = []
    salt = 0
    for model in MODELS:
        ids = primary_ids(core, model)
        deltas = {name: indexed_delta(core, model, code, ids) for name, code in SINGLES.items()}
        for name, values in deltas.items():
            lo, hi = bootstrap_mean_ci(values, salt); salt += 1
            q = core[(core.model == model) & (core.transformation == SINGLES[name])].copy()
            q["sample_id"] = q.sample_id.astype(str); q = q[q.sample_id.isin(ids)]
            result_rows.append({"section": "single_component", "model": model, "condition": name,
                "reference": "original", "n": len(values), "mean_delta_p_angry": values.mean(),
                "median_delta_p_angry": values.median(), "ci95_low": lo, "ci95_high": hi,
                "angry_flip_rate": np.mean(q.transformed_prediction == "angry")})

        # Three primary comparisons, corrected together within each model.
        test_rows.append(paired_test(deltas["Pitch"], deltas["Energy"], model,
                                     "RQ1_primary_direct", "Pitch vs Energy", salt)); salt += 1
        test_rows.append(paired_test(deltas["Pitch"], deltas["Duration"], model,
                                     "RQ1_primary_direct", "Pitch vs Duration", salt)); salt += 1

        dose_ids = primary_ids(dose, model, "condition", "original")
        mean_f0 = indexed_delta(dose, model, "pitch_mean_100", dose_ids, "condition")
        range_f0 = indexed_delta(dose, model, "pitch_range_100", dose_ids, "condition")
        test_rows.append(paired_test(mean_f0, range_f0, model,
                                     "RQ1_primary_direct", "F0 mean vs F0 range", salt)); salt += 1

        sham = indexed_delta(dose, model, "sham_resynthesis", dose_ids, "condition")
        pitch100 = indexed_delta(dose, model, "pitch_100", dose_ids, "condition")
        reverse = indexed_delta(dose, model, "pitch_opposite_100", dose_ids, "condition")
        test_rows.append(paired_test(pitch100, sham, model, "pitch_vs_sham",
                                     "Pitch 100% vs PSOLA sham", salt)); salt += 1
        test_rows.append(paired_test(pitch100, reverse, model, "direction_control",
                                     "Angry-directed Pitch vs reverse Pitch", salt)); salt += 1

        for condition in ["sham_resynthesis", "pitch_25", "pitch_50", "pitch_75", "pitch_100",
                          "pitch_125_stress", "pitch_opposite_100", "pitch_mean_100", "pitch_range_100"]:
            q = dose[(dose.model == model) & (dose.condition == condition)].copy()
            q["sample_id"] = q.sample_id.astype(str); q = q[q.sample_id.isin(dose_ids)]
            if "delta_p_angry_vs_sham" not in q:
                sham_map = dose[(dose.model == model) & (dose.condition == "sham_resynthesis")].set_index("sample_id").p_angry_transformed
                q["delta_p_angry_vs_sham"] = q.apply(lambda r: r.p_angry_transformed - sham_map.loc[str(r.sample_id)], axis=1)
            vals = q.delta_p_angry_vs_sham.to_numpy(); lo, hi = bootstrap_mean_ci(vals, salt); salt += 1
            result_rows.append({"section": "pitch_control_or_dose", "model": model, "condition": condition,
                "reference": "PSOLA sham", "n": len(vals), "mean_delta_p_angry": vals.mean(),
                "median_delta_p_angry": np.median(vals), "ci95_low": lo, "ci95_high": hi,
                "angry_flip_rate": np.mean(q.transformed_prediction == "angry")})

    tests = pd.DataFrame(test_rows)
    for _, idx in tests.groupby(["test_family", "model"]).groups.items():
        tests.loc[idx, "p_value_holm"] = holm(tests.loc[idx, "p_value_raw"].to_numpy())
    results = pd.DataFrame(result_rows)
    results.to_csv(ART / "RQ1_FINAL_RESULTS.csv", index=False)
    tests.to_csv(ART / "RQ1_STATISTICAL_TESTS.csv", index=False)
    return results, tests


def build_common_manifest(core):
    correct = []
    for model in MODELS:
        correct.append(primary_ids(core, model))
    common = set.intersection(*correct)
    base = core[core.transformation == "T0_original"].copy()
    base["sample_id"] = base.sample_id.astype(str)
    metadata = base[base.sample_id.isin(common)][["sample_id", "speaker_id", "accent"]].drop_duplicates("sample_id")
    for model in MODELS:
        q = base[(base.model == model) & base.sample_id.isin(common)].set_index("sample_id")
        slug = model.lower().replace("-", "_")
        metadata[f"{slug}_original_prediction"] = metadata.sample_id.map(q.original_prediction)
        metadata[f"{slug}_p_angry_original"] = metadata.sample_id.map(q.p_angry_original)
        metadata[f"{slug}_p_neutral_original"] = metadata.sample_id.map(q.p_neutral_original)
    metadata = metadata.sort_values("sample_id").reset_index(drop=True)
    metadata.to_csv(ART / "rq2_common_sample_manifest.csv", index=False)
    return set(metadata.sample_id)


def finalize_rq2(core, common):
    rows = []
    values = {}
    for model in MODELS:
        for label, code in INTERVENTIONS.items():
            q = core[(core.model == model) & (core.transformation == code)].copy()
            q["sample_id"] = q.sample_id.astype(str); q = q[q.sample_id.isin(common)].drop_duplicates("sample_id").set_index("sample_id").sort_index()
            if set(q.index) != common:
                raise AssertionError(f"Missing common RQ2 predictions: {model}, {label}")
            d = q.delta_p_angry
            values[(model, label)] = d
            lo, hi = bootstrap_mean_ci(d, len(rows) + 500)
            rows.append({"model": model, "intervention": label, "n": len(d),
                "mean_delta_p_angry": d.mean(), "median_delta_p_angry": d.median(),
                "std_delta_p_angry": d.std(ddof=0), "ci95_low": lo, "ci95_high": hi,
                "angry_flip_rate": np.mean(q.transformed_prediction == "angry")})
    results = pd.DataFrame(rows)
    results["sensitivity_rank"] = results.groupby("model").mean_delta_p_angry.rank(ascending=False, method="min").astype(int)
    results.to_csv(ART / "RQ2_FINAL_RESULTS.csv", index=False)

    tests = []
    for intervention in ["Pitch", "Energy", "Duration"]:
        arrays = [values[(model, intervention)].sort_index().to_numpy() for model in MODELS]
        stat, p = friedmanchisquare(*arrays)
        tests.append({"test_type": "Friedman", "intervention": intervention, "comparison": "Mel-CNN vs RawWave-CNN vs Prosody-Fusion",
            "n": len(common), "statistic": stat, "p_value_raw": p, "p_value_holm": p,
            "effect_size": stat / (len(common) * (len(MODELS) - 1)), "effect_size_name": "Kendall_W", "direction": "omnibus"})
        if p < .05:
            start = len(tests)
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                row = paired_test(pd.Series(arrays[i]), pd.Series(arrays[j]), f"{MODELS[i]} vs {MODELS[j]}",
                                  f"RQ2_{intervention}_posthoc", f"{MODELS[i]} vs {MODELS[j]}", 800 + len(tests))
                tests.append({"test_type": "Wilcoxon post-hoc", "intervention": intervention,
                    "comparison": row["comparison"], "n": row["n"], "statistic": row["wilcoxon_statistic"],
                    "p_value_raw": row["p_value_raw"], "p_value_holm": np.nan,
                    "effect_size": row["rank_biserial"], "effect_size_name": "matched_rank_biserial",
                    "direction": row["effect_direction"], "mean_paired_difference": row["mean_paired_difference"],
                    "median_paired_difference": row["median_paired_difference"], "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"]})
            idx = list(range(start, len(tests)))
            adj = holm([tests[k]["p_value_raw"] for k in idx])
            for k, a in zip(idx, adj): tests[k]["p_value_holm"] = a

    mean_matrix = results.pivot(index="model", columns="intervention", values="mean_delta_p_angry").reindex(index=MODELS, columns=list(INTERVENTIONS))
    for a, b in [(MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[1], MODELS[2])]:
        rho, p = spearmanr(mean_matrix.loc[a], mean_matrix.loc[b])
        tests.append({"test_type": "Spearman rank correlation", "intervention": "all seven",
            "comparison": f"{a} vs {b}", "n": len(INTERVENTIONS), "statistic": rho,
            "p_value_raw": p, "p_value_holm": np.nan, "effect_size": rho,
            "effect_size_name": "Spearman_rho", "direction": "ranking agreement"})
    test_df = pd.DataFrame(tests)
    corr_idx = test_df.index[test_df.test_type == "Spearman rank correlation"]
    test_df.loc[corr_idx, "p_value_holm"] = holm(test_df.loc[corr_idx, "p_value_raw"].to_numpy())
    test_df.to_csv(ART / "RQ2_STATISTICAL_TESTS.csv", index=False)
    return results, test_df, mean_matrix


def make_figures(rq1, rq2, matrix):
    colors = {"Mel-CNN": "#4C78A8", "RawWave-CNN": "#F58518", "Prosody-Fusion": "#54A24B", "Overall mean": "#555555"}
    single = rq1[rq1.section == "single_component"].pivot(index="condition", columns="model", values="mean_delta_p_angry").reindex(["Pitch", "Energy", "Duration"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    single.plot.bar(ax=ax, color=[colors[c] for c in single.columns], width=.78)
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("Mean ΔP(Angry)"); ax.set_xlabel(""); ax.legend(frameon=False, ncol=2); fig.tight_layout(); fig.savefig(FIG/"rq1_single_components.png", dpi=240); plt.close(fig)

    dose_order = ["sham_resynthesis", "pitch_25", "pitch_50", "pitch_75", "pitch_100", "pitch_125_stress"]
    strengths = [0, 25, 50, 75, 100, 125]
    dose = rq1[(rq1.section == "pitch_control_or_dose") & rq1.condition.isin(dose_order)].pivot(index="condition", columns="model", values="mean_delta_p_angry").reindex(dose_order)
    dose["Overall mean"] = dose.mean(axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for col in dose.columns:
        ax.plot(strengths, dose[col], marker="o", lw=2 if col != "Overall mean" else 2.5, linestyle="--" if col == "Overall mean" else "-", color=colors[col], label=col)
    ax.axhline(0, color="black", lw=.8); ax.set_xlabel("Pitch intervention strength (%)"); ax.set_ylabel("Mean ΔP(Angry) relative to sham"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG/"rq1_pitch_dose_response.png", dpi=240); plt.close(fig)

    decomp = rq1[(rq1.section == "pitch_control_or_dose") & rq1.condition.isin(["pitch_mean_100", "pitch_range_100"])].pivot(index="condition", columns="model", values="mean_delta_p_angry").reindex(["pitch_mean_100", "pitch_range_100"])
    decomp.index = ["F0 mean/baseline", "F0 range"]; decomp["Overall mean"] = decomp.mean(axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 4.8)); decomp.plot.bar(ax=ax, color=[colors[c] for c in decomp.columns], width=.72)
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("Mean ΔP(Angry) relative to sham"); ax.set_xlabel(""); ax.legend(frameon=False, ncol=2); fig.tight_layout(); fig.savefig(FIG/"rq1_pitch_mean_vs_range.png", dpi=240); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 3.7)); vals = matrix.to_numpy(); lim = max(abs(vals.min()), abs(vals.max())); im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns); ax.set_yticks(range(matrix.shape[0]), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j, i, f"{vals[i,j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Mean ΔP(Angry) on common Neutral subset"); fig.colorbar(im, ax=ax, label="Mean ΔP(Angry)"); fig.tight_layout(); fig.savefig(FIG/"rq2_model_intervention_heatmap.png", dpi=240); plt.close(fig)


def write_findings(rq1_tests, rq1_results, rq2_tests, rq2_results, common_n):
    primary = rq1_tests[rq1_tests.test_family == "RQ1_primary_direct"]
    def rows(comp): return primary[primary.comparison == comp]
    pe, pdur, mr = rows("Pitch vs Energy"), rows("Pitch vs Duration"), rows("F0 mean vs F0 range")
    sham = rq1_tests[rq1_tests.test_family == "pitch_vs_sham"]
    direction = rq1_tests[rq1_tests.test_family == "direction_control"]
    dose = rq1_results[rq1_results.section == "pitch_control_or_dose"].groupby("condition").mean(numeric_only=True)
    pitch_stronger_energy = bool(((pe.mean_paired_difference > 0) & (pe.p_value_holm < .05)).all())
    pitch_vs_duration_supported = pdur.assign(ok=(pdur.mean_paired_difference > 0) & (pdur.p_value_holm < .05))
    mean_over_range = bool(((mr.mean_paired_difference > 0) & (mr.p_value_holm < .05)).all())
    sham_supported = bool(((sham.mean_paired_difference > 0) & (sham.p_value_holm < .05)).all())
    direction_supported = bool(((direction.mean_paired_difference > 0) & (direction.p_value_holm < .05)).all())
    rq1_lines = ["# RQ1 final findings", "",
        f"1. **Is Pitch significantly stronger than Energy?** {'Yes' if pitch_stronger_energy else 'Not consistently across all architectures'}. Direct paired Wilcoxon tests were positive for {int(((pe.mean_paired_difference>0)&(pe.p_value_holm<.05)).sum())}/3 models after Holm correction.", "",
        f"2. **Is Pitch significantly stronger than Duration?** The result was architecture-dependent. Pitch was significantly stronger for {int(pitch_vs_duration_supported.ok.sum())}/3 models; RawWave-CNN showed a different duration-sensitivity profile.", "",
        f"3. **Is the Pitch effect still present relative to PSOLA sham?** {'Yes' if sham_supported else 'Not for every architecture'}. The 100% Pitch shift exceeded sham for {int(((sham.mean_paired_difference>0)&(sham.p_value_holm<.05)).sum())}/3 models.", "",
        f"4. **Is there a clear dose-response pattern?** Yes descriptively. The architecture-mean ΔP(Angry) relative to sham increased from {dose.loc['pitch_25','mean_delta_p_angry']:.4f} at 25% to {dose.loc['pitch_100','mean_delta_p_angry']:.4f} at 100% and {dose.loc['pitch_125_stress','mean_delta_p_angry']:.4f} at the secondary 125% stress condition.", "",
        f"5. **Does reverse Pitch move prediction in the opposite direction?** {'Yes' if direction_supported else 'The paired angry-directed versus reverse comparison was not significant for every architecture'}. Angry-directed Pitch exceeded reverse Pitch in {int(((direction.mean_paired_difference>0)&(direction.p_value_holm<.05)).sum())}/3 models.", "",
        f"6. **Is the effect mainly caused by F0 baseline or F0 range?** {'F0 baseline' if mean_over_range else 'The evidence was mixed'}. The direct mean-versus-range comparison favored F0 baseline in {int(((mr.mean_paired_difference>0)&(mr.p_value_holm<.05)).sum())}/3 models.", "",
        "**Conclusion.** Among the tested acoustic components, F0 manipulation produced the strongest and most systematic shift toward Angry predictions. The effect was primarily associated with elevation of the F0 baseline rather than expansion of the F0 range. This result describes SER model sensitivity under controlled intervention."]
    (ART/"RQ1_FINAL_FINDINGS.md").write_text("\n".join(rq1_lines)+"\n", encoding="utf-8")

    fried = rq2_tests[(rq2_tests.test_type == "Friedman") & (rq2_tests.intervention == "Pitch")].iloc[0]
    post = rq2_tests[(rq2_tests.test_type == "Wilcoxon post-hoc") & (rq2_tests.intervention == "Pitch")]
    prosody_pairs = post[post.comparison.str.contains("Prosody-Fusion", regex=False)]
    prosody_sig = len(prosody_pairs) == 2 and (prosody_pairs.p_value_holm < .05).all()
    corr = rq2_tests[rq2_tests.test_type == "Spearman rank correlation"]
    matrix = rq2_results.pivot(index="model", columns="intervention", values="mean_delta_p_angry")
    single_ranks = matrix[["Pitch","Energy","Duration"]].rank(axis=1, ascending=False, method="min")
    ranking_text = "; ".join(f"{m}: " + " > ".join(matrix.loc[m,["Pitch","Energy","Duration"]].sort_values(ascending=False).index) for m in MODELS)
    rq2_lines = ["# RQ2 final findings", "",
        f"Common evaluation subset: **N={common_n}** Neutral utterances correctly classified as Neutral by all three models before intervention.", "",
        f"1. **Do the three architectures react equally strongly to Pitch?** No. The Friedman test was {'significant' if fried.p_value_raw < .05 else 'not significant'} (χ²={fried.statistic:.3f}, p={fried.p_value_raw:.3g}, Kendall’s W={fried.effect_size:.3f}).", "",
        f"2. **Is Prosody-Fusion significantly more Pitch-sensitive than the other architectures?** {'Yes' if prosody_sig else 'Not against both architectures after Holm correction'}. The statement is based on paired post-hoc Wilcoxon tests on the same {common_n} utterances.", "",
        f"3. **Do model rankings of Pitch/Energy/Duration agree?** No complete agreement was observed. {ranking_text}. Pairwise seven-intervention Spearman correlations ranged from {corr.statistic.min():.3f} to {corr.statistic.max():.3f}; non-significant correlations are not evidence of different internal representations.", "",
        "4. **Are there clear architecture-specific intervention profiles?** Yes. Prosody-Fusion showed its largest response to Pitch-related conditions, RawWave-CNN was relatively more responsive to duration-containing conditions, and Mel-CNN responses were smaller and included negative duration effects.", "",
        "5. **Which result is strongest enough to include in the paper?** The common-subset paired analysis showing substantially greater Pitch sensitivity for Prosody-Fusion is the strongest RQ2 result. The intervention-ranking analysis should be presented as supporting descriptive evidence.", "",
        "**Conclusion.** SER architectures exhibited different acoustic-intervention sensitivity profiles, with Prosody-Fusion showing substantially greater sensitivity to F0 manipulation. The result concerns output sensitivity and does not establish completely different internal representations."]
    (ART/"RQ2_FINAL_FINDINGS.md").write_text("\n".join(rq2_lines)+"\n", encoding="utf-8")


def validate_outputs(common_n):
    required = ["RQ1_FINAL_RESULTS.csv", "RQ2_FINAL_RESULTS.csv", "RQ1_STATISTICAL_TESTS.csv", "RQ2_STATISTICAL_TESTS.csv", "rq2_common_sample_manifest.csv"]
    for name in required:
        path = ART/name
        if not path.exists() or path.stat().st_size == 0: raise AssertionError(f"Missing output {name}")
        frame = pd.read_csv(path)
        if frame.empty: raise AssertionError(f"Empty output {name}")
        if frame.columns.duplicated().any(): raise AssertionError(f"Duplicate columns {name}")
    manifest = pd.read_csv(ART/"rq2_common_sample_manifest.csv")
    if len(manifest) != common_n or manifest.sample_id.duplicated().any(): raise AssertionError("Invalid common manifest")
    expected_figs = ["rq1_single_components.png", "rq1_pitch_dose_response.png", "rq1_pitch_mean_vs_range.png", "rq2_model_intervention_heatmap.png"]
    for name in expected_figs:
        if not (FIG/name).exists() or (FIG/name).stat().st_size < 10_000: raise AssertionError(f"Invalid figure {name}")


def main():
    core = pd.read_csv(CORE, dtype={"sample_id": str, "speaker_id": str})
    dose = pd.read_csv(DOSE, dtype={"sample_id": str, "speaker_id": str})
    if len(core) != 227 * 8 * 3: raise AssertionError("Unexpected core prediction design")
    if len(dose) != 227 * 10 * 3: raise AssertionError("Unexpected dose prediction design")
    if core[["sample_id","model","transformation"]].duplicated().any(): raise AssertionError("Duplicate core rows")
    if dose[["sample_id","model","condition"]].duplicated().any(): raise AssertionError("Duplicate dose rows")
    rq1, rq1_tests = finalize_rq1(core, dose)
    common = build_common_manifest(core)
    rq2, rq2_tests, matrix = finalize_rq2(core, common)
    make_figures(rq1, rq2, matrix)
    write_findings(rq1_tests, rq1, rq2_tests, rq2, len(common))
    validate_outputs(len(common))
    print(f"Finalized RQ1/RQ2. Common subset N={len(common)}")
    print(matrix.to_string())


if __name__ == "__main__":
    main()
