"""Paired four-pipeline robustness analysis on all 227 true-Neutral items.

This script reuses stored probabilities. It performs no model training or
inference and does not replace the strict baseline-correct RQ2 analysis.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PREDICTIONS = ARTIFACTS / "rq2_4model_predictions.csv"
OUT_CSV = ARTIFACTS / "RQ2_ALL_NEUTRAL_PAIRED_STATISTICS.csv"
OUT_MD = ARTIFACTS / "RQ2_ALL_NEUTRAL_PAIRED_FINDINGS.md"

MODELS = ["Mel-CNN", "RawWave-CNN", "Wav2Vec2", "Prosody-Fusion"]
CUES = {
    "Pitch": "T1_pitch",
    "Energy": "T2_energy",
    "Duration": "T3_duration",
}
SEED = 2025
BOOTSTRAPS = 5000


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def matched_rank_biserial(differences):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def bootstrap_mean_ci(values, salt):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED + salt)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    frame = pd.read_csv(PREDICTIONS, dtype={"sample_id": str})
    selected = frame[frame.transformation.isin(CUES.values())].copy()
    pivot = selected.pivot(
        index="sample_id",
        columns=["model", "transformation"],
        values="delta_p_angry",
    ).sort_index()

    required = pd.MultiIndex.from_product([MODELS, CUES.values()])
    missing_columns = required.difference(pivot.columns)
    if len(missing_columns):
        raise RuntimeError(f"Missing model-condition columns: {list(missing_columns)}")
    pivot = pivot.loc[:, required]
    if len(pivot) != 227 or pivot.isna().any().any():
        raise RuntimeError(f"Expected complete N=227 paired matrix, found {pivot.shape}")

    rows = []
    salt = 0
    for cue, transformation in CUES.items():
        arrays = [pivot[(model, transformation)].to_numpy() for model in MODELS]
        statistic, p_value = friedmanchisquare(*arrays)
        rows.append(
            {
                "test_type": "Friedman",
                "intervention": cue,
                "comparison": " vs ".join(MODELS),
                "n": len(pivot),
                "statistic": statistic,
                "p_value_raw": p_value,
                "p_value_holm": p_value,
                "mean_paired_difference": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "effect_size": statistic / (len(pivot) * (len(MODELS) - 1)),
                "effect_size_name": "Kendall_W",
                "direction": "omnibus",
            }
        )

        posthoc = []
        for left, right in combinations(range(len(MODELS)), 2):
            difference = arrays[left] - arrays[right]
            try:
                stat, raw_p = wilcoxon(
                    difference,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
            except ValueError:
                stat, raw_p = 0.0, 1.0
            low, high = bootstrap_mean_ci(difference, salt)
            salt += 1
            posthoc.append(
                {
                    "test_type": "Wilcoxon post-hoc",
                    "intervention": cue,
                    "comparison": f"{MODELS[left]} vs {MODELS[right]}",
                    "n": len(pivot),
                    "statistic": stat,
                    "p_value_raw": raw_p,
                    "p_value_holm": np.nan,
                    "mean_paired_difference": difference.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "effect_size": matched_rank_biserial(difference),
                    "effect_size_name": "matched_rank_biserial",
                    "direction": "A > B" if difference.mean() > 0 else "A < B",
                }
            )
        adjusted = holm([row["p_value_raw"] for row in posthoc])
        for row, value in zip(posthoc, adjusted):
            row["p_value_holm"] = value
        rows.extend(posthoc)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_CSV, index=False)

    lines = [
        "# RQ2 paired all-Neutral robustness analysis",
        "",
        "This secondary analysis reuses stored predictions for all 227 true-Neutral test utterances. "
        "It performs no training or inference and does not replace the strict four-model "
        "baseline-correct common-subset analysis (N=36).",
        "",
        "All tests are paired by utterance across the four complete pipelines. Angry Flip Rate "
        "is not reported for this all-Neutral set because its denominator is defined only for "
        "items predicted Neutral at baseline.",
        "",
        "## Omnibus tests",
        "",
        "| Cue | N | Friedman chi-square | p-value | Kendall's W |",
        "|---|---:|---:|---:|---:|",
    ]
    omnibus = results[results.test_type.eq("Friedman")]
    for row in omnibus.itertuples(index=False):
        lines.append(
            f"| {row.intervention} | {row.n} | {row.statistic:.3f} | "
            f"{row.p_value_raw:.3g} | {row.effect_size:.3f} |"
        )

    lines.extend(["", "## Pairwise results", ""])
    for cue in CUES:
        lines.extend(
            [
                f"### {cue}",
                "",
                "| Comparison | Mean paired difference | 95% bootstrap CI | Holm p | Rank-biserial |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        subset = results[
            results.test_type.eq("Wilcoxon post-hoc")
            & results.intervention.eq(cue)
        ]
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.comparison} | {row.mean_paired_difference:.4f} | "
                f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] | "
                f"{row.p_value_holm:.3g} | {row.effect_size:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These paired all-Neutral results test whether complete pipelines respond differently "
            "on a broader fixed set. They do not isolate neural architecture from preprocessing, "
            "feature access, calibration, or training, and they do not estimate Neutral-to-Angry "
            "flip rates on a common baseline-correct population.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
