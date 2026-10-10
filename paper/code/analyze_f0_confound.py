"""Quantify how the train-derived Neutral-to-Angry F0 contrast varies by metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cf_ser.config import ARTIFACTS, SPLIT_CSV


SEED = 2025


def bootstrap_ci(values, iterations=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(iterations)]
    return np.percentile(means, [2.5, 97.5])


def contrast(frame, group_name, group_value):
    neutral = frame.loc[frame.emotion == "neutral", "logf0_mean"].dropna()
    angry = frame.loc[frame.emotion == "angry", "logf0_mean"].dropna()
    delta = float(angry.mean() - neutral.mean())
    return {
        "scope": group_name, "group": group_value,
        "n_neutral": len(neutral), "n_angry": len(angry),
        "mean_logf0_neutral": neutral.mean(), "mean_logf0_angry": angry.mean(),
        "delta_logf0": delta, "f0_ratio": float(np.exp(delta)),
        "semitones": float(12 * delta / np.log(2)),
    }


def main():
    split = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    features = pd.read_csv(ARTIFACTS / "audio_features.csv")
    train = split[split.split == "train"].merge(features, on="path")
    train = train[train.emotion.isin(["neutral", "angry"])].copy()

    rows = [contrast(train, "overall", "all")]
    for accent, frame in train.groupby("accent"):
        rows.append(contrast(frame, "accent", accent))
    rows.append(contrast(train[train.speaker == "0"], "speaker_status", "missing_id_bucket"))
    rows.append(contrast(train[train.speaker != "0"], "speaker_status", "identified"))

    known = train[train.speaker != "0"]
    speaker_means = known.groupby(["speaker", "emotion"]).logf0_mean.mean().unstack()
    speaker_means = speaker_means.dropna(subset=["neutral", "angry"])
    speaker_means["delta_logf0"] = speaker_means.angry - speaker_means.neutral
    speaker_means["f0_ratio"] = np.exp(speaker_means.delta_logf0)
    speaker_means["semitones"] = 12 * speaker_means.delta_logf0 / np.log(2)
    speaker_means.reset_index().to_csv(ARTIFACTS / "F0_WITHIN_SPEAKER_CONTRASTS.csv", index=False)

    if len(speaker_means):
        low, high = bootstrap_ci(speaker_means.delta_logf0)
        try:
            stat, p_value = wilcoxon(speaker_means.delta_logf0)
        except ValueError:
            stat, p_value = 0.0, 1.0
        rows.append({
            "scope": "within_speaker", "group": "identified_with_both_emotions",
            "n_neutral": len(speaker_means), "n_angry": len(speaker_means),
            "mean_logf0_neutral": speaker_means.neutral.mean(),
            "mean_logf0_angry": speaker_means.angry.mean(),
            "delta_logf0": speaker_means.delta_logf0.mean(),
            "f0_ratio": float(np.exp(speaker_means.delta_logf0.mean())),
            "semitones": float(12 * speaker_means.delta_logf0.mean() / np.log(2)),
            "ci95_low_logf0": low, "ci95_high_logf0": high,
            "wilcoxon_statistic": stat, "wilcoxon_p_value": p_value,
        })

    results = pd.DataFrame(rows)
    results.to_csv(ARTIFACTS / "F0_CONFOUND_AUDIT.csv", index=False)
    overall = results[(results.scope == "overall")].iloc[0]
    within = results[(results.scope == "within_speaker")]
    accents = results[results.scope == "accent"]
    lines = [
        "# Train-derived F0 contrast audit", "",
        f"The pooled Neutral-to-Angry contrast is {overall.semitones:.3f} semitones ({overall.f0_ratio:.3f}x).", "",
        "| Accent | Neutral N | Angry N | Semitones | F0 ratio |", "|---|---:|---:|---:|---:|",
    ]
    for row in accents.itertuples(index=False):
        lines.append(f"| {row.group} | {row.n_neutral} | {row.n_angry} | {row.semitones:.3f} | {row.f0_ratio:.3f} |")
    lines += ["", f"Gender cannot be audited because it is absent from the available ViSEC metadata. Speaker ID `0` is treated as a missing-ID bucket, never as one speaker.", ""]
    if len(within):
        row = within.iloc[0]
        lines += [
            f"Among {int(row.n_neutral)} identified speakers represented in both emotions, the mean within-speaker contrast is {row.semitones:.3f} semitones (Wilcoxon p={row.wilcoxon_p_value:.6g}).", "",
        ]
    lines += [
        "The pooled corpus contrast is retained only as a secondary corpus-derived condition. Fixed semitone doses are the primary intervention scale for reviewer-hardened analysis.",
    ]
    (ARTIFACTS / "F0_CONFOUND_AUDIT_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
