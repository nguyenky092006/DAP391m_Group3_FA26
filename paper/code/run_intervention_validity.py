"""Objective quality checks for the fixed 100% F0-mean intervention."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

from cf_ser.acoustics import pitch_transform
from cf_ser.config import ARTIFACTS, SAMPLE_RATE, SPLIT_CSV
from cf_ser.data import load_audio


def local_log_f0(waveform):
    pitch = parselmouth.Sound(waveform, sampling_frequency=SAMPLE_RATE).to_pitch_ac(
        time_step=0.01, pitch_floor=60, pitch_ceiling=500
    )
    values = pitch.selected_array["frequency"].astype(float)
    out = np.full(values.shape, np.nan, dtype=float)
    mask = (values > 0) & np.isfinite(values)
    out[mask] = np.log(values[mask])
    return out


def rms_db(waveform):
    rms = np.sqrt(np.mean(np.square(np.asarray(waveform, float))))
    return float(20 * np.log10(max(rms, 1e-12)))


def analyze_pair(original, transformed):
    f0_original = local_log_f0(original)
    f0_transformed = local_log_f0(transformed)
    n = min(len(f0_original), len(f0_transformed))
    common = np.isfinite(f0_original[:n]) & np.isfinite(f0_transformed[:n])
    if common.sum() >= 3 and np.std(f0_original[:n][common]) > 0 and np.std(f0_transformed[:n][common]) > 0:
        contour_correlation = float(np.corrcoef(f0_original[:n][common], f0_transformed[:n][common])[0, 1])
    else:
        contour_correlation = np.nan
    achieved = float(np.mean(f0_transformed[:n][common] - f0_original[:n][common])) if common.any() else np.nan
    duration_original = len(original) / SAMPLE_RATE
    duration_transformed = len(transformed) / SAMPLE_RATE
    return {
        "achieved_mean_log_f0_shift": achieved,
        "achieved_mean_f0_ratio": float(np.exp(achieved)) if np.isfinite(achieved) else np.nan,
        "local_log_f0_contour_correlation": contour_correlation,
        "common_voiced_frames": int(common.sum()),
        "duration_original_seconds": duration_original,
        "duration_transformed_seconds": duration_transformed,
        "duration_difference_seconds": duration_transformed - duration_original,
        "duration_difference_percent": 100 * (duration_transformed / duration_original - 1),
        "rms_original_dbfs": rms_db(original),
        "rms_transformed_dbfs": rms_db(transformed),
        "rms_difference_db": rms_db(transformed) - rms_db(original),
        "transformed_peak_absolute": float(np.max(np.abs(transformed))),
        "clipping_fraction_ge_0_999": float(np.mean(np.abs(transformed) >= 0.999)),
        "finite_audio": bool(np.isfinite(transformed).all()),
    }


def describe(frame, label):
    return {
        "group": label,
        "n": len(frame),
        "mean_achieved_log_f0_shift": frame.achieved_mean_log_f0_shift.mean(),
        "median_achieved_f0_ratio": frame.achieved_mean_f0_ratio.median(),
        "median_contour_correlation": frame.local_log_f0_contour_correlation.median(),
        "max_absolute_duration_difference_seconds": frame.duration_difference_seconds.abs().max(),
        "mean_rms_difference_db": frame.rms_difference_db.mean(),
        "max_absolute_rms_difference_db": frame.rms_difference_db.abs().max(),
        "mean_clipping_fraction": frame.clipping_fraction_ge_0_999.mean(),
        "max_clipping_fraction": frame.clipping_fraction_ge_0_999.max(),
        "all_finite": bool(frame.finite_audio.all()),
    }


def main():
    split = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    neutral = split[(split.split == "test") & (split.emotion == "neutral") & split.accent.isin(["north", "mid", "south"])].copy()
    params = json.loads((ARTIFACTS / "transformation_params.json").read_text(encoding="utf-8"))
    rows = []
    for number, row in enumerate(neutral.itertuples(index=False), start=1):
        original = load_audio(row.path)
        transformed, requested_shift = pitch_transform(original, params, strength=1.0, mode="mean")
        metrics = analyze_pair(original, transformed)
        rows.append({
            "sample_id": Path(row.path).stem, "speaker_id": row.speaker, "accent": row.accent,
            "condition": "F0_mean_100", "requested_tier_mean_log_f0_shift": requested_shift,
            **metrics,
        })
        if number % 25 == 0:
            print(f"Objective validity {number}/{len(neutral)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(ARTIFACTS / "INTERVENTION_VALIDITY_RESULTS.csv", index=False)
    summaries = [describe(result, "Overall")]
    for accent in ["north", "mid", "south"]:
        summaries.append(describe(result[result.accent == accent], accent.capitalize() if accent != "mid" else "Central"))
    summary = pd.DataFrame(summaries)
    overall = summary.iloc[0]
    lines = [
        "# Objective intervention validity", "",
        "These checks evaluate the unchanged 100% F0-mean intervention on all true Neutral ViSEC test utterances with valid accent metadata. No intervention parameter was optimized from these results.", "",
        summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Interpretation", "",
        f"- The median achieved F0 ratio was {overall.median_achieved_f0_ratio:.3f}× (log-F0 shift {overall.mean_achieved_log_f0_shift:.4f}).",
        f"- Median local log-F0 contour correlation was {overall.median_contour_correlation:.4f}; the intervention primarily shifts the baseline while retaining local contour shape.",
        f"- Maximum absolute duration difference was {overall.max_absolute_duration_difference_seconds:.6f} s.",
        f"- Mean RMS difference was {overall.mean_rms_difference_db:.4f} dB; maximum absolute RMS difference was {overall.max_absolute_rms_difference_db:.4f} dB.",
        f"- Maximum sample clipping fraction was {overall.max_clipping_fraction:.6%}; all transformed arrays were finite: {bool(overall.all_finite)}.",
        "", "These are processing-quality checks, not evidence of human emotion perception.",
    ]
    (ARTIFACTS / "INTERVENTION_VALIDITY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
