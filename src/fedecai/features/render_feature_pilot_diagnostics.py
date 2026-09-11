"""Render reproducible diagnostics for a completed B4 feature-pilot run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACCENT_ORDER = ("central", "north", "south")
EMOTION_ORDER = ("angry", "happy", "neutral", "sad")
ACCENT_COLORS = {
    "central": "#0072B2",
    "north": "#009E73",
    "south": "#D55E00",
}
EMOTION_MARKERS = {
    "angry": "o",
    "happy": "s",
    "neutral": "^",
    "sad": "D",
}
SAMPLE_RATE = 16_000
HOP_LENGTH = 160


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def expected_centered_frame_count(duration_sec: float) -> int:
    """Return librosa's centered STFT frame count for the B4 10 ms hop."""
    sample_count = int(round(duration_sec * SAMPLE_RATE))
    return sample_count // HOP_LENGTH + 1


def build_pilot_grid(
    manifest_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    expected_cells = {
        (accent, emotion) for accent in ACCENT_ORDER for emotion in EMOTION_ORDER
    }
    grid: dict[tuple[str, str], dict[str, str]] = {}
    for row in manifest_rows:
        cell = (row["accent"], row["emotion"])
        if cell not in expected_cells:
            raise ValueError(f"Unexpected accent-emotion cell: {cell}")
        if cell in grid:
            raise ValueError(f"Duplicate accent-emotion cell: {cell}")
        grid[cell] = row
    missing = sorted(expected_cells - set(grid))
    if missing:
        raise ValueError(f"Missing accent-emotion cells: {missing}")
    return grid


def build_feature_lookup(
    repo_root: Path, feature_rows: list[dict[str, str]]
) -> dict[tuple[str, str], Path]:
    lookup: dict[tuple[str, str], Path] = {}
    for row in feature_rows:
        if row["status"] != "ok":
            continue
        key = (row["sample_id"], row["feature_family"])
        if key in lookup:
            raise ValueError(f"Duplicate feature artifact: {key}")
        artifact = (repo_root / row["artifact_path"]).resolve()
        try:
            artifact.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Artifact is outside repository: {artifact}") from exc
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        lookup[key] = artifact
    return lookup


def _panel_title(row: dict[str, str]) -> str:
    return (
        f"{row['emotion']} | spk {row['speaker_id']}\n"
        f"{float(row['duration_sec']):.2f} s"
    )


def render_log_mel_grid(
    output_path: Path,
    grid: dict[tuple[str, str], dict[str, str]],
    features: dict[tuple[str, str], Path],
) -> None:
    figure, axes = plt.subplots(
        len(ACCENT_ORDER),
        len(EMOTION_ORDER),
        figsize=(15, 9),
        sharey=True,
        layout="constrained",
    )
    image = None
    for row_index, accent in enumerate(ACCENT_ORDER):
        for column_index, emotion in enumerate(EMOTION_ORDER):
            axis = axes[row_index, column_index]
            row = grid[(accent, emotion)]
            value = np.load(
                features[(row["sample_id"], "log_mel_80")], allow_pickle=False
            )
            duration = float(row["duration_sec"])
            image = axis.imshow(
                value,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=-80,
                vmax=0,
                extent=(0, duration, 0, value.shape[0]),
                interpolation="nearest",
            )
            axis.set_title(_panel_title(row), fontsize=9)
            axis.set_xlabel("Time (s)")
            if column_index == 0:
                axis.set_ylabel(f"{accent}\nMel bin")
    if image is not None:
        figure.colorbar(image, ax=axes, label="Power (dB)", shrink=0.82, pad=0.02)
    figure.suptitle(
        "B4 pilot log-Mel diagnostics — one sample per accent × emotion",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def render_pitch_grid(
    output_path: Path,
    grid: dict[tuple[str, str], dict[str, str]],
    features: dict[tuple[str, str], Path],
) -> list[float]:
    figure, axes = plt.subplots(
        len(ACCENT_ORDER),
        len(EMOTION_ORDER),
        figsize=(15, 9),
        sharey=True,
        layout="constrained",
    )
    voiced_fractions: list[float] = []
    for row_index, accent in enumerate(ACCENT_ORDER):
        for column_index, emotion in enumerate(EMOTION_ORDER):
            axis = axes[row_index, column_index]
            row = grid[(accent, emotion)]
            with np.load(
                features[(row["sample_id"], "pitch_contour")], allow_pickle=False
            ) as archive:
                f0 = archive["f0_hz"].astype(np.float64, copy=True)
            voiced = np.isfinite(f0)
            voiced_fraction = float(voiced.mean())
            voiced_fractions.append(voiced_fraction)
            time = np.arange(f0.size) * HOP_LENGTH / SAMPLE_RATE
            axis.plot(time[voiced], f0[voiced], ".", color="#0072B2", markersize=2)
            axis.set_xlim(0, max(float(row["duration_sec"]), 0.01))
            axis.set_ylim(50, 500)
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{_panel_title(row)} | voiced {voiced_fraction:.0%}", fontsize=9
            )
            axis.set_xlabel("Time (s)")
            if column_index == 0:
                axis.set_ylabel(f"{accent}\nF0 (Hz)")
    figure.suptitle(
        "B4 pilot pitch diagnostics — unvoiced frames remain missing",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return voiced_fractions


def render_frame_duration_check(
    output_path: Path,
    manifest_rows: list[dict[str, str]],
    features: dict[tuple[str, str], Path],
) -> tuple[list[int], list[int]]:
    actual_counts: list[int] = []
    expected_counts: list[int] = []
    durations: list[float] = []
    figure, axis = plt.subplots(figsize=(10, 6.5), layout="constrained")
    duration_values = [float(row["duration_sec"]) for row in manifest_rows]
    labelled_durations = {min(duration_values), max(duration_values)}
    for row in manifest_rows:
        duration = float(row["duration_sec"])
        value = np.load(
            features[(row["sample_id"], "log_mel_80")], allow_pickle=False
        )
        actual = int(value.shape[1])
        expected = expected_centered_frame_count(duration)
        durations.append(duration)
        actual_counts.append(actual)
        expected_counts.append(expected)
        axis.scatter(
            duration,
            actual,
            color=ACCENT_COLORS[row["accent"]],
            marker=EMOTION_MARKERS[row["emotion"]],
            s=75,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        if duration in labelled_durations:
            axis.annotate(
                row["sample_id"].removeprefix("visec_hf_"),
                (duration, actual),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
    x_values = np.linspace(0, max(durations) * 1.04, 300)
    y_values = [expected_centered_frame_count(float(value)) for value in x_values]
    axis.plot(x_values, y_values, color="#333333", linewidth=1.5, label="Expected 10 ms hop")
    for accent in ACCENT_ORDER:
        axis.scatter([], [], color=ACCENT_COLORS[accent], label=accent, s=55)
    for emotion in EMOTION_ORDER:
        axis.scatter(
            [], [], color="white", edgecolor="black", marker=EMOTION_MARKERS[emotion],
            label=emotion, s=55,
        )
    max_error = max(abs(a - e) for a, e in zip(actual_counts, expected_counts))
    axis.text(
        0.02,
        0.95,
        f"Maximum absolute frame error: {max_error}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )
    axis.set_xlabel("Audio duration (s)")
    axis.set_ylabel("log-Mel frame count")
    axis.set_title("B4 pilot sequence length follows duration (no padding or cropping)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8, loc="lower right")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return actual_counts, expected_counts


def render_diagnostics(
    repo_root: Path, run_root: Path, figure_dir: Path, report_path: Path
) -> dict[str, Any]:
    manifest_rows = read_csv(run_root / "pilot_manifest.csv")
    feature_rows = read_csv(run_root / "feature_index.csv")
    grid = build_pilot_grid(manifest_rows)
    features = build_feature_lookup(repo_root, feature_rows)
    required = {
        (row["sample_id"], family)
        for row in manifest_rows
        for family in ("log_mel_80", "pitch_contour")
    }
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(f"Missing diagnostic feature artifacts: {missing}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "log_mel_grid": figure_dir / "b4_pilot_log_mel_grid.png",
        "pitch_grid": figure_dir / "b4_pilot_pitch_grid.png",
        "frame_duration_check": figure_dir / "b4_pilot_frame_duration_check.png",
    }
    render_log_mel_grid(figure_paths["log_mel_grid"], grid, features)
    voiced_fractions = render_pitch_grid(figure_paths["pitch_grid"], grid, features)
    actual_counts, expected_counts = render_frame_duration_check(
        figure_paths["frame_duration_check"], manifest_rows, features
    )
    frame_errors = [actual - expected for actual, expected in zip(actual_counts, expected_counts)]
    report: dict[str, Any] = {
        "status": "PASS" if max(map(abs, frame_errors), default=0) == 0 else "REVIEW",
        "scope": "12-sample balanced B4 pilot; diagnostic only, not population evidence.",
        "run_root": run_root.resolve().relative_to(repo_root.resolve()).as_posix(),
        "sample_count": len(manifest_rows),
        "accent_emotion_cells": len(grid),
        "figures": {
            name: path.resolve().relative_to(repo_root.resolve()).as_posix()
            for name, path in figure_paths.items()
        },
        "log_mel_frames": {
            "minimum": min(actual_counts),
            "maximum": max(actual_counts),
            "maximum_absolute_expected_error": max(map(abs, frame_errors), default=0),
            "hop_length_samples": HOP_LENGTH,
            "hop_duration_ms": HOP_LENGTH / SAMPLE_RATE * 1000,
        },
        "pitch_voiced_fraction": {
            "minimum": min(voiced_fractions),
            "mean": float(np.mean(voiced_fractions)),
            "maximum": max(voiced_fractions),
        },
        "interpretation_guardrail": (
            "Panels are one sample per cell and may reveal extraction problems; "
            "they must not be used to claim accent or emotion population differences."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="balanced_12")
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "data/interim/features/b4_features_v1",
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=repo_root / "reports/figures/b4"
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_pilot_diagnostic_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    report = render_diagnostics(
        repo_root=repo_root,
        run_root=args.root.resolve() / args.run_name,
        figure_dir=args.figure_dir.resolve(),
        report_path=args.report_path.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
