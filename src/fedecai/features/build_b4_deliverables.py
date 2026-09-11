"""Build the full B4 feature table, three advanced figures, and dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .wav2vec2_chunk_contract import (
        eligible_manifest_rows,
        file_sha256,
        read_manifest_csv,
    )
except ImportError:
    from wav2vec2_chunk_contract import (  # type: ignore
        eligible_manifest_rows,
        file_sha256,
        read_manifest_csv,
    )


ACCENTS = ("central", "north", "south")
EMOTIONS = ("angry", "happy", "neutral", "sad")
COLORS = {
    "angry": "#d62728",
    "happy": "#ffbf00",
    "neutral": "#1f77b4",
    "sad": "#9467bd",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def require_passing_report(path: Path, expected_rows: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("failures") != 0:
        raise ValueError(f"Required extraction report did not pass: {path}")
    reported_rows = report.get("eligible_rows", report.get("eligible_samples"))
    if int(reported_rows) != expected_rows:
        raise ValueError(f"Extraction report row count differs: {path}")
    return report


def load_index(
    path: Path, expected_families: set[str], expected_records: int
) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    if len(rows) != expected_records:
        raise ValueError(f"Unexpected index size for {path}: {len(rows)}")
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["sample_id"], row["feature_family"])
        if key in indexed:
            raise ValueError(f"Duplicate feature-index key: {key}")
        if row["feature_family"] not in expected_families:
            raise ValueError(f"Unexpected feature family: {row['feature_family']}")
        if row["status"] != "ok" or row["error"]:
            raise ValueError(f"Unclean feature-index row: {key}")
        indexed[key] = row
    return indexed


def load_npy(repo_root: Path, row: dict[str, str], shape: tuple[int, ...]) -> np.ndarray:
    path = repo_root / row["artifact_path"]
    if not path.is_file():
        raise ValueError(f"Missing feature artifact: {path}")
    value = np.load(path, allow_pickle=False)
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"Invalid feature artifact: {path}")
    return value


def load_pitch_summary(repo_root: Path, row: dict[str, str]) -> np.ndarray:
    path = repo_root / row["artifact_path"]
    with np.load(path, allow_pickle=False) as archive:
        value = archive["summary"]
    if value.shape != (6,) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"Invalid pitch summary: {path}")
    return value


def build_feature_table(
    repo_root: Path,
    manifest_path: Path,
    handcrafted_index_path: Path,
    wav2vec_index_path: Path,
    handcrafted_report_path: Path,
    wav2vec_report_path: Path,
    egemaps_names_path: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    eligible = eligible_manifest_rows(read_manifest_csv(manifest_path))
    sample_count = len(eligible)
    handcrafted_report = require_passing_report(handcrafted_report_path, sample_count)
    wav2vec_report = require_passing_report(wav2vec_report_path, sample_count)
    if file_sha256(handcrafted_index_path) != handcrafted_report.get("index_sha256"):
        raise ValueError("Handcrafted feature index differs from its passing report")
    if file_sha256(wav2vec_index_path) != wav2vec_report.get("aggregate_index_sha256"):
        raise ValueError("Wav2Vec2 feature index differs from its passing report")
    handcrafted = load_index(
        handcrafted_index_path,
        {"mfcc40_summary", "egemaps_v02", "log_mel_80", "pitch_contour"},
        sample_count * 4,
    )
    wav2vec = load_index(wav2vec_index_path, {"wav2vec2_embedding"}, sample_count)
    egemaps_names = json.loads(egemaps_names_path.read_text(encoding="utf-8"))
    if len(egemaps_names) != 88 or len(egemaps_names) != len(set(egemaps_names)):
        raise ValueError("Expected 88 unique eGeMAPS feature names")

    mfcc = np.empty((sample_count, 80), dtype=np.float32)
    egemaps = np.empty((sample_count, 88), dtype=np.float32)
    pitch = np.empty((sample_count, 6), dtype=np.float32)
    wav2vec_matrix = np.empty((sample_count, 768), dtype=np.float32)
    metadata: list[dict[str, Any]] = []
    for index, row in enumerate(eligible):
        sample_id = row["sample_id"]
        mfcc[index] = load_npy(repo_root, handcrafted[(sample_id, "mfcc40_summary")], (80,))
        egemaps[index] = load_npy(repo_root, handcrafted[(sample_id, "egemaps_v02")], (88,))
        pitch[index] = load_pitch_summary(
            repo_root, handcrafted[(sample_id, "pitch_contour")]
        )
        wav2vec_matrix[index] = load_npy(
            repo_root, wav2vec[(sample_id, "wav2vec2_embedding")], (768,)
        )
        metadata.append(
            {
                "sample_id": sample_id,
                "source_row_index": int(row["source_row_index"]),
                "speaker_id": row["speaker_id"],
                "accent": row["accent"],
                "emotion": row["emotion"],
                "audio_sha256": row["audio_sha256"],
                "log_mel_artifact_path": handcrafted[(sample_id, "log_mel_80")][
                    "artifact_path"
                ],
                "pitch_artifact_path": handcrafted[(sample_id, "pitch_contour")][
                    "artifact_path"
                ],
                "wav2vec_artifact_path": wav2vec[(sample_id, "wav2vec2_embedding")][
                    "artifact_path"
                ],
            }
        )
        if (index + 1) % 512 == 0:
            print(f"feature table loaded {index + 1}/{sample_count}", flush=True)

    columns: dict[str, Any] = {key: [row[key] for row in metadata] for key in metadata[0]}
    for index in range(40):
        columns[f"mfcc_mean_{index + 1:02d}"] = mfcc[:, index]
        columns[f"mfcc_std_{index + 1:02d}"] = mfcc[:, 40 + index]
    egemaps_columns = []
    for index, name in enumerate(egemaps_names):
        column = f"egemaps_{index:03d}_{safe_name(name)}"
        egemaps_columns.append(column)
        columns[column] = egemaps[:, index]
    pitch_names = (
        "pitch_voiced_fraction",
        "pitch_f0_mean_hz",
        "pitch_f0_std_hz",
        "pitch_f0_median_hz",
        "pitch_f0_q10_hz",
        "pitch_f0_q90_hz",
    )
    for index, name in enumerate(pitch_names):
        columns[name] = pitch[:, index]
    for index in range(768):
        columns[f"wav2vec2_{index:03d}"] = wav2vec_matrix[:, index]
    frame = pd.DataFrame(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    summary = {
        "status": "PASS",
        "rows": len(frame),
        "columns": len(frame.columns),
        "metadata_columns": len(metadata[0]),
        "mfcc_columns": 80,
        "egemaps_columns": 88,
        "pitch_summary_columns": 6,
        "wav2vec2_columns": 768,
        "variable_length_paths": ["log_mel_artifact_path", "pitch_artifact_path"],
        "table_path": output_path.relative_to(repo_root).as_posix(),
        "table_sha256": file_sha256(output_path),
        "egemaps_columns_ordered": egemaps_columns,
    }
    return frame, wav2vec_matrix, summary


def randomized_pca_2d(matrix: np.ndarray, seed: int = 391) -> tuple[np.ndarray, list[float]]:
    centered = matrix.astype(np.float64) - matrix.mean(axis=0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], 12))
    projected = centered @ omega
    for _ in range(2):
        projected = centered @ (centered.T @ projected)
    basis, _ = np.linalg.qr(projected, mode="reduced")
    compact = basis.T @ centered
    _, singular_values, right = np.linalg.svd(compact, full_matrices=False)
    components = right[:2].T
    coordinates = (centered @ components).astype(np.float32)
    total_variance = float(np.square(centered).sum())
    explained = [float(value * value / total_variance) for value in singular_values[:2]]
    return coordinates, explained


def projection_insight(frame: pd.DataFrame) -> dict[str, float]:
    centroids = {
        emotion: frame.loc[frame["emotion"] == emotion, ["pca_1", "pca_2"]]
        .mean()
        .to_numpy()
        for emotion in EMOTIONS
    }
    between = np.mean(
        [np.linalg.norm(centroids[left] - centroids[right]) for left, right in combinations(EMOTIONS, 2)]
    )
    within = np.mean(
        [
            np.linalg.norm(
                frame.loc[frame["emotion"] == emotion, ["pca_1", "pca_2"]].to_numpy()
                - centroids[emotion],
                axis=1,
            ).mean()
            for emotion in EMOTIONS
        ]
    )
    return {
        "mean_pairwise_emotion_centroid_distance": float(between),
        "mean_within_emotion_distance": float(within),
        "centroid_to_within_ratio": float(between / within),
    }


def render_projection(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for axis, accent in zip(axes, ACCENTS):
        subset = frame[frame["accent"] == accent]
        for emotion in EMOTIONS:
            values = subset[subset["emotion"] == emotion]
            axis.scatter(
                values["pca_1"], values["pca_2"], s=9, alpha=0.45,
                color=COLORS[emotion], label=emotion.capitalize(), rasterized=True,
            )
        axis.set_title(f"{accent.capitalize()} (n={len(subset):,})")
        axis.set_xlabel("Wav2Vec2 PCA 1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Wav2Vec2 PCA 2")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("B4 Wav2Vec2 embedding projection by accent and emotion")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(1.0 - np.dot(left, right) / denominator) if denominator else 0.0


def build_shift_heatmap(
    frame: pd.DataFrame, wav2vec: np.ndarray
) -> pd.DataFrame:
    result = pd.DataFrame(index=ACCENTS, columns=EMOTIONS, dtype=float)
    emotions = frame["emotion"].to_numpy()
    accents = frame["accent"].to_numpy()
    for emotion in EMOTIONS:
        global_centroid = wav2vec[emotions == emotion].mean(axis=0)
        for accent in ACCENTS:
            centroid = wav2vec[(emotions == emotion) & (accents == accent)].mean(axis=0)
            result.loc[accent, emotion] = cosine_distance(centroid, global_centroid)
    return result


def render_heatmap(values: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    matrix = values.to_numpy(dtype=float)
    image = axis.imshow(matrix, cmap="magma", aspect="auto")
    axis.set_xticks(range(len(values.columns)), values.columns)
    axis.set_yticks(
        range(len(values.index)), [value.capitalize() for value in values.index]
    )
    midpoint = float((matrix.min() + matrix.max()) / 2.0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.4f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < midpoint else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Cosine distance")
    axis.set_title("Accent-conditioned shift from each global emotion centroid")
    axis.set_xlabel("Emotion")
    axis.set_ylabel("Accent")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def find_column(frame: pd.DataFrame, token: str, fallback: str) -> str:
    matches = [column for column in frame.columns if token in column]
    return matches[0] if matches else fallback


def build_radar_profiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    egemaps_columns = [column for column in frame.columns if column.startswith("egemaps_")]
    if len(egemaps_columns) != 88:
        raise ValueError("Radar input must contain all 88 eGeMAPS columns")
    selected = [
        "pitch_voiced_fraction",
        "pitch_f0_mean_hz",
        "pitch_f0_std_hz",
        "mfcc_mean_01",
        find_column(frame, "loudness_sma3_amean", egemaps_columns[0]),
        find_column(frame, "hnrdbacf_sma3nz_amean", egemaps_columns[1]),
    ]
    values = frame[selected].astype(float)
    standard_deviation = values.std(ddof=0).replace(0, 1.0)
    standardized = (values - values.mean()) / standard_deviation
    standardized["accent"] = frame["accent"].to_numpy()
    profiles = standardized.groupby("accent", sort=False)[selected].mean().loc[list(ACCENTS)]
    return profiles, selected


def render_radar(profiles: pd.DataFrame, output: Path) -> None:
    labels = [
        "Voiced\nfraction",
        "F0 mean",
        "F0 std",
        "MFCC-1 mean",
        "Loudness",
        "HNR",
    ]
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, axis = plt.subplots(figsize=(8.5, 7.5), subplot_kw={"polar": True})
    colors = {"central": "#2ca02c", "north": "#1f77b4", "south": "#d62728"}
    for accent in ACCENTS:
        values = profiles.loc[accent].tolist()
        values += values[:1]
        axis.plot(angles, values, linewidth=2, label=accent.capitalize(), color=colors[accent])
        axis.fill(angles, values, alpha=0.08, color=colors[accent])
    axis.set_xticks(angles[:-1], labels, fontsize=10)
    axis.tick_params(axis="x", pad=12)
    axis.set_title("Standardized acoustic profile by accent", pad=24)
    axis.legend(loc="upper right", bbox_to_anchor=(1.16, 1.10))
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_dashboard(frame: pd.DataFrame, insight: dict[str, Any], output: Path) -> None:
    dashboard_rows = frame[
        ["sample_id", "speaker_id", "accent", "emotion", "pca_1", "pca_2"]
    ].to_dict(orient="records")
    payload = json.dumps(dashboard_rows, ensure_ascii=False, separators=(",", ":"))
    insight_payload = json.dumps(insight, ensure_ascii=False, indent=2)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B4 ViSEC Feature Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f7f8fb;color:#182230}} .controls{{display:flex;gap:16px;margin:16px 0}} select{{padding:8px}} #chart{{height:72vh;background:white}} pre{{background:white;padding:16px;overflow:auto}}</style>
</head><body><h1>B4 ViSEC Wav2Vec2 Feature Dashboard</h1>
<p>Exploratory feature projection only; these coordinates are not reused for model evaluation.</p>
<div class="controls"><label>Accent <select id="accent"><option value="all">All</option><option>central</option><option>north</option><option>south</option></select></label>
<label>Emotion <select id="emotion"><option value="all">All</option><option>angry</option><option>happy</option><option>neutral</option><option>sad</option></select></label></div>
<div id="chart"></div><h2>Reproducible quantitative evidence</h2><pre>{insight_payload}</pre>
<script>const rows={payload}; const colors={json.dumps(COLORS)};
function draw(){{const a=document.getElementById('accent').value,e=document.getElementById('emotion').value;
const filtered=rows.filter(r=>(a==='all'||r.accent===a)&&(e==='all'||r.emotion===e));
const traces=['angry','happy','neutral','sad'].map(label=>{{const x=filtered.filter(r=>r.emotion===label);return{{type:'scattergl',mode:'markers',name:label,x:x.map(r=>r.pca_1),y:x.map(r=>r.pca_2),text:x.map(r=>`sample=${{r.sample_id}}<br>speaker=${{r.speaker_id}}<br>accent=${{r.accent}}<br>emotion=${{r.emotion}}`),hoverinfo:'text',marker:{{size:6,opacity:.55,color:colors[label]}}}};}});
Plotly.react('chart',traces,{{title:`Filtered samples: ${{filtered.length.toLocaleString()}}`,xaxis:{{title:'PCA 1'}},yaxis:{{title:'PCA 2'}},dragmode:'zoom'}},{{responsive:true}});}}
document.getElementById('accent').addEventListener('change',draw);document.getElementById('emotion').addEventListener('change',draw);draw();</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def build_deliverables(repo_root: Path) -> dict[str, Any]:
    tables = repo_root / "reports/tables/b4"
    figures = repo_root / "reports/figures/b4"
    table_path = repo_root / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet"
    frame, wav2vec, table_summary = build_feature_table(
        repo_root,
        repo_root / "data/manifests/clean_manifest.csv",
        tables / "b4_handcrafted_full_feature_index.csv",
        tables / "b4_wav2vec2_full_feature_index.csv",
        tables / "b4_handcrafted_full_extraction_report.json",
        tables / "b4_wav2vec2_full_extraction_report.json",
        tables / "b4_egemaps_feature_names.json",
        table_path,
    )
    coordinates, explained = randomized_pca_2d(wav2vec)
    projection = frame[["sample_id", "speaker_id", "accent", "emotion"]].copy()
    projection["pca_1"] = coordinates[:, 0]
    projection["pca_2"] = coordinates[:, 1]
    projection_path = tables / "b4_wav2vec2_pca_projection.csv"
    projection.to_csv(projection_path, index=False)
    projection_result = projection_insight(projection)
    projection_result["approximate_explained_variance_ratio"] = explained
    render_projection(projection, figures / "b4_wav2vec2_pca_by_accent_emotion.png")

    heatmap = build_shift_heatmap(frame, wav2vec)
    heatmap_path = tables / "b4_accent_emotion_embedding_shift.csv"
    heatmap.to_csv(heatmap_path, index_label="accent")
    render_heatmap(heatmap, figures / "b4_accent_emotion_embedding_shift_heatmap.png")
    maximum_cell = np.unravel_index(np.argmax(heatmap.to_numpy()), heatmap.shape)
    heatmap_result = {
        "maximum_shift_accent": heatmap.index[maximum_cell[0]],
        "maximum_shift_emotion": heatmap.columns[maximum_cell[1]],
        "maximum_cosine_distance": float(heatmap.iloc[maximum_cell]),
    }

    profiles, profile_columns = build_radar_profiles(frame)
    radar_path = tables / "b4_accent_acoustic_radar_profiles.csv"
    profiles.to_csv(radar_path, index_label="accent")
    render_radar(profiles, figures / "b4_accent_acoustic_radar.png")
    spread = profiles.max(axis=0) - profiles.min(axis=0)
    radar_result = {
        "largest_accent_profile_spread_feature": str(spread.idxmax()),
        "largest_standardized_mean_spread": float(spread.max()),
        "profile_columns": profile_columns,
    }
    insights = {
        "scope": "Exploratory full-feature analysis; projections and standardization are not reused by models.",
        "rq1_emotion_structure_in_wav2vec2_projection": projection_result,
        "rq2_accent_shift_within_emotion": heatmap_result,
        "rq2_acoustic_profile_difference_by_accent": radar_result,
    }
    insights_path = tables / "b4_advanced_visualization_insights.json"
    insights_path.write_text(
        json.dumps(insights, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dashboard_path = repo_root / "reports/dashboard/b4_feature_dashboard.html"
    dashboard_frame = projection.copy()
    write_dashboard(dashboard_frame, insights, dashboard_path)
    report = {
        "status": "PASS",
        "feature_table": table_summary,
        "figures": [
            "reports/figures/b4/b4_wav2vec2_pca_by_accent_emotion.png",
            "reports/figures/b4/b4_accent_emotion_embedding_shift_heatmap.png",
            "reports/figures/b4/b4_accent_acoustic_radar.png",
        ],
        "tables": [
            projection_path.relative_to(repo_root).as_posix(),
            heatmap_path.relative_to(repo_root).as_posix(),
            radar_path.relative_to(repo_root).as_posix(),
            insights_path.relative_to(repo_root).as_posix(),
        ],
        "dashboard": dashboard_path.relative_to(repo_root).as_posix(),
    }
    report_path = tables / "b4_deliverables_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(build_deliverables(repo_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
