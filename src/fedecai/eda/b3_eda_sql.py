"""Build the reproducible B3 EDA database, SQL evidence tables, and figures.

The script treats the reviewed manifest as the source of truth. It never edits
raw audio and does not fit any preprocessing transformation. EDA is performed
on all B2-eligible rows; split fields are used only for leakage/coverage checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


QUERY_MARKER = "-- name:"


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _as_float(value: Any) -> float | None:
    text = str(value).strip()
    return None if not text else float(text)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_database(rows: list[dict[str, str]], output_path: Path) -> None:
    """Create a compact SQLite database from the reviewed CSV manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    connection = sqlite3.connect(output_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            CREATE TABLE utterances (
                sample_id TEXT PRIMARY KEY,
                source_row_index INTEGER NOT NULL,
                speaker_id TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                accent TEXT NOT NULL,
                emotion TEXT NOT NULL,
                gender TEXT NOT NULL,
                is_eligible INTEGER NOT NULL CHECK (is_eligible IN (0, 1)),
                split TEXT NOT NULL,
                possible_clipping INTEGER NOT NULL CHECK (possible_clipping IN (0, 1)),
                high_silence INTEGER NOT NULL CHECK (high_silence IN (0, 1)),
                silent_frame_ratio REAL NOT NULL,
                peak_amplitude_ratio REAL NOT NULL,
                cleaning_action TEXT NOT NULL,
                exclusion_reason TEXT NOT NULL,
                audio_sha256 TEXT NOT NULL
            );
            """
        )
        records = []
        for row in rows:
            accent = row.get("accent_clean") or row.get("accent") or ""
            gender = row.get("gender_clean") or row.get("gender") or ""
            records.append(
                (
                    row["sample_id"],
                    int(row["source_row_index"]),
                    row["speaker_id"],
                    float(row["duration_sec_measured"]),
                    accent,
                    row["emotion"],
                    gender,
                    int(_as_bool(row["is_eligible_for_split"])),
                    row.get("split") or ("eligible" if _as_bool(row["is_eligible_for_split"]) else "excluded"),
                    int(_as_bool(row["possible_clipping"])),
                    int(_as_bool(row["high_silence_ratio"])),
                    float(row["silent_frame_ratio"]),
                    float(row["peak_amplitude_ratio"]),
                    row["cleaning_action"],
                    row.get("exclusion_reason", ""),
                    row["audio_sha256"],
                )
            )

        connection.executemany(
            """
            INSERT INTO utterances VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.executescript(
            """
            CREATE INDEX idx_utterances_eligible ON utterances(is_eligible);
            CREATE INDEX idx_utterances_speaker ON utterances(speaker_id);
            CREATE INDEX idx_utterances_accent_emotion ON utterances(accent, emotion);
            CREATE INDEX idx_utterances_split ON utterances(split);

            CREATE VIEW eligible_utterances AS
            SELECT * FROM utterances WHERE is_eligible = 1;
            """
        )
        connection.commit()
    finally:
        connection.close()


def load_named_queries(path: Path) -> dict[str, str]:
    """Read SQL blocks introduced by '-- name: query_id' markers."""
    queries: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith(QUERY_MARKER):
            if current_name is not None:
                queries[current_name] = "\n".join(current_lines).strip().rstrip(";")
            current_name = line.split(":", 1)[1].strip()
            if not current_name:
                raise ValueError("SQL query marker is missing a name")
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        queries[current_name] = "\n".join(current_lines).strip().rstrip(";")
    if not queries or any(not query for query in queries.values()):
        raise ValueError(f"No complete named SQL queries found in {path}")
    return queries


def execute_queries(database_path: Path, sql_path: Path, output_dir: Path) -> dict[str, int]:
    """Run each named read-only query and export its result as UTF-8 CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = load_named_queries(sql_path)
    row_counts: dict[str, int] = {}
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        for name, query in queries.items():
            if not query.lstrip().upper().startswith(("SELECT", "WITH")):
                raise ValueError(f"Query {name!r} is not read-only")
            cursor = connection.execute(query)
            records = cursor.fetchall()
            destination = output_dir / f"{name}.csv"
            with destination.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([column[0] for column in cursor.description])
                writer.writerows([tuple(record) for record in records])
            row_counts[name] = len(records)
    finally:
        connection.close()
    return row_counts


def build_summary(database_path: Path, query_row_counts: dict[str, int]) -> dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        counts = connection.execute(
            """
            SELECT COUNT(*) AS source_rows,
                   SUM(is_eligible) AS eligible_rows,
                   SUM(1 - is_eligible) AS excluded_rows,
                   COUNT(DISTINCT CASE WHEN is_eligible = 1 THEN speaker_id END) AS speakers
            FROM utterances
            """
        ).fetchone()
        durations = [
            row[0]
            for row in connection.execute(
                "SELECT duration_sec FROM eligible_utterances ORDER BY duration_sec"
            )
        ]
        accent_counts = dict(
            connection.execute(
                "SELECT accent, COUNT(*) FROM eligible_utterances GROUP BY accent ORDER BY accent"
            ).fetchall()
        )
        emotion_counts = dict(
            connection.execute(
                "SELECT emotion, COUNT(*) FROM eligible_utterances GROUP BY emotion ORDER BY emotion"
            ).fetchall()
        )
        quality = connection.execute(
            """
            SELECT SUM(possible_clipping) AS possible_clipping,
                   SUM(high_silence) AS high_silence,
                   AVG(silent_frame_ratio) AS mean_silent_frame_ratio
            FROM eligible_utterances
            """
        ).fetchone()
        largest_speaker = connection.execute(
            """
            SELECT speaker_id, accent, COUNT(*) AS samples
            FROM eligible_utterances
            GROUP BY speaker_id, accent
            ORDER BY samples DESC, CAST(speaker_id AS INTEGER)
            LIMIT 1
            """
        ).fetchone()
        split_speakers = dict(
            connection.execute(
                """
                SELECT split, COUNT(DISTINCT speaker_id)
                FROM eligible_utterances
                GROUP BY split ORDER BY split
                """
            ).fetchall()
        )
        missing_split_cells = [
            dict(row)
            for row in connection.execute(
                """
                WITH splits(split) AS (
                    VALUES ('train'), ('validation'), ('test')
                ),
                accents(accent) AS (
                    VALUES ('central'), ('north'), ('south')
                ),
                emotions(emotion) AS (
                    VALUES ('angry'), ('happy'), ('neutral'), ('sad')
                ),
                observed AS (
                    SELECT split, accent, emotion, COUNT(*) AS utterances
                    FROM eligible_utterances
                    GROUP BY split, accent, emotion
                )
                SELECT s.split, a.accent, e.emotion
                FROM splits s
                CROSS JOIN accents a
                CROSS JOIN emotions e
                LEFT JOIN observed o
                  ON o.split = s.split AND o.accent = a.accent AND o.emotion = e.emotion
                WHERE COALESCE(o.utterances, 0) = 0
                ORDER BY s.split, a.accent, e.emotion
                """
            )
        ]
    finally:
        connection.close()

    eligible_rows = int(counts["eligible_rows"])
    return {
        "b3_version": "b3.1-v1",
        "scope": "All B2-eligible rows; split labels are used only for coverage and leakage checks.",
        "source_rows": int(counts["source_rows"]),
        "eligible_rows": eligible_rows,
        "excluded_rows": int(counts["excluded_rows"]),
        "eligible_speakers": int(counts["speakers"]),
        "duration_seconds": {
            "minimum": min(durations),
            "median": statistics.median(durations),
            "mean": statistics.fmean(durations),
            "maximum": max(durations),
        },
        "accent_counts": accent_counts,
        "emotion_counts": emotion_counts,
        "quality_flags": {
            "possible_clipping": int(quality["possible_clipping"]),
            "high_silence": int(quality["high_silence"]),
            "mean_silent_frame_ratio": float(quality["mean_silent_frame_ratio"]),
        },
        "largest_speaker": {
            "speaker_id": largest_speaker["speaker_id"],
            "accent": largest_speaker["accent"],
            "samples": int(largest_speaker["samples"]),
            "eligible_share": int(largest_speaker["samples"]) / eligible_rows,
        },
        "split_speaker_counts": split_speakers,
        "split_cross_stratum_complete": not missing_split_cells,
        "missing_split_cells": missing_split_cells,
        "sql_result_rows": query_row_counts,
    }


def render_figures(database_path: Path, output_dir: Path) -> list[str]:
    """Render the B3 univariate, bivariate, and multivariate EDA figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        frame = pd.read_sql_query("SELECT * FROM eligible_utterances", connection)
    finally:
        connection.close()

    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"angry": "#C44E52", "happy": "#E5AE38", "neutral": "#4C72B0", "sad": "#8172B3"}
    outputs: list[str] = []

    q99_duration = frame["duration_sec"].quantile(0.99)
    outlier_count = int((frame["duration_sec"] > q99_duration).sum())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.histplot(
        frame.loc[frame["duration_sec"] <= q99_duration],
        x="duration_sec",
        bins=40,
        color="#4C72B0",
        edgecolor="white",
        ax=ax,
    )
    median_duration = frame["duration_sec"].median()
    ax.axvline(median_duration, color="#C44E52", linestyle="--", linewidth=2,
               label=f"Median = {median_duration:.2f} s")
    ax.text(
        0.98,
        0.80,
        f"99th percentile = {q99_duration:.2f} s\n{outlier_count} utterances above panel range",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set(
        title="ViSEC eligible utterance duration (up to 99th percentile)",
        xlabel="Duration (seconds)",
        ylabel="Utterances",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "b3_duration_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path.name)

    proportions = pd.crosstab(frame["accent"], frame["emotion"], normalize="index") * 100
    proportions = proportions.reindex(index=["central", "north", "south"], columns=list(palette))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    proportions.plot(kind="bar", stacked=True, color=[palette[column] for column in proportions], ax=ax)
    ax.set(title="Emotion composition within each accent", xlabel="Accent", ylabel="Share within accent (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    accent_sizes = frame.groupby("accent").size().reindex(proportions.index)
    ax.set_xticklabels([f"{accent}\nn={accent_sizes[accent]:,}" for accent in proportions.index])
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    path = output_dir / "b3_emotion_composition_by_accent.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path.name)

    duration_matrix = frame.pivot_table(
        index="accent", columns="emotion", values="duration_sec", aggfunc="median"
    ).reindex(index=["central", "north", "south"], columns=list(palette))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sns.heatmap(duration_matrix, annot=True, fmt=".2f", cmap="Blues", linewidths=0.7,
                linecolor="white", cbar_kws={"label": "Median seconds"}, ax=ax)
    ax.set(title="Median duration by accent and emotion", xlabel="Emotion", ylabel="Accent")
    fig.tight_layout()
    path = output_dir / "b3_duration_accent_emotion_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path.name)
    return outputs


def validate_summary(summary: dict[str, Any]) -> None:
    if summary["source_rows"] != summary["eligible_rows"] + summary["excluded_rows"]:
        raise ValueError("Source row accounting is inconsistent")
    if summary["eligible_rows"] <= 0 or summary["eligible_speakers"] <= 0:
        raise ValueError("No eligible data available for B3")
    if sum(summary["accent_counts"].values()) != summary["eligible_rows"]:
        raise ValueError("Accent counts do not cover the eligible data")
    if sum(summary["emotion_counts"].values()) != summary["eligible_rows"]:
        raise ValueError("Emotion counts do not cover the eligible data")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repo_root / "data/manifests/split_manifest.csv")
    parser.add_argument("--database", type=Path, default=repo_root / "data/processed/visec_metadata.sqlite")
    parser.add_argument("--sql", type=Path, default=repo_root / "sql/B3_rq_queries.sql")
    parser.add_argument("--tables", type=Path, default=repo_root / "reports/tables/b3")
    parser.add_argument("--figures", type=Path, default=repo_root / "reports/figures/b3")
    parser.add_argument("--summary", type=Path, default=repo_root / "reports/tables/b3/b3_summary.json")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.input.resolve())
    build_database(rows, args.database.resolve())
    query_counts = execute_queries(args.database.resolve(), args.sql.resolve(), args.tables.resolve())
    summary = build_summary(args.database.resolve(), query_counts)
    if not args.skip_figures:
        summary["figures"] = render_figures(args.database.resolve(), args.figures.resolve())
    validate_summary(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
