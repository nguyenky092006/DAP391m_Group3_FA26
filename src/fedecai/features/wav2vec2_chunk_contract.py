"""Lightweight deterministic selection contract shared by Wav2Vec2 chunks."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def read_manifest_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_manifest_row(row: dict[str, str]) -> dict[str, str]:
    required = (
        "sample_id",
        "source_row_index",
        "speaker_id",
        "accent_clean",
        "emotion",
        "audio_sha256",
    )
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(f"Eligible manifest row is missing fields: {missing}")
    return {
        "sample_id": row["sample_id"],
        "source_row_index": row["source_row_index"],
        "speaker_id": row["speaker_id"],
        "accent": row["accent_clean"],
        "emotion": row["emotion"],
        "audio_sha256": row["audio_sha256"],
    }


def eligible_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    eligible = [row for row in rows if _as_bool(row.get("is_eligible_for_split"))]
    eligible.sort(key=lambda row: int(row["source_row_index"]))
    normalized = [normalize_manifest_row(row) for row in eligible]
    sample_ids = [row["sample_id"] for row in normalized]
    source_indexes = [int(row["source_row_index"]) for row in normalized]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Eligible manifest contains duplicate sample IDs")
    if len(source_indexes) != len(set(source_indexes)):
        raise ValueError("Eligible manifest contains duplicate source row indexes")
    return normalized


def select_eligible_chunk(
    rows: list[dict[str, str]], offset: int, chunk_size: int
) -> tuple[list[dict[str, str]], int]:
    if offset < 0:
        raise ValueError("Chunk offset must be non-negative")
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1")
    eligible = eligible_manifest_rows(rows)
    if offset >= len(eligible):
        raise ValueError(
            f"Chunk offset {offset} is outside {len(eligible)} eligible rows"
        )
    return eligible[offset : offset + chunk_size], len(eligible)


def selection_sha256(rows: list[dict[str, str]]) -> str:
    canonical = "\n".join(
        "|".join(
            (
                row["source_row_index"],
                row["sample_id"],
                row["speaker_id"],
                row["accent"],
                row["emotion"],
                row["audio_sha256"],
            )
        )
        for row in rows
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
