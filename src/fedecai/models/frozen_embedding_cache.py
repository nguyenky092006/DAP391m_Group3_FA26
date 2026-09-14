"""Validated, split-aware frozen-embedding cache used by B6 baselines."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fedecai.models.b6_common import sha256_file, write_csv_atomic, write_json_atomic


CACHE_SCHEMA_VERSION = "fedecai.frozen_embedding_cache.v1"


def _metadata_matches(actual: dict[str, Any], expected: dict[str, Any], prefix: str = "") -> None:
    for key, expected_value in expected.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in actual:
            raise ValueError(f"Cache metadata is missing {name}")
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                raise ValueError(f"Cache metadata {name} must be an object")
            _metadata_matches(actual_value, expected_value, name)
        elif actual_value != expected_value:
            raise ValueError(
                f"Cache metadata mismatch for {name}: {actual_value!r} != {expected_value!r}"
            )


def write_embedding_cache(
    cache_dir: Path,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish a complete cache after structural validation."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(frame) or matrix.shape[1] < 1:
        raise ValueError("Embedding cache must be a non-empty rows x dimensions matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Refusing to cache NaN/Inf embeddings")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Refusing to cache duplicate sample IDs")

    embedding_path = cache_dir / "embeddings.npy"
    temporary = cache_dir / "embeddings.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, matrix, allow_pickle=False)
    os.replace(temporary, embedding_path)

    index_rows = [
        {
            "embedding_row": offset,
            "sample_id": str(row.sample_id),
            "speaker_id": str(row.speaker_id),
            "split": str(row.split),
        }
        for offset, row in enumerate(frame.itertuples(index=False))
    ]
    index_path = cache_dir / "index.csv"
    write_csv_atomic(index_rows, index_path)
    cache_metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        **metadata,
        "sample_count": int(len(frame)),
        "embedding_dimension": int(matrix.shape[1]),
        "dtype": "float32",
        "embedding_file": "embeddings.npy",
        "index_file": "index.csv",
        "embedding_sha256": sha256_file(embedding_path),
        "index_sha256": sha256_file(index_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(cache_metadata, cache_dir / "metadata.json")
    return cache_metadata


def validate_embedding_cache(
    cache_dir: Path,
    authoritative_frame: pd.DataFrame,
    expected_metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fail loudly unless cache metadata, membership, split and values match."""
    metadata_path = cache_dir / "metadata.json"
    index_path = cache_dir / "index.csv"
    embedding_path = cache_dir / "embeddings.npy"
    for path in (metadata_path, index_path, embedding_path):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete frozen-embedding cache: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _metadata_matches(metadata, {"schema_version": CACHE_SCHEMA_VERSION, **expected_metadata})
    if metadata.get("embedding_sha256") != sha256_file(embedding_path):
        raise ValueError("Cached embedding checksum mismatch")
    if metadata.get("index_sha256") != sha256_file(index_path):
        raise ValueError("Cached index checksum mismatch")

    index = pd.read_csv(
        index_path, dtype={"sample_id": "string", "speaker_id": "string"}
    )
    expected = authoritative_frame[["sample_id", "speaker_id", "split"]].copy()
    expected["sample_id"] = expected["sample_id"].astype("string")
    expected["speaker_id"] = expected["speaker_id"].astype("string")
    if len(index) != len(expected) or int(metadata.get("sample_count", -1)) != len(expected):
        raise ValueError(f"Cache sample count mismatch: {len(index)} != {len(expected)}")
    if index["sample_id"].duplicated().any():
        raise ValueError("Cache contains duplicate sample IDs")
    missing = set(expected["sample_id"]) - set(index["sample_id"])
    extra = set(index["sample_id"]) - set(expected["sample_id"])
    if missing or extra:
        raise ValueError(f"Cache membership mismatch: missing={len(missing)}, extra={len(extra)}")
    joined = index.merge(
        expected, on="sample_id", suffixes=("_cache", "_expected"),
        how="inner", validate="one_to_one",
    )
    if not (
        (joined["speaker_id_cache"] == joined["speaker_id_expected"]).all()
        and (joined["split_cache"] == joined["split_expected"]).all()
    ):
        raise ValueError("Cache speaker or split membership does not match B5")
    rows = index["embedding_row"].to_numpy(dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or set(rows) != set(range(len(rows))):
        raise ValueError("Cache embedding-row mapping is invalid")

    matrix = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    dimension = int(metadata.get("embedding_dimension", -1))
    if matrix.shape != (len(index), dimension) or matrix.dtype != np.float32:
        raise ValueError(
            f"Cache tensor mismatch: shape={matrix.shape}, dtype={matrix.dtype}"
        )
    for start in range(0, len(matrix), 1024):
        if not np.isfinite(matrix[start : start + 1024]).all():
            raise ValueError("Cache contains NaN/Inf embeddings")

    row_by_id = dict(zip(index["sample_id"].astype(str), rows))
    ordered_rows = [row_by_id[str(value)] for value in authoritative_frame["sample_id"]]
    return np.asarray(matrix[ordered_rows], dtype=np.float32), metadata
