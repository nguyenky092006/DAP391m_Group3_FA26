"""Resume-safe full extraction for MFCC, eGeMAPS, log-Mel, and pitch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .extract_feature_pilot import (
        FEATURE_VERSION,
        PILOT_FAMILIES,
        build_smile_extractor,
        decode_wav,
        extract_egemaps,
        extract_log_mel,
        extract_mfcc,
        extract_pitch,
        load_embedded_audio,
        validate_feature_value,
    )
    from .wav2vec2_chunk_contract import (
        eligible_manifest_rows,
        file_sha256,
        read_manifest_csv,
    )
except ImportError:
    from extract_feature_pilot import (  # type: ignore
        FEATURE_VERSION,
        PILOT_FAMILIES,
        build_smile_extractor,
        decode_wav,
        extract_egemaps,
        extract_log_mel,
        extract_mfcc,
        extract_pitch,
        load_embedded_audio,
        validate_feature_value,
    )
    from wav2vec2_chunk_contract import (  # type: ignore
        eligible_manifest_rows,
        file_sha256,
        read_manifest_csv,
    )


INDEX_FIELDS = (
    "sample_id",
    "source_row_index",
    "speaker_id",
    "emotion",
    "accent",
    "audio_sha256",
    "feature_family",
    "feature_version",
    "artifact_path",
    "artifact_sha256",
    "shape",
    "dtype",
    "finite_fraction",
    "status",
    "error",
)


def connect_progress(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_records (
            sample_id TEXT NOT NULL,
            source_row_index INTEGER NOT NULL,
            speaker_id TEXT NOT NULL,
            emotion TEXT NOT NULL,
            accent TEXT NOT NULL,
            audio_sha256 TEXT NOT NULL,
            feature_family TEXT NOT NULL,
            feature_version TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            shape TEXT NOT NULL,
            dtype TEXT NOT NULL,
            finite_fraction TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT NOT NULL,
            PRIMARY KEY (sample_id, feature_family)
        )
        """
    )
    connection.commit()
    return connection


def expected_artifact(run_root: Path, family: str, sample_id: str) -> Path:
    suffix = ".npz" if family == "pitch_contour" else ".npy"
    return run_root / family / f"{sample_id}{suffix}"


def load_artifact(family: str, path: Path) -> Any:
    if family == "pitch_contour":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    return np.load(path, allow_pickle=False)


def artifact_shape(family: str, value: Any) -> str:
    if family == "pitch_contour":
        return f"frames={value['f0_hz'].shape[0]};summary=6"
    return "x".join(str(size) for size in value.shape)


def artifact_finite_fraction(family: str, value: Any) -> float:
    if family == "pitch_contour":
        return float(np.isfinite(value["f0_hz"]).mean())
    return float(np.isfinite(value).mean())


def atomic_save(path: Path, family: str, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if family == "pitch_contour":
        temporary = path.with_name(f"{path.stem}.tmp.npz")
        np.savez_compressed(temporary, **value)
    else:
        temporary = path.with_name(f"{path.stem}.tmp.npy")
        np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def relative_path(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def reusable_record(
    record: sqlite3.Row | None,
    manifest_row: dict[str, str],
    family: str,
    artifact_path: Path,
    repo_root: Path,
) -> bool:
    if record is None or record["status"] != "ok" or record["error"]:
        return False
    expected = {
        "source_row_index": int(manifest_row["source_row_index"]),
        "speaker_id": manifest_row["speaker_id"],
        "emotion": manifest_row["emotion"],
        "accent": manifest_row["accent"],
        "audio_sha256": manifest_row["audio_sha256"],
        "feature_version": FEATURE_VERSION,
        "artifact_path": relative_path(artifact_path, repo_root),
    }
    if any(record[field] != value for field, value in expected.items()):
        return False
    if not artifact_path.is_file():
        return False
    if file_sha256(artifact_path) != record["artifact_sha256"]:
        return False
    try:
        value = load_artifact(family, artifact_path)
        validate_feature_value(family, value)
    except Exception:
        return False
    return True


def extract_family(
    family: str, signal: np.ndarray, sample_rate: int, smile: Any
) -> tuple[Any, list[str] | None]:
    if family == "mfcc40_summary":
        return extract_mfcc(signal, sample_rate), None
    if family == "egemaps_v02":
        value, names = extract_egemaps(signal, sample_rate, smile)
        return value, names
    if family == "log_mel_80":
        return extract_log_mel(signal, sample_rate), None
    if family == "pitch_contour":
        return extract_pitch(signal, sample_rate), None
    raise ValueError(f"Unsupported family: {family}")


def upsert_record(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
    placeholders = ",".join("?" for _ in INDEX_FIELDS)
    updates = ",".join(
        f"{field}=excluded.{field}" for field in INDEX_FIELDS if field != "sample_id"
    )
    connection.execute(
        f"""
        INSERT INTO feature_records ({','.join(INDEX_FIELDS)})
        VALUES ({placeholders})
        ON CONFLICT(sample_id, feature_family) DO UPDATE SET {updates}
        """,
        [record[field] for field in INDEX_FIELDS],
    )
    connection.commit()


def write_index(connection: sqlite3.Connection, path: Path) -> list[dict[str, Any]]:
    family_order = {family: index for index, family in enumerate(PILOT_FAMILIES)}
    rows = [dict(row) for row in connection.execute("SELECT * FROM feature_records")]
    rows.sort(key=lambda row: (int(row["source_row_index"]), family_order[row["feature_family"]]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return rows


def run_full_handcrafted(
    repo_root: Path,
    manifest_path: Path,
    parquet_path: Path,
    run_root: Path,
    index_path: Path,
    report_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    eligible = eligible_manifest_rows(read_manifest_csv(manifest_path))
    connection = connect_progress(run_root / "progress.sqlite3")
    existing = {
        (row["sample_id"], row["feature_family"]): row
        for row in connection.execute("SELECT * FROM feature_records")
    }
    pending_by_sample: dict[str, list[str]] = {}
    resumed = invalidated = 0
    for row in eligible:
        for family in PILOT_FAMILIES:
            artifact = expected_artifact(run_root, family, row["sample_id"])
            record = existing.get((row["sample_id"], family))
            if reusable_record(record, row, family, artifact, repo_root):
                resumed += 1
            else:
                if record is not None:
                    invalidated += 1
                pending_by_sample.setdefault(row["sample_id"], []).append(family)

    smile = (
        build_smile_extractor()
        if any("egemaps_v02" in families for families in pending_by_sample.values())
        else None
    )
    egemaps_names_path = run_root / "egemaps_feature_names.json"
    tracked_egemaps_names_path = report_path.parent / "b4_egemaps_feature_names.json"
    expected_egemaps_names = (
        json.loads(egemaps_names_path.read_text(encoding="utf-8"))
        if egemaps_names_path.is_file()
        else (
            json.loads(tracked_egemaps_names_path.read_text(encoding="utf-8"))
            if tracked_egemaps_names_path.is_file()
            else None
        )
    )
    processed = failures = 0
    pending_rows = [row for row in eligible if row["sample_id"] in pending_by_sample]
    for batch_start in range(0, len(pending_rows), batch_size):
        batch = pending_rows[batch_start : batch_start + batch_size]
        audio = load_embedded_audio(
            parquet_path, [int(row["source_row_index"]) for row in batch]
        )
        for row in batch:
            source_index = int(row["source_row_index"])
            audio_bytes = audio[source_index]["bytes"]
            actual_audio_hash = hashlib.sha256(audio_bytes).hexdigest()
            if actual_audio_hash != row["audio_sha256"]:
                raise ValueError(f"Audio SHA-256 mismatch for {row['sample_id']}")
            signal, sample_rate, info = decode_wav(audio_bytes)
            if sample_rate != 16_000 or info["channels"] != 1:
                raise ValueError(f"Audio contract mismatch for {row['sample_id']}")
            for family in pending_by_sample[row["sample_id"]]:
                artifact = expected_artifact(run_root, family, row["sample_id"])
                record = {
                    "sample_id": row["sample_id"],
                    "source_row_index": source_index,
                    "speaker_id": row["speaker_id"],
                    "emotion": row["emotion"],
                    "accent": row["accent"],
                    "audio_sha256": actual_audio_hash,
                    "feature_family": family,
                    "feature_version": FEATURE_VERSION,
                    "artifact_path": relative_path(artifact, repo_root),
                    "artifact_sha256": "",
                    "shape": "",
                    "dtype": "float32",
                    "finite_fraction": "",
                    "status": "error",
                    "error": "",
                }
                try:
                    value, names = extract_family(family, signal, sample_rate, smile)
                    validate_feature_value(family, value)
                    if names is not None:
                        if expected_egemaps_names is None:
                            expected_egemaps_names = names
                            egemaps_names_path.parent.mkdir(parents=True, exist_ok=True)
                            egemaps_names_path.write_text(
                                json.dumps(names, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8",
                            )
                            tracked_egemaps_names_path.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            shutil.copyfile(
                                egemaps_names_path, tracked_egemaps_names_path
                            )
                        elif names != expected_egemaps_names:
                            raise ValueError("eGeMAPS column order changed")
                    atomic_save(artifact, family, value)
                    reopened = load_artifact(family, artifact)
                    validate_feature_value(family, reopened)
                    record.update(
                        {
                            "artifact_sha256": file_sha256(artifact),
                            "shape": artifact_shape(family, reopened),
                            "finite_fraction": f"{artifact_finite_fraction(family, reopened):.8f}",
                            "status": "ok",
                        }
                    )
                    processed += 1
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    failures += 1
                upsert_record(connection, record)
        completed_samples = min(batch_start + len(batch), len(pending_rows))
        print(
            f"handcrafted samples {completed_samples}/{len(pending_rows)} "
            f"processed_features={processed} failures={failures}",
            flush=True,
        )

    rows = write_index(connection, index_path)
    connection.close()
    if egemaps_names_path.is_file():
        tracked_egemaps_names_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(egemaps_names_path, tracked_egemaps_names_path)
    expected_records = len(eligible) * len(PILOT_FAMILIES)
    clean_records = sum(row["status"] == "ok" and not row["error"] for row in rows)
    report = {
        "status": "PASS" if failures == 0 and clean_records == expected_records else "FAIL",
        "feature_version": FEATURE_VERSION,
        "eligible_samples": len(eligible),
        "feature_families": list(PILOT_FAMILIES),
        "expected_records": expected_records,
        "clean_records": clean_records,
        "processed_this_invocation": processed,
        "resumed_this_invocation": resumed,
        "invalidated_this_invocation": invalidated,
        "failures": failures,
        "manifest_sha256": file_sha256(manifest_path),
        "index_path": relative_path(index_path, repo_root),
        "index_sha256": file_sha256(index_path),
        "egemaps_feature_names_path": relative_path(
            tracked_egemaps_names_path, repo_root
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "PASS":
        raise RuntimeError(
            f"Handcrafted extraction incomplete: {clean_records}/{expected_records} clean, "
            f"{failures} failures"
        )
    return report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--manifest", type=Path, default=repo_root / "data/manifests/clean_manifest.csv"
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=repo_root / "data/raw/visec_hf/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=repo_root / "data/interim/features/b4_features_v1/handcrafted_full",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_handcrafted_full_feature_index.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_handcrafted_full_extraction_report.json",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    report = run_full_handcrafted(
        repo_root,
        args.manifest.resolve(),
        args.parquet.resolve(),
        args.run_root.resolve(),
        args.index.resolve(),
        args.report.resolve(),
        args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
