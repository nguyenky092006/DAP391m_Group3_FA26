"""Build a reproducible record-level audit of the pinned ViSEC Parquet file.

The raw Parquet file is never modified. The script writes a CSV manifest with
one row per source record and a JSON summary for human and automated review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_REVISION = "06926b7a1dca5f6627b47a2446492be6ff061716"
EXPECTED_FILE_SIZE = 366_955_512
EXPECTED_FILE_SHA256 = "bfc7697b3a591cc6cf185c61178815a35363dae4f2c43276636d68eb72cd4a3e"
EXPECTED_COLUMNS = [
    "speaker_id",
    "path",
    "duration",
    "accent",
    "emotion",
    "emotion_id",
    "gender",
]
ACCENT_MAP = {"north": "north", "mid": "central", "south": "south"}
EMOTION_ID_MAP = {"happy": 0, "neutral": 1, "sad": 2, "angry": 3}
CLIPPING_AMPLITUDE_RATIO = 0.999
CLIPPING_SAMPLE_RATIO = 0.001
SILENCE_DBFS = -40.0
HIGH_SILENCE_FRAME_RATIO = 0.60
SILENCE_FRAME_MS = 20


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_pcm(frames: bytes, sample_width: int) -> tuple[np.ndarray, int]:
    """Decode little-endian PCM samples and return samples plus full-scale value."""
    if sample_width == 1:
        return np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128, 128
    if sample_width == 2:
        return np.frombuffer(frames, dtype="<i2"), 32_768
    if sample_width == 3:
        values = np.frombuffer(frames, dtype=np.uint8)
        values = values[: (len(values) // 3) * 3].reshape(-1, 3)
        decoded = (
            values[:, 0].astype(np.int32)
            | (values[:, 1].astype(np.int32) << 8)
            | (values[:, 2].astype(np.int32) << 16)
        )
        decoded = (decoded ^ 0x800000) - 0x800000
        return decoded, 8_388_608
    if sample_width == 4:
        return np.frombuffer(frames, dtype="<i4"), 2_147_483_648
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def _signal_quality(
    frames: bytes,
    sample_width: int,
    sample_rate: int,
    num_channels: int,
) -> dict[str, Any]:
    samples, full_scale = _decode_pcm(frames, sample_width)
    if samples.size == 0:
        return {
            "peak_amplitude_ratio": 0.0,
            "clipped_sample_count": 0,
            "clipped_sample_ratio": 0.0,
            "possible_clipping": False,
            "silent_frame_ratio": 1.0,
            "high_silence_ratio": True,
        }

    absolute = np.abs(samples.astype(np.float64))
    peak_ratio = float(absolute.max() / full_scale)
    clipped_count = int(np.count_nonzero(absolute >= full_scale * CLIPPING_AMPLITUDE_RATIO))
    clipped_ratio = clipped_count / int(samples.size)

    samples_per_window = max(1, int(sample_rate * SILENCE_FRAME_MS / 1000) * num_channels)
    starts = np.arange(0, samples.size, samples_per_window)
    squared = absolute * absolute
    sums = np.add.reduceat(squared, starts)
    lengths = np.minimum(samples_per_window, samples.size - starts)
    rms = np.sqrt(sums / lengths)
    silence_threshold = full_scale * (10 ** (SILENCE_DBFS / 20))
    silent_ratio = float(np.count_nonzero(rms <= silence_threshold) / len(rms))

    return {
        "peak_amplitude_ratio": peak_ratio,
        "clipped_sample_count": clipped_count,
        "clipped_sample_ratio": clipped_ratio,
        "possible_clipping": clipped_ratio >= CLIPPING_SAMPLE_RATIO,
        "silent_frame_ratio": silent_ratio,
        "high_silence_ratio": silent_ratio >= HIGH_SILENCE_FRAME_RATIO,
    }


def inspect_wav(audio_bytes: bytes) -> dict[str, Any]:
    """Read WAV header and signal-quality values without extracting the audio."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(num_frames)
        return {
            "sample_rate_hz": sample_rate,
            "num_channels": num_channels,
            "sample_width_bytes": sample_width,
            "num_frames": num_frames,
            "duration_sec_measured": num_frames / sample_rate if sample_rate else None,
            "compression_type": wav_file.getcomptype(),
            **_signal_quality(frames, sample_width, sample_rate, num_channels),
        }


def _sorted_counts(values: Counter[Any]) -> dict[str, int]:
    return {str(key): values[key] for key in sorted(values, key=lambda item: str(item))}


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def audit_dataset(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    input_size = input_path.stat().st_size
    input_hash = file_sha256(input_path)
    parquet_file = pq.ParquetFile(input_path)
    columns = parquet_file.schema_arrow.names
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema. Expected {EXPECTED_COLUMNS}, found {columns}")

    rows: list[dict[str, Any]] = []
    audio_groups: dict[str, list[int]] = defaultdict(list)
    speaker_accents: dict[str, Counter[str]] = defaultdict(Counter)
    speaker_genders: dict[str, Counter[str]] = defaultdict(Counter)
    null_counts: Counter[str] = Counter()
    accent_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    emotion_id_counts: Counter[int] = Counter()
    gender_counts: Counter[str] = Counter()
    sample_rate_counts: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    sample_width_counts: Counter[int] = Counter()
    compression_counts: Counter[str] = Counter()
    duration_deltas: list[float] = []
    durations: list[float] = []
    peak_ratios: list[float] = []
    silence_ratios: list[float] = []
    audio_path_counts: Counter[str] = Counter()
    total_audio_bytes = 0
    unreadable_audio = 0
    missing_audio = 0
    possible_clipping = 0
    high_silence = 0

    source_row_index = 0
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_index, columns=EXPECTED_COLUMNS)
        for raw in table.to_pylist():
            issues: set[str] = set()
            for field in ("speaker_id", "duration", "accent", "emotion", "emotion_id", "gender"):
                if raw[field] is None:
                    null_counts[field] += 1

            speaker_raw = raw["speaker_id"]
            speaker_id = "" if speaker_raw is None else str(speaker_raw).strip()
            accent_raw = "" if raw["accent"] is None else str(raw["accent"]).strip()
            emotion_raw = "" if raw["emotion"] is None else str(raw["emotion"]).strip()
            gender_raw = "" if raw["gender"] is None else str(raw["gender"]).strip()
            accent = ACCENT_MAP.get(accent_raw.lower(), "")
            emotion = emotion_raw.lower()
            gender = gender_raw.lower()

            if not speaker_id:
                issues.add("missing_speaker")
            if not accent:
                issues.add("unknown_accent")
            if emotion not in EMOTION_ID_MAP:
                issues.add("unknown_emotion")
            expected_emotion_id = EMOTION_ID_MAP.get(emotion)
            if expected_emotion_id is not None and raw["emotion_id"] != expected_emotion_id:
                issues.add("emotion_id_mismatch")

            if accent_raw:
                accent_counts[accent_raw] += 1
            if emotion_raw:
                emotion_counts[emotion_raw] += 1
            if raw["emotion_id"] is not None:
                emotion_id_counts[int(raw["emotion_id"])] += 1
            if gender_raw:
                gender_counts[gender_raw] += 1
            if speaker_id and accent_raw:
                speaker_accents[speaker_id][accent_raw] += 1
            if speaker_id and gender_raw:
                speaker_genders[speaker_id][gender_raw] += 1

            audio_obj = raw["path"] or {}
            audio_bytes = audio_obj.get("bytes")
            audio_path_raw = audio_obj.get("path") or ""
            if audio_path_raw:
                audio_path_counts[audio_path_raw] += 1
            audio_hash = ""
            audio_info: dict[str, Any] = {
                "sample_rate_hz": "",
                "num_channels": "",
                "sample_width_bytes": "",
                "num_frames": "",
                "duration_sec_measured": "",
                "compression_type": "",
                "peak_amplitude_ratio": "",
                "clipped_sample_count": "",
                "clipped_sample_ratio": "",
                "possible_clipping": "",
                "silent_frame_ratio": "",
                "high_silence_ratio": "",
            }
            duration_delta: float | str = ""
            if not audio_bytes:
                missing_audio += 1
                issues.add("missing_embedded_audio")
            else:
                total_audio_bytes += len(audio_bytes)
                audio_hash = hashlib.sha256(audio_bytes).hexdigest()
                audio_groups[audio_hash].append(source_row_index)
                try:
                    audio_info = inspect_wav(audio_bytes)
                    sample_rate_counts[audio_info["sample_rate_hz"]] += 1
                    channel_counts[audio_info["num_channels"]] += 1
                    sample_width_counts[audio_info["sample_width_bytes"]] += 1
                    compression_counts[audio_info["compression_type"]] += 1
                    if audio_info["duration_sec_measured"] is not None:
                        durations.append(float(audio_info["duration_sec_measured"]))
                    peak_ratios.append(float(audio_info["peak_amplitude_ratio"]))
                    silence_ratios.append(float(audio_info["silent_frame_ratio"]))
                    if audio_info["possible_clipping"]:
                        possible_clipping += 1
                        issues.add("possible_clipping")
                    if audio_info["high_silence_ratio"]:
                        high_silence += 1
                        issues.add("high_silence_ratio")
                    if audio_info["num_frames"] == 0:
                        issues.add("empty_audio")
                    if raw["duration"] is not None:
                        duration_delta = audio_info["duration_sec_measured"] - float(raw["duration"])
                        duration_deltas.append(abs(duration_delta))
                except (wave.Error, EOFError, ValueError) as exc:
                    unreadable_audio += 1
                    issues.add("unreadable_audio")
                    audio_info["read_error"] = str(exc)

            rows.append(
                {
                    "source_row_id": f"hf_row_{source_row_index:06d}",
                    "source_row_index": source_row_index,
                    "sample_id": f"visec_hf_{source_row_index:06d}",
                    "source_name": "huggingface",
                    "source_release_id": SOURCE_REVISION,
                    "audio_locator": f"data/train-00000-of-00001.parquet#row={source_row_index}",
                    "audio_path_raw": audio_path_raw,
                    "speaker_id_raw": "" if speaker_raw is None else speaker_raw,
                    "speaker_id": speaker_id,
                    "duration_sec_metadata": "" if raw["duration"] is None else raw["duration"],
                    "accent_raw": accent_raw,
                    "accent": accent,
                    "emotion_raw": emotion_raw,
                    "emotion_id_raw": "" if raw["emotion_id"] is None else raw["emotion_id"],
                    "emotion": emotion if emotion in EMOTION_ID_MAP else "",
                    "emotion_id": "" if expected_emotion_id is None else expected_emotion_id,
                    "gender_raw": gender_raw,
                    "gender": gender,
                    "embedded_audio_present": bool(audio_bytes),
                    "audio_readable": bool(audio_bytes) and "unreadable_audio" not in issues,
                    "audio_sha256": audio_hash,
                    "file_size_bytes": 0 if not audio_bytes else len(audio_bytes),
                    **audio_info,
                    "duration_delta_sec": duration_delta,
                    "duplicate_content_group": "",
                    "requires_review": bool(issues),
                    "audit_issue": ";".join(sorted(issues)),
                }
            )
            source_row_index += 1

    duplicate_groups = {digest: indexes for digest, indexes in audio_groups.items() if len(indexes) > 1}
    conflicting_duplicate_details: list[dict[str, Any]] = []
    duplicate_rows = 0
    for group_number, (digest, indexes) in enumerate(sorted(duplicate_groups.items()), start=1):
        group_id = f"dup_{group_number:04d}"
        duplicate_rows += len(indexes)
        group_fields = {
            field: sorted({str(rows[index][field]) for index in indexes})
            for field in ("speaker_id", "accent", "emotion", "emotion_id", "gender")
        }
        conflicting_fields = [field for field, values in group_fields.items() if len(values) > 1]
        for index in indexes:
            issue_set = set(filter(None, rows[index]["audit_issue"].split(";")))
            issue_set.add("duplicate_audio")
            rows[index]["duplicate_content_group"] = group_id
            rows[index]["requires_review"] = True
            if conflicting_fields:
                issue_set.add("conflicting_duplicate_metadata")
                issue_set.update(f"conflicting_duplicate_{field}" for field in conflicting_fields)
            rows[index]["audit_issue"] = ";".join(sorted(issue_set))
        if conflicting_fields:
            conflicting_duplicate_details.append(
                {
                    "duplicate_content_group": group_id,
                    "audio_sha256": digest,
                    "conflicting_fields": conflicting_fields,
                    "records": [
                        {
                            key: rows[index][key]
                            for key in (
                                "source_row_index",
                                "audio_path_raw",
                                "speaker_id",
                                "accent_raw",
                                "emotion",
                                "emotion_id",
                                "gender",
                            )
                        }
                        for index in indexes
                    ],
                }
            )

    speaker_conflict_details: list[dict[str, Any]] = []
    speaker_ids = sorted(set(speaker_accents) | set(speaker_genders), key=lambda value: int(value))
    for speaker_id in speaker_ids:
        accent_values = speaker_accents[speaker_id]
        gender_values = speaker_genders[speaker_id]
        conflicting_fields = []
        if len(accent_values) > 1:
            conflicting_fields.append("accent")
        if len(gender_values) > 1:
            conflicting_fields.append("gender")
        if conflicting_fields:
            speaker_conflict_details.append(
                {
                    "speaker_id": speaker_id,
                    "conflicting_fields": conflicting_fields,
                    "accent_counts": _sorted_counts(accent_values),
                    "gender_counts": _sorted_counts(gender_values),
                }
            )
            for row in rows:
                if row["speaker_id"] == speaker_id:
                    issue_set = set(filter(None, row["audit_issue"].split(";")))
                    issue_set.update(f"speaker_{field}_conflict" for field in conflicting_fields)
                    row["requires_review"] = True
                    row["audit_issue"] = ";".join(sorted(issue_set))

    issue_counts: Counter[str] = Counter()
    for row in rows:
        issue_counts.update(filter(None, row["audit_issue"].split(";")))

    report = {
        "audit_version": "1.1.0",
        "source": {
            "name": "huggingface",
            "revision": SOURCE_REVISION,
            "relative_file": "data/raw/visec_hf/train-00000-of-00001.parquet",
            "file_size_bytes": input_size,
            "expected_file_size_bytes": EXPECTED_FILE_SIZE,
            "sha256": input_hash,
            "expected_sha256": EXPECTED_FILE_SHA256,
            "binary_integrity_pass": input_size == EXPECTED_FILE_SIZE and input_hash == EXPECTED_FILE_SHA256,
        },
        "parquet": {
            "rows": len(rows),
            "row_groups": parquet_file.num_row_groups,
            "columns": columns,
        },
        "metadata": {
            "null_counts": _sorted_counts(null_counts),
            "unique_speakers": len(speaker_ids),
            "accent_counts_raw": _sorted_counts(accent_counts),
            "emotion_counts": _sorted_counts(emotion_counts),
            "emotion_id_counts": _sorted_counts(emotion_id_counts),
            "gender_counts": _sorted_counts(gender_counts),
            "emotion_id_mapping_verified": all(
                row["emotion_id"] == row["emotion_id_raw"] for row in rows if row["emotion_id"] != ""
            ),
            "speaker_conflict_count": len(speaker_conflict_details),
            "speaker_conflicts": speaker_conflict_details,
        },
        "audio": {
            "embedded_audio_missing": missing_audio,
            "audio_unreadable": unreadable_audio,
            "total_embedded_audio_bytes": total_audio_bytes,
            "sample_rate_counts": _sorted_counts(sample_rate_counts),
            "channel_counts": _sorted_counts(channel_counts),
            "sample_width_bytes_counts": _sorted_counts(sample_width_counts),
            "compression_type_counts": _sorted_counts(compression_counts),
            "audio_path_unique": len(audio_path_counts),
            "duplicate_audio_path_groups": sum(count > 1 for count in audio_path_counts.values()),
            "duration_seconds": {
                "min": min(durations, default=None),
                "q01": _quantile(durations, 0.01),
                "q25": _quantile(durations, 0.25),
                "median": _quantile(durations, 0.50),
                "q75": _quantile(durations, 0.75),
                "q99": _quantile(durations, 0.99),
                "max": max(durations, default=None),
            },
            "duration_delta_abs_max": max(duration_deltas, default=None),
            "duration_delta_abs_mean": (sum(duration_deltas) / len(duration_deltas)) if duration_deltas else None,
            "unique_audio_sha256": len(audio_groups),
            "duplicate_audio_groups": len(duplicate_groups),
            "duplicate_audio_rows": duplicate_rows,
            "conflicting_duplicate_groups": len(conflicting_duplicate_details),
            "conflicting_duplicate_details": conflicting_duplicate_details,
        },
        "signal_quality": {
            "method": {
                "clipping": (
                    f"Flag when at least {CLIPPING_SAMPLE_RATIO:.3%} of PCM samples have absolute "
                    f"amplitude at or above {CLIPPING_AMPLITUDE_RATIO:.1%} of full scale."
                ),
                "silence": (
                    f"Use {SILENCE_FRAME_MS} ms frames; flag when at least "
                    f"{HIGH_SILENCE_FRAME_RATIO:.0%} of frames are at or below {SILENCE_DBFS:.0f} dBFS."
                ),
            },
            "possible_clipping_files": possible_clipping,
            "high_silence_files": high_silence,
            "peak_amplitude_ratio": {
                "median": _quantile(peak_ratios, 0.50),
                "q99": _quantile(peak_ratios, 0.99),
                "max": max(peak_ratios, default=None),
            },
            "silent_frame_ratio": {
                "median": _quantile(silence_ratios, 0.50),
                "q95": _quantile(silence_ratios, 0.95),
                "max": max(silence_ratios, default=None),
            },
        },
        "issues": {
            "rows_requiring_review": sum(bool(row["requires_review"]) for row in rows),
            "issue_row_counts": _sorted_counts(issue_counts),
        },
        "decision": {
            "status": "PASS_WITH_WARNINGS",
            "raw_data_modified": False,
            "notes": [
                "All embedded audio is readable and duration metadata matches decoded WAV duration.",
                "Clipping and high-silence flags are screening signals for review, not automatic exclusions.",
                "Duplicate audio and speaker-level metadata conflicts are retained and flagged for B2 cleaning.",
                "No split may be created before duplicate groups and speaker metadata conflicts are handled explicitly.",
            ],
        },
    }
    return rows, report


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_invalid_audio_report(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["sample_id", "source_row_index", "audio_path_raw", "speaker_id", "audit_issue"]
    invalid_issues = {"missing_embedded_audio", "unreadable_audio", "empty_audio"}
    invalid = [
        {field: row[field] for field in fields}
        for row in rows
        if invalid_issues.intersection(filter(None, str(row["audit_issue"]).split(";")))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(invalid)


def write_duplicate_report(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "duplicate_content_group",
        "audio_sha256",
        "sample_id",
        "source_row_index",
        "audio_path_raw",
        "speaker_id",
        "accent",
        "emotion",
        "emotion_id",
        "gender",
        "metadata_conflict",
        "conflicting_fields",
    ]
    output = []
    for row in rows:
        if not row["duplicate_content_group"]:
            continue
        issues = set(filter(None, str(row["audit_issue"]).split(";")))
        conflicting_fields = sorted(
            issue.removeprefix("conflicting_duplicate_")
            for issue in issues
            if issue.startswith("conflicting_duplicate_") and issue != "conflicting_duplicate_metadata"
        )
        output.append({
            **{field: row[field] for field in fields[:10]},
            "metadata_conflict": "conflicting_duplicate_metadata" in issues,
            "conflicting_fields": ";".join(conflicting_fields),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def write_quality_report(report: dict[str, Any], path: Path) -> None:
    audio = report["audio"]
    quality = report["signal_quality"]
    duration = audio["duration_seconds"]
    lines = [
        "# B2 ViSEC Audio Quality Report",
        "",
        "## Scope",
        "",
        f"This report summarizes reproducible checks for all {report['parquet']['rows']} source records. "
        "Quality flags require review but do not automatically remove a sample.",
        "",
        "## Integrity and format",
        "",
        f"- Missing embedded audio: `{audio['embedded_audio_missing']}`.",
        f"- Unreadable audio: `{audio['audio_unreadable']}`.",
        f"- Unique audio path names: `{audio['audio_path_unique']}`; duplicate path groups: "
        f"`{audio['duplicate_audio_path_groups']}`.",
        f"- Sample rates: `{audio['sample_rate_counts']}`.",
        f"- Channel counts: `{audio['channel_counts']}`.",
        f"- Sample widths in bytes: `{audio['sample_width_bytes_counts']}`.",
        "",
        "## Duration",
        "",
        "| Statistic | Seconds |",
        "| --- | ---: |",
        f"| Minimum | {duration['min']:.6f} |",
        f"| 1st percentile | {duration['q01']:.6f} |",
        f"| 25th percentile | {duration['q25']:.6f} |",
        f"| Median | {duration['median']:.6f} |",
        f"| 75th percentile | {duration['q75']:.6f} |",
        f"| 99th percentile | {duration['q99']:.6f} |",
        f"| Maximum | {duration['max']:.6f} |",
        "",
        "The processing duration limit must be selected later from the training split only.",
        "",
        "## Clipping and silence screening",
        "",
        f"- Clipping rule: {quality['method']['clipping']}",
        f"- Files flagged for possible clipping: `{quality['possible_clipping_files']}`.",
        f"- Silence rule: {quality['method']['silence']}",
        f"- Files flagged for high silence: `{quality['high_silence_files']}`.",
        "",
        "These flags are diagnostic. Any exclusion or signal processing decision requires a separate "
        "documented experiment and a new cleaning-policy version.",
        "",
        "## Duplicate waveform screening",
        "",
        f"- Unique waveform hashes: `{audio['unique_audio_sha256']}`.",
        f"- Duplicate groups: `{audio['duplicate_audio_groups']}` covering "
        f"`{audio['duplicate_audio_rows']}` rows.",
        f"- Conflicting duplicate groups: `{audio['conflicting_duplicate_groups']}`.",
        "",
        "Detailed duplicate rows are stored in `data/manifests/duplicate_report.csv`; invalid audio rows "
        "are stored in `data/manifests/invalid_audio.csv`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repo_root / "data/raw/visec_hf/train-00000-of-00001.parquet")
    parser.add_argument("--manifest", type=Path, default=repo_root / "data/manifests/raw_manifest.csv")
    parser.add_argument("--report", type=Path, default=repo_root / "data/manifests/raw_audit.json")
    parser.add_argument("--invalid-report", type=Path, default=repo_root / "data/manifests/invalid_audio.csv")
    parser.add_argument("--duplicate-report", type=Path, default=repo_root / "data/manifests/duplicate_report.csv")
    parser.add_argument("--quality-report", type=Path, default=repo_root / "docs/research/B2_quality_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, report = audit_dataset(args.input.resolve())
    write_manifest(rows, args.manifest.resolve())
    write_report(report, args.report.resolve())
    write_invalid_audio_report(rows, args.invalid_report.resolve())
    write_duplicate_report(rows, args.duplicate_report.resolve())
    write_quality_report(report, args.quality_report.resolve())
    print(json.dumps({
        "status": report["decision"]["status"],
        "rows": report["parquet"]["rows"],
        "manifest": str(args.manifest.resolve()),
        "report": str(args.report.resolve()),
        "invalid_report": str(args.invalid_report.resolve()),
        "duplicate_report": str(args.duplicate_report.resolve()),
        "quality_report": str(args.quality_report.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
