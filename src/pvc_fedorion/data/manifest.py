"""Build a deterministic, leakage-aware manifest for the supplied VESC corpus.

The raw folder label is preserved as the training label. Filename-derived metadata
is used only for auditing and grouping; suspicious records are never silently fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

CANONICAL_LABELS = ("angry", "anxiety", "happy", "neutral", "sad")
LABEL_ALIASES = {
    "angry": "angry",
    "anxiety": "anxiety",
    "auxiety": "anxiety",
    "happy": "happy",
    "happ": "happy",
    "neutral": "neutral",
    "sad": "sad",
}
VIEW_SUFFIX = "_vocals"
SOURCE_RE = re.compile(r"_(p\d{2})_", re.IGNORECASE)
SPEAKER_RE = re.compile(r"_([mf])(\d+)_", re.IGNORECASE)

MANIFEST_FIELDS = (
    "sample_id",
    "original_split",
    "relative_path",
    "file_name",
    "sha256",
    "utterance_id",
    "view",
    "folder_label",
    "filename_label_raw",
    "filename_label",
    "source_id",
    "speaker_local_id",
    "speaker_id",
    "gender",
    "is_child",
    "sample_rate_hz",
    "channels",
    "sample_width_bytes",
    "frame_count",
    "duration_seconds",
    "compression_type",
    "issues",
)


@dataclass(frozen=True)
class ParsedFilename:
    """Metadata inferred from a VESC filename without modifying the source data."""

    utterance_id: str
    view: str
    filename_label_raw: str
    filename_label: str
    source_id: str
    speaker_local_id: str
    speaker_id: str
    gender: str
    is_child: bool
    issues: tuple[str, ...]


def _canonical_folder_label(path: Path) -> str:
    label = path.parent.name.strip().lower()
    if label not in CANONICAL_LABELS:
        raise ValueError(
            f"Expected parent folder to be one of {CANONICAL_LABELS}, got {path.parent.name!r}"
        )
    return label


def parse_vesc_filename(file_name: str) -> ParsedFilename:
    """Parse grouping metadata from a VESC WAV filename.

    Unknown or malformed fields are retained as explicit issues. A normalized alias
    is provided for auditing, while the raw filename token remains available.
    """

    path = Path(file_name)
    stem = path.stem.lower()
    issues: list[str] = []

    view = "processed" if stem.endswith(VIEW_SUFFIX) else "raw"
    utterance_id = stem[: -len(VIEW_SUFFIX)] if view == "processed" else stem

    raw_label = stem.split("_", maxsplit=1)[0]
    filename_label = LABEL_ALIASES.get(raw_label, "unknown")
    if filename_label == "unknown":
        issues.append("unknown_filename_label")
    elif raw_label != filename_label:
        issues.append("filename_label_alias")

    source_match = SOURCE_RE.search(f"_{stem}_")
    source_id = source_match.group(1).lower() if source_match else "unknown"
    if source_id == "unknown":
        issues.append("missing_source_id")

    speaker_match = SPEAKER_RE.search(f"_{stem}_")
    if speaker_match:
        gender = speaker_match.group(1).upper()
        speaker_local_id = f"{gender}{speaker_match.group(2)}"
        speaker_id = f"{source_id}:{speaker_local_id}"
    else:
        gender = "unknown"
        speaker_local_id = "unknown"
        speaker_id = "unknown"
        issues.append("missing_speaker_id")

    return ParsedFilename(
        utterance_id=utterance_id,
        view=view,
        filename_label_raw=raw_label,
        filename_label=filename_label,
        source_id=source_id,
        speaker_local_id=speaker_local_id,
        speaker_id=speaker_id,
        gender=gender,
        is_child="_eb_" in f"_{stem}_",
        issues=tuple(issues),
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_wave_metadata(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as wav:
        frame_count = wav.getnframes()
        sample_rate = wav.getframerate()
        return {
            "sample_rate_hz": sample_rate,
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "frame_count": frame_count,
            "duration_seconds": round(frame_count / sample_rate, 9),
            "compression_type": wav.getcomptype(),
        }


def _scan_split(root: Path, split: str) -> list[dict[str, object]]:
    if not root.is_dir():
        raise FileNotFoundError(f"{split} root does not exist or is not a directory: {root}")

    wav_paths = sorted(
        (path for path in root.rglob("*.wav") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    if not wav_paths:
        raise ValueError(f"No WAV files found below {split} root: {root}")

    records: list[dict[str, object]] = []
    for path in wav_paths:
        folder_label = _canonical_folder_label(path)
        parsed = parse_vesc_filename(path.name)
        issues = list(parsed.issues)
        if parsed.filename_label != "unknown" and parsed.filename_label != folder_label:
            issues.append("folder_filename_label_mismatch")

        metadata = _read_wave_metadata(path)
        digest = _sha256(path)
        relative_path = path.relative_to(root).as_posix()
        records.append(
            {
                "sample_id": digest[:16],
                "original_split": split,
                "relative_path": relative_path,
                "file_name": path.name,
                "sha256": digest,
                "utterance_id": parsed.utterance_id,
                "view": parsed.view,
                "folder_label": folder_label,
                "filename_label_raw": parsed.filename_label_raw,
                "filename_label": parsed.filename_label,
                "source_id": parsed.source_id,
                "speaker_local_id": parsed.speaker_local_id,
                "speaker_id": parsed.speaker_id,
                "gender": parsed.gender,
                "is_child": parsed.is_child,
                **metadata,
                "issues": ";".join(sorted(set(issues))),
            }
        )
    return records


def _counter_dict(values: Iterable[object]) -> dict[str, int]:
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items()))


def audit_manifest(records: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summarize integrity, leakage, pairing, and audio-format properties."""

    families: dict[str, list[dict[str, object]]] = defaultdict(list)
    hashes: dict[str, list[dict[str, object]]] = defaultdict(list)
    issue_counts: Counter[str] = Counter()
    for record in records:
        families[str(record["utterance_id"])].append(record)
        hashes[str(record["sha256"])].append(record)
        issue_counts.update(filter(None, str(record["issues"]).split(";")))

    paired_families: list[list[dict[str, object]]] = []
    singleton_families = 0
    malformed_families = 0
    cross_split_pairs = 0
    pair_label_conflicts = 0
    pair_sample_rate_mismatches = 0
    pair_channel_mismatches = 0
    pair_counts_by_label: Counter[str] = Counter()

    for family in families.values():
        views = Counter(str(record["view"]) for record in family)
        is_pair = len(family) == 2 and views == {"raw": 1, "processed": 1}
        if is_pair:
            paired_families.append(family)
            labels = {str(record["folder_label"]) for record in family}
            if len(labels) == 1:
                pair_counts_by_label.update(labels)
            else:
                pair_label_conflicts += 1
            if len({str(record["original_split"]) for record in family}) > 1:
                cross_split_pairs += 1
            if len({int(record["sample_rate_hz"]) for record in family}) > 1:
                pair_sample_rate_mismatches += 1
            if len({int(record["channels"]) for record in family}) > 1:
                pair_channel_mismatches += 1
        elif len(family) == 1:
            singleton_families += 1
        else:
            malformed_families += 1

    train_speakers = {
        str(record["speaker_id"])
        for record in records
        if record["original_split"] == "train" and record["speaker_id"] != "unknown"
    }
    test_records = [record for record in records if record["original_split"] == "test"]
    test_speakers = {
        str(record["speaker_id"])
        for record in test_records
        if record["speaker_id"] != "unknown"
    }
    test_files_with_train_speaker = sum(
        str(record["speaker_id"]) in train_speakers for record in test_records
    )

    exact_duplicate_groups = [
        sorted(str(record["relative_path"]) for record in group)
        for group in hashes.values()
        if len(group) > 1
    ]

    return {
        "schema_version": 1,
        "total_files": len(records),
        "counts_by_original_split": _counter_dict(
            record["original_split"] for record in records
        ),
        "counts_by_folder_label": _counter_dict(record["folder_label"] for record in records),
        "counts_by_view": _counter_dict(record["view"] for record in records),
        "counts_by_source": _counter_dict(record["source_id"] for record in records),
        "sample_rates_hz": _counter_dict(record["sample_rate_hz"] for record in records),
        "channel_counts": _counter_dict(record["channels"] for record in records),
        "speaker_count": len(
            {
                str(record["speaker_id"])
                for record in records
                if record["speaker_id"] != "unknown"
            }
        ),
        "utterance_family_count": len(families),
        "paired_family_count": len(paired_families),
        "singleton_family_count": singleton_families,
        "malformed_family_count": malformed_families,
        "paired_families_by_label": dict(sorted(pair_counts_by_label.items())),
        "cross_original_split_pair_count": cross_split_pairs,
        "pair_label_conflict_count": pair_label_conflicts,
        "pair_sample_rate_mismatch_count": pair_sample_rate_mismatches,
        "pair_channel_mismatch_count": pair_channel_mismatches,
        "train_speaker_count": len(train_speakers),
        "test_speaker_count": len(test_speakers),
        "overlapping_train_test_speaker_count": len(train_speakers & test_speakers),
        "test_files_with_train_speaker_count": test_files_with_train_speaker,
        "issue_counts": dict(sorted(issue_counts.items())),
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "exact_duplicate_groups": exact_duplicate_groups,
    }


def _write_manifest(records: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def build_manifest(
    train_root: Path,
    test_root: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, object]]:
    """Scan raw VESC directories and write a stable manifest plus audit report."""

    records = _scan_split(train_root.resolve(), "train")
    records.extend(_scan_split(test_root.resolve(), "test"))
    records.sort(
        key=lambda record: (
            str(record["original_split"]),
            str(record["folder_label"]),
            str(record["relative_path"]).lower(),
        )
    )

    manifest_path = output_dir.resolve() / "manifest.csv"
    audit_path = output_dir.resolve() / "audit.json"
    _write_manifest(records, manifest_path)

    audit = audit_manifest(records)
    audit["manifest_sha256"] = _sha256(manifest_path)
    audit["train_root"] = str(train_root.resolve())
    audit["test_root"] = str(test_root.resolve())
    with audit_path.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    return manifest_path, audit_path, audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and audit a deterministic raw-data manifest for VESC."
    )
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path, audit_path, audit = build_manifest(
        train_root=args.train_root,
        test_root=args.test_root,
        output_dir=args.output_dir,
    )
    summary = {
        "manifest": str(manifest_path),
        "audit": str(audit_path),
        "total_files": audit["total_files"],
        "utterance_families": audit["utterance_family_count"],
        "paired_families": audit["paired_family_count"],
        "cross_split_pairs": audit["cross_original_split_pair_count"],
        "overlapping_train_test_speakers": audit["overlapping_train_test_speaker_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

