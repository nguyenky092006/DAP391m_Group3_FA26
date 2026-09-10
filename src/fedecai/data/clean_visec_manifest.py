"""Apply the approved B2 cleaning policy to the ViSEC raw manifest.

The output keeps every source row for traceability and marks whether each row
is eligible for a later speaker-disjoint split. No raw column is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CLEANING_POLICY_VERSION = "b2.2-v2"
VALID_ACCENTS = {"central", "north", "south"}
VALID_EMOTIONS = {"angry", "happy", "neutral", "sad"}


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _row_index(row: dict[str, Any]) -> int:
    return int(row["source_row_index"])


def _majority(values: Iterable[str]) -> str:
    counts = Counter(value for value in values if value)
    if not counts:
        return ""
    highest = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == highest)
    if len(winners) != 1:
        raise ValueError(f"Cannot resolve a tied speaker-level label: {dict(counts)}")
    return winners[0]


def normalized_text_sha256(path: Path) -> str:
    """Hash UTF-8 CSV content independently of LF/CRLF and trailing blank lines."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n") + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_exclusion(row: dict[str, Any]) -> tuple[str, str] | None:
    if not _is_true(row.get("audio_readable", False)):
        return "exclude_invalid_audio", "audio_missing_empty_or_unreadable"
    if not str(row.get("speaker_id", "")).strip():
        return "exclude_missing_speaker", "missing_speaker_id"
    if row.get("emotion") not in VALID_EMOTIONS:
        return "exclude_invalid_emotion", "missing_or_invalid_emotion"
    if row.get("accent") not in VALID_ACCENTS:
        return "exclude_invalid_accent", "missing_or_invalid_accent"
    return None


def apply_cleaning_policy(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["duplicate_content_group"]:
            duplicate_groups[row["duplicate_content_group"]].append(row)

    conflicting_groups = {
        group_id
        for group_id, members in duplicate_groups.items()
        if any("conflicting_duplicate_metadata" in member["audit_issue"].split(";") for member in members)
    }
    duplicate_copy_ids: set[str] = set()
    duplicate_keeper_ids: set[str] = set()
    for group_id, members in duplicate_groups.items():
        if group_id in conflicting_groups:
            continue
        ordered = sorted(members, key=_row_index)
        duplicate_keeper_ids.add(ordered[0]["sample_id"])
        duplicate_copy_ids.update(member["sample_id"] for member in ordered[1:])

    # Build speaker-level labels only from unique, non-conflicting audio.
    label_reference_rows = [
        row
        for row in rows
        if row["duplicate_content_group"] not in conflicting_groups
        and row["sample_id"] not in duplicate_copy_ids
        and _base_exclusion(row) is None
    ]
    speaker_accents: dict[str, list[str]] = defaultdict(list)
    speaker_genders: dict[str, list[str]] = defaultdict(list)
    for row in label_reference_rows:
        speaker_accents[row["speaker_id"]].append(row["accent"])
        speaker_genders[row["speaker_id"]].append(row["gender"])

    accent_by_speaker = {
        speaker_id: _majority(values) for speaker_id, values in speaker_accents.items()
    }
    gender_by_speaker = {
        speaker_id: _majority(values) for speaker_id, values in speaker_genders.items()
    }

    cleaned: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        group_id = row["duplicate_content_group"]
        accent_clean = accent_by_speaker.get(row["speaker_id"], row["accent"])
        gender_clean = gender_by_speaker.get(row["speaker_id"], row["gender"])
        accent_corrected = bool(accent_clean and accent_clean != row["accent"])
        gender_corrected = bool(gender_clean and gender_clean != row["gender"])

        base_exclusion = _base_exclusion(row)
        if group_id in conflicting_groups:
            eligible = False
            action = "exclude_conflicting_duplicate"
            exclusion_reason = "identical_audio_has_conflicting_metadata"
        elif row["sample_id"] in duplicate_copy_ids:
            eligible = False
            action = "exclude_duplicate_copy"
            exclusion_reason = "identical_audio_represented_by_lower_source_row"
        elif base_exclusion is not None:
            eligible = False
            action, exclusion_reason = base_exclusion
        else:
            eligible = True
            action = "keep_with_metadata_normalization" if accent_corrected or gender_corrected else "keep"
            exclusion_reason = ""

        notes = []
        if row["sample_id"] in duplicate_keeper_ids:
            notes.append("kept_as_duplicate_group_representative")
        if accent_corrected:
            notes.append(f"accent:{row['accent']}->{accent_clean}")
        if gender_corrected:
            notes.append(f"gender:{row['gender']}->{gender_clean}")

        row.update(
            {
                "accent_clean": accent_clean,
                "gender_clean": gender_clean,
                "accent_corrected": accent_corrected,
                "gender_corrected": gender_corrected,
                "is_eligible_for_split": eligible,
                "cleaning_action": action,
                "exclusion_reason": exclusion_reason,
                "cleaning_note": ";".join(notes),
                "raw_manifest_version": "1.1.0",
                "cleaning_policy_version": CLEANING_POLICY_VERSION,
            }
        )
        cleaned.append(row)

    eligible_rows = [row for row in cleaned if row["is_eligible_for_split"]]
    eligible_hashes = [row["audio_sha256"] for row in eligible_rows]
    speaker_to_accents: dict[str, set[str]] = defaultdict(set)
    speaker_to_genders: dict[str, set[str]] = defaultdict(set)
    for row in eligible_rows:
        speaker_to_accents[row["speaker_id"]].add(row["accent_clean"])
        speaker_to_genders[row["speaker_id"]].add(row["gender_clean"])

    report = {
        "cleaning_policy_version": CLEANING_POLICY_VERSION,
        "input_rows": len(rows),
        "output_rows_retained_for_traceability": len(cleaned),
        "eligible_rows": len(eligible_rows),
        "excluded_rows": len(cleaned) - len(eligible_rows),
        "actions": dict(sorted(Counter(row["cleaning_action"] for row in cleaned).items())),
        "corrections": {
            "accent_rows": sum(row["accent_corrected"] for row in cleaned),
            "gender_rows": sum(row["gender_corrected"] for row in cleaned),
            "accent_speakers": sorted(
                {row["speaker_id"] for row in cleaned if row["accent_corrected"]}, key=int
            ),
            "gender_speakers": sorted(
                {row["speaker_id"] for row in cleaned if row["gender_corrected"]}, key=int
            ),
        },
        "eligible_distributions": {
            "speakers": len({row["speaker_id"] for row in eligible_rows}),
            "accent": dict(sorted(Counter(row["accent_clean"] for row in eligible_rows).items())),
            "emotion": dict(sorted(Counter(row["emotion"] for row in eligible_rows).items())),
            "gender": dict(sorted(Counter(row["gender_clean"] for row in eligible_rows).items())),
        },
        "validation": {
            "eligible_audio_hashes_unique": len(eligible_hashes) == len(set(eligible_hashes)),
            "all_eligible_audio_readable": all(_is_true(row["audio_readable"]) for row in eligible_rows),
            "all_eligible_speakers_present": all(bool(row["speaker_id"]) for row in eligible_rows),
            "all_eligible_emotions_valid": all(
                row["emotion"] in VALID_EMOTIONS for row in eligible_rows
            ),
            "all_eligible_accents_valid": all(row["accent_clean"] in VALID_ACCENTS for row in eligible_rows),
            "one_clean_accent_per_speaker": all(len(values) == 1 for values in speaker_to_accents.values()),
            "one_clean_gender_per_speaker": all(len(values) == 1 for values in speaker_to_genders.values()),
        },
        "policy": {
            "consistent_duplicates": "Keep the lowest source_row_index and exclude the other copies.",
            "conflicting_duplicates": "Exclude every row in the checksum group from supervised modeling.",
            "speaker_accent_and_gender": "Use the unique non-conflicting audio rows to select the unambiguous majority label for each speaker; preserve raw labels.",
            "emotion": "Never repair or infer a conflicting emotion label automatically.",
        },
    }
    if not all(report["validation"].values()):
        raise ValueError(f"Clean manifest validation failed: {report['validation']}")
    return cleaned, report


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repo_root / "data/manifests/raw_manifest.csv")
    parser.add_argument("--output", type=Path, default=repo_root / "data/manifests/clean_manifest.csv")
    parser.add_argument("--report", type=Path, default=repo_root / "data/manifests/cleaning_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.input.resolve())
    cleaned, report = apply_cleaning_policy(rows)
    report["input_manifest"] = {
        "path": args.input.name,
        "normalized_text_sha256": normalized_text_sha256(args.input.resolve()),
        "normalization": "UTF-8 text; CRLF converted to LF; exactly one trailing newline",
    }
    write_manifest(cleaned, args.output.resolve())
    write_report(report, args.report.resolve())
    print(json.dumps({
        "eligible_rows": report["eligible_rows"],
        "excluded_rows": report["excluded_rows"],
        "output": str(args.output.resolve()),
        "report": str(args.report.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
