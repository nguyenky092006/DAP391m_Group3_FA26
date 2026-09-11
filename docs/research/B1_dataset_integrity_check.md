# B1 ViSEC Dataset Integrity and Record Audit

## Final status

- Check date: 2026-09-09
- Source revision: `06926b7a1dca5f6627b47a2446492be6ff061716`
- Project file: `data/raw/visec_hf/train-00000-of-00001.parquet`
- Binary integrity: PASS
- Parquet container and schema: PASS
- Embedded audio readability: PASS
- Record-level result: PASS WITH WARNINGS
- Raw data modified: NO

The data file is authentic and technically readable. The warnings are annotation and duplication issues that must be handled explicitly in B2 before any train/validation/test split is created.

## Binary and container checks

| Check | Measured result | Status |
| --- | --- | --- |
| File size | `366955512` bytes; matches pinned source | PASS |
| SHA-256 | `bfc7697b3a591cc6cf185c61178815a35363dae4f2c43276636d68eb72cd4a3e`; exact match | PASS |
| Parquet rows | `5280` | PASS |
| Row groups | `53` | PASS |
| Top-level fields | `speaker_id`, `path`, `duration`, `accent`, `emotion`, `emotion_id`, `gender` | PASS |
| Missing metadata values | `0` in all six metadata variables | PASS |

The `path` column contains both the original WAV filename and embedded audio bytes. The raw Parquet file remains ignored by Git.

## Metadata summary

| Variable | Measured distribution |
| --- | --- |
| Speakers | `147` unique IDs (`0` to `146`) |
| Accent | south `3618`, north `1336`, mid `326` |
| Emotion | angry `1466`, neutral `1507`, happy `1228`, sad `1079` |
| Gender | female `3675`, male `1605` |
| Emotion mapping | happy=`0`, neutral=`1`, sad=`2`, angry=`3`; consistent in all rows |

The accent distribution is strongly imbalanced, especially for the Central group (`mid`). Later evaluation must report macro metrics and per-accent results rather than accuracy alone.

## Audio summary

| Check | Result |
| --- | --- |
| Embedded audio missing | `0` |
| Unreadable audio | `0` |
| Sample rate | all `5280` files are `16000 Hz` |
| Channels | all `5280` files are mono |
| Sample width | `5178` files use 2 bytes; `102` files use 3 bytes |
| Compression | all report `NONE` in the WAV header |
| Metadata vs measured duration | maximum absolute difference `0.0` seconds |
| Unique audio checksums | `5002` |

## Warnings that affect later processing

### Duplicate audio

- `277` checksum groups contain repeated audio, covering `555` rows.
- `267` groups have the same speaker/accent/emotion/gender metadata.
- `10` groups contain at least one metadata difference.
- Among those 10 groups: 3 differ in emotion, 8 differ in speaker ID, and 1 differs in gender. No duplicate group differs in accent.

Identical audio must never appear in different data splits. No duplicate is deleted in B1. In B2, a documented cleaning rule must retain only an approved representative or exclude the ambiguous group from supervised modeling.

### Speaker metadata consistency

- Six speaker IDs contain more than one accent value: `15`, `26`, `28`, `34`, `38`, `113`.
- Two speaker IDs contain more than one gender value: `25`, `63`.
- Most conflicts are isolated minority rows, so they are flagged as possible annotation errors rather than automatically corrected.

Speaker IDs must still be treated as indivisible groups. A record cannot be moved to another split because its accent or gender label differs from the majority label of that speaker.

## Reproducible evidence

- Audit script: `src/fedecai/data/audit_visec.py`
- Current record manifest: `data/manifests/raw_manifest.csv`
- Current machine-readable report: `data/manifests/raw_audit.json`
- Unified project dependencies: `requirements.txt`
- Unit tests: `tests/test_audit_visec.py`

The script preserves all source rows and raw labels. It adds normalized fields, WAV header measurements, checksums, duplicate group IDs, and machine-readable issue flags.

## B1.6 decision

B1.6 is complete as a data collection and audit step. The dataset is suitable to continue to B2, with these mandatory constraints:

1. Do not create a split from the unreviewed raw rows.
2. Resolve or exclude duplicate-audio groups before freezing the split.
3. Keep every speaker entirely within one split.
4. Preserve the raw manifest; write all cleaning decisions to a new manifest version.
5. Record both source label and the reason for every exclusion or correction.
