# B2 — Dataset Cleaning and Quality Control

## Objective and immutable boundary

Create a traceable eligible population without modifying raw ViSEC audio or
silently changing source emotion labels. Input is `raw_manifest.csv`; output is
`clean_manifest.csv` plus `cleaning_report.json`. Policy version is `b2.2-v2`.

## Cleaning policy

- Retain all 5,280 source rows for traceability.
- Require readable audio, a released speaker ID and one of angry/happy/neutral/sad.
- For exact waveform duplicates with consistent metadata, retain the lowest
  `source_row_index` and exclude later copies.
- For identical audio with conflicting speaker/emotion/gender/accent metadata,
  exclude the entire group; never guess a label.
- Derive one speaker-level `accent_clean` and `gender_clean` by majority vote
  over unique non-conflicting audio. Stop on ties.
- Preserve raw values and record corrections, exclusion reason, cleaning note,
  policy version and raw-manifest version.
- Clipping and high-silence flags are diagnostic. They do not change eligibility
  without a new policy version and sensitivity experiment.

## Verified result

| Measure | Value |
| --- | ---: |
| Source rows | 5,280 |
| Eligible rows | 4,992 |
| Consistent duplicate copies excluded | 268 |
| Conflicting-duplicate rows excluded | 20 |
| Released speaker IDs retained | 147 |

Eligible distributions are Central 275, North 1,183 and South 3,534; Angry
1,465, Happy 973, Neutral 1,505 and Sad 1,049. Every eligible checksum is
unique, audio is readable, required labels exist and speaker-level cleaned
accent/gender values are consistent.

## Evidence and reproduction

- `src/fedecai/data/audit_visec.py`
- `src/fedecai/data/clean_visec_manifest.py`
- `data/manifests/raw_manifest.csv`
- `data/manifests/raw_audit.json`
- `data/manifests/clean_manifest.csv`
- `data/manifests/cleaning_report.json`
- Generated quality detail: `reports/tables/b2/b2_quality_report.md`

## Definition of Done

Every source row remains traceable; eligible waveforms are unique and readable;
labels and speaker metadata validate; raw artifacts remain unchanged; and no
train/validation/test assignment is created by B2. Status: **DONE**.
