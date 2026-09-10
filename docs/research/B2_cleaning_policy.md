# B2 ViSEC Cleaning Policy (v2)

## Goal

Create a traceable dataset for later speaker-disjoint splitting without modifying the raw ViSEC file or silently changing source labels.

## Input and output

- Input: `data/manifests/raw_manifest.csv`
- Output: `data/manifests/clean_manifest.csv`
- Summary: `data/manifests/cleaning_report.json`
- Policy version: `b2.2-v2`

The clean manifest still contains all 5,280 source rows. The field `is_eligible_for_split` determines which records can enter the next step.

## Cleaning rules

### Readability and required labels

A record is eligible only if its audio is readable and it has a valid speaker ID and one of the four emotion labels: angry, happy, neutral, or sad.

### Exact duplicate audio with consistent metadata

When multiple rows have the same audio SHA-256 and consistent metadata, keep the row with the lowest `source_row_index`. Mark all later copies as `exclude_duplicate_copy`.

This ensures that identical audio cannot leak across train, validation, and test sets.

### Exact duplicate audio with conflicting metadata

When identical audio has different speaker, emotion, gender, or accent metadata, exclude every row in that duplicate group from supervised modeling. Do not choose an emotion label automatically.

### Speaker-level accent and gender

Accent and gender should be stable attributes of one speaker. For each speaker, calculate the majority label using only unique audio that does not belong to a conflicting duplicate group. Store the result in `accent_clean` and `gender_clean`.

The original `accent_raw`, `accent`, `gender_raw`, and `gender` fields are preserved. Corrections are recorded by `accent_corrected`, `gender_corrected`, and `cleaning_note`.

If a speaker-level vote is tied, the script stops instead of guessing.

## Fields added by B2 cleaning

| Field | Meaning |
| --- | --- |
| `accent_clean` | One reviewed canonical accent label per speaker |
| `gender_clean` | One reviewed canonical gender label per speaker |
| `accent_corrected` | Whether the canonical accent differs from the row-level source label |
| `gender_corrected` | Whether the canonical gender differs from the row-level source label |
| `is_eligible_for_split` | Whether the row may enter the split step |
| `cleaning_action` | Keep or exclusion decision applied to the row |
| `exclusion_reason` | Machine-readable reason when a row is excluded |
| `cleaning_note` | Trace of duplicate representative or metadata normalization |
| `cleaning_policy_version` | Version of the cleaning rules used |
| `raw_manifest_version` | Raw manifest version used as the cleaning input |

## Acceptance checks

Before B2 cleaning is considered complete:

1. every source row remains traceable;
2. eligible audio checksums are unique;
3. all eligible audio is readable;
4. every eligible record has a speaker and valid emotion;
5. each speaker has exactly one `accent_clean` value;
6. each speaker has exactly one `gender_clean` value; and
7. the raw manifest remains unchanged.

The cleaning script does not create train, validation, or test assignments.

## Execution result

| Item | Result |
| --- | --- |
| Source rows retained for traceability | `5280` |
| Rows eligible for splitting | `4992` |
| Consistent duplicate copies excluded | `268` |
| Rows from conflicting duplicate groups excluded | `20` |
| Source rows with accent normalization recorded | `9` |
| Source rows with gender normalization recorded | `2` |
| Speakers remaining | `147` |

Eligible-set distributions:

- Accent: Central `275`, Northern `1183`, Southern `3534`.
- Emotion: Angry `1465`, Happy `973`, Neutral `1505`, Sad `1049`.
- Gender: Female `3542`, Male `1450`.

All acceptance checks passed. B2 cleaning is complete. Dataset partitioning is handled separately in B5.

## B2 quality-audit extension

Audit version 1.1 adds reproducible duration quantiles, duplicate path checks,
PCM clipping screening, and frame-level silence screening. The following
artifacts contain the results:

- `data/manifests/raw_manifest.csv`
- `data/manifests/raw_audit.json`
- `data/manifests/invalid_audio.csv`
- `data/manifests/duplicate_report.csv`
- `docs/research/B2_quality_report.md`

The v2 quality flags are diagnostic and do not cause automatic exclusion.
Therefore, the 4,992-row eligible set and the frozen `split_manifest.csv` remain
unchanged. Any future exclusion based on clipping or silence requires a new
cleaning-policy version and a separate sensitivity experiment. The cleaning
report records the input manifest name and a normalized text SHA-256 so its
lineage can be verified consistently with either Windows CRLF or LF line
endings.
