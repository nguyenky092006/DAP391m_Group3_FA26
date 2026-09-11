# B5 Speaker Disjoint Split Version 2 Plan

> Status: completed and frozen on 2026-09-11. The final assignment SHA-256 is
> `da5610e76cf0ce3cd5de8e282e5819824b5251ff22d28e2cf27294598d80ef77`.
> See `B5_split_v2_finalization.md` for the reviewed result.

## Reason for a new version

B3 found that `speaker_disjoint_v1` has no Central-Sad utterance in the test set. The existing version remains preserved and must not be overwritten. Version 2 adds a hard requirement that every train, validation, and test combination of accent and emotion contains at least one speaker and at least one utterance.

## Files created by the script

- `data/manifests/split_manifest_v2.csv`
- `data/manifests/split_report_v2.json`

The version 1 files remain unchanged:

- `data/manifests/split_manifest.csv`
- `data/manifests/split_report.json`

## Reproduce in PowerShell

    .\.venv\Scripts\python.exe src\fedecai\data\run_b5_split_pipeline.py

Then inspect the validation section:

    Get-Content data/manifests/split_report_v2.json

Every value under `validation` must be `true`. The `cross_strata` section must contain 36 records and every record must have `samples > 0` and `speakers > 0`.

## Determinism check

The consolidated runner performs the split optimization twice before writing
outputs and fails if the assignments differ. All later models must use version 2.

## Acceptance criteria

1. Exactly 103 train, 22 validation, and 22 test speakers.
2. Speaker overlap between splits is zero.
3. Eligible audio hashes remain unique.
4. All 36 split-accent-emotion cells contain data.
5. Speaker 0 remains in train.
6. All 288 B2-excluded rows remain excluded.
7. Version 1 artifacts remain byte-for-byte unchanged.
8. Repeated execution with seed 391 produces the same assignment hash.

All eight criteria passed. The original v1 files remain available only for
provenance and must not be used for model comparison.
