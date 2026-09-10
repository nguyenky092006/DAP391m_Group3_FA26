# B2 ViSEC Audio Quality Report

## Scope

This report summarizes reproducible checks for all 5280 source records. Quality flags require review but do not automatically remove a sample.

## Integrity and format

- Missing embedded audio: `0`.
- Unreadable audio: `0`.
- Unique audio path names: `5280`; duplicate path groups: `0`.
- Sample rates: `{'16000': 5280}`.
- Channel counts: `{'1': 5280}`.
- Sample widths in bytes: `{'2': 5178, '3': 102}`.

## Duration

| Statistic | Seconds |
| --- | ---: |
| Minimum | 1.000000 |
| 1st percentile | 1.024000 |
| 25th percentile | 1.408000 |
| Median | 1.944531 |
| 75th percentile | 2.368000 |
| 99th percentile | 6.874880 |
| Maximum | 24.704000 |

The processing duration limit must be selected later from the training split only.

## Clipping and silence screening

- Clipping rule: Flag when at least 0.100% of PCM samples have absolute amplitude at or above 99.9% of full scale.
- Files flagged for possible clipping: `80`.
- Silence rule: Use 20 ms frames; flag when at least 60% of frames are at or below -40 dBFS.
- Files flagged for high silence: `408`.

These flags are diagnostic. Any exclusion or signal processing decision requires a separate documented experiment and a new cleaning-policy version.

## Duplicate waveform screening

- Unique waveform hashes: `5002`.
- Duplicate groups: `277` covering `555` rows.
- Conflicting duplicate groups: `10`.

Detailed duplicate rows are stored in `data/manifests/duplicate_report.csv`; invalid audio rows are stored in `data/manifests/invalid_audio.csv`.

