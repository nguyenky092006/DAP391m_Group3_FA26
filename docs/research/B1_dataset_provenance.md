# B1 ViSEC Dataset Provenance and License

## Purpose

This document records where ViSEC comes from, what the official sources currently state, how the team will store the data, and which facts still require a local audit. Published metadata is kept separate from measurements produced by this project.

## Source register

| Source | Role in this project | Information currently supported | Access date |
| --- | --- | --- | --- |
| [Official ViSEC repository](https://github.com/thanhpv2102/ViSEC) | Primary project reference and historical original-download reference | Identifies the repository as the official Pitch-Fusion and ViSEC release. Its Google Drive link is currently unavailable. The README states that the original `visec.zip` contained a WAV folder and `data.csv`, and defines the three accents as Northern, Central, and Southern. | 2026-09-09 |
| [Hugging Face ViSEC dataset card](https://huggingface.co/datasets/hustep-lab/ViSEC) | Primary acquisition source, dataset metadata, schema, and license reference | Reports 5,280 rows, one train split, approximately 367 MB, three accent values, four emotion values, and the fields `speaker_id`, `path`, `duration`, `accent`, `emotion`, `emotion_id`, and `gender`. Declares the dataset license as CC BY 4.0. | 2026-09-09 |
| [ICASSP 2024 paper](https://doi.org/10.1109/ICASSP48485.2024.10448373) | Dataset and Pitch-Fusion research reference | Pham Viet Thanh, Ngo Thi Thu Huyen, Pham Ngoc Quan, and Nguyen Thi Thu Trang. “A Robust Pitch-Fusion Model for Speech Emotion Recognition in Tonal Languages,” ICASSP 2024, pages 12386-12390. | 2026-09-09 |

## Published dataset description

The current Project Planning document and Hugging Face dataset card describe ViSEC as a Vietnamese speech emotion corpus with:

- 5,280 audio records;
- four emotion labels;
- three accent groups;
- speaker identifiers;
- gender metadata; and
- audio duration metadata.

These are published claims, not yet local audit results. B1 data collection and audit must verify the downloaded copy before the numbers are used in analysis or reporting.

## Planned raw-to-canonical label handling

Raw labels must be retained in the manifest. Canonical labels may be added in separate columns after verification.

| Source value currently visible | Planned canonical value | Display label | Verification status |
| --- | --- | --- | --- |
| `north` | `north` | Northern | Pending local audit |
| `mid` | `central` | Central | Pending local audit |
| `south` | `south` | Southern | Pending local audit |
| `angry` | `angry` | Angry | Pending local audit |
| `happy` | `happy` | Happy | Pending local audit |
| `neutral` | `neutral` | Neutral | Pending local audit |
| `sad` | `sad` | Sad | Pending local audit |

The original value must be stored as `accent_raw` or `emotion_raw`. Normalization must not overwrite the source label.

## License decision

The Hugging Face dataset card declares ViSEC under CC BY 4.0. Project use must therefore retain attribution to the dataset authors and cite the ICASSP 2024 paper.

The official GitHub repository did not display a separate `LICENSE` file during this review. The dataset license must not automatically be applied to the repository's model code. The team may inspect and reproduce the public implementation for coursework, but it must not claim that the code is CC BY 4.0 or redistribute substantial source code under that license without separate confirmation from the authors.

## Acquisition decision

- Primary acquisition source: the versioned Parquet file hosted in the Hugging Face ViSEC repository.
- Pinned dataset commit: `06926b7a1dca5f6627b47a2446492be6ff061716`.
- File: `data/train-00000-of-00001.parquet`.
- Published file size: `366955512` bytes.
- Published SHA-256: `bfc7697b3a591cc6cf185c61178815a35363dae4f2c43276636d68eb72cd4a3e`.
- Pinned download URL: `https://huggingface.co/datasets/hustep-lab/ViSEC/resolve/06926b7a1dca5f6627b47a2446492be6ff061716/data/train-00000-of-00001.parquet?download=true`.
- The Google Drive link in the official model repository is retained only as provenance evidence because it was unavailable when checked on 2026-09-09.
- Do not combine a future recovered Drive archive with the Hugging Face records unless a record-level comparison confirms that they represent the same release.
- Store the chosen raw release under `data/raw/visec/` or in another local directory referenced through configuration.
- Never commit raw audio or the downloaded archive to GitHub.
- Do not rename, resample, trim, or overwrite raw audio files.
- Extracted or transformed copies belong in `data/interim/` or `data/processed/`, never in the raw layer.

## Download record required after acquisition

After downloading the dataset, create a machine-readable record containing at least:

- dataset name;
- source URL;
- access date and time;
- retrieval method;
- downloaded filename;
- file size in bytes;
- SHA-256 checksum of the downloaded archive or snapshot;
- extraction directory;
- metadata filename;
- stated dataset license;
- source revision or commit identifier when available; and
- notes describing any download or extraction problem.

The download record must describe the actual local copy. Do not enter a checksum, file size, or revision value before measuring it.

## Repository storage rules

| Location | Contents | Git policy |
| --- | --- | --- |
| `data/raw/` | Original downloaded and extracted ViSEC files | Ignored except for `.gitkeep` |
| `data/manifests/` | Reviewed metadata, checksums, split assignments, and audit summaries without raw audio | Tracked when small and free of machine-specific absolute paths |
| `data/interim/` | Temporary normalized metadata, resampled audio, and extracted features under development | Ignored except for `.gitkeep` |
| `data/processed/` | Final processed inputs used by models | Ignored except for `.gitkeep` |
| `artifacts/` | Predictions, metrics, model files, and experiment outputs | Ignored by default; selected small tables may later be copied to a tracked reporting location |

Manifests should store a stable sample identifier and a path relative to the configured dataset root. Do not store a developer's absolute Windows path as the only audio identifier.

## Privacy and handling rules

- Treat voice recordings and speaker identifiers as sensitive research data even when the dataset license permits reuse.
- Do not publish raw audio again in the project repository.
- Do not upload the corpus to unrelated public services.
- Application uploads must not be retained by default when the demo is implemented.
- Reports and dashboards should use aggregate statistics or non-identifying sample identifiers.

## Facts that remain unresolved until local audit

1. The exact number of downloaded audio files and metadata rows.
2. The number of unique speakers in the chosen release.
3. Whether every metadata row resolves to one readable audio file.
4. Whether there are unreferenced audio files, duplicate paths, duplicate checksums, or conflicting labels.
5. The observed emotion, accent, gender, and speaker distributions.
6. The exact mapping between `emotion` and `emotion_id`.
7. Audio sample rates, channel counts, encodings, durations, and corrupt files.
8. Whether the unavailable original Google Drive release and the Hugging Face copy contain equivalent records, if the original release becomes accessible later.

## B1.4 acceptance criteria

- The official repository, dataset card, and paper are recorded.
- Published claims are separated from locally measured facts.
- Dataset licensing is distinguished from source-code licensing.
- Raw, interim, processed, manifest, and artifact storage rules are explicit.
- Raw labels will be preserved before normalization.
- Required download provenance fields are defined.
- Remaining audit questions are listed without invented answers.
