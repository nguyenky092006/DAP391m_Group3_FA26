# B1 ViSEC Data Dictionary

## Status and scope

- Schema version: `1.0.0`
- Status: confirmed against the pinned local Hugging Face Parquet file
- Source reference: the current Hugging Face ViSEC card exposes `speaker_id`, `path`, `duration`, `accent`, `emotion`, `emotion_id`, and `gender`.
- Important: the original Google Drive release may use different column names or types. This project uses the pinned Hugging Face revision recorded in `B1_dataset_provenance.md`.

This dictionary defines the fields observed in the local source and retained in `raw_manifest.csv`. Raw values are kept separately from normalized fields.

## Variable roles

| Role | Meaning |
| --- | --- |
| Identifier | Identifies a record, speaker, source file, run, or derived item. It is not a model feature. |
| Main target | The four-class emotion label predicted by the application. |
| Domain and analysis | Used to measure accent differences and in GRL/federated experiments. It is not the main application output. |
| Analysis only | Used for descriptive analysis, not as a model input unless a later approved experiment explicitly requires it. |
| Split only | Used to prevent leakage. It must not be passed to the model as a predictive feature. |
| Audit | Produced by code to verify file integrity, metadata consistency, or duplication. |
| Configuration | Records a controlled experimental assignment such as split, fold, or client. |

## Raw metadata fields

These fields preserve what is read from the selected ViSEC release. Values must not be silently corrected in place.

| Field | Planned type | Unit or format | Currently published values | Role | Validation and handling |
| --- | --- | --- | --- | --- | --- |
| `source_row_id` | String | Stable row reference | Not supplied by the dataset card | Identifier | Create deterministically from the original metadata row position or source key. It must remain stable across repeated manifest builds from the same release. |
| `source_row_index` | Integer | Zero-based Parquet row number | `0` to `5279` | Identifier | Preserves the exact row position in the pinned source file. |
| `audio_locator` | String | Source-relative Parquet row locator | `data/train-00000-of-00001.parquet#row=<index>` | Identifier | Locates embedded audio without storing a developer-specific absolute Windows path. |
| `audio_path_raw` | String | Path exactly as supplied | Dataset-dependent | Identifier | Preserve the source value. Do not use it as the only stable sample identifier. Flag blank paths and paths that resolve outside the configured dataset root. |
| `speaker_id_raw` | String | Source value converted to text without changing meaning | Hugging Face currently displays integer IDs from 0 to 146 | Split only | Preserve the raw value. A missing speaker ID blocks safe speaker-disjoint assignment until reviewed. Do not use speaker ID as a model feature. |
| `duration_sec_metadata` | Float or null | Seconds | Hugging Face currently displays approximately 1.0 to 24.7 seconds | Analysis only | Parse numeric values without changing the raw source file. Flag missing, non-numeric, zero, or negative values. Compare with measured audio duration later. |
| `accent_raw` | String | Original label | `north`, `mid`, `south` are currently visible | Domain and analysis | Preserve exact source text. Unknown or missing values remain in the audit manifest and require review before accent-level evaluation. |
| `emotion_raw` | String | Original label | `angry`, `happy`, `neutral`, `sad` are currently visible | Main target | Preserve exact source text. Missing or unknown target labels cannot enter supervised training until resolved. |
| `emotion_id_raw` | String or integer | Source code | Hugging Face displays integer IDs 0 to 3 | Identifier for source label | Do not assume the ID-to-label mapping from isolated preview rows. Verify the mapping across the full metadata table and flag conflicts. |
| `gender_raw` | String or null | Original label | `female`, `male` are currently visible | Analysis only | Preserve exact source text. Missing or unknown values do not get imputed automatically. Gender is not a default model input. |

## Canonical manifest fields

These fields are created without overwriting the raw fields. The current canonical raw manifest is `data/manifests/raw_manifest.csv`.

| Field | Type | Unit or format | Allowed values | Role | Derivation and rule |
| --- | --- | --- | --- | --- | --- |
| `sample_id` | String | `visec_<stable-id>` | Unique and non-empty | Identifier | Generate deterministically from a stable raw record reference. Do not use a random UUID that changes on every run. |
| `audio_path_rel` | String or null | POSIX-style path relative to the configured dataset root | Existing file path after optional extraction | Identifier | The source stores audio inside Parquet, so this remains null until a later extraction step creates individual files. Never store a developer-specific absolute Windows path. |
| `speaker_id` | String | Canonical text ID | Non-empty for records eligible for splitting | Split only | Normalize representation without merging different raw speaker IDs. Preserve `speaker_id_raw` for traceability. |
| `accent` | String or null | Lowercase category | `north`, `central`, `south` | Domain and analysis | Planned mapping: `north` to `north`, `mid` to `central`, and `south` to `south`. Apply only after the local audit confirms the raw labels. |
| `emotion` | String or null | Lowercase category | `angry`, `happy`, `neutral`, `sad` | Main target | Normalize case and surrounding whitespace after validation. Do not infer emotion from filenames when metadata is present but conflicting. |
| `emotion_id` | Integer or null | Canonical class index | `0`, `1`, `2`, `3` after verified mapping | Main target encoding | Create from the verified canonical `emotion` mapping in one configuration source. Do not rely on row order or alphabetical order implicitly. |
| `gender` | String or null | Lowercase category | Expected `female`, `male`; additional observed values retained for review | Analysis only | Normalize only after listing all raw values. Do not impute missing gender. |
| `source_name` | String | Dataset source label | `official_drive` or `huggingface` | Identifier | Record which acquisition source produced the row. One manifest version must not silently mix sources. |
| `source_release_id` | String or null | Revision, archive checksum prefix, or documented release ID | Dataset-dependent | Identifier | Populate from the download record. Use null until a release identifier or checksum has actually been recorded. |

## Audio audit fields

These values must be measured from the local audio files rather than copied from assumptions.

| Field | Type | Unit or format | Role | Validation and interpretation |
| --- | --- | --- | --- | --- |
| `embedded_audio_present` | Boolean | `true` or `false` | Audit | True only when the Parquet audio object contains non-empty bytes. This is the relevant raw-layer check for the selected release. |
| `audio_readable` | Boolean | `true` or `false` | Audit | True only when the selected audio library can open the file and read its header. An unreadable file remains in the manifest with an issue description. |
| `audio_sha256` | String or null | 64 lowercase hexadecimal characters | Audit | Hash the raw file bytes. Identical checksums indicate identical content and require duplicate review; they do not trigger automatic deletion. |
| `file_size_bytes` | Integer or null | Bytes | Audit | Measure from the local file. Must be greater than zero for a usable file. |
| `sample_rate_hz` | Integer or null | Hertz | Audit | Read from the audio header. List observed values before choosing a processing sample rate. |
| `num_channels` | Integer or null | Channels | Audit | Read from the audio header. Do not silently convert stereo to mono in the raw layer. |
| `sample_width_bytes` | Integer or null | Bytes per PCM sample | Audit | Read from the WAV header. ViSEC currently contains 16-bit and 24-bit PCM represented by values 2 and 3. |
| `num_frames` | Integer or null | Frames | Audit | Read from the audio header. Must be non-negative and consistent with readable audio. |
| `duration_sec_measured` | Float or null | Seconds | Audit | Calculate from frames and sample rate or a verified audio reader. Must be positive for a usable record. |
| `duration_delta_sec` | Float or null | Seconds | Audit | `duration_sec_measured - duration_sec_metadata`. Set the warning tolerance only after checking the metadata precision. |
| `compression_type` | String or null | WAV compression code | Audit | Read from the WAV header. The current release reports `NONE`; other values require review. |
| `peak_amplitude_ratio` | Float or null | Fraction of PCM full scale | Audit | Maximum absolute sample amplitude divided by the full-scale value. Values are bounded from 0 to 1. |
| `clipped_sample_count` | Integer or null | Samples | Audit | Number of PCM samples at or above 99.9% of full scale. |
| `clipped_sample_ratio` | Float or null | Fraction of samples | Audit | `clipped_sample_count` divided by the number of decoded PCM samples. |
| `possible_clipping` | Boolean | `true` or `false` | Audit | True when at least 0.1% of samples are at or above 99.9% of full scale. It is a review flag, not an automatic exclusion. |
| `silent_frame_ratio` | Float or null | Fraction of 20 ms frames | Audit | Fraction of frames whose RMS level is at or below -40 dBFS. |
| `high_silence_ratio` | Boolean | `true` or `false` | Audit | True when at least 60% of frames meet the silence rule. It is a review flag, not an automatic exclusion. |
| `duplicate_content_group` | String or null | Deterministic group ID | Audit | Assign only when two or more rows share the same audio checksum. Review labels and paths before deciding how duplicates are handled. |
| `requires_review` | Boolean | `true` or `false` | Audit | True when the row has at least one audit issue, duplicate warning, speaker metadata conflict, clipping flag, or high-silence flag. |
| `audit_issue` | String or null | Short machine-readable issue list | Audit | Record specific issues such as `missing_file`, `unreadable_audio`, `missing_speaker`, `unknown_emotion`, or `conflicting_duplicate_label`. Null means no detected issue under the current checks, not that the sample is scientifically perfect. |

## Split and experiment assignment fields

These fields are not part of raw data. They are added only after the raw manifest has passed review.

| Field | Type | Allowed values | Role | Rule |
| --- | --- | --- | --- | --- |
| `split` | String | `train`, `validation`, `test` | Configuration | Assign by speaker. A speaker must occur in exactly one split. Freeze the test assignment before model tuning. |
| `fold_id` | String or null | Project-defined grouped fold ID | Configuration | Use only for grouped cross-validation. All samples from one speaker must remain in the same fold. |
| `client_id` | String or null | Project-defined federated client ID | Configuration | Assign speakers, not individual recordings, to federated clients. Record the client-partition strategy and random seed. |
| `is_augmented` | Boolean | `true` or `false` | Configuration | False for raw records. Augmented records may exist only in training-derived data. |
| `parent_sample_id` | String or null | Existing `sample_id` | Identifier | Required for augmented or transformed records so they can be traced to the original sample. |
| `processing_version` | String or null | Versioned pipeline/config ID | Configuration | Identifies the code and configuration that produced a derived record. It must not be populated for untouched raw audio unless explicitly set to `raw`. |

## Rules for missing, invalid, and duplicate records

1. Keep every source metadata row in the raw audit manifest, including invalid rows.
2. Never replace a missing target, accent, speaker ID, or gender with a guessed value.
3. A missing speaker ID blocks the record from speaker-disjoint splitting until it is resolved or explicitly excluded.
4. An invalid or unknown emotion blocks the record from supervised emotion training.
5. Missing accent does not justify guessing from speaker name, filename, or geography. Its inclusion in training and exclusion from accent-level reporting must be decided after the audit.
6. Duplicate checksums must be reviewed before splitting so identical audio cannot leak across partitions.
7. Conflicting labels attached to identical audio must be reported separately from ordinary duplicates.
8. Cleaning decisions must produce a new manifest version. The raw manifest remains unchanged.

## Model-use restrictions

- `speaker_id`, `speaker_id_raw`, file paths, source row IDs, and checksums must never be predictive features.
- `emotion` is the main supervised target.
- `accent` is used for stratified analysis, accent probing, GRL, and federated partition design. It is not the main application output.
- `gender` is descriptive by default and must not become a model input without an approved research reason.
- Audit fields may filter invalid records through an explicit cleaning rule, but they are not model features.

## B1.5 acceptance criteria

- Raw source fields and canonical fields are separate.
- Every field has a type, meaning, role, and validation rule.
- Units are stated for duration, sample rate, frames, channels, and file size.
- Speaker ID is restricted to splitting and leakage checks.
- Emotion is identified as the main target and accent as a domain/analysis variable.
- Raw label values are preserved before normalization.
- Missing values and duplicates are flagged without silent repair or deletion.
- Split, fold, federated-client, and augmentation fields are defined but not assigned prematurely.
