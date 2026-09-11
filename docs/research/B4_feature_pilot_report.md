# B4 ViSEC Feature Pilot Report

## Result

The B4.2 pilot passed on 12 reviewed ViSEC utterances. The selection contains one utterance for every combination of three accents and four emotions, with 12 distinct speakers. All 48 feature artifacts were created, reopened, structurally validated, and matched against their recorded SHA-256 hashes.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.14.3 |
| NumPy | 2.5.3 |
| PyArrow | 25.0.1 |
| librosa | 1.0.0 |
| openSMILE | 2.6.0 |
| Matplotlib | 3.11.1 |

The environment is isolated under `.venv/`. Direct dependency versions are recorded in `requirements-feature-pilot.txt`.

## Pilot coverage

| Measure | Result |
| --- | ---: |
| Selected utterances | 12 |
| Selected speakers | 12 |
| Accent-emotion cells | 12 of 12 |
| Selected rows with clipping or high-silence flags | 0 |
| Minimum duration | 1.600 s |
| Median duration | 2.368 s |
| Mean duration | 3.227 s |
| Maximum duration | 8.000 s |

The pilot selection is a functional coverage sample, not a statistically representative sample and not a model evaluation set.

## Feature validation

| Feature family | Successful | Failed | Observed shape |
| --- | ---: | ---: | --- |
| MFCC summary | 12 | 0 | 80 values |
| eGeMAPSv02 Functionals | 12 | 0 | 88 values |
| Log-Mel spectrogram | 12 | 0 | 80 x 161 to 801 frames |
| Pitch contour | 12 | 0 | 161 to 801 frames plus 6 summary values |

All MFCC, eGeMAPS, and log-Mel values were finite. Pitch contours preserve NaN for unvoiced frames by design; the finite-frame fraction ranged from 0.121 to 0.930, with a mean of approximately 0.557. Every six-value pitch summary was finite.

The variable sequence length is expected because the pilot preserves full utterance duration. No padding or cropping length has been selected. That decision remains deferred until the final B5 split exists and must use training data only.

## Integrity and lineage checks

1. Embedded WAV bytes were loaded only from Parquet row groups containing selected records.
2. Every decoded audio SHA-256 matched `clean_manifest.csv`.
3. Every pilot audio was 16 kHz and mono.
4. Each output records sample ID, speaker, accent, emotion, feature version, artifact path, shape, status, and SHA-256.
5. All 48 saved files were reopened with pickle disabled.
6. All 48 recorded artifact hashes matched the saved files.
7. The eGeMAPS feature-name list contains 88 unique names in a stable order.

## Storage decision

Feature arrays are generated under `data/interim/features/b4_features_v1/` and remain excluded from Git. Small reproducibility evidence is tracked under `reports/tables/b4/`:

- `balanced_12_pilot_manifest.csv`
- `balanced_12_feature_index.csv`
- `balanced_12_pilot_report.json`
- `balanced_12_egemaps_feature_names.json`
- `b4_pilot_diagnostic_report.json`

The complete pilot artifact directory contains 52 files totaling 1,294,324 bytes. The four evidence files do not contain raw audio or feature arrays.

## Scope boundaries

- Wav2Vec2 was not included in this 12-sample handcrafted-feature pilot. Its checkpoint and embedding protocol were subsequently pinned in B4.5 and validated on one reviewed utterance in B4.7.
- This pilot does not perform normalization, feature selection, PCA/UMAP fitting, padding, cropping, augmentation, or modeling.
- B5 split version 1 is not used to select feature parameters.
- Full-dataset extraction must not begin until pilot evidence and the Wav2Vec2 decision are reviewed.

## Acceptance decision

B4.2 pilot extraction passes for MFCC, eGeMAPS, log-Mel, and pitch. B4.3 also
passes: all 12 pilot cells were rendered, reviewed, and checked against the
expected frame-count formula. The diagnostic details and figures are recorded
in `B4_feature_pilot_diagnostics.md`.
