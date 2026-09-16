# B4 — Feature Engineering and Representation Analysis

## Objective

Provide reproducible features for all eight centralized models, one shared
three-channel Mel adapter, and accent-probe diagnostics under the active split.

## Technical Details

- Preserve deterministic split-independent MFCC-40 summaries, eGeMAPSv02,
  log-Mel-80, pYIN pitch and frozen Wav2Vec2 embeddings for all 4,992 eligible samples.
- `mel_3ch_mode=repeat_normalized_log_mel`: fixed `[-80,0]` dB to `[0,1]`,
  repeated to `(3,80,T)` on load without duplicating the corpus.
- Duration/target-frame decisions, scalers and any learned transform are fitted
  from `emotion_stratified_v1` training rows only. Augmentation is training-only.
- Accent probes use active train/validation membership and never load test for
  development selection. Required evidence remains Macro-F1/UAR, majority
  baseline, permutation baseline and sample-level bootstrap stratified by emotion.
- Because speaker overlap is allowed, probe performance may partially reflect
  speaker-associated information and is not purely speaker-independent accent evidence.

## Metrics

- Feature coverage, dimensions, dtype, finite-value and lineage checks.
- Three-channel Mel contract coverage.
- Probe Macro-F1/UAR, majority/permutation baselines and sample-bootstrap CI.
- Speaker overlap is saved as `INFO`; test-loaded count remains a hard boundary.

## Products & Deliverables

- `configs/features/visec_features_v1.json`
- `configs/features/b4_accent_probe_v1.json`
- feature extraction/adapter/probe source under `src/fedecai/features/`
- split-independent feature cache under `data/interim/features/b4_features_v1/`
- new-protocol probe evidence under `reports/tables/b4/`

## Definition of Done

Split-independent extraction and Mel adaptation validate; all fitted decisions
use active training membership; accent probe and uncertainty outputs are
regenerated under `emotion_stratified_v1` with the required limitation.
Status after reset: **PARTIAL** — reusable raw features are preserved, while the
old probe outputs and split-dependent B6 caches were removed and must be rebuilt.
