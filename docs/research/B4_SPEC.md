# B4 — Feature Engineering and Representation Analysis

## Objective

Provide reproducible feature contracts for all eight required model families, including a shared three-channel Mel adapter for ResNet-18, MobileNetV3, and ViT, plus speaker-disjoint accent-probe diagnostics. Preserve valid partner artifacts and avoid using exploratory full-data transforms in model pipelines.

## Technical Details

- Reuse the complete 4,992-row MFCC, eGeMAPS, log-Mel, pitch, and frozen Wav2Vec2 artifacts; do not re-extract them when their lineage remains valid.
- `mel_3ch_mode` is `repeat_normalized_log_mel`: clip existing 80-band log-Mel values to the fixed `[-80, 0] dB` contract, scale to `[0, 1]`, and repeat into channels-first shape `(3, 80, T)`.
- Three-channel Mel is materialized on load, not duplicated on disk. ResNet-18, MobileNetV3, and ViT must use the same adapter.
- Target-frame selection is train-only under canonical `speaker_disjoint_v2`; the shared operation is center crop or right pad with normalized value `0.0` after a target is chosen.
- Accent probes use frozen `handcrafted_174` and `wav2vec2_768` representations. Scalers and linear probes fit on 3,681 training samples/103 speakers and evaluate on 646 validation samples/22 unseen speakers.
- The B4 probe must load zero test samples. It is development evidence only and cannot confirm H1 or replace final probe evaluation.
- Majority-label and training-label permutation baselines are required. Confidence intervals use validation-speaker bootstrap resampling.
- Existing PCA, heatmap, radar, and dashboard remain exploratory only and are not reused as model transforms.

## Metrics

- Feature coverage, shape, dtype, finite-value and lineage checks.
- Three-channel contract coverage and source-frame range.
- Accent-probe validation Macro-F1 and UAR.
- Majority-baseline Macro-F1 and UAR.
- Permutation-baseline Macro-F1/UAR distribution and empirical p-value.
- Speaker-bootstrap 95% confidence intervals for Macro-F1 and UAR.
- Split audit: train/validation speaker overlap and test samples loaded must both be zero.
- No model-performance, final-test, causal, privacy, or diagnosis claim may be inferred from B4 diagnostics.

## Products & Deliverables

- `configs/features/visec_features_v1.json`
- `configs/features/b4_accent_probe_v1.json`
- `src/fedecai/features/mel_3ch.py`
- `src/fedecai/features/validate_b4_mel_3ch.py`
- `src/fedecai/features/run_b4_accent_probe.py`
- B4 unit tests under `tests/`
- `reports/tables/b4/b4_mel_3ch_contract_report.json`
- Accent-probe report, metrics, predictions, permutation samples, and speaker-bootstrap samples under `reports/tables/b4/`
- Existing five-family artifacts, feature table, three advanced figures, and Plotly dashboard
- This consolidated `docs/research/B4_SPEC.md`

## Definition of Done

- Existing five extracted feature families remain complete for 4,992 eligible utterances with valid lineage.
- A documented `mel_3ch_mode` produces finite float32 `(3, 80, T)` inputs consistently for ResNet-18, MobileNetV3, and ViT.
- All 4,992 existing log-Mel artifacts pass the three-channel adapter without re-extraction or duplicate materialization.
- Target-frame selection and every fitted transform are train-only; augmentation remains train-only.
- Speaker-disjoint accent probes run on frozen representations with zero train/validation speaker overlap and zero test samples loaded.
- Majority and permutation baselines plus speaker-bootstrap 95% confidence intervals are saved with predictions and full lineage.
- At least three advanced visualizations and one interactive dashboard remain available with quantitative, RQ-linked insights and exploratory-only boundaries.
- Current feature-table hash/status and environment evidence are consistent across B4 documentation.
- Relevant tests pass, experiment/decision logs are updated, and B4 results are not promoted to final RQ conclusions.

## Verified feature inventory and findings

All 4,992 eligible utterances have 80 MFCC summaries, 88 ordered eGeMAPS
functionals, an 80-band variable-length log-Mel matrix, six pitch summaries plus
contour, and a finite float32[768] frozen Wav2Vec2 embedding. The combined table
has 4,992 rows and 951 columns. The current table hash and detailed lineage are
recorded in `reports/tables/b4/b4_deliverables_report.json`.

The shared image adapter passed 4,992/4,992 log-Mel artifacts, with source widths
101–2,471 frames. It creates `(3,80,T)` tensors on load and stores no duplicate
three-channel corpus.

Development-only accent probes used 3,681 train samples/103 speakers and 646
validation samples/22 unseen speakers, with zero overlap and zero test samples.
Handcrafted-174 achieved validation Macro-F1 0.4257/UAR 0.4937; frozen
Wav2Vec2-768 achieved 0.4905/0.5081; majority baseline was 0.2615/0.3333.
These values are not final test model metrics and do not confirm H1.

The three report figures and dashboard remain under `reports/figures/b4/` and
`reports/dashboard/`. Full-data PCA/standardization is exploratory and cannot be
reused by a model pipeline. Pilot/chunk-level evidence remains machine-readable
under `reports/tables/b4/`; this specification replaces the former narrative
pilot/checklist files. Status: **DONE**.

## B6 cache reuse verified on 2026-09-14

B6 reuses B4 representations without changing the B4 feature contract. The
official frozen Wav2Vec2 runner validated and stacked only the 3,681 train and
646 validation embeddings into
`data/interim/features/b6_frozen_embeddings/wav2vec2_raw/`; its metadata pins
the B4 feature/report hashes, 16 kHz normalization, model revision, masked-mean
pooling and `speaker_disjoint_v2`. Historical B4 encoder-extraction duration is
not available and remains null rather than being reconstructed. ViT similarly
stores a validated train+validation-only frozen-pretrained cache; neither cache
contains the 665 fixed-test samples.
