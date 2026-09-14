# B5 — Dataset Partition and Federated Clients

## Objective

Maintain one frozen, versioned, speaker-disjoint ViSEC train/validation/test split and reproducible grouped-CV and federated-client assignments. B5 defines data membership only; it does not train models, tune on test data, or claim federated privacy.

## Technical Details

- Canonical split version: `speaker_disjoint_v2`, seed `391`.
- Eligible population: 4,992 unique-waveform samples from 147 released speaker IDs; 288 B2-excluded rows remain excluded.
- Frozen global membership: train 3,681 samples/103 speakers, validation 646/22, and test 665/22.
- Speaker overlap must be zero for train–validation, train–test, and validation–test.
- All 36 split–accent–emotion cells contain at least one sample and speaker; low-support cells remain documented.
- Grouped CV assigns each training speaker to exactly one of five folds. Validation/test speakers receive no CV fold.
- Federated main scenario: `main_mixed_accent_v1`, three mixed-accent clients.
- Federated stress scenario: `stress_accent_domain_v1`, three single-accent domain clients.
- Every training speaker is assigned exactly once in each scenario; no validation/test speaker is assigned.
- Augmentation and train-fitted preprocessing may operate only on training membership. The fixed test set is never a tuning source.
- Speaker-disjoint claims are limited to the released `speaker_id` metadata. Source documentation does not establish whether ID `0` is one person or a catch-all identifier.

## Metrics

- Samples and distinct speakers by global split.
- Pairwise speaker overlap counts.
- Duplicate eligible waveform-hash count.
- Split–accent–emotion coverage: 36/36 populated cells.
- Grouped-CV fold count and one-fold-per-training-speaker coverage.
- Main/stress client counts, samples, speakers, accents, emotions, and assignment completeness.
- Deterministic speaker-assignment SHA-256 and protected-v1 hash checks.
- No model metric, privacy guarantee, or fabricated statistic is part of B5.

## Products & Deliverables

- `data/manifests/split_manifest_v2.csv`
- `data/manifests/split_manifest_v2_enriched.csv`
- `data/manifests/split_report_v2.json`
- `data/manifests/federated_speaker_partitions_v1.csv`
- B5 summaries under `reports/tables/b5/`
- `src/fedecai/data/create_speaker_split_v2.py`
- `src/fedecai/data/run_b5_split_pipeline.py`
- `src/fedecai/data/build_b5_federated_scenarios.py`
- `tests/test_create_speaker_split_v2.py`
- `tests/test_b5_federated_scenarios.py`
- This consolidated `docs/research/B5_SPEC.md`

## Definition of Done

- Canonical split v2 is versioned, frozen, deterministic, and preserves v1 artifacts unchanged.
- Train/validation/test are speaker-disjoint according to released metadata, and eligible waveform hashes are unique.
- All eligible/excluded row counts, split counts, speaker counts, and 36 cross-stratum cells validate.
- The same fixed test membership is available to every required centralized model and is not used for tuning.
- Five grouped-CV folds cover every training speaker exactly once and exclude non-training speakers.
- Both main and stress federated scenarios exist, are versioned, and assign all training speakers exactly once.
- Per-client sample/speaker/accent/emotion summaries exist and expose the dominant-speaker/client imbalance.
- Augmentation and fitted preprocessing are restricted to training data.
- Source limitation for speaker ID `0` is explicit.
- Relevant tests and read-only artifact validation pass without regenerating or overwriting split files.

## Verified finalization

The frozen v2 split contains train 3,681/103 speakers, validation 646/22 and test
665/22. Same-seed regeneration is deterministic, protected v1 hashes remain
unchanged, all pairwise speaker overlaps equal zero and all 36
split–accent–emotion cells are populated. Five grouped CV folds cover training
speakers only. Three main mixed-accent and three stress accent-domain clients
assign every training speaker exactly once and exclude validation/test speakers.

Authoritative evidence is `split_report_v2.json`,
`reports/tables/b5/b5_pipeline_summary.json`, speaker assignments, grouped-CV
summary and client summary. The old planning/finalization narratives have been
merged here. Status: **DONE**.
