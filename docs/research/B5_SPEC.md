# B5 — Dataset Partition and Federated Clients

## Objective

Maintain one deterministic sample-level emotion-stratified global split,
sample-stratified classical CV folds and versioned main/stress federated clients.

## Technical Details

- Split version: `emotion_stratified_v1`; seed 391.
- Eligible population: 4,992 unique-waveform samples.
- Preserve established counts: train 3,681, validation 646, test 665.
- Stratify train/validation/test with emotion as the primary target and accent
  as a secondary joint balancing stratum; do not group by speaker.
- `speaker_id` remains metadata. Pairwise overlap is allowed and reported as `INFO`.
- Hard-fail exact waveform or file identity duplicated across splits.
- Classical CV uses deterministic shuffled `StratifiedKFold` by emotion on
  training samples. Fitted preprocessing remains fold-local.
- Fixed test IDs are shared by all eight models and never used for tuning.
- Main mixed-accent and stress accent-domain clients use training samples only;
  speaker metadata may be used descriptively inside client statistics.

## Metrics

- Dataset and split totals; emotion/accent counts by split.
- Unique speakers by split and pairwise overlaps as `INFO`.
- Cross-split duplicate waveform/file identity counts.
- Assignment and fixed-test SHA-256 hashes.
- CV and federated-client coverage/version checks.

## Products & Deliverables

- `src/fedecai/data/create_emotion_stratified_split.py`
- `src/fedecai/data/run_b5_split_pipeline.py`
- `src/fedecai/data/build_b5_federated_scenarios.py`
- `data/manifests/emotion_stratified_v1.csv`
- `data/manifests/emotion_stratified_v1_report.json`
- `data/manifests/federated_client_assignments_emotion_stratified_v1.csv`
- B5 tests and compact validation/client summaries

## Definition of Done

Every eligible sample is assigned once; reruns are deterministic; emotion
stratification and fixed-test hash validate; cross-split duplicate hashes/file
identities equal zero; speaker overlap is informational; sample-level CV and
new-training-membership federated clients are persisted. Status is determined
only by the generated validation evidence.

## Verified migration result

`emotion_stratified_v1_report.json` records PASS: 4,992 eligible samples,
3,681/646/665 split counts, all four emotions in every split, 147 total released
speaker IDs, speaker overlaps 95/102/78 as `INFO`, and zero cross-split waveform
hash or file-identity duplicates. Assignment and test-membership hashes are
persisted. Main/stress v2 client scenarios were rebuilt from the 3,681 training
samples and passed their coverage checks. Status: **DONE**.
