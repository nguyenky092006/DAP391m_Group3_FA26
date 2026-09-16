# FedECAI Project Log

## 2026-09-14 — Approved project-wide protocol reset

- Replaced the prior global grouping protocol with deterministic sample-level
  emotion-stratified train/validation/test evaluation.
- Created `emotion_stratified_v1` with seed 391 and retained the established
  3,681/646/665 sample counts. All four emotions occur in every split.
- Speaker identity is no longer a grouping constraint. Pairwise speaker overlap
  is 95 train/validation, 102 train/test and 78 validation/test, all `INFO`.
- Cross-split duplicate waveform hashes and duplicate file identities both
  validate at zero.
- Replaced speaker-grouped classical CV with sample-level emotion-stratified CV.
- Rebuilt main/stress federated client assignments from the new training membership.
- Reset old active checkpoints, metrics, predictions, model-selection reports,
  B6 figures, B4 accent-probe results, run logs and experiment rows. Large binary
  artifacts were moved to a recoverable local quarantine after bulk permanent
  deletion was rejected by the execution safety layer.
- Preserved raw ViSEC, raw/clean manifests, B2 quality audits, split-independent
  B4 raw features, pretrained encoder caches, reusable source and dependency files.
- Established ResNet-18 and MobileNetV3 as equal candidates. No backbone is
  selected and adversarial/federated execution remains blocked.
- No model training, benchmark metric generation or backbone selection occurred
  during the protocol-reset action itself.
- The active manifest is deterministic: rerun SHA-256 remained
  `d1ff69387dedfee1a07ec46c7e157b04eb322b32adddf4bef6248b56faa9025f`.
- Secondary accent balancing keeps all 36 split–accent–emotion cells populated.
- The shared Mel adapter revalidated 4,992/4,992 split-independent artifacts.
- Final lightweight verification passed 93/93 repository unit tests.

## 2026-09-14 — Centralized B1–B6 development benchmark

- Verified execution through `DAP301m/Scripts/python.exe` with Torch
  `2.11.0+cu126`, CUDA 12.6 and an NVIDIA GeForce RTX 4060 Laptop GPU.
- Regenerated B3 SQL tables/figures and the B4 accent probe under
  `emotion_stratified_v1`; the Mel adapter validated 4,992/4,992 artifacts.
- Centralized infrastructure preflight returned `READY`; all eight model paths
  passed as `SMOKE_TEST_ONLY` before full training.
- Trained SVM, XGBoost, CNN2D, ResNet-18, MobileNetV3, frozen ViT, raw-waveform
  frozen Wav2Vec2 and paper-aligned Pitch-Fusion. Each produced a checkpoint,
  canonical metrics JSON and exactly 646 validation predictions.
- `reports/tables/b6/b6_all_models_summary.json` returned `PASS`,
  `models_completed=8` and `test_untouched=true`. The complete run took
  6,231.4 seconds wall clock.
- Generated two validation report tables and a no-winner ResNet-18 versus
  MobileNetV3 candidate table. Backbone status is now
  `AWAITING_HUMAN_DECISION`; no GRL, FedAvg, FedProx or FedECAI run occurred.
- Post-run verification passed 93/93 repository unit tests.

## 2026-09-15 — B6 performance-oriented centralized screening

- Preserved all eight existing centralized baseline metrics, predictions,
  configs, and checkpoints; their hashes were recorded in
  `reports/tables/b6_performance/frozen_baseline_audit.csv`.
- Disclosed speaker overlap (95 train/validation, 102 train/test, 78
  validation/test) and labeled this validation screening
  `emotion_stratified_with_speaker_overlap`; fixed test remained untouched.
- Fine-tuned five deep paths under new experiment IDs: CNN2D, ResNet-18,
  MobileNetV3, raw Wav2Vec2, and Pitch-Fusion. ViT fine-tuning was skipped by
  human time constraint without removing its implementation or old evidence.
- Final validation Macro-F1: CNN2D 0.4885, ResNet-18 0.6218, MobileNetV3
  0.5361, Wav2Vec2 0.5245, Pitch-Fusion 0.4574.
- Final consolidated smoke gate passed 5/5. No full training run failed or
  OOMed. Earlier resolved smoke failures remain traceable in `logs/run_index.csv`.
- Generated traceable rankings, delta/resource tables, per-model metrics, and
  `reports/tables/b6_performance/human_decision_packet.md`.
- Verification used DAP301m and passed 97/97 tests with `unittest`. `pytest`
  was not installed in DAP301m and was not added.
- No backbone was selected; no GRL, federated training, FedECAI, or VESC run.

## 2026-09-15 — Controlled Wav2Vec2 and Pitch-Fusion validation reruns

- Fixed the checkpoint-overwrite defect by preventing both performance runners
  from saving best checkpoints during `--smoke-test`; controlled smoke audits
  confirm that checkpoint existence/hash did not change.
- Ran exactly one new Wav2Vec2 controlled experiment with the prior
  initialization policy, upper-four-layer freeze policy, parameter-group LRs,
  head, preprocessing and effective batch; only the budget/cosine horizon was
  extended to 12 epochs. Early stopping ended it at epoch 9; best epoch was 7.
- Ran exactly one new Pitch-Fusion experiment with fully frozen Wav2Vec2,
  constant fusion LR `1e-5`, 10 epochs, patience 3 and effective batch 16; best
  epoch was 7.
- Saved epoch histories, per-class validation metrics and artifact integrity in
  `reports/tables/b6_regression_controls/`. Both prediction files contain 646
  validation samples, both checkpoint hashes match their reports, and both run
  audits record `test_samples_loaded=0`.
- Validation Macro-F1/UAR were 0.536264/0.542677 for Wav2Vec2 and
  0.566057/0.563146 for Pitch-Fusion. Happy F1/recall were 0.200000/0.126984
  and 0.365957/0.341270, respectively.
- Four regression-control contract tests passed with `unittest`. DAP301m does
  not contain `pytest`; no dependency was installed.
- No additional retraining, test evaluation, backbone selection, GRL,
  federated learning, FedAvg, FedProx or FedECAI execution occurred.

## 2026-09-15 — Final B6 lock and safe archive

- Locked eight explicit FINAL_ACTIVE experiment IDs in
  `configs/models/b6_final_lock_v1.json`; centralized B6 runners now refuse
  training/smoke execution.
- Selected `resnet18_mel3` with `full_finetune` for the future federated stage
  and retained `mobilenet_v3_mel3` as the efficiency fallback.
- Safely archived 294 files/snapshots under
  `archive/b6_superseded/20260915_final_b6_v1/`. Each moved artifact was copied,
  SHA-256 verified and only then removed from the active namespace. The archive
  manifest retains original/archived paths, experiment/model, type, hash,
  status, reason and timestamp.
- Active artifact directories now contain exactly eight checkpoints, eight
  metric JSON files and eight 646-row validation prediction files. Final source
  reports and classical CV evidence remain active; obsolete rankings, smoke
  outputs, controlled/non-selected experiments and invalid checkpoint lineage
  are archive-only.
- Generated the canonical 8-model table and final gate evidence in
  `reports/tables/b6_final/`. All deep checkpoint hashes match their source
  reports; SVM/XGBoost hashes are established by the canonical final table.
- Preserved the mandated ResNet-18 metric/cost values exactly. No test sample
  was loaded, no centralized retraining occurred, and no FL/GRL/FedECAI process
  was started.
- Final focused verification passed 18/18 unit tests in DAP301m with
  `PYTHONPATH=src`; the full archive re-hash passed 294/294 entries.
- Generated `reports/figures/b6/final_centralized_b6_ranking.png` directly
  from the locked CSV, including Accent Gap and traceable validation-only labels.

## 2026-09-15 — ResNet18 E1-E7 master-plan reconciliation

- Archived pre-edit snapshots of the master plan, compact rules/specs and old
  FedECAI config with SHA-256 lineage in
  `archive/spec_superseded/20260915_resnet18_ladder_v1/`.
- Reconciled the active master plan and B1/B6/B7/B8/B10 compact specs to one
  official ResNet18 E1-E7 ladder. Removed obsolete current-state wording that
  left the backbone undecided or made E8 active.
- Recorded E1 as the existing locked B6 run
  `b6_resnet18_mel3_finetune_seed391`; no duplicate E1 training is required.
- Specified the actual 512-dimensional pooled pre-FC ResNet18 tap, canonical
  detached predicted-posterior emotion conditioning, controlled E1-E3
  ablations, shared B5 client contract, lambda grid/utility rule and required
  E1-E7 metrics.
- Replaced the old CNN-oriented FedECAI configuration with a non-executable
  `SPEC_ONLY_NOT_IMPLEMENTED` plan contract. Existing runner code remains
  deliberately blocked and is a declared implementation blocker for E2-E7.
- No training, retraining, GRL, FedAvg, FedProx, FedECAI or lambda tuning ran;
  the fixed 665-sample test was not accessed.
- Static plan/config validation passed, all 8 archived snapshot hashes matched,
  and the existing FedECAI helper/config tests passed 3/3 under DAP301m with
  `PYTHONPATH=src`.

## 2026-09-15 — Authorized ResNet18 E2-E7 implementation and validation execution

- Preserved locked B6/E1 and archived four superseded implementation snapshots
  with verified SHA-256 hashes under
  `archive/spec_superseded/20260915_resnet18_e2_e7_v1/`.
- Added the checkpoint-compatible 512-dimensional ResNet18 feature interface,
  unconditional and predicted-posterior conditional adversaries, and analytical
  GRL direction/scale tests. E1 replay matched logits exactly.
- Ran E2-E7 sequentially after smoke and hard gates. All six experiments use
  the same 646 validation IDs, pass checkpoint replay/hash checks, and record
  `test_samples_loaded=0`.
- Primary validation Macro-F1 for E2-E7 was respectively 0.610973, 0.587763,
  0.646339, 0.648969, 0.643101, and 0.642503. Worst-accent Macro-F1 was
  0.455529, 0.543859, 0.552033, 0.588607, 0.406917, and 0.551200.
- E5 was the strongest primary federated validation result. E7 did not
  establish accent invariance versus E4 because its accent-probe score was
  higher despite a smaller accent gap.
- Generated traceable RQ4/RQ5/full-ladder, delta, per-emotion, per-accent,
  accent-probe and resource tables plus the decision packet in
  `reports/tables/fedecai/`.
- Final integrity audit returned `PASS`. The fixed 665-sample test remained
  untouched; RQ6 lambda sweep and final lambda selection were not started.
