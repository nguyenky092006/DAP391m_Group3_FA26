# FedECAI Decision Log

## 2026-09-14 — PROJECT PROTOCOL RESET

### Decision 1 — Primary centralized evaluation

Primary centralized evaluation changed to deterministic sample-level
emotion-stratified train/validation/test splitting.

Consequences:

- `speaker_id` is not a grouping constraint;
- speakers may occur across train/validation/test;
- speaker overlap is diagnostic `INFO` only;
- exact waveform/file duplication across splits remains forbidden;
- old model results are not valid evidence for the new benchmark;
- classical CV uses emotion-stratified sample folds.

### Decision 2 — FedECAI backbone

The FedECAI backbone is not preselected.

Candidate backbones:

- ResNet-18
- MobileNetV3

The human project owner will select the final backbone only after comparable
centralized benchmark performance, accent-robustness, efficiency, stability
and federated-practicality evidence is presented.

```text
FEDECAI_BACKBONE_STATUS = UNDECIDED
FEDECAI_BACKBONE = NONE
HUMAN_DECISION_REQUIRED = YES
FEDERATED_TRAINING_ALLOWED = NO
```

## 2026-09-15 — Temporary centralized performance screening disposition

This entry records experiment scope, not a permanent protocol change.

- Preserve the eight-model official benchmark and all existing baselines.
- Temporarily skip only the ViT performance-oriented fine-tune because of the
  human time constraint; keep ViT source, config, checkpoint, and prior result.
- Treat the five new deep-model results as validation-only evidence under
  `emotion_stratified_with_speaker_overlap`.
- Leave the ResNet-18 versus MobileNetV3 backbone choice to the human owner.
- Keep federated training, GRL, FedECAI, and VESC blocked until an explicit
  subsequent decision.

```text
VIT_FINETUNE_STATUS = SKIPPED_PENDING_HUMAN_DECISION
FEDERATED_BACKBONE_STATUS = WAITING_FOR_HUMAN_DECISION
FEDECAI_STATUS = NOT_STARTED
VESC_STATUS = NOT_STARTED
```

## 2026-09-15 — Final B6 lock and backbone selection

The human project owner permanently closed centralized B6 search/retraining
and selected ResNet-18 after reviewing the saved candidate evidence.

```text
B6_STATUS = LOCKED
CENTRALIZED_HYPERPARAMETER_SEARCH = CLOSED
CENTRALIZED_RETRAINING = DISABLED_BY_HUMAN_DECISION
FEDERATED_BACKBONE_STATUS = SELECTED_RESNET18
FEDERATED_BACKBONE = resnet18_mel3
BACKBONE_TRAINING_POLICY = full_finetune
EFFICIENCY_FALLBACK = mobilenet_v3_mel3
TEST_SET_STATUS = UNTOUCHED
FEDERATED_STAGE = READY_TO_START
FEDECAI_STATUS = NOT_STARTED
```

ResNet-18 is the strongest finalized neural model, still exhibits meaningful
accent disparity, is substantially smaller/faster than Pitch-Fusion, and is
practical for simulated federated experiments. This decision selects a future
backbone but does not authorize or start FL, GRL, FedAvg, FedProx, or FedECAI.

Only the eight experiments in `configs/models/b6_final_lock_v1.json` remain
FINAL_ACTIVE. Superseded evidence is preserved with SHA-256 lineage under
`archive/b6_superseded/20260915_final_b6_v1/`.

## 2026-09-15 — Final ResNet18 FedECAI experiment ladder

The active post-B6 experiment plan is ResNet18 E1-E7. E1 references locked run
`b6_resnet18_mel3_finetune_seed391` and has status
`BASELINE_AVAILABLE_FROM_LOCKED_B6`; it is not a new training run. E2 and E3
are controlled centralized unconditional and emotion-conditioned GRL ablations.
E4-E7 are respectively FedAvg, FedProx, FedAvg + unconditional GRL, and
FedECAI emotion-conditioned GRL with the same retained B5 client contract.

The shared adversarial tap is the 512-dimensional pooled ResNet18 feature before
the emotion FC. Conditional experiments use the model-predicted emotion
posterior with a stop-gradient policy in both training and inference. The fixed
test is excluded from implementation, tuning and selection. Wav2Vec2 E8 is
future work only. At that historical plan-only checkpoint, E2-E7 remained
`NOT_STARTED`; the later execution disposition below supersedes that status.

## 2026-09-15 — E2-E7 validation ladder disposition

The separately authorized ResNet18 E2-E7 implementation and execution is now
complete on the fixed 646-sample validation membership. This execution does
not reopen B6 or change the selected backbone.

```text
E2_STATUS = COMPLETE
E3_STATUS = COMPLETE
E4_STATUS = COMPLETE
E5_STATUS = COMPLETE
E6_STATUS = COMPLETE
E7_STATUS = COMPLETE
FIXED_TEST_STATUS = UNTOUCHED
RQ6_LAMBDA_SWEEP_STATUS = NOT_STARTED
FINAL_LAMBDA_STATUS = NOT_SELECTED
```

E5 currently provides the strongest primary federated validation evidence.
E7 is retained as a valid completed experiment but is not declared superior or
accent-invariant: relative to E4 it has slightly lower utility and worst-accent
performance, a smaller accent gap, and a higher representation accent-probe
score. No further training or fixed-test evaluation follows automatically.
