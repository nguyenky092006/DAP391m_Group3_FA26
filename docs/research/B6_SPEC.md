# B6 — Modelling

## Objective

Maintain the locked eight-model centralized benchmark under
`emotion_stratified_v1`. ResNet-18 is selected by explicit human decision for
later adversarial/federated work; MobileNetV3 is the efficiency fallback.

## Technical Details

The exact model list remains SVM, XGBoost, CNN2D, ResNet-18 3-channel Mel,
MobileNetV3 3-channel Mel, ViT 3-channel Mel, raw-waveform frozen Wav2Vec2 and
Pitch-Fusion. Efficient frozen/cached ViT and Wav2Vec2 strategies remain valid,
but caches must be rebuilt for the active membership. Wav2Vec2 deployment
latency includes its encoder.

Classical CV uses emotion-stratified sample folds. All preprocessing and model
fitting are training-only; validation selects configurations/checkpoints; the
fixed 665-sample test is evaluated once after freezing.

Old-split checkpoints, predictions, metrics, summaries and model-selection
outputs are invalid for the new benchmark and cannot be relabelled or reused.

## Backbone Gate

ResNet-18 and MobileNetV3 were originally compared without an automatic winner.
After reviewing performance, accent robustness, cost, stability and FL
practicality evidence, the human selected `resnet18_mel3` with `full_finetune`
on 2026-09-15 and retained MobileNetV3 as the efficiency fallback. This gate is
closed and must not be reopened.

E1 references locked run `b6_resnet18_mel3_finetune_seed391`; it must not
retrain merely to create a new baseline. E2-E7 completed under the separately
authorized ResNet18 execution order. The official
ResNet18 ladder and implementation contracts are defined in
`PLAN_FEDECAI.md`; Wav2Vec2-based E8 is future work only.

## Metrics

- Overall Macro-F1, UAR, weighted-F1, Accuracy and per-class metrics.
- North/Central/South Macro-F1, worst/best accent, gap/std and per-emotion F1.
- Parameter count, deployable model size, training time and full-path latency.
- Four-axis Pareto membership from saved predictions/metrics only.

## Products & Deliverables

- centralized model/config/evaluator/cost/output-contract source
- new checkpoints, predictions, metrics and per-run logs after future training
- eight-model report/Pareto outputs
- archived ResNet-18 versus MobileNetV3 candidate evidence and recorded human decision

## Definition of Done

Eight new-protocol models have traceable identical-validation evidence and
costs; the candidate comparison and explicit human selection are recorded; B6
is locked without opening the fixed test.

Current verified status (2026-09-15): **LOCKED**. Eight FINAL_ACTIVE models
have traceable checkpoints, metrics, and identical 646-row validation
predictions. Superseded evidence is stored in the versioned B6 archive. The
fixed 665-row test remains untouched by explicit human decision.

```text
FEDECAI_BACKBONE_STATUS = SELECTED
FEDECAI_BACKBONE = resnet18_mel3
BACKBONE_TRAINING_POLICY = full_finetune
EFFICIENCY_FALLBACK = mobilenet_v3_mel3
B6_CENTRALIZED_RETRAINING = DISABLED_BY_HUMAN_DECISION
FEDERATED_STAGE = E2_E7_VALIDATION_COMPLETE
FEDECAI_STATUS = VALIDATION_ONLY_COMPLETE
FIXED_TEST_STATUS = UNTOUCHED
RQ6_LAMBDA_SWEEP_STATUS = NOT_STARTED
```

## Performance-oriented validation rerun (2026-09-15)

An isolated screening rerun compared the preserved centralized baselines with
stronger training for CNN2D, ImageNet-pretrained ResNet-18, ImageNet-pretrained
MobileNetV3, raw-waveform Wav2Vec2 (upper four transformer layers trainable),
and raw-waveform plus pitch Pitch-Fusion (upper two transformer layers and the
fusion network trainable). All five final smoke tests passed and all five runs
produced distinct checkpoints, 646-row validation predictions, and canonical
metrics. The fixed 665-sample test was not loaded for selection or reporting.

Validation Macro-F1 was 0.4885 CNN2D, 0.6218 ResNet-18, 0.5361 MobileNetV3,
0.5245 Wav2Vec2, and 0.4574 Pitch-Fusion. Relative to the preserved baseline,
the deltas were -0.0045, +0.0720, +0.1055, -0.0333, and -0.1166 respectively.
These are development results under `emotion_stratified_with_speaker_overlap`,
not unseen-speaker or final-test evidence. ViT fine-tuning was skipped by human
time constraint; its source, config, checkpoint, and existing result remain.

The superseded human-review packet is archived. Canonical active B6 evidence is
`reports/tables/b6_final/centralized_b6_final_locked.csv`, and the decision is
recorded in `reports/tables/b6_final/FINAL_B6_LOCK_REPORT.md`. B6 itself was not
retrained during E2-E7 execution.

## Post-B6 E2-E7 validation execution (2026-09-15)

The separately authorized ResNet18 ladder is complete on validation only.
E2/E3 are centralized GRL ablations; E4-E7 are simulated federated main/stress
experiments. All final predictions contain the identical 646 validation IDs,
checkpoint replay/hash checks pass, and every run records
`test_samples_loaded=0`. See `reports/tables/fedecai/` and
`artifacts/fedecai/final_integrity_audit.json`. This does not reopen B6.
