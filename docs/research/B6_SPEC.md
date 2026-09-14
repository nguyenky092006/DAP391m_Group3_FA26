# B6 Specification — Modelling

## Scope

B6 currently ends at the eight-model centralized benchmark and a human
backbone-selection gate. Model/preprocessing selection uses train/validation;
after configuration is frozen, every model receives one evaluation on the same
665-sample `speaker_disjoint_v2` test membership. No adversarial or federated
implementation/run is authorized in this phase.

The required centralized benchmark contains exactly:

1. SVM;
2. XGBoost;
3. CNN 2D;
4. ResNet-18 over B4 three-channel Mel;
5. MobileNetV3 over the same B4 three-channel Mel;
6. ViT over the same B4 three-channel Mel;
7. Wav2Vec2;
8. Pitch-Fusion.

The legacy frozen-embedding Wav2Vec2 MLP is not the required raw-waveform
Wav2Vec2 benchmark and cannot silently substitute for it. The paper-aligned
Pitch-Fusion boundary must remain explicit.

## Leakage and selection contract

- Model fitting and preprocessing fitting use training speakers only.
- Grouped CV uses the versioned B5 folds and never includes validation/test rows.
- Validation may select a checkpoint/config; test is evaluated exactly once
  afterward and never used for tuning.
- Sample-level fixed-test predictions and all comparison/cost metrics must be
  saved for each model before Pareto/backbone reporting.
- The three image models consume the common B4
  `repeat_normalized_log_mel` adapter. Target frames are selected from the
  current training partition only, followed by shared center-crop/right-pad.

## Current hard stop

Do not implement or run unconditional GRL, emotion-conditioned GRL, FedAvg,
FedProx, federated GRL or FedECAI. ResNet-18 and MobileNetV3 are compared only
as centralized candidates. The report must end with
`BACKBONE_SELECTION_STATUS = WAITING_FOR_HUMAN_DECISION`; only an explicit
human decision may unlock later work. Compact-CNN E1–E7 outputs remain legacy
history and are not current-phase evidence.

## Reproducibility requirements

Every new B6 experiment records config path/hash, seed, split version, git commit,
dirty-worktree state, fit/evaluation partitions, test rows loaded, checkpoint
path/hash and output paths in its report and `logs/experiment_registry.csv`.
No missing historical field may be guessed or backfilled without evidence.

## Definition of Done

- All eight required centralized models either complete or have a persisted,
  explicit technical-failure record; none is silently excluded.
- Every successful model has a checkpoint and sample-level predictions on the
  identical 665 test IDs; reported metrics recompute from predictions.
- ResNet-18 and MobileNetV3 use identical B4 Mel input construction.
- Macro-F1, UAR, weighted-F1, accuracy, class/accent metrics, parameter count,
  checkpoint size, training time and documented inference latency exist.
- Four-axis Pareto and ResNet-18/MobileNetV3 trade-off reports exist without an
  automatic backbone decision.
- All accepted new experiments satisfy the reproducibility record above.
- The phase stops and waits for explicit human backbone selection; no new
  adversarial/federated implementation or run is started.

## Verified status at 2026-09-14

B6 centralized infrastructure is **READY before training**. The exact
uv-managed interpreter is `DAP301m/Scripts/python.exe`. Eight runner smoke paths,
including raw-waveform frozen Wav2Vec2, pass; the shared evaluator, normalized
prediction schema, parameter/size/time/latency protocols, output contract and
report generator are present. `artifacts/audit/benchmark_preflight.json` has no
blocking failures. Torchaudio is absent but is not required by these paths.

The centralized development pass is now complete for all eight models. Each has
one canonical checkpoint, a 646-row sample-level validation prediction file and
a common-schema validation metrics JSON. All eight files use the same fixed
validation IDs, and the 665-sample final test remains unopened.

Current centralized configs include `b6_classical_centralized_v1.json` and
`b6_image_models_v1.json`. Legacy federated config/code is not an authorized
execution path in this phase.

## Consolidated implementation and current evidence

Inputs are fixed as MFCC/eGeMAPS for SVM/XGBoost, log-Mel for CNN2D, the shared
three-channel Mel adapter for ResNet-18/MobileNetV3/ViT, mono 16 kHz waveform
through a fully frozen encoder for official Wav2Vec2, and waveform plus pitch
contour for Pitch-Fusion. The common evaluator derives all metrics from saved
sample predictions. Primary RQ3 latency is median batch-1 model-forward latency;
`model_size_mb` is effective deployable bytes / 1024².

Infrastructure preflight and all 8 smoke paths pass in uv-managed DAP301m.
Current validation Macro-F1 values, recomputed by the common evaluator from
saved predictions, are: SVM 0.4663, XGBoost 0.4236, CNN2D 0.3430, ResNet-18
0.3518, MobileNetV3 0.3702, frozen-pretrained ViT 0.4142, official raw-waveform
frozen Wav2Vec2 0.4193 and Pitch-Fusion 0.6023. These are development values,
not fixed-test benchmark results. The consolidated source-of-truth table is
`artifacts/metrics/b6_centralized_validation_summary.csv`; full class/accent and
cost detail remains in the eight canonical JSON files under
`artifacts/metrics/`.

ViT and Wav2Vec2 train lightweight heads over validated split-specific frozen
embedding caches. Their primary latency still measures the full deployable
inference path, including the frozen encoder, and `model_size_mb` includes the
externally referenced pretrained encoder bytes. Pitch-Fusion reuses the
completed epoch-30 training state, whose best validation checkpoint was epoch
24. FP32 softmax export is required after autocast so saved probabilities pass
the common probability-simplex validation.

B6 remains **PARTIAL** against its Definition of Done: the one-time identical
665-ID final-test evaluation, report-ready tables/figures, four-axis Pareto
analysis and human backbone decision are still absent. No final-test metric,
Pareto front or backbone choice may be inferred from validation.

The former compact-CNN FedECAI model card and modelling-protocol narrative are
merged here. Detailed legacy results are retained only as a compressed archive
and are not authorized current-phase evidence.

`CENTRALIZED_INFRASTRUCTURE_STATUS = READY`  
`BACKBONE_SELECTION_STATUS = WAITING_FOR_HUMAN_DECISION`
