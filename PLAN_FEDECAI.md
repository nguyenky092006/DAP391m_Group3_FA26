# PLAN_FEDECAI.md

> Master Project Plan — FedECAI
> Protocol reset effective 2026-09-14. This is the single active master
> specification; results tied to an earlier split are not current evidence.

# 0. Context Policy

Read `PROJECT_RULES.md` for every task and only the relevant compact B-step
specification. Read this entire plan only for project-wide changes, RQ changes,
split changes, backbone policy changes or conflicts between steps.

# 1. Project Definition

FedECAI studies emotion-conditioned accent-invariant learning for federated
Vietnamese speech emotion recognition using ViSEC.

Verified modelling population:

- 4,992 B2-eligible, unique-waveform utterances from 5,280 source rows;
- emotions: Angry, Happy, Neutral and Sad;
- accents: North, Central and South;
- 147 released speaker IDs.

## Active centralized evaluation

The primary protocol is deterministic sample-level emotion-stratified
train/validation/test evaluation, version `emotion_stratified_v1`, seed 391.
Speaker identity is not used to group samples. A speaker may occur in more than
one split. Speaker overlap is descriptive only and does not support claims
about generalization to entirely new speakers.

Exact waveform duplication and duplicate file identity across splits remain
forbidden hard failures. The same fixed test sample IDs are used by all eight
centralized models. Test is never used for tuning; preprocessing fitting,
feature selection and augmentation are training-only.

# 2. Research Questions and Hypotheses

## RQ1 — Accent information in representations

How much accent information remains in MFCC/eGeMAPS, log-Mel, pitch-related and
Wav2Vec2 representations under the emotion-stratified evaluation protocol?

Evidence: accent-probe Macro-F1/UAR, majority baseline, permutation-label
baseline and an explicitly labelled sample-level confidence interval. Because
speakers may cross splits, the probe may partially reflect speaker-associated
information and is not a purely speaker-independent accent measurement.

## RQ2 — Performance disparity across accents

How much does emotion-recognition performance differ across Northern, Central
and Southern accents under the same emotion-stratified protocol?

Evidence: overall and per-accent Macro-F1, worst/best accent Macro-F1, accent
gap/std and per-emotion F1 within accent.

## RQ3 — Centralized benchmark and Pareto trade-off

Compare exactly:

1. SVM — MFCC/eGeMAPS
2. XGBoost — MFCC/eGeMAPS
3. CNN 2D — log-Mel
4. ResNet-18 — 3-channel Mel
5. MobileNetV3 — 3-channel Mel
6. ViT — 3-channel Mel
7. Wav2Vec2 — raw waveform, frozen encoder baseline
8. Pitch-Fusion — waveform plus pitch

Primary Pareto dimensions are exactly Overall Macro-F1 ↑, Worst-accent
Macro-F1 ↑, inference latency ↓ and model size ↓. Parameter count and training
time are supporting efficiency metrics.

## RQ4 — Conditional adversarial learning

Using the selected `resnet18_mel3` backbone, compare the locked no-GRL E1
reference, centralized unconditional GRL (E2), and centralized
emotion-conditioned GRL (E3). E2/E3 are controlled ablations against E1.

## RQ5 — Federated FedECAI comparison

Compare ResNet18 FedAvg (E4), ResNet18 FedProx (E5), ResNet18 FedAvg plus
unconditional GRL (E6), and ResNet18 FedECAI with emotion-conditioned GRL (E7)
on the retained B5 main and stress client scenarios.

## RQ6 — Utility versus accent invariance

For the selected ResNet18 ladder, study the utility–invariance trade-off
controlled by `lambda_adv` using development evidence only. Never select
lambda from the fixed test set.

H1–H6 remain preregistered expectations rather than findings: measurable
residual accent information; accent-associated emotion-performance differences;
a non-trivial centralized Pareto frontier; a competitive conditional-GRL
trade-off; a possible federated benefit in at least one scenario; and at least
one useful non-zero lambda setting. No hypothesis is confirmed without valid
new-protocol evidence.

# 3. Project-Wide Rules

1. Primary global evaluation uses deterministic sample-level emotion stratification.
2. `speaker_id` is not a split grouping constraint.
3. Speaker overlap is allowed and reported as `INFO`.
4. Exact duplicate waveform or file identity across splits is forbidden.
5. All centralized models use the same fixed test membership.
6. Test is excluded from tuning and model selection.
7. Fitted preprocessing is training-only and fold-local where CV applies.
8. Augmentation is training-only.
9. Macro-F1 is the primary emotion metric; Accuracy is never the only metric.
10. Per-accent evaluation and Worst-accent Macro-F1 are mandatory.
11. Accent probes use the active split and carry the speaker-associated-information limitation.
12. Statistical resampling uses samples, stratified by emotion where practical.
13. UMAP/t-SNE are illustrative only.
14. Every experiment records config, seed, dataset/split version, commit and outputs.
15. Never fabricate statistics, metrics, costs, citations or conclusions.
16. Federated learning is simulated unless explicitly changed and is not a privacy guarantee.
17. The system is not for psychological or medical diagnosis.
18. Major protocol decisions are recorded in the decision log.

# 4. FedECAI Backbone Decision Contract

Before 2026-09-15, candidate backbones were ResNet-18 and MobileNetV3 and
neither was preselected. This records the original protocol, not current state.

Before comparable centralized evidence:

```text
FEDECAI_BACKBONE_STATUS = UNDECIDED
FEDECAI_BACKBONE = NONE
```

Compare both candidates using Overall Macro-F1, Worst-accent Macro-F1,
per-accent results, accent gap, latency, model size, parameter count, training
time, training/implementation stability and expected FL communication/compute
burden. Do not select with Accuracy alone, highest Macro-F1 alone or an
unapproved weighted score.

After the table exists:

```text
FEDECAI_BACKBONE_STATUS = AWAITING_HUMAN_DECISION
```

Only an explicit human instruction may set `SELECTED` and name the backbone.
The adversarial/federated E1–E7 ladder is prohibited before selection.

Final human decision (2026-09-15):

```text
FEDECAI_BACKBONE_STATUS = SELECTED
FEDECAI_BACKBONE = resnet18_mel3
ACTIVE_FEDECAI_BACKBONE = resnet18_mel3
BACKBONE_TRAINING_POLICY = full_finetune
EFFICIENCY_FALLBACK = mobilenet_v3_mel3
B6_STATUS = LOCKED
```

Centralized B6 hyperparameter search and retraining are closed. The federated
stage is ready to start only under a separate execution instruction; no FL,
GRL or FedECAI run was started by the selection decision.

# 5. B1 — Problem Definition and Research Scope

Define the problem, evidence-based gap, contributions, RQ1–RQ6, H1–H6,
foundational ViSEC/Pitch-Fusion, DANN/GRL, FedAvg and FedProx literature, scope,
owners and evidence mapping. The framing must use the active evaluation
protocol and final ResNet18 backbone contract.

Definition of Done: all RQs/hypotheses and evidence mappings exist; RQ3 lists
eight models and four Pareto dimensions; no unsupported novelty, privacy,
diagnosis or experiment claim exists.

# 6. B2 — Data Understanding and Quality Audit

Verify provenance, schema, labels, speaker metadata, audio readability,
duration, sample rate/channels, clipping/silence, missingness and duplicate
waveforms without modifying raw audio. B2 produces the canonical split-independent
clean manifest. Exact duplicates are resolved before B5.

Definition of Done: source rows remain traceable, exclusions have reasons,
labels are verified and the eligible population contains unique waveforms.

# 7. B3 — EDA and SQL

Describe data composition, accent/emotion imbalance, speaker concentration,
quality context, split distributions and federated-client composition. Retain
the speaker-overlap query as a diagnostic whose non-zero value is expected and
not a pass/fail gate. SQL remains read-only and RQ-linked.

Definition of Done: univariate/bivariate/multivariate evidence, advanced SQL,
quantitative takeaways and correct active-split interpretation exist.

# 8. B4 — Feature Engineering and Representation Analysis

Retain deterministic split-independent MFCC, eGeMAPS, log-Mel, pitch and frozen
Wav2Vec2 extraction. Three-channel Mel uses the shared
`repeat_normalized_log_mel` adapter. Any duration target, scaler, learned
normalization, feature selection or augmentation is fitted from training only.

Accent probes use the active emotion-stratified train/validation membership,
frozen representations, a simple classifier, Macro-F1/UAR, majority and
permutation baselines, and sample-level bootstrap stratified by emotion where
practical. Test is not loaded during development probing.

# 9. B5 — Global Split and Federated Clients

Create `emotion_stratified_v1` from the 4,992 eligible samples with seed 391 and
the established 3,681/646/665 counts. Stratify at sample level with emotion as
the primary target and accent as a secondary joint balancing stratum.
Persist sample identity/path, split, emotion, accent, speaker metadata, seed,
dataset version and split version.

SVM/XGBoost CV uses shuffled deterministic `StratifiedKFold` on training
samples. Speakers are not CV groups. Report unique speaker counts and overlaps
as information only. Hard-fail cross-split duplicate waveform/file identity.

Retain main mixed-accent and stress accent-domain federated scenarios, rebuilt
from the new training membership when necessary. Client statistics may use
speaker metadata descriptively. No validation/test samples enter clients.

Definition of Done: deterministic split and fixed-test hashes validate, all
samples are assigned once, emotion distributions are stratified, duplicate
cross-split count is zero, CV is sample-stratified, and client assignments are
versioned.

# 10. B6 — Modelling

Train the eight centralized models only after new-split preflight and smoke
tests pass. Existing old-split checkpoints, caches, predictions and metrics are
invalid and cannot be relabelled or evaluated as new benchmark evidence.

Current efficient strategies remain: normal SVM/XGBoost/CNN/ResNet/MobileNet
training; frozen pretrained ViT with cache; frozen Wav2Vec2 with cache while
retaining full encoder latency; limited Pitch-Fusion reproduction.

E1 does not trigger training. It references the locked ResNet18 run
`b6_resnet18_mel3_finetune_seed391` and its saved validation predictions and
checkpoint. A compatibility replay, if ever required, must match architecture,
split, seed, full-finetune policy and preprocessing, use no test data, and be
labelled a reproducibility replay rather than a tuned baseline.

Definition of Done after the explicit final lock: eight new-protocol models
have identical 646-sample validation predictions, prediction-derived metrics
and measured costs; human selection is recorded before any E1–E7 run. The
fixed test remains untouched.

## Final E1–E7 Experiment Ladder

### E1 — Centralized ResNet18 baseline

- Backbone: `resnet18_mel3`; training policy: `full_finetune`.
- Reference: `b6_resnet18_mel3_finetune_seed391`.
- No adversarial branch and no duplicate training run.
- `E1_STATUS = BASELINE_AVAILABLE_FROM_LOCKED_B6`.

### E2 — Centralized ResNet18 + unconditional GRL

Use the same ResNet18 backbone and emotion head as E1. The accent adversary
receives the pooled ResNet18 representation through GRL without emotion
conditioning.

### E3 — Centralized ResNet18 + emotion-conditioned GRL

This is the centralized FedECAI mechanism. Let `h` be the 512-dimensional
pooled vector immediately before the ResNet18 emotion FC, let
`p_emotion = softmax(emotion_logits)`, and define
`z = concat(h, stop_gradient(p_emotion))`. The accent discriminator consumes
`GRL(z)`. Training and inference both use the predicted posterior; true emotion
labels are never required for inference conditioning.

### E4 — Federated ResNet18 with FedAvg

Use the same ResNet18 architecture and task head without GRL. This is the main
federated utility baseline.

### E5 — Federated ResNet18 with FedProx

Keep the E4 architecture, clients and schedule; change only federated
optimization to FedProx and log `mu` separately.

### E6 — Federated ResNet18 with FedAvg + unconditional GRL

Keep the E4 client/data scenario and add the E2 unconditional accent adversary.

### E7 — FedECAI

Use federated ResNet18 with the E3 predicted-posterior emotion-conditioned
adversary. Compare directly with E4, E5 and E6.

E8 is not active. Wav2Vec2-based FedECAI is future work only.

```text
E1_STATUS = BASELINE_AVAILABLE_FROM_LOCKED_B6
E2_STATUS = COMPLETE
E3_STATUS = COMPLETE
E4_STATUS = COMPLETE
E5_STATUS = COMPLETE
E6_STATUS = COMPLETE
E7_STATUS = COMPLETE
ACTIVE_EXPERIMENT_LADDER = RESNET18_E1_TO_E7
E8_STATUS = FUTURE_WORK_ONLY
FIXED_TEST_STATUS = UNTOUCHED
RQ6_LAMBDA_SWEEP_STATUS = NOT_STARTED
```

### Shared representation and objective contract

Input is the shared three-channel Mel tensor. ResNet18 produces `h` by global
pooling immediately before the final emotion FC; the actual torchvision
ResNet18 implementation exposes 512 features at this point. The emotion head
and GRL branch both originate from `h`; do not attach the adversary to emotion
logits.

Conceptually, E1 uses `L_total = L_emotion`. E2 uses emotion loss plus the GRL
objective from an unconditional North/Central/South discriminator. E3/E7 use
emotion loss plus the emotion-conditioned GRL objective. The implementation
must follow the sign convention of its tested GRL primitive rather than
hard-coding an unverified sign.

E1–E3 hold constant ResNet18 architecture, Mel representation, emotion head,
train/validation membership, seed 391, optimizer family, budget, augmentation,
batch policy, checkpoint selection and evaluator wherever technically possible.
The intended change is only no adversary versus unconditional versus
emotion-conditioned adversary.

E4–E7 use identical B5 training-only client membership, initial architecture,
client sampling, rounds, local epochs, local batch, baseline optimizer and
evaluation schedule unless the method itself requires a logged difference.
Validation/test samples never enter clients. FedProx `mu` and GRL `lambda_adv`
are logged independently.

### Locked E1 validation reference

```text
Overall Macro-F1 = 0.6218351697860597
UAR = 0.6216308745720511
Accuracy = 0.6439628482972136
North Macro-F1 = 0.5651249841390686
Central Macro-F1 = 0.5184027777777778
South Macro-F1 = 0.6469934829083388
Worst-accent Macro-F1 = 0.5184027777777778
Accent Gap = 0.12859070513056103
```

Central is the lowest-performing and South the highest-performing accent group
in this validation reference; the gap is approximately 0.1286 absolute
Macro-F1. These are descriptive associations, not causal accent claims, and
speaker overlap prevents unseen-speaker generalization claims.

# 11. B7 — Evaluation and Pareto Analysis

For every E1-E7 experiment, compute metrics from saved sample-level predictions:
Overall Macro-F1, UAR, weighted-F1, Accuracy; North/Central/South Macro-F1;
worst/best accent Macro-F1, accent gap and accent standard deviation;
per-emotion F1/recall and per-emotion-within-accent metrics where supported.
Also report the chosen-representation accent-probe Macro-F1/UAR, total and
trainable parameters, model size, training time, inference latency and
communication cost where applicable. FedECAI must not be judged by Overall
Macro-F1 alone. Use sample-level bootstrap, stratified by emotion where
practical, and name the resampling unit accurately.

The comparison stage originally produced a no-winner ResNet-18 versus
MobileNetV3 candidate table. After human review, that table was archived and
the final state is `FEDECAI_BACKBONE_STATUS = SELECTED` with
`FEDECAI_BACKBONE = resnet18_mel3` and MobileNetV3 as efficiency fallback.
This human selection is complete; B7 must not reopen or automatically rescore
the backbone decision.

The authorized E2-E7 validation ladder completed on 2026-09-15. All six runs
use the same 646 validation sample IDs and report `test_samples_loaded=0`.
Canonical comparisons are in `reports/tables/fedecai/`; integrity evidence is
in `artifacts/fedecai/final_integrity_audit.json`. E5 is the strongest primary
federated validation result (Macro-F1 0.648969; Worst-accent Macro-F1 0.588607).
E7 does not establish accent invariance: compared with E4 it has slightly lower
Macro-F1 and Worst-accent Macro-F1, a smaller accent gap, but a higher accent
probe score. These findings are validation-only and do not authorize RQ6 or
fixed-test evaluation.

# 12. B8 — Tuning and Cross-Validation

Classical models use emotion-stratified CV with fold-local preprocessing. Deep
models use the fixed development/validation protocol. Federated tuning uses
development clients only. The initial lambda grid remains 0, 0.05, 0.10, 0.25,
0.50 and 1.00, with the 0.01–0.02 absolute Macro-F1 utility-budget planning
range against the matched non-adversarial baseline. Prefer settings that improve
Worst-accent Macro-F1, reduce Accent Gap and reduce accent-probe performance
without exceeding that utility budget. The backbone gate is satisfied, but no
tuning is authorized by this plan update. Select no lambda from test evidence;
freeze configs before the one-time final test.

# 13. B9 — Pipeline, API and Cloud Services

Deploy a reproducible selected model using FastAPI, Docker, GCP Cloud Run,
Pub/Sub, webhook/email alerts, Dialogflow and Streamlit or Dash. Required API
endpoints are `/health` and `/predict`; validate inputs, do not hard-code
secrets or retain/log raw audio unnecessarily, and do not provide diagnosis.

# 14. B10 — Final Report and Reproducibility

The report methodology must state that centralized evaluation uses a
deterministic sample-level emotion-stratified split and does not group by
speaker. Required limitation: because a speaker may occur across splits,
centralized performance is not evidence of generalization to completely new
speakers.

Explain that ResNet-18 and MobileNetV3 were candidates evaluated under an
identical protocol and that a human selected the final backbone after reviewing
performance and efficiency evidence on 2026-09-15. Report E1 as the locked B6
reference, E2-E7 as separate traceable experiments, and Wav2Vec2-based E8 only
as future work. Do not rewrite the chronology as if ResNet18 had been
preselected before that decision.

The final report has exactly seven top-level sections: Abstract, Introduction,
Methodology, Results, Discussion, Conclusion and Future Work, References.
Every reported number must trace to config, split, seed, prediction and run.

# 15. Artifacts, Logs and Reproducibility

- Raw audio, source metadata, B2 clean manifests, reusable extraction code,
  dependency files and deterministic split-independent raw features are retained.
- Anything fitted, selected, trained or cached using a superseded split is not
  active evidence and must be removed before new benchmarking.
- `logs/experiment_registry.csv` uses the declared schema and contains only
  new-protocol runs.
- `logs/ai_audit_log.md` preserves irreplaceable human AI-use evidence and
  records reset boundaries without reconstructing missing history.
- Benchmark outputs include predictions, metrics, checkpoints and complete run evidence.

# 16. Completion Gates

Data gate: new split deterministic, emotion-stratified, fixed test, duplicate
waveform/file overlap zero; speaker overlap recorded as information.

Benchmark gate: all eight models retrained/evaluated on the new protocol with
saved predictions and cost metrics.

Backbone gate: comparison presented, explicit human selection recorded.

E1 gate: locked B6 run `b6_resnet18_mel3_finetune_seed391` and its saved
validation predictions/checkpoint provide the baseline; no duplicate training
is required.

E2/E3 implementation gate: expose the checkpoint-compatible 512-dimensional
pooled pre-FC ResNet18 representation and reusable emotion head; implement and
test GRL, the unconditional adversary, and canonical predicted-posterior
conditioning; use the shared evaluator/output contract; do not access test.

Federated implementation gate: E4-E7 share the retained B5 training-only client
membership, architecture, initialization policy, client sampling, rounds, local
epochs, local batch, optimizer baseline and evaluation schedule. Validation and
test samples are excluded from clients; method-specific `mu` and `lambda_adv`
are logged separately.

FedECAI completion gate: selected ResNet18 backbone plus completed E1-E7 ladder.
E8 is not part of this gate.

Deployment gate: API, Docker, Cloud Run, Pub/Sub, Dialogflow and UI verified.

Documentation gate: RQ evidence, seven-section report, reproduction guide and
audit trail are complete without fabricated values or contradictory protocol.
