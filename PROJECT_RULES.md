# PROJECT_RULES.md

> Compact always-on context for Codex. This file is the current project-wide
> source of truth. Read it before every task.

# Project

**FedECAI — Emotion-Conditioned Accent-Invariant Learning for Federated Vietnamese Speech Emotion Recognition**

- Dataset: ViSEC
- Eligible modelling population: 4,992 unique-waveform samples
- Emotion classes: Angry, Happy, Neutral, Sad
- Accent groups: North, Central, South
- Seed: 391 unless an experiment explicitly records another approved seed

# Primary Split

The only active global evaluation protocol is deterministic sample-level
emotion-stratified train/validation/test splitting.

- Split version: `emotion_stratified_v1`
- Manifest: `data/manifests/emotion_stratified_v1.csv`
- Preserve the established 3,681/646/665 train/validation/test sample counts.
- Use emotion as the primary stratification target and `accent_clean` as a
  secondary joint stratum to preserve rare accent/emotion support.
- Use the same fixed test sample IDs for all eight centralized models.
- Never tune, select a checkpoint, fit preprocessing, or fit feature selection on test.
- Fit scalers/preprocessing on training only.
- Apply augmentation on training only.

# Speaker Policy

`speaker_id` is metadata, not a train/validation/test grouping constraint.
Speakers may occur in multiple global splits. Report train/validation/test
speaker counts and pairwise overlap as `INFO`; overlap is not an error and must
not be described as leakage. Do not claim generalization to entirely new speakers.

# Duplicate Policy

Exact waveform duplication or duplicate file identity across global splits is
forbidden and is a hard failure. This is distinct from allowed speaker overlap.

# Cross-Validation

- SVM and XGBoost use deterministic emotion-stratified CV on training samples.
- Every fitted transform stays inside its training fold.
- Validation/test do not participate in CV fitting or tuning.

# Research Questions

- RQ1 measures residual accent information in MFCC/eGeMAPS, log-Mel,
  pitch-related and Wav2Vec2 representations under the active split.
- RQ2 measures emotion-recognition differences across North, Central and South
  under the same protocol.
- RQ3 compares exactly eight centralized models and the four Pareto dimensions.
- RQ4 compares the locked ResNet18 E1 reference with ResNet18 E2 unconditional
  GRL and E3 emotion-conditioned GRL.
- RQ5 compares ResNet18 E4 FedAvg, E5 FedProx, E6 FedAvg plus unconditional
  GRL, and E7 FedECAI.
- RQ6 tunes `lambda_adv` for the selected ResNet18 ladder on development
  evidence only, subject to the matched-baseline utility budget.
- Accent probes may reflect speaker-associated information; never interpret them
  as purely speaker-independent accent measurements.
- Statistical resampling uses sample-level bootstrap, stratified by emotion
  where practical, and must name the bootstrap unit accurately.

# RQ3 — Required Eight Centralized Models

Exactly:

1. SVM — MFCC/eGeMAPS
2. XGBoost — MFCC/eGeMAPS
3. CNN 2D — log-Mel
4. ResNet-18 — 3-channel Mel-Spectrogram
5. MobileNetV3 — 3-channel Mel-Spectrogram
6. ViT — 3-channel Mel-Spectrogram
7. Wav2Vec2 — raw waveform through frozen encoder and classifier
8. Pitch-Fusion — waveform plus pitch

# RQ3 — Mandatory Pareto Dimensions

Compare all eight models using:

- Overall Macro-F1 ↑
- Worst-accent Macro-F1 ↑
- Inference Latency (ms) ↓
- Model Size (MB) ↓

Parameter count and training time are supporting efficiency evidence. Accuracy
must never replace Macro-F1 or be the sole selection metric.

# FedECAI Backbone Policy

```text
CANDIDATE_BACKBONES = ResNet-18, MobileNetV3
FEDECAI_BACKBONE_STATUS = SELECTED
FEDECAI_BACKBONE = resnet18_mel3
ACTIVE_FEDECAI_BACKBONE = resnet18_mel3
BACKBONE_TRAINING_POLICY = full_finetune
EFFICIENCY_FALLBACK = mobilenet_v3_mel3
HUMAN_DECISION_REQUIRED = NO
```

The explicit final human decision on 2026-09-15 selected ResNet-18 and retained
MobileNetV3 as the efficiency fallback after comparable centralized evidence
was reviewed. The decision is recorded in the final B6 lock report.

Historically, comparable metrics first led to `AWAITING_HUMAN_DECISION`; only
the explicit human choice then set `SELECTED`. That chronology is retained but
is not the current state and must not reopen selection.

B6 is permanently locked. Centralized hyperparameter search is closed and all
centralized B6 training/smoke entrypoints must refuse execution. Superseded B6
evidence belongs under `archive/b6_superseded/`; active runners must never scan
that namespace. The fixed test set remains untouched.

# Federated Gate

The official active ladder is exactly:

- E1: locked centralized ResNet18 baseline
  `b6_resnet18_mel3_finetune_seed391`; status
  `BASELINE_AVAILABLE_FROM_LOCKED_B6`; do not retrain.
- E2: centralized ResNet18 plus unconditional GRL.
- E3: centralized ResNet18 plus emotion-conditioned GRL.
- E4: federated ResNet18 with FedAvg and no GRL.
- E5: federated ResNet18 with FedProx and no GRL; only the federated
  optimization method changes relative to E4.
- E6: federated ResNet18 with FedAvg plus unconditional GRL.
- E7: FedECAI, federated ResNet18 plus emotion-conditioned GRL.

E8 is not active. Wav2Vec2-based FedECAI is future work only.

Current verified execution state (2026-09-15): E2-E7 are `COMPLETE` on the
fixed 646-sample validation membership. The fixed 665-sample test is
`UNTOUCHED`; the RQ6 lambda sweep is `NOT_STARTED`. The canonical decision
packet is `reports/tables/fedecai/FEDECAI_E1_E7_DECISION_PACKET.md`. No result
from this ladder may be described as a final-test or unseen-speaker result.

E2/E3 are controlled ablations against E1. Hold constant, wherever technically
possible, the ResNet18/Mel/emotion-head architecture, train/validation
membership, seed 391, optimizer family, budget, augmentation, batch policy,
checkpoint rule and evaluator.

The shared adversarial representation is the actual 512-dimensional pooled
ResNet18 feature immediately before the final emotion FC. The emotion and
adversarial heads branch from that representation. Canonical conditioning is:
`p_emotion = softmax(emotion_logits)`,
`z = concat(h, stop_gradient(p_emotion))`, then
`accent_logits = accent_discriminator(GRL(z))`. Training and inference both use
the predicted posterior; true emotion labels are never required for inference.
The emotion task predicts Angry/Happy/Neutral/Sad and the adversary predicts
North/Central/South. The implemented loss sign must be verified against the GRL
primitive rather than hard-coded from prose.

E4-E7 use identical retained B5 training-only client membership, architecture,
client sampling, rounds, local epochs, local batch, optimizer baseline and
evaluation schedule unless a method requires a logged difference. Validation
and test never enter clients. Federated main/stress partitions, accent skew and
emotion skew remain required. Log FedProx `mu` and GRL `lambda_adv` separately.

The preregistered `lambda_adv` grid is 0, 0.05, 0.10, 0.25, 0.50 and 1.00.
Tune only on development evidence. The planning utility budget is approximately
0.01-0.02 absolute Overall Macro-F1 degradation against the matched baseline,
while preferably increasing Worst-accent Macro-F1, reducing Accent Gap and
reducing accent-probe performance. Never select lambda from test results.

# Metrics and Evidence

- Emotion: Macro-F1, UAR, weighted-F1, accuracy and per-class metrics.
- Accent robustness: per-accent Macro-F1, worst/best accent, gap, std and
  per-emotion F1 within accent.
- Accent probe: Macro-F1, UAR, majority baseline, permutation baseline and an
  explicitly labelled sample-level confidence interval.
- Federated: global, mean-client and worst-client Macro-F1 plus convergence and
  communication evidence.
- E1-E7 additionally require per-emotion F1/recall,
  per-emotion-within-accent where supported, representation accent-probe
  Macro-F1/UAR, total/trainable parameters, model size, training time and
  inference latency.
- UMAP/t-SNE are illustrative only.
- Never fabricate counts, metrics, latency, size, parameters, training time,
  citations or conclusions.
- Do not claim FL guarantees privacy or that the system provides diagnosis.
- Every experiment records config, seed, dataset/split version, git commit and outputs.

# Services and Final Report

- B9 stack: FastAPI, Docker, GCP Cloud Run, Pub/Sub, webhook/email alert path,
  Dialogflow and Streamlit or Dash.
- Final report has exactly seven top-level sections: Abstract, Introduction,
  Methodology, Results, Discussion, Conclusion and Future Work, References.

# Context Loading

- B1–B6: `docs/research/B1_SPEC.md` through `B6_SPEC.md`
- B7: `docs/research/B7_SPEC.md`
- B8: `docs/research/B8_SPEC.md`
- B9: relevant B9 section in `PLAN_FEDECAI.md` until a compact spec exists
- B10: `docs/research/B10_SPEC.md`

Read `PLAN_FEDECAI.md` only for project-wide changes or conflicts.

# Documentation and Logging

- `PLAN_FEDECAI.md`: master specification
- `PROJECT_RULES.md`: compact always-on rules
- `docs/research/B*_SPEC.md`: step-level context
- `doc.md`: verified human-readable current status
- `logs/project_log.md`: current protocol work log
- `logs/decision_log.md`: approved decisions
- `logs/experiment_registry.csv`: experiment registry; never add invented results
- `logs/ai_audit_log.md`: preserve human AI-usage evidence and reset boundaries

After a task, update only relevant code/config, step documentation, `doc.md` if
human-visible state changed, and the applicable logs. Major protocol changes
must be explicit and must not leave contradictory active rules.
