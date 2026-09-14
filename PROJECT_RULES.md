# PROJECT_RULES.md

> Compact always-on context for Codex.
> Read this file for every task.
> Do not read the full `PLAN_FEDECAI.md` unless a project-wide rule is changing.

# Project

**FedECAI — Emotion-Conditioned Accent-Invariant Learning for Federated Vietnamese Speech Emotion Recognition**

Dataset: ViSEC  
Planned size: 5,280 samples  
Emotion classes: Angry, Happy, Neutral, Sad  
Accent groups: North, Central, South

# Core Evaluation Rule

**Speaker-disjoint split is mandatory.**

The same speaker must never appear across train/validation/test.

All centralized models must use the same fixed test set.

Never tune using the final test set.

# RQ3 — Required 8 Centralized Models

Exactly:

1. SVM
2. XGBoost
3. CNN 2D
4. ResNet-18 — 3-channel Mel-Spectrogram
5. MobileNetV3 — 3-channel Mel-Spectrogram
6. ViT — 3-channel Mel-Spectrogram
7. Wav2Vec2
8. Pitch-Fusion

# RQ3 — Mandatory Pareto Dimensions

Compare all 8 models using:

- Overall Macro-F1 ↑
- Worst-accent Macro-F1 ↑
- Inference Latency (ms) ↓
- Model Size (MB) ↓

Do not replace this with Accuracy.

# Federated Backbone

Primary candidate:

- ResNet-18

Efficiency alternative:

- MobileNetV3

The final choice must be evidence-based using:

- Overall Macro-F1
- Worst-accent Macro-F1
- latency
- model size
- FL training/communication practicality

# FedECAI Components

Required:

- shared encoder
- emotion classifier
- Gradient Reversal Layer
- accent discriminator
- emotion condition

Required experiment ladder:

- baseline
- unconditional GRL
- emotion-conditioned GRL
- FedAvg
- FedProx
- FedAvg + unconditional GRL
- FedECAI

# Main Metrics

Emotion:

- Macro-F1
- UAR
- per-class metrics

Accent robustness:

- per-accent Macro-F1
- Worst-accent Macro-F1
- accent gap

Accent invariance:

- accent-probe Macro-F1
- accent-probe UAR
- majority baseline
- permutation baseline

Federated:

- Global Macro-F1
- Mean-client Macro-F1
- Worst-client Macro-F1
- communication rounds
- convergence stability

# Non-Negotiable Rules

- No speaker leakage.
- No test-set tuning.
- Preprocessing/scalers fit on training only.
- Augmentation only on training.
- Accent probe must be speaker-disjoint.
- UMAP/t-SNE are illustrative only.
- Do not fabricate dataset statistics.
- Do not fabricate metrics.
- Do not fabricate citations.
- Do not claim FL guarantees privacy.
- Do not claim medical/psychological diagnosis.
- Every experiment must record config, seed, split version, git commit, and outputs.

# Services Architecture

Required B9 stack:

- FastAPI
- Docker
- GCP Cloud Run
- Pub/Sub
- webhook/email alert path
- Dialogflow
- Streamlit or Dash

# Final Report

Exactly 7 top-level sections:

1. Abstract
2. Introduction
3. Methodology
4. Results
5. Discussion
6. Conclusion and Future Work
7. References

No additional top-level sections.

# Context Loading

Read only the relevant step spec:

- B1 → `docs/research/B1_SPEC.md`
- B2 → `docs/research/B2_SPEC.md`
- B3 → `docs/research/B3_SPEC.md`
- B4 → `docs/research/B4_SPEC.md`
- B5 → `docs/research/B5_SPEC.md`
- B6 → `docs/research/B6_SPEC.md`
- B7 → `docs/research/B7_SPEC.md`
- B8 → `docs/research/B8_SPEC.md`
- B9 → `docs/research/B9_SPEC.md`
- B10 → `docs/research/B10_SPEC.md`

Do not read all specs by default.

Read `PLAN_FEDECAI.md` only for project-wide changes or conflicts.

# Documentation

`PLAN_FEDECAI.md` = master specification  
`PROJECT_RULES.md` = compact always-on rules  
`B*_SPEC.md` = step-level context  
`doc.md` = human-readable project documentation  
`logs/*` = evidence/history

# After Each Task

Update only relevant artifacts:

- code/config
- relevant B-step documentation
- `doc.md` if human-visible state changed
- `logs/project_log.md`
- `logs/decision_log.md` if a major decision changed
- `logs/experiment_registry.csv` if an experiment ran
- AI audit log when applicable

Do not rewrite unrelated files.
