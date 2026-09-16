# B1 — Problem Definition and Research Scope

## Objective

Define FedECAI, the evidence-based research gap, intended contributions, six
research questions and hypotheses, scope, owners and evidence requirements
under the active sample-level emotion-stratified evaluation protocol.

## Technical Details

- Dataset: ViSEC; four emotions and three accent groups.
- Evaluation: deterministic sample-level emotion-stratified train/validation/test,
  fixed seed and one fixed test membership across all centralized models.
- `speaker_id` is descriptive metadata, not a split grouping constraint.
- RQ1 asks how much accent information remains in MFCC/eGeMAPS, log-Mel,
  pitch-related and Wav2Vec2 representations under the active protocol.
- RQ2 asks how emotion-recognition performance differs among North, Central
  and South under the same protocol.
- RQ3 compares exactly SVM, XGBoost, CNN2D, ResNet-18, MobileNetV3, ViT,
  Wav2Vec2 and Pitch-Fusion using Overall Macro-F1, Worst-accent Macro-F1,
  latency and model size as the four primary Pareto dimensions.
- RQ4 compares the locked ResNet18 E1 baseline, E2 unconditional GRL and E3
  emotion-conditioned GRL as controlled centralized ablations.
- RQ5 compares E4 FedAvg, E5 FedProx, E6 FedAvg + unconditional GRL and E7
  FedECAI, all using the selected ResNet18 architecture.
- RQ6 studies the ResNet18 `lambda_adv` utility-invariance trade-off using
  development evidence only.
- ResNet-18 and MobileNetV3 were originally candidates. The human selected
  `resnet18_mel3` with `full_finetune` on 2026-09-15; MobileNetV3 remains the
  efficiency fallback. Wav2Vec2-based E8 is future work, not active.
- Foundational literature remains ViSEC/Pitch-Fusion, DANN/GRL, FedAvg and FedProx.
- Because speakers may occur across splits, no result may be framed as
  generalization to completely new speakers or purely speaker-independent accent evidence.

H1–H6 are expectations, not findings: residual accent information; differences
among accent groups; a non-trivial RQ3 Pareto frontier; a competitive
emotion-conditioned GRL trade-off; possible FedECAI benefit in at least one
federated scenario; and at least one useful non-zero lambda setting.

## Metrics

- 6/6 RQs and 6/6 hypotheses.
- Exactly eight RQ3 models and four Pareto dimensions.
- Complete RQ-to-method/metric/evidence/decision/owner mapping.
- Zero unsupported novelty, privacy, diagnosis or experiment-result claims.

## Products & Deliverables

- `docs/research/B1_SPEC.md`
- `PROJECT_RULES.md`
- `PLAN_FEDECAI.md`
- `doc.md`
- decision/project logs

## Definition of Done

The problem, gap, intended contributions, RQ1–RQ6, H1–H6, foundational
literature, scope and ownership are explicit; RQ3 has the exact model/Pareto
contract; the active split, completed human backbone decision and ResNet18
E1-E7 ladder are unambiguous.
Status after protocol migration: **DONE**.
