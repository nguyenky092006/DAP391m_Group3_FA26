# FedECAI — Vietnamese Speech Emotion Recognition

FedECAI studies emotion recognition and accent robustness on ViSEC. The only
active centralized protocol is deterministic sample-level emotion-stratified
train/validation/test evaluation.

## Current protocol

- Dataset version: `visec_clean_eligible_v1`
- Split version: `emotion_stratified_v1`
- Seed: 391
- Train/validation/test: 3,681/646/665 samples
- Speaker identity: metadata only; overlap across splits is allowed and diagnostic
- Cross-split duplicate waveform/file identity: forbidden
- Fixed test membership shared by all eight centralized models
- FedECAI backbone: `resnet18_mel3`, selected by final human B6 decision
- Backbone policy: full fine-tuning; efficiency fallback: `mobilenet_v3_mel3`
- B6 centralized training/search: permanently locked
- ResNet18 E2-E7 validation ladder: complete; fixed test untouched

Only the eight experiment IDs in `configs/models/b6_final_lock_v1.json` are
FINAL_ACTIVE. Superseded evidence is preserved outside the active runner
namespace under `archive/b6_superseded/20260915_final_b6_v1/`.

## Environment

Use the existing uv-managed environment and never silently use global Python:

```powershell
$python = (Resolve-Path '.\DAP301m\Scripts\python.exe').Path
$env:PYTHONPATH = (Resolve-Path '.\src').Path
& $python -c "import sys; print(sys.executable); print(sys.version)"
```

## Lightweight data validation

```powershell
& $python src\fedecai\data\run_b5_split_pipeline.py --seed 391
& $python -m unittest tests.test_emotion_stratified_split tests.test_b5_federated_scenarios -v
```

## B6 lock

Do not invoke centralized B6 training or smoke runners. They enforce the final
lock and refuse execution. The authorized ResNet18 E2-E7 ladder has completed
on validation only. Do not start the RQ6 lambda sweep or fixed-test evaluation
without a separate explicit instruction.

## Evidence layout

- Current split/report: `data/manifests/emotion_stratified_v1.csv` and
  `data/manifests/emotion_stratified_v1_report.json`
- Federated client assignment: `data/manifests/federated_client_assignments_emotion_stratified_v1.csv`
- FINAL_ACTIVE checkpoints/predictions/metrics: `artifacts/`
- Active B6 run evidence: `logs/runs/`
- Final B6 table/report: `reports/tables/b6_final/`
- E1-E7 validation tables/decision packet: `reports/tables/fedecai/`
- E1-E7 run evidence: `logs/runs/fedecai_e1_e7_20260915/`
- Superseded B6 evidence and manifest: `archive/b6_superseded/20260915_final_b6_v1/`
- Current decisions/history: `logs/decision_log.md`, `logs/project_log.md`

Read `PROJECT_RULES.md` before work and the relevant compact B-step spec under
`docs/research/`. Never relabel old results as new-protocol evidence.
