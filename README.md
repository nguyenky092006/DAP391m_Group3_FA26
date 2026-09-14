# FedECAI — Vietnamese Speech Emotion Recognition

This repository studies speaker-disjoint Vietnamese speech emotion recognition on ViSEC, including performance differences across North, Central, and South accent groups.

## Current verified state

- B1–B5 are complete and documented in `docs/research/B1_SPEC.md` through `B5_SPEC.md`.
- B6 centralized infrastructure is ready for the fixed 8-model benchmark: SVM, XGBoost, CNN 2D, ResNet-18, MobileNetV3, ViT, raw-waveform frozen Wav2Vec2, and Pitch-Fusion.
- Existing CNN 2D, ResNet-18, MobileNetV3, and ViT outputs are development/validation evidence only; they are not final test results.
- Federated learning, GRL, FedAvg, FedProx, and FedECAI are not the current execution phase.
- No backbone has been selected.

The authoritative rules are `PROJECT_RULES.md`; the master protocol is `PLAN_FEDECAI.md`. Use only the relevant `docs/research/B*_SPEC.md` for a normal step task. `doc.md` is the concise reader-facing status.

## Environment

All benchmark commands must use the existing uv-managed `DAP301m` interpreter. Do not silently fall back to a system Python. The exact environment captured by the latest validation is stored under `logs/runs/` and `artifacts/audit/benchmark_preflight.json`.

From the repository root, resolve and verify the interpreter before running anything:

```powershell
$python = (Resolve-Path '.\DAP301m\Scripts\python.exe').Path
& $python -c "import sys; print(sys.executable); print(sys.version)"
```

If the environment is located elsewhere, use its exact resolved `python.exe` path and record it in the run evidence.

## Reproducible entry points

Data and feature steps:

```powershell
& $python src\fedecai\data\audit_visec.py
& $python src\fedecai\data\clean_visec_manifest.py
& $python src\fedecai\eda\b3_eda_sql.py
& $python src\fedecai\data\run_b5_split_pipeline.py
```

Centralized infrastructure validation only (no expensive benchmark training):

```powershell
& $python scripts\run_centralized_infrastructure_validation.py
```

The hard gate is `artifacts/audit/benchmark_preflight.json`. Full centralized training is allowed only when it reports `CENTRALIZED_INFRASTRUCTURE_STATUS = READY`. Do not start federated work or choose a backbone during this phase.

## Evidence layout

- Machine-readable predictions: `artifacts/predictions/`
- Checkpoints: `artifacts/checkpoints/`
- Metrics: `artifacts/metrics/`
- Current run evidence: `logs/runs/`
- Run registries/history: `logs/run_index.csv`, `logs/experiment_registry.csv`, `logs/infrastructure_run_history.csv`
- Generated tables and figures: `reports/`
- Cleanup decisions and hard-gate audit: `artifacts/audit/`

Raw and generated data are described in `data/README.md`. Never change the frozen split, labels, accent groups, benchmark list, or RQ3 Pareto dimensions without an explicit protocol-change record.
