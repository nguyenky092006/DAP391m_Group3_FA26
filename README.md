# PVC-FedOrion

Leakage-aware paired-view federated learning for view-robust Vietnamese Speech
Emotion Recognition on VESC.

## Current phase

The repository follows the B1–B10 Data Science workflow. **B1 research framing
was frozen by the project owner on 2026-09-08. B2 is the active step.** No model
should be trained before B5 freezes a speaker-disjoint, pair-safe protocol.

Current B1 artifacts:

- [`docs/research/B1_research_charter.md`](docs/research/B1_research_charter.md)
- [`docs/research/B1_rq_evidence_matrix.csv`](docs/research/B1_rq_evidence_matrix.csv)
- [`docs/research/B1_literature_matrix.csv`](docs/research/B1_literature_matrix.csv)
- [`docs/research/B1_acceptance_checklist.md`](docs/research/B1_acceptance_checklist.md)

The existing raw-manifest utility is preparatory work for B2. Its output must not
be treated as a frozen split or permission to start modeling.

The raw dataset is deliberately excluded from Git. Keep it outside the repository
or under `data/raw/` and pass its paths explicitly to the manifest command.

## Step 1: build and audit the raw-data manifest

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m pvc_fedorion.data.manifest `
  --train-root "C:\path\to\VESC" `
  --test-root "C:\path\to\VESC_test" `
  --output-dir "data\manifests\raw_v1"
```

Generated files:

- `manifest.csv`: one row per WAV file, with canonical utterance, speaker, source,
  view, label, audio-format and checksum fields.
- `audit.json`: aggregate counts, metadata anomalies, pair leakage and speaker
  overlap in the original split.

This command does **not** alter labels, move audio, or silently repair filenames.
Raw anomalies remain visible until the team records a reviewed decision.

## Run tests

The initial test suite uses only the Python standard library:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

See [`docs/00_pre_code_decisions.md`](docs/00_pre_code_decisions.md) for the
scientific decisions that must remain true throughout implementation.
