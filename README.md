# DAP391m Group 3 - Federated Vietnamese Speech Emotion Recognition

This project studies Vietnamese speech emotion recognition using the ViSEC dataset. The main objective is to evaluate emotion recognition performance on unseen speakers and examine performance differences among Northern, Central, and Southern accents.

The project compares five models: SVM, XGBoost, CNN 2D, Wav2Vec2, and Pitch-Fusion. It also investigates emotion-conditioned accent-invariant learning and federated learning methods.

## Main rules

- All evaluation splits must be speaker-disjoint.
- The test set must remain unchanged after it is finalized.
- Data augmentation is applied only to training data.
- Accent information is used for analysis and domain-invariance experiments, not as the main application output.
- All models must use the same finalized data split for fair comparison.

## Current progress

- B1: research framing and dataset provenance documented.
- B2: record audit, cleaning policy, and audio-quality screening completed.
- B3: metadata EDA and RQ-linked SQL completed with reproducible tables and
  figures. The audit found that frozen split v1 has no Central-Sad test sample.
- B4: complete. All five feature families were extracted for 4,992 eligible
  utterances. The final 4,992-row, 951-column feature table combines 80 MFCC,
  88 eGeMAPS, six pitch-summary, and 768 Wav2Vec2 values plus metadata and
  variable-length artifact paths. Three RQ-linked advanced figures and an
  interactive Plotly dashboard were generated and visually reviewed. All
  generated arrays and the Parquet table remain under ignored `data/interim/`.
- B5: complete. The frozen speaker-disjoint v2 split has full 36/36
  split-accent-emotion coverage. Same-seed reproducibility, preservation of v1,
  grouped five-fold CV, and three accent-domain federated clients all passed.

The reproducible version 2 split runner is available at
`src/fedecai/data/run_b5_split_pipeline.py`. It writes new versioned files,
checks same-seed reproducibility, and verifies that the frozen version 1 files
remain byte-for-byte unchanged. Run it once from the repository root:

```powershell
.\.venv\Scripts\python.exe src\fedecai\data\run_b5_split_pipeline.py
```

## Reproduce B3 EDA and SQL

```powershell
python -m pip install -r requirements-eda.txt
python src/fedecai/eda/b3_eda_sql.py
```

See `docs/research/B3_eda_sql_report.md` and
`notebooks/B3_eda_sql.ipynb` for the findings and review workflow.

## Reproduce B4 pilot diagnostics

```powershell
.\.venv\Scripts\python.exe src\fedecai\features\render_feature_pilot_diagnostics.py --run-name balanced_12
```

The command requires the ignored pilot arrays under
`data/interim/features/b4_features_v1/balanced_12/`. See
`docs/research/B4_feature_pilot_diagnostics.md` for the three tracked figures,
quantitative checks, and interpretation limits.

## Reproduce the B4 Wav2Vec2 one-sample pilot

```powershell
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
$env:HF_TOKEN = ''
.\.venv\Scripts\python.exe src\fedecai\features\extract_wav2vec2_pilot.py --local-files-only
```

The offline command requires the ignored checkpoint cache created by the first
online run. See `docs/research/B4_wav2vec2_one_sample_pilot.md` for the pinned
model, sample lineage, output checks, and scope limits.

## Reproduce the B4 Wav2Vec2 three-sample resume pilot

```powershell
.\.venv\Scripts\python.exe src\fedecai\features\extract_wav2vec2_resume_pilot.py
```

The first run computes any missing or invalid artifact. Repeating the command
reopens and verifies lineage, SHA-256, shape, dtype, and finite values before it
skips a sample. See `docs/research/B4_wav2vec2_resume_pilot.md`.

## Reproduce the B4 Wav2Vec2 eligible-chunk preflight

```powershell
.\.venv\Scripts\python.exe src\fedecai\features\extract_wav2vec2_chunk.py --chunk-offset 0 --chunk-size 12
```

Chunk offsets are applied after filtering the B2 manifest to eligible rows and
sorting numerically by source row. The report records both the complete
manifest SHA-256 and selected-row SHA-256. See
`docs/research/B4_wav2vec2_chunk_preflight.md`.

## Reproduce the B4 Wav2Vec2 full-run plan

```powershell
.\.venv\Scripts\python.exe src\fedecai\features\plan_wav2vec2_full_run.py --chunk-size 256
```

This command reads only the clean manifest and writes a 20-row execution plan.
It does not load audio, the model, or embeddings. See
`docs/research/B4_wav2vec2_full_run_plan.md`.

## Run the complete Wav2Vec2 extraction with one command

```powershell
.\.venv\Scripts\python.exe src\fedecai\features\run_wav2vec2_full_extraction.py
```

The runner executes all 20 planned chunks sequentially, safely resumes partial
runs, validates every chunk, and writes one aggregate 4,992-row feature index.
If the terminal is interrupted, run the same command again.

## Run all remaining B4 deliverables with one command

```powershell
.\.venv\Scripts\python.exe src\fedecai\features\run_b4_remaining_pipeline.py
```

This resume-safe runner extracts full MFCC, eGeMAPS, log-Mel, and pitch
features, joins them with the completed Wav2Vec2 index, writes the full Parquet
feature table, renders three advanced figures, and creates the interactive
Plotly HTML dashboard. If interrupted during extraction, run the same command
again.
