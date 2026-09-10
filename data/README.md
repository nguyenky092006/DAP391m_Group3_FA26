# Data directory

This directory separates immutable source data from generated project data.

## Layout

- `raw/`: the original ViSEC download and extracted files. Do not modify these files.
- `manifests/`: small reviewed CSV or JSON records containing provenance, metadata, checksums, and later split assignments.
- `interim/`: temporary normalized metadata, resampled audio, or features under development.
- `processed/`: finalized processed model inputs.

The pinned ViSEC file is expected at `raw/visec_hf/train-00000-of-00001.parquet`.

Raw audio, archives, processed audio, and large generated features are excluded from Git. A tracked manifest must use stable sample identifiers and repository-safe relative paths rather than relying only on one person's absolute local path.

See `docs/research/B1_dataset_provenance.md` before downloading or moving ViSEC data.

## Reproduce the ViSEC download

Run these commands from the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force data/raw/visec_hf | Out-Null
$datasetUrl = 'https://huggingface.co/datasets/hustep-lab/ViSEC/resolve/06926b7a1dca5f6627b47a2446492be6ff061716/data/train-00000-of-00001.parquet?download=true'
Invoke-WebRequest -Uri $datasetUrl -OutFile 'data/raw/visec_hf/train-00000-of-00001.parquet'
(Get-FileHash -Algorithm SHA256 'data/raw/visec_hf/train-00000-of-00001.parquet').Hash.ToLower()
```

Expected SHA-256:

`bfc7697b3a591cc6cf185c61178815a35363dae4f2c43276636d68eb72cd4a3e`

## Reproduce the B2 quality audit

```powershell
python -m pip install -r requirements-audit.txt
python src/fedecai/data/audit_visec.py
python src/fedecai/data/clean_visec_manifest.py
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

Audit v2 adds signal-quality screening to the original record audit. It creates
`raw_manifest.csv`, `raw_audit.json`, `invalid_audio.csv`,
`duplicate_report.csv`, and `docs/research/B2_quality_report.md`. Clipping and
silence flags are review signals only. Cleaning reads `raw_manifest.csv`
and creates the canonical `clean_manifest.csv` and `cleaning_report.json`.
Version numbers are stored inside the artifacts instead of in filenames.

## Reproduce B3 EDA and SQL

```powershell
python -m pip install -r requirements-eda.txt
python src/fedecai/eda/b3_eda_sql.py
```

The command builds `data/processed/visec_metadata.sqlite`, which is generated
and ignored by Git. It also writes the tracked SQL evidence tables under
`reports/tables/b3/` and figures under `reports/figures/b3/`. B3 uses all
B2-eligible rows for descriptive EDA; split labels are read only for coverage
and leakage checks.
