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
- B5: the global speaker-disjoint split was completed early, but it requires a
  documented version decision before modelling because accent-emotion test
  coverage is incomplete. Grouped cross-validation and federated client
  partitions remain future work.

## Reproduce B3 EDA and SQL

```powershell
python -m pip install -r requirements-eda.txt
python src/fedecai/eda/b3_eda_sql.py
```

See `docs/research/B3_eda_sql_report.md` and
`notebooks/B3_eda_sql.ipynb` for the findings and review workflow.
