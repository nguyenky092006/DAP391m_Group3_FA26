# B3 — Exploratory Data Analysis and Advanced SQL

## Objective

Provide reproducible descriptive analysis and read-only SQL evidence for ViSEC
composition, accent/emotion imbalance, speaker concentration, data quality,
active split diagnostics and federated-client composition.

## Technical Details

- Canonical input: `data/manifests/emotion_stratified_v1.csv`.
- Federated input: `data/manifests/federated_client_assignments_emotion_stratified_v1.csv`.
- The 4,992 eligible rows are the descriptive modelling population.
- Keep univariate, bivariate and multivariate analysis plus RQ-linked CTE/window SQL.
- Report train/validation/test speaker counts and pairwise overlap, but interpret
  all speaker-overlap values as descriptive `INFO` only.
- Exact cross-split waveform/file duplication is the leakage hard gate.
- Federated tables remain descriptive and are not model results.

## Metrics

- Sample/speaker/accent/emotion and split-cell counts.
- Emotion proportions and ranks within accent.
- Speaker concentration and duration/quality summaries.
- Speaker-overlap diagnostics labelled `INFO`.
- Main/stress client sample, speaker, accent and emotion distributions.
- No fabricated model or causal conclusion.

## Products & Deliverables

- `src/fedecai/eda/b3_eda_sql.py`
- `sql/B3_rq_queries.sql`
- `notebooks/B3_eda_sql.ipynb`
- `tests/test_b3_eda_sql.py`
- regenerated outputs under `reports/tables/b3/` and `reports/figures/b3/`

## Definition of Done

Code/notebook defaults use the active manifest; overlap SQL is diagnostic;
advanced SQL and EDA outputs are regenerated and interpreted under the current
protocol. Status after reset: **PARTIAL** — reusable code is migrated, but
split-dependent B3 generated tables/figures must be regenerated.
