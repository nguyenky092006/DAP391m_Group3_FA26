# B3 — Exploratory Data Analysis and Advanced SQL

## Objective

Provide reproducible descriptive analysis and read-only SQL evidence for ViSEC data composition, speaker concentration, quality context, speaker-disjoint split integrity, and federated-client composition. B3 describes data and protocol risks; it does not fit model preprocessing, tune models, or establish RQ hypotheses.

## Technical Details

- Canonical input: `data/manifests/split_manifest_v2_enriched.csv` with split version `speaker_disjoint_v2`.
- Federated input: `data/manifests/federated_speaker_partitions_v1.csv`, containing the versioned main mixed-accent and stress accent-domain assignments for training speakers.
- Population: all 5,280 reviewed rows are retained in SQLite for traceability; descriptive EDA uses the 4,992 B2-eligible rows.
- Existing univariate, bivariate, and multivariate EDA figures are retained without regeneration unless explicitly requested.
- Exactly eight named, read-only SQL evidence groups are required:
  1. samples and speakers per accent;
  2. emotion distribution and rank within accent;
  3. speaker concentration using `RANK()`;
  4. quality by accent and emotion;
  5. split–accent–emotion coverage;
  6. speaker overlap for every train/validation/test pair;
  7. main and stress federated-client distributions;
  8. explicit manifest-to-quality JOIN.
- CTEs and window functions must remain visible in the SQL deliverable.
- Query mappings use the current RQ1–RQ6 vocabulary. B3 outputs are descriptive/support evidence and do not replace model, accent-probe, or federated evaluation results.

## Metrics

- Row, speaker, accent, emotion, split-cell, and client counts.
- Within-accent emotion proportions and ranks.
- Per-accent speaker concentration.
- Duration, clipping, and silence summaries.
- Pairwise speaker-overlap count; required value is zero for every split pair.
- Main/stress client utterance share, speaker count, accent count, and size rank.
- SQL completeness: 8/8 named query groups execute successfully.
- No fabricated model metric, causal conclusion, or hypothesis confirmation.

## Products & Deliverables

- `src/fedecai/eda/b3_eda_sql.py`
- `sql/B3_rq_queries.sql`
- `tests/test_b3_eda_sql.py`
- `notebooks/B3_eda_sql.ipynb`
- This consolidated `docs/research/B3_SPEC.md`
- `reports/tables/b3/b3_summary.json`
- Eight CSV query outputs in `reports/tables/b3/`
- Three retained EDA figures in `reports/figures/b3/`

## Definition of Done

- Univariate, bivariate, and multivariate EDA artifacts exist with quantitative takeaways.
- All eight named SQL groups exist, are read-only, map to the current research/evaluation scope, and execute against canonical split v2.
- Samples and distinct speakers are reported per accent.
- Emotion proportions/ranks and speaker concentration are reported with CTE/window SQL.
- Pairwise train/validation/test speaker overlap is checked in SQL and equals zero.
- Both main and stress federated-client distributions are reported from versioned assignments.
- An explicit manifest-to-quality SQL JOIN is present.
- Pipeline defaults and notebook use `split_manifest_v2_enriched.csv`; split v1 evidence is not presented as current.
- B3 tests pass without model training, and no existing completed figure is needlessly regenerated.

## Verified findings

The current run contains 5,280 source rows, 4,992 eligible rows and 147 released
speaker IDs. Eligible accent support is Central 275 (5.51%), North 1,183
(23.70%) and South 3,534 (70.79%). Speaker `0` contributes 2,217 utterances,
so speaker-aware interpretation is mandatory. Median duration is 1.920 s;
77 eligible rows carry possible-clipping flags and 403 high-silence flags.

Canonical split v2 has 36/36 populated split–accent–emotion cells and zero
speaker overlap for train–validation, train–test and validation–test. Some cells
remain extremely small (test Central-Sad: one utterance/one speaker), so nominal
coverage does not imply stable subgroup estimates. Main and stress client tables
describe heterogeneity only and are not federated-model results.

The authoritative generated evidence is `reports/tables/b3/b3_summary.json`,
the eight CSV query outputs, three figures under `reports/figures/b3/`, the SQL
source and `notebooks/B3_eda_sql.ipynb`. Status: **DONE**.
