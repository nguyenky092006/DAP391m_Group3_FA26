# B4 Feature and Advanced Visualization Completion Checklist

## Audit result

B4 is complete. The feature contract, full five-family extraction, 4,992-row
feature table, three advanced visualizations with RQ-linked quantitative
insights, and interactive Plotly dashboard all pass. The deterministic PCA
projection fulfills the planned dimensionality-reduction view without adding
a fitted transform to any modelling pipeline; the heatmap and radar fulfill
the other planned visualization roles.

The earlier B3 descriptive figures and B4.3 pilot diagnostics remain evidence
for their stated purposes but are not counted among the three final B4 figures.

## Source requirements

| Source | B4 requirement |
| --- | --- |
| DAP391m Guide, ten-step mapping | B4 is Feature and Advanced Visualization |
| DAP391m Guide, mandatory products | Feature table, at least three advanced visualizations, Plotly or Dash dashboard |
| DAP391m Guide, advanced visualization rubric | Correct chart choice, interactive zoom/filter/drill-down, one insight linked to an RQ for every figure |
| DAP391m Guide, final checklist | Dashboard and at least three advanced visualizations must be present |
| Group Project Planning, week 3 | Extract MFCC/eGeMAPS, log-Mel, pitch, and Wav2Vec2 features |
| Group Project Planning, week 4 | Produce UMAP or t-SNE, heatmap, radar or Pareto visualization, and a dashboard |

## Current evidence

| Requirement | Evidence | Status |
| --- | --- | --- |
| Versioned feature table and extraction contract | `configs/features/visec_features_v1.json`; `B4_feature_specification.md` | Pass |
| Leakage controls documented | Train-fitted normalization, feature selection, dimensionality reduction, duration selection, and augmentation are deferred until the finalized split | Pass |
| MFCC/eGeMAPS/log-Mel/pitch extraction works | 19,968 of 19,968 full-data feature records pass; 78 all-unvoiced pitch records use the documented deterministic sentinel | Pass |
| Feature artifact lineage | Full handcrafted and Wav2Vec2 indexes retain sample/audio/model lineage and artifact SHA-256 | Pass |
| Pilot visual diagnostics | Log-Mel grid, pitch grid, and duration-frame check | Pass for diagnostics |
| Wav2Vec2 representation | Pinned revision produced 4,992 finite 768-value embeddings across 20 validated chunks | Pass |
| Wav2Vec2 runtime | Exact Torch/Transformers packages import successfully on Python 3.14; CPU only, no CUDA | Pass |
| Full eligible-data feature extraction | All five families cover all 4,992 eligible utterances; final table has 951 columns | Pass |
| Three final advanced visualizations | Wav2Vec2 PCA projection, accent-emotion embedding-shift heatmap, and standardized acoustic radar | Pass |
| Quantitative insight linked to an RQ for each final figure | `b4_advanced_visualization_insights.json` records reproducible values and scope | Pass |
| Interactive dashboard | Plotly HTML includes accent/emotion filters, hover detail, and zoom | Pass |

## Completion evidence

See `B4_full_feature_and_visualization_report.md` for final counts, hashes,
quantitative insights, visual-review results, and interpretation limits.

## Dependency on B5 and later model steps

Feature-distribution visualizations may be produced before B5 only when they
are clearly marked exploratory and their fitted projection is never reused by a
model. Any figure that claims unseen-speaker performance, compares model
metrics, selects preprocessing parameters, or performs Pareto model selection
must use the finalized B5 split and later B6-B8 outputs. This boundary prevents
the order of the checklist from introducing evaluation leakage.

## Next step

B4 is closed. Return to the pending B5 version decision before any model
training or evaluation. The B4 exploratory PCA and standardization are not
reused by B5 or later modelling.
