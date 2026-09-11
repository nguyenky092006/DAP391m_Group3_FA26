# B4 Full Feature and Advanced Visualization Report

## Final status

B4 completed on 2026-09-11. All 4,992 B2-eligible utterances have complete
handcrafted and Wav2Vec2 feature evidence. The final table, three advanced
figures, their quantitative insights, and the interactive dashboard pass.

## Feature outputs

| Output | Result |
| --- | --- |
| Wav2Vec2 | 4,992 finite `float32[768]` embeddings; 20 chunks; zero failures |
| MFCC | 4,992 records; 80 summary values per utterance |
| eGeMAPS | 4,992 records; 88 ordered Functionals per utterance |
| Log-Mel | 4,992 variable-length 80-band matrices |
| Pitch | 4,992 contours and six-value summaries |
| Handcrafted index | 19,968 of 19,968 clean records |
| Final feature table | 4,992 rows and 951 columns |

Seventy-eight utterances contained no pYIN voiced frame. Their contours retain
unvoiced `NaN` values, while the six-value summary uses a deterministic zero
sentinel: voiced fraction zero and F0 statistics zero Hz. Zero is outside the
configured 50-500 Hz F0 range, requires no population fit, and preserves a
distinguishable all-unvoiced state.

The ignored Parquet table is
`data/interim/features/b4_features_v1/b4_full_feature_table.parquet`, SHA-256
`22c98b9243914ea7c3b05742aeb8a8a7dc0870345c39643226a1688326cfad28`.

## Advanced visualization findings

1. **Wav2Vec2 PCA by accent and emotion.** The first two deterministic
   approximate principal components explain 18.42% and 12.24% of embedding
   variation. Mean emotion-centroid distance divided by mean within-emotion
   distance is 0.442, showing visible structure but substantial class overlap.
2. **Accent-emotion embedding-shift heatmap.** Central-Sad has the largest
   cosine shift from its global emotion centroid, 0.1585. This is exploratory
   and must be interpreted with the imbalanced accent counts: Central 275,
   North 1,183, and South 3,534.
3. **Standardized acoustic radar.** MFCC-1 mean has the largest accent-level
   standardized mean spread, 1.034, among the six displayed acoustic measures.

Machine-readable values are in
`reports/tables/b4/b4_advanced_visualization_insights.json`. These full-data
visual transforms are exploratory only and must not be reused for model
training, preprocessing selection, or evaluation.

## Interactive dashboard

`reports/dashboard/b4_feature_dashboard.html` contains all 4,992 projected
samples and provides accent filtering, emotion filtering, sample/speaker hover
detail, and Plotly zoom. It uses the Plotly CDN and therefore needs internet
access when first opened in a browser.

## Visual review

The three PNG outputs were reviewed at their generated resolution. The PCA
panels have consistent axes and readable legends. Heatmap annotations retain
contrast on both dark and light cells. Radar labels and legend are not clipped
or overlapping. The dashboard source contains its Plotly renderer, both
filters, hover content, and zoom configuration.

## Boundary with B5

No B4 figure reports unseen-speaker accuracy or model performance. B5 must
finalize the speaker-disjoint split version before train-fitted normalization,
model selection, training, or evaluation begins.
