# B3 ViSEC EDA and SQL Report

## Status and scope

B3 metadata EDA and RQ-linked SQL are reproducible from the reviewed manifest. The analysis uses all 4,992 B2-eligible utterances for descriptive statistics. The existing B5 split labels are used only to audit coverage and leakage risk; no model selection or preprocessing parameter is fitted from validation or test data.

## Reproducible outputs

- Notebook: `notebooks/B3_eda_sql.ipynb`
- Pipeline: `src/fedecai/eda/b3_eda_sql.py`
- Named SQL queries: `sql/B3_rq_queries.sql`
- Generated database: `data/processed/visec_metadata.sqlite` (ignored by Git)
- SQL result tables: `reports/tables/b3/`
- EDA figures: `reports/figures/b3/`

Run from the repository root:

    python -m pip install -r requirements-eda.txt
    python src/fedecai/eda/b3_eda_sql.py

## Dataset accounting

| Measure | Result |
| --- | ---: |
| Source rows | 5,280 |
| Eligible rows | 4,992 |
| Excluded rows retained for traceability | 288 |
| Eligible speakers | 147 |
| Median duration | 1.920 s |
| Mean duration | 2.129 s |
| Minimum to maximum duration | 1.000 to 24.704 s |
| Possible clipping flags | 77 (1.54%) |
| High-silence flags | 403 (8.07%) |

The duration distribution is right-skewed. Fixed-length padding or truncation must therefore be selected from the training partition only and reported with a sensitivity check rather than chosen from the full dataset.

## RQ-linked findings

### RQ1 accent information and nuisance structure

Accent sample counts are highly imbalanced: Central has 275 utterances (5.51%), North has 1,183 (23.70%), and South has 3,534 (70.79%). Speaker concentration is also extreme. Speaker 0 contributes 2,217 utterances, equal to 44.41% of all eligible data and 62.73% of the South-accent rows. Within Central, speaker 6 contributes 151 of 275 rows (54.91%).

These results mean that an accent probe can learn speaker-specific or recording-specific structure if the evaluation is not speaker-disjoint. RQ1 results must report speaker counts in addition to utterance counts and should use speaker-aware confidence intervals.

### RQ2 emotion performance by accent

Emotion composition differs across accents. Central contains 42.91% Neutral and 42.55% Happy but only 9.09% Angry and 5.45% Sad. North is more balanced, with class shares between 20.54% and 28.83%. South contains 31.38% Angry, 29.60% Neutral, 22.38% Sad, and 16.64% Happy.

The existing split version has one empty cross-stratum cell: the test set contains no Central-Sad utterance. Test Central-Angry has 12 utterances from only one speaker. Therefore, a four-class Central-accent Macro-F1 and speaker-aware uncertainty estimate would be incomplete or unstable. The frozen version 1 split must not be overwritten silently. If full accent-by-emotion test coverage is required for RQ2, create split version 2 with an explicit coverage constraint and document the reason for superseding version 1 before any model comparison begins.

### Audio-quality context for RQ1 and RQ2

Quality flags are not distributed uniformly. For example, the high-silence rate ranges from 0.00% to 1.49% in the Central and North accent-emotion cells except where noted, while South cells range from 7.46% to 14.43%. North-Sad has a 10.70% possible-clipping rate. These are diagnostic flags, not exclusion rules. Later experiments should retain the current B2 population and use sensitivity analysis before adopting a new cleaning policy.

Median duration also varies across accent-emotion cells: Central-Happy is 3.26 s, while South-Angry is 1.58 s. Duration may therefore act as a nuisance feature. This is descriptive association and does not establish causality.

## SQL query map

| Query | RQ | Purpose | SQL requirement |
| --- | --- | --- | --- |
| `rq2_emotion_distribution_by_accent` | RQ2 | Complete accent-emotion grid, within-accent percentages, and class rank | CTE, cross join, window sum, window rank |
| `rq1_rq2_speaker_concentration` | RQ1 and RQ2 | Top five speakers and their share within each accent | CTE, window sum, window rank |
| `rq1_rq2_quality_by_accent_emotion` | RQ1 and RQ2 | Duration, clipping, and silence summaries by accent and emotion | CTE and grouped aggregation |
| `b5_split_coverage_check` | Evaluation protocol | Full split-accent-emotion grid with zero cells visible | CTE, cross join, window sum, window rank |

## Acceptance result

- Univariate EDA: duration histogram and numeric summary completed.
- Bivariate EDA: within-accent emotion composition completed.
- Multivariate EDA: accent-by-emotion duration heatmap and quality table completed.
- SQL: four RQ-linked queries with CTEs and window functions completed.
- Leakage review: speaker concentration quantified.
- Split coverage review: one missing test cross-stratum identified and not hidden.
- Raw data and B2 cleaning decisions remain unchanged.

B3 is ready for review. The split-coverage decision should be resolved before B5-B8 model evaluation, and B4 advanced visualization should not treat the three descriptive B3 plots as its final required advanced figures.

