-- B3 SQL evidence queries for the reviewed ViSEC metadata.
-- Every query is read-only, uses B2-eligible records, and is tied to an RQ.

-- name: rq1_rq2_samples_speakers_by_accent
SELECT accent,
       COUNT(*) AS utterances,
       COUNT(DISTINCT speaker_id) AS speakers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_eligible_utterances
FROM eligible_utterances
GROUP BY accent
ORDER BY utterances DESC, accent;

-- name: rq2_emotion_distribution_by_accent
WITH accents AS (
    SELECT 'central' AS accent UNION ALL
    SELECT 'north' UNION ALL
    SELECT 'south'
),
emotions AS (
    SELECT 'angry' AS emotion UNION ALL
    SELECT 'happy' UNION ALL
    SELECT 'neutral' UNION ALL
    SELECT 'sad'
),
grid AS (
    SELECT a.accent, e.emotion FROM accents a CROSS JOIN emotions e
),
counts AS (
    SELECT accent, emotion, COUNT(*) AS utterances
    FROM eligible_utterances
    GROUP BY accent, emotion
),
completed AS (
    SELECT g.accent, g.emotion, COALESCE(c.utterances, 0) AS utterances
    FROM grid g
    LEFT JOIN counts c USING (accent, emotion)
)
SELECT accent,
       emotion,
       utterances,
       ROUND(100.0 * utterances / SUM(utterances) OVER (PARTITION BY accent), 2) AS pct_within_accent,
       RANK() OVER (PARTITION BY accent ORDER BY utterances DESC) AS frequency_rank
FROM completed
ORDER BY accent, frequency_rank, emotion;

-- name: rq1_rq2_speaker_concentration
WITH speaker_counts AS (
    SELECT accent, speaker_id, COUNT(*) AS utterances
    FROM eligible_utterances
    GROUP BY accent, speaker_id
),
ranked AS (
    SELECT accent,
           speaker_id,
           utterances,
           SUM(utterances) OVER (PARTITION BY accent) AS accent_utterances,
           RANK() OVER (PARTITION BY accent ORDER BY utterances DESC) AS speaker_rank
    FROM speaker_counts
)
SELECT accent,
       speaker_id,
       utterances,
       accent_utterances,
       ROUND(100.0 * utterances / accent_utterances, 2) AS pct_of_accent,
       speaker_rank
FROM ranked
WHERE speaker_rank <= 5
ORDER BY accent, speaker_rank, CAST(speaker_id AS INTEGER);

-- name: rq1_rq2_quality_by_accent_emotion
WITH grouped AS (
    SELECT accent,
           emotion,
           COUNT(*) AS utterances,
           AVG(duration_sec) AS mean_duration_sec,
           AVG(silent_frame_ratio) AS mean_silent_frame_ratio,
           SUM(possible_clipping) AS possible_clipping,
           SUM(high_silence) AS high_silence
    FROM eligible_utterances
    GROUP BY accent, emotion
)
SELECT accent,
       emotion,
       utterances,
       ROUND(mean_duration_sec, 3) AS mean_duration_sec,
       ROUND(100.0 * possible_clipping / utterances, 2) AS clipping_pct,
       ROUND(100.0 * high_silence / utterances, 2) AS high_silence_pct,
       ROUND(100.0 * mean_silent_frame_ratio, 2) AS mean_silent_frame_pct
FROM grouped
ORDER BY accent, emotion;

-- name: b5_split_coverage_check
WITH splits(split, split_order) AS (
    VALUES ('train', 1), ('validation', 2), ('test', 3)
),
accents(accent) AS (
    VALUES ('central'), ('north'), ('south')
),
emotions(emotion) AS (
    VALUES ('angry'), ('happy'), ('neutral'), ('sad')
),
grid AS (
    SELECT s.split, s.split_order, a.accent, e.emotion
    FROM splits s CROSS JOIN accents a CROSS JOIN emotions e
),
split_cells AS (
    SELECT split,
           accent,
           emotion,
           COUNT(*) AS utterances,
           COUNT(DISTINCT speaker_id) AS speakers
    FROM eligible_utterances
    GROUP BY split, accent, emotion
),
completed AS (
    SELECT g.split,
           g.split_order,
           g.accent,
           g.emotion,
           COALESCE(c.utterances, 0) AS utterances,
           COALESCE(c.speakers, 0) AS speakers
    FROM grid g
    LEFT JOIN split_cells c
      ON c.split = g.split AND c.accent = g.accent AND c.emotion = g.emotion
),
ranked AS (
    SELECT *,
           SUM(utterances) OVER (PARTITION BY split) AS split_utterances,
           RANK() OVER (PARTITION BY split, accent ORDER BY utterances DESC) AS emotion_rank
    FROM completed
)
SELECT split,
       accent,
       emotion,
       utterances,
       speakers,
       ROUND(100.0 * utterances / split_utterances, 2) AS pct_of_split,
       emotion_rank
FROM ranked
ORDER BY split_order, accent, emotion;

-- Diagnostic only under emotion_stratified_v1. Non-zero speaker overlap is allowed.
-- name: evaluation_speaker_overlap_across_splits
WITH speaker_splits AS (
    SELECT DISTINCT speaker_id, split
    FROM eligible_utterances
    WHERE split IN ('train', 'validation', 'test')
),
split_pairs(left_split, right_split, pair_order) AS (
    VALUES ('train', 'validation', 1),
           ('train', 'test', 2),
           ('validation', 'test', 3)
)
SELECT p.left_split,
       p.right_split,
       COUNT(DISTINCT CASE WHEN r.speaker_id IS NOT NULL THEN l.speaker_id END)
           AS overlapping_speakers
FROM split_pairs p
LEFT JOIN speaker_splits l ON l.split = p.left_split
LEFT JOIN speaker_splits r
  ON r.split = p.right_split AND r.speaker_id = l.speaker_id
GROUP BY p.left_split, p.right_split, p.pair_order
ORDER BY p.pair_order;

-- name: rq5_federated_client_distributions
WITH assignments AS (
    SELECT 'main' AS scenario,
           main_client_id AS client_id,
           speaker_id,
           samples,
           accent
    FROM federated_speaker_partitions
    UNION ALL
    SELECT 'stress' AS scenario,
           stress_client_id AS client_id,
           speaker_id,
           samples,
           accent
    FROM federated_speaker_partitions
),
client_totals AS (
    SELECT scenario,
           client_id,
           COUNT(DISTINCT speaker_id) AS speakers,
           SUM(samples) AS utterances,
           COUNT(DISTINCT accent) AS accents
    FROM assignments
    GROUP BY scenario, client_id
)
SELECT scenario,
       client_id,
       speakers,
       utterances,
       accents,
       ROUND(100.0 * utterances / SUM(utterances) OVER (PARTITION BY scenario), 2)
           AS pct_of_scenario_utterances,
       RANK() OVER (PARTITION BY scenario ORDER BY utterances DESC) AS size_rank
FROM client_totals
ORDER BY scenario, size_rank, client_id;

-- name: rq1_rq2_manifest_quality_join
SELECT m.accent,
       m.emotion,
       COUNT(*) AS utterances,
       ROUND(AVG(q.duration_sec), 3) AS mean_duration_sec,
       SUM(q.possible_clipping) AS possible_clipping,
       SUM(q.high_silence) AS high_silence,
       ROUND(100.0 * AVG(q.silent_frame_ratio), 2) AS mean_silent_frame_pct
FROM manifest_records m
JOIN quality_records q ON q.sample_id = m.sample_id
WHERE m.is_eligible = 1
GROUP BY m.accent, m.emotion
ORDER BY m.accent, m.emotion;
