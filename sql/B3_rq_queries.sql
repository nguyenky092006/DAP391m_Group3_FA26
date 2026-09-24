-- B3 read-only SQL evidence queries. The executable definitions are also
-- embedded in src/pitchfusion/prepare_foundation.py to keep one runner.

-- 1. Emotion distribution by accent, including a window rank.
WITH counts AS (
  SELECT accent_clean AS accent, emotion, COUNT(*) AS samples
  FROM eligible GROUP BY accent_clean, emotion
)
SELECT *, RANK() OVER (PARTITION BY accent ORDER BY samples DESC) AS frequency_rank
FROM counts ORDER BY accent, frequency_rank;

-- 2. Average duration by emotion.
SELECT emotion, COUNT(*) AS samples, AVG(duration_sec_measured) AS mean_seconds
FROM eligible GROUP BY emotion ORDER BY emotion;

-- 3. Top speakers per accent using CTE and window functions.
WITH counts AS (
  SELECT accent_clean AS accent, speaker_id, COUNT(*) AS samples
  FROM eligible GROUP BY accent_clean, speaker_id
), ranked AS (
  SELECT *, RANK() OVER (PARTITION BY accent ORDER BY samples DESC) AS speaker_rank
  FROM counts
)
SELECT * FROM ranked WHERE speaker_rank <= 5 ORDER BY accent, speaker_rank;

-- 4. Quality by accent and emotion.
SELECT accent_clean AS accent, emotion, COUNT(*) AS samples,
       AVG(duration_sec_measured) AS mean_duration_seconds,
       100.0 * AVG(possible_clipping) AS clipping_percent,
       100.0 * AVG(CASE WHEN high_silence_ratio >= 0.60 THEN 1.0 ELSE 0.0 END) AS high_silence_percent
FROM eligible GROUP BY accent_clean, emotion ORDER BY accent, emotion;

-- 5. Split coverage.
SELECT split, accent_clean AS accent, emotion, COUNT(*) AS samples,
       COUNT(DISTINCT speaker_id) AS speakers
FROM eligible GROUP BY split, accent_clean, emotion ORDER BY split, accent, emotion;
