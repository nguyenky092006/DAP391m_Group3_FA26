# B4.9 Wav2Vec2 Eligible-Chunk Preflight

## Status

PASS on 2026-09-10. The generalized extractor filters the B2 clean manifest to
eligible rows, sorts them numerically by `source_row_index`, and applies offset
and size only after that ordering. This checkpoint ran only offset 0, size 12.

## Selection lineage

| Item | Result |
| --- | --- |
| Eligible rows in manifest | 4,992 |
| Requested offset / size | 0 / 12 |
| Actual rows | 12 |
| Source row range | 0 through 11 |
| Distinct speakers | 9 |
| Accents present | Central, North, South |
| Manifest SHA-256 | `dfde353769f23bbac8aecf1f3d7c3e3211604e7cc8c15c28bdafd055f784704c` |
| Ordered selection SHA-256 | `153fec4090e9b0bf55c651b7ad822a8be34deda2472dbcadb1d36aaa929ef20b` |

This first sequential chunk is not class-balanced: it is an operational
preflight and cannot support feature-distribution or research-question claims.

## First run and replay

The first run loaded the pinned local checkpoint once and produced 12 valid
`float32[768]` embeddings with zero failures. The index was written atomically
after each sample. The immediate replay rechecked lineage, path, SHA-256, shape,
dtype, and finite values for all 12 artifacts.

| Check | First run | Replay |
| --- | ---: | ---: |
| Processed | 12 | 0 |
| Resumed | 0 | 12 |
| Invalidated | 0 | 0 |
| Failures | 0 | 0 |
| Model loaded | Yes | No |

Artifact hashes, manifest hash, and selection hash matched across both runs.

## Evidence

- `reports/tables/b4/b4_wav2vec2_chunk_o000000_n000012_first_run.json`
- `reports/tables/b4/b4_wav2vec2_chunk_o000000_n000012_replay.json`
- `reports/tables/b4/b4_wav2vec2_chunk_o000000_n000012_feature_index.csv`

The 12 embeddings remain under ignored `data/interim/`.

## Next checkpoint

Completed in B4.10. The plan-only coordinator partitions all 4,992 eligible
rows into 20 bounded chunks and reports exact, gap-free, non-overlapping
coverage. See `B4_wav2vec2_full_run_plan.md`.
