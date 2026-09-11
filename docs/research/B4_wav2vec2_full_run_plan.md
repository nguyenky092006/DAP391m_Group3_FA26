# B4.10 Wav2Vec2 Full-Run Plan

## Status

PASS on 2026-09-10 in `plan_only` mode. The coordinator read only the B2 clean
manifest. It did not open embedded audio, load the checkpoint, or start
inference.

## Locked plan

| Item | Result |
| --- | --- |
| Eligible rows | 4,992 |
| Chunk size | 256 |
| Chunk count | 20 |
| Full chunks | 19 |
| Final chunk | offset 4,864; 128 samples |
| Unique covered samples | 4,992 |
| Gaps / overlaps | 0 / 0 |
| First / last sample | `visec_hf_000000` / `visec_hf_005279` |
| Manifest SHA-256 | `dfde353769f23bbac8aecf1f3d7c3e3211604e7cc8c15c28bdafd055f784704c` |
| Eligible selection SHA-256 | `ad5c09f67daa2b92626b814ec4ce3a1149f1b5d5c5610fbe50b685e09e3c5ae5` |
| Plan SHA-256 | `06bc28e91464a23e8ce8743d8f6ef40f8480be603f05d81c065cec58003a3cb9` |

Eligible offsets, rather than raw source-row numbers, are contiguous. Source
rows may skip excluded B2 records; this does not indicate a plan gap.

## Storage estimate

The raw 768-value float32 payload for 4,992 embeddings is 15,335,424 bytes.
Using the observed 3,200-byte `.npy` artifact size, the complete embedding set
is estimated at 15,974,400 bytes (about 15.2 MiB), excluding the already cached
checkpoint and small CSV/JSON evidence.

## Execution policy

Run chunks sequentially on CPU. A chunk may advance only after its report has
zero failures. A replay may skip an artifact only when lineage, model revision,
path, SHA-256, shape, dtype, and finite-value checks all pass. The plan CSV
contains the exact command for every chunk.

Tracked evidence:

- `reports/tables/b4/b4_wav2vec2_full_chunk_plan.json`
- `reports/tables/b4/b4_wav2vec2_full_chunk_plan.csv`

## Next checkpoint

Execute planned chunk 0 only: eligible offset 0 and requested size 256. Then
replay that chunk and verify all 256 artifacts without loading the model before
advancing to any later chunk.
