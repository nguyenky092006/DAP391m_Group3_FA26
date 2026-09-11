# B4.8 Wav2Vec2 Three-Sample Resume Pilot

## Status

PASS on 2026-09-10. The run is deliberately limited to three reviewed Angry
utterances: one Central, one North, and one South, all from distinct speakers.
It validates resumability and failure detection, not population extraction or
model performance.

## Samples and first run

| Sample | Speaker | Accent | Valid frames | Artifact SHA-256 |
| --- | ---: | --- | ---: | --- |
| `visec_hf_000111` | 37 | Central | 98 | `5ce4952cbcd3961cf883b796b994efa7f6491d92e2726a8a28753097514e560e` |
| `visec_hf_000017` | 12 | North | 332 | `16785213f296ebf4485b0f1e167c683910881012c0b6c6cd047ccc9944c79587` |
| `visec_hf_000009` | 0 | South | 95 | `0414ea0a4b9a9e41cd5076c716c64416dcad49f89cedb7d6abd1572f4ae95fc7` |

The first run loaded the pinned model once from the local cache, processed all
three samples, atomically saved each embedding, reopened it, and atomically
updated the versioned feature index after every sample. Result: three
processed, zero resumed, zero invalidated, and zero failures.

## Resume and recovery checks

The immediate replay checked sample/audio/model lineage, fixed artifact path,
SHA-256, shape `[768]`, `float32` dtype, and finite values. All three artifacts
passed, so the replay processed zero, resumed three, and did not load the model.
All three hashes matched the first run.

Unit tests use temporary directories to verify two failure paths without
damaging real evidence:

1. Replacing an indexed artifact with different bytes produces
   `artifact_sha256_mismatch` and forces recomputation.
2. Changing the pinned model revision produces
   `lineage_mismatch:model_revision` and forces recomputation.

## Tracked and ignored outputs

Tracked evidence:

- `reports/tables/b4/b4_wav2vec2_resume3_first_run.json`
- `reports/tables/b4/b4_wav2vec2_resume3_replay.json`
- `reports/tables/b4/b4_wav2vec2_resume3_feature_index.csv`

The three `.npy` embeddings and the checkpoint remain under ignored
`data/interim/`.

## Next checkpoint

Completed in B4.9. The chunk runner filters the B2 manifest to eligible rows,
sorts by numeric source row, records manifest and selection hashes, and passed
a 12-row run plus replay. See `B4_wav2vec2_chunk_preflight.md`.
