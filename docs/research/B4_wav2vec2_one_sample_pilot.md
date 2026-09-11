# B4.7 Wav2Vec2 One-Sample Pilot

## Status

PASS on 2026-09-10. This checkpoint validates only deterministic feature
extraction for one reviewed utterance on CPU. It does not constitute
full-dataset extraction, model training, or model evaluation.

## Locked inputs

| Item | Value |
| --- | --- |
| Sample | `visec_hf_000111` (source row 111) |
| Speaker / labels | speaker 37 / Central / Angry |
| Duration | 1.984 seconds |
| Audio SHA-256 | `fbb1b077e1869a3d76ce97e575bc2e3e0039a2cf995f122b85b82278b77bc4e1` |
| Model | `nguyenvulebinh/wav2vec2-base-vi` |
| Model revision | `86bde51fa76dd7f2c4c1bb28d7475d622639f869` |
| Runtime | Torch 2.14.0, Transformers 4.57.6, CPU |

The waveform is mono at 16 kHz. The checkpoint feature extractor performs
waveform normalization. The extractor uses the final transformer hidden state
and computes a mean only over frames projected as valid by the input attention
mask.

## Acceptance evidence

| Check | Result |
| --- | --- |
| Input samples | 31,744 |
| Feature frames | 98 total, 98 valid |
| Output | `float32`, shape `[768]` |
| Finite values | 100% |
| Maximum difference between two CPU inferences | 0.0 |
| Saved artifact SHA-256 | `5ce4952cbcd3961cf883b796b994efa7f6491d92e2726a8a28753097514e560e` |
| Offline replay | PASS with the same artifact SHA-256 |

The tracked machine-readable evidence is
`reports/tables/b4/b4_wav2vec2_one_sample_report.json`. The `.npy` vector and
downloaded model remain under ignored `data/interim/` and are not committed.

## Reproduction

After the checkpoint has been downloaded once:

```powershell
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
$env:HF_TOKEN = ''
.\.venv\Scripts\python.exe src\fedecai\features\extract_wav2vec2_pilot.py --local-files-only
```

The environment variables avoid sending an expired implicit Hugging Face token
for this public repository. They affect only the current terminal session.

## Next checkpoint

Completed in B4.8. The three-sample pilot added a versioned index, strict
lineage and SHA-256 checks, atomic writes, resume behavior, and isolated corrupt
artifact recovery tests. See `B4_wav2vec2_resume_pilot.md`.
