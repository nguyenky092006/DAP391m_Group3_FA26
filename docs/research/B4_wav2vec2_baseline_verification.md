# B4 Wav2Vec2 Baseline Verification

## Decision

B4.5 verifies and pins the Wav2Vec2 source contract without downloading model
weights. The project will use `nguyenvulebinh/wav2vec2-base-vi` at revision
`86bde51fa76dd7f2c4c1bb28d7475d622639f869`. This is the checkpoint loaded by
the official ViSEC standalone Wav2Vec2 baseline and Pitch-Fusion training
scripts.

The fixed embedding used for RQ1 and feature visualization is defined as the
final transformer hidden sequence followed by an attention-mask-aware mean
over valid frames. Its output dimension is 768. This is a project analysis
protocol. Exact baseline reproduction remains a separate supervised model path
using `Wav2Vec2ForSequenceClassification`.

## Pinned sources

| Source | Pinned reference | Evidence used |
| --- | --- | --- |
| Official ViSEC and Pitch-Fusion repository | `7fbd5d88f3fbe864bc12aa7a7b71c6f52e5f5953` | `ser_pitch_model.py`, `train_no_joint.py`, `train_interpolated_pitch.py` |
| Vietnamese Wav2Vec2 checkpoint | `86bde51fa76dd7f2c4c1bb28d7475d622639f869` | model card, `config.json`, `preprocessor_config.json` |

Primary links:

- <https://github.com/thanhpv2102/ViSEC>
- <https://github.com/thanhpv2102/ViSEC/blob/main/train_no_joint.py>
- <https://github.com/thanhpv2102/ViSEC/blob/main/train_interpolated_pitch.py>
- <https://github.com/thanhpv2102/ViSEC/blob/main/ser_pitch_model.py>
- <https://huggingface.co/nguyenvulebinh/wav2vec2-base-vi>

## Verified model contract

| Item | Verified value |
| --- | --- |
| Architecture | Wav2Vec2 Base |
| Pretraining data reported by model author | 13,000 hours of Vietnamese YouTube audio |
| Model size reported by model author | Approximately 95 million parameters |
| Input sample rate | 16,000 Hz |
| Input channels used by project | Mono |
| Waveform preprocessing | Float waveform with normalization enabled |
| Transformer layers | 12 |
| Hidden dimension | 768 |
| Analysis layer | Final transformer hidden state |
| Analysis pooling | Mean over valid frames selected with an attention mask |
| Analysis embedding dimension | 768 |
| Model license reported by host | CC BY-NC 4.0 |

## Baseline reproduction boundary

The official `train_no_joint.py` constructs
`Wav2Vec2ForSequenceClassification` with four labels, a 256-dimensional
classifier projection, and a frozen convolutional feature extractor. It loads
the Vietnamese checkpoint and pads batches on the right. Those settings define
the supervised baseline path; a standalone pooled embedding is not saved by
the script.

The official script manually creates a feature extractor with normalization
enabled and `return_attention_mask=False`. The checkpoint's current
`preprocessor_config.json` instead records `return_attention_mask=true`. The
official repository also does not pin a Transformers version. Exact historical
padding behavior therefore cannot be inferred safely beyond the checked-in
script. The project will preserve the official model class for baseline
comparison while requiring a valid-frame mask for its separate RQ1 embedding.

## Pitch-Fusion boundary

The custom Pitch-Fusion model reads the final acoustic hidden sequence from its
Wav2Vec2 backbone, combines it with an encoded pitch sequence through
bidirectional cross-attention and self-attention, projects the fused sequence,
and takes an unmasked temporal mean before classification. Its training script
loads audio model weights from the Vietnamese checkpoint but instantiates the
processor from `facebook/wav2vec2-base-100h`.

The project must not describe its masked 768-dimensional embedding as the
official Pitch-Fusion pooled representation. Pitch-Fusion reproduction and the
RQ1 embedding share a backbone checkpoint but have different downstream
representations and purposes.

## Execution state

- No Wav2Vec2 weights were downloaded during B4.5.
- No inference or training was run.
- No GPU or PyTorch compatibility assumption was made.
- The next checkpoint must test the runtime and one sample before any batch or
  full-dataset extraction.
