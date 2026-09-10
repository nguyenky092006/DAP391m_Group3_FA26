# B5.1 Frozen Speaker-Disjoint Split

## Protocol

- Split unit: `speaker_id`, never individual audio files.
- Target sample ratio: 70% train, 15% validation, 15% test.
- Fixed speaker counts: 103 train, 22 validation, 22 test.
- Random seed: `391`.
- Shared split: every model and feature representation must use `split_manifest.csv`.
- Test status: frozen after the validation checks pass.
- Augmentation: may be applied only to rows assigned to train and only after this split.

The optimizer assigns whole speakers while minimizing differences from the target sample proportions for emotion, accent, and gender. It also balances the number of speakers per accent and gender.

## Dominant speaker constraint

Speaker `0` contains 2,217 of the 4,992 eligible samples. It is forced into train because placing it in validation or test would make the evaluation set mostly represent one speaker.

The [official Hugging Face release](https://huggingface.co/datasets/hustep-lab/ViSEC) represents `speaker_id` as integers from 0 to 146 but does not explain whether ID 0 is a single verified person or a catch-all identifier. The [authors' repository](https://github.com/thanhpv2102/ViSEC) also does not document this point. Therefore, this project describes its protocol precisely as **speaker-disjoint according to the released `speaker_id` metadata**. Results must mention this source limitation.

## Required validation

1. No speaker appears in more than one of train, validation, and test.
2. No eligible audio checksum appears more than once.
3. Every split contains all four emotions and all three accents.
4. Speaker counts match the frozen protocol.
5. Rows excluded during B2 cleaning v2 remain `excluded`.
6. The speaker-to-split assignment hash remains unchanged across models.

## Frozen artifacts

- `data/manifests/split_manifest.csv`
- `data/manifests/split_report.json`
- `src/fedecai/data/create_speaker_split.py`

Any future change requires a new split version and a written reason. Do not overwrite version 1.

## Final split result

| Split | Samples | Sample share | Speakers | Central speakers | Northern speakers | Southern speakers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 3547 | 71.05% | 103 | 13 | 41 | 49 |
| Validation | 724 | 14.50% | 22 | 3 | 7 | 12 |
| Test | 721 | 14.44% | 22 | 3 | 8 | 11 |

Every split contains all four emotions and all three accents. Speaker overlap is zero, all eligible audio checksums are unique, and all 288 rows excluded in B2 cleaning v2 remain excluded.

Frozen assignment SHA-256:

`9e4d37b62db267c7748b5d097022d8412aff18427e6d567e6c87323cc35b8fc0`

Running the script twice with seed `391` produced the same assignment and report. B2.2 is complete.
