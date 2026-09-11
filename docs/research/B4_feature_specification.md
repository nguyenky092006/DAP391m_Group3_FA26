# B4 ViSEC Feature Specification

## B4.1 status

The feature contract is defined before extraction. This step does not extract audio, choose a fixed duration, fit normalization, or use validation/test data for decisions.

The isolated pilot environment uses Python 3.14 with NumPy 2.5.3, PyArrow
25.0.1, librosa 1.0.0, openSMILE 2.6.0, and Matplotlib 3.11.1. Exact direct
dependencies are recorded in `requirements-feature-pilot.txt`.

## B4.2 pilot status

The 12-sample balanced pilot passed for MFCC, eGeMAPS, log-Mel, and pitch.
It covered all 12 accent-emotion combinations with distinct speakers and
produced 48 of 48 valid artifacts. Saved artifacts and their SHA-256 hashes
were reopened and verified. See `B4_feature_pilot_report.md` for details.

## B4.3 diagnostic status

The balanced pilot passed visual and quantitative diagnostics. Log-Mel and
pitch panels were rendered for every accent-emotion cell. Log-Mel sequence
lengths exactly matched the expected centered-frame calculation for a 10 ms
hop, with zero maximum absolute frame error. These panels test feature
plausibility only; 12 selected utterances cannot support population claims.
See `B4_feature_pilot_diagnostics.md` for details.

## B4.5 Wav2Vec2 specification status

The official Pitch-Fusion repository was reviewed at commit
`7fbd5d88f3fbe864bc12aa7a7b71c6f52e5f5953`. Its standalone baseline loads
`nguyenvulebinh/wav2vec2-base-vi`; the checkpoint is pinned at Hugging Face
revision `86bde51fa76dd7f2c4c1bb28d7475d622639f869`. The project embedding
protocol uses the final 768-dimensional transformer hidden sequence and an
attention-mask-aware mean over valid frames. This project embedding is kept
separate from exact baseline model reproduction. B4.7-B4.9 subsequently
validated one sample, multi-row resume/recovery, and a deterministic 12-row
eligible chunk on CPU. Full extraction has not been run.

## Shared audio contract

| Item | Decision |
| --- | --- |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Numeric type | Float32 |
| Silence trimming | Disabled; B2 quality flags remain diagnostic |
| Peak normalization | Disabled during the pilot |
| Duration | Preserve each utterance's full duration |
| Augmentation | Deferred; train only after B5 |

## Feature families

| Feature family | Pilot definition | Shape before post-split processing | Planned use | Status |
| --- | --- | --- | --- | --- |
| MFCC | 40 coefficients, 25 ms window, 10 ms hop; mean and standard deviation | 80 values | SVM, XGBoost | Ready for pilot |
| eGeMAPS | eGeMAPSv02 Functionals | 88 values | SVM, XGBoost, accent probe | Ready for pilot |
| Log-Mel | 80 Mel bands, 25 ms window, 10 ms hop | 80 x variable frames | CNN 2D | Ready for pilot |
| Pitch | pYIN from 50 to 500 Hz plus voicing and summary statistics; all-unvoiced utterances use zero as a deterministic out-of-range F0 sentinel | Variable sequence plus six summary values | Classical models, Pitch-Fusion | Full extraction recovery in progress |
| Wav2Vec2 | Vietnamese base checkpoint, final hidden sequence, valid-frame masked mean | 768 values | Wav2Vec2, Pitch-Fusion, accent probe | Full extraction validated for 4,992 rows |

## Leakage controls

Deterministic feature extraction that processes each utterance independently may run on all B2-eligible rows. Operations that learn from a population must wait until the final B5 split and fit on train only. These include normalization, feature selection, dimensionality reduction, fixed-duration selection, and augmentation policy decisions.

Every generated feature artifact must be traceable through `feature_index.csv` using sample ID, speaker ID, labels, feature version, relative artifact path, SHA-256, status, and error fields. Generated features remain under `data/interim/` and are not committed to Git.

## B4.1 acceptance criteria

1. The JSON specification passes the validator.
2. All four pilot-ready feature families use 16 kHz mono input.
3. No pilot operation fits statistics from the complete dataset.
4. Fixed-duration padding or cropping remains undecided until B5.
5. Wav2Vec2 advances only after its baseline checkpoint and representation are verified.
6. Pilot extraction and diagnostic review occur before full-dataset extraction.
