# B4 Wav2Vec2 Runtime Compatibility Check

## Result

B4.6 passes. PyPI dependency resolution confirmed that Windows CPython 3.14
wheels exist for Torch 2.14.0 and Transformers 4.57.6. The complete dependency
set resolved without a version conflict against the existing B4 feature
environment, and all pinned packages were subsequently installed successfully
from a normal terminal.

The strict runtime validator imported Torch and `Wav2Vec2Model` successfully.
This machine uses Torch 2.14.0 CPU and reports no CUDA device. No model weights
were downloaded during this checkpoint.

## Pinned runtime

| Component | Version |
| --- | --- |
| Python | 3.14.3 |
| Platform | Windows 11, AMD64 |
| Torch | 2.14.0 |
| Transformers | 4.57.6 |
| huggingface-hub | 0.36.2 |
| safetensors | 0.8.0 |

The Wav2Vec2 and handcrafted-feature dependencies are pinned together in the
unified `requirements.txt` so B1-B6 can be reproduced with one installation.

## Automated runtime gate

Run:

```powershell
.\.venv\Scripts\python.exe src\fedecai\features\validate_wav2vec_runtime.py
```

The validator reports `BLOCKED` when a required distribution is missing or has
a different version. Once all versions match, it imports Torch and
`Wav2Vec2Model`, reports CUDA availability, and returns `PASS`. It never loads
or downloads model weights.

The accepted machine-readable result is stored at
`reports/tables/b4/b4_wav2vec_runtime.json`.

For a strict terminal or CI gate, add `--require-pass` so a blocked runtime
exits with code 1.

## One-sample pilot design

The pilot will use one already reviewed `balanced_12` utterance. It must:

1. load the exact model and repository revisions recorded in the feature spec;
2. require 16 kHz mono float audio and normalized processor input;
3. request an attention mask during right-padding;
4. obtain the final transformer hidden sequence;
5. derive a frame-level mask and exclude padded frames from mean pooling;
6. require a finite 768-value Float32 embedding;
7. write the artifact, SHA-256, input sample ID, model revision, shape, device,
   and elapsed time to a versioned pilot report;
8. reopen the artifact with pickle disabled and reproduce its hash.

The first execution must run on CPU unless the runtime validator confirms CUDA.
It will use a local Hugging Face cache after the single approved checkpoint
download and must support offline reruns. Full extraction remains out of scope.

## Next action

B4.7 implemented and executed the one-sample extractor. Because the runtime is
CPU-only and the checkpoint download is much larger than package metadata, the
first run must remain limited to one reviewed utterance and record download,
load, and inference timing separately.
