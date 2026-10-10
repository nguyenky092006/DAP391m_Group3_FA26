from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "visec_staged"
WAV_DIR = DATA_DIR / "wav"
META_CSV = DATA_DIR / "data.csv"
SPLIT_CSV = ROOT / "data" / "splits.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
AUDIO_EXAMPLES = ARTIFACTS / "audio_examples"
CHECKPOINTS = ROOT / "checkpoints"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"
SAMPLE_RATE = 16000
CLIP_SECONDS = 4.5
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
SEED = 2025
LABELS = ["happy", "neutral", "sad", "angry"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
TRANSFORMS = {
    "T0_original": (), "T1_pitch": ("pitch",), "T2_energy": ("energy",),
    "T3_duration": ("duration",), "T4_pitch_energy": ("pitch", "energy"),
    "T5_pitch_duration": ("pitch", "duration"),
    "T6_energy_duration": ("energy", "duration"),
    "T7_all": ("pitch", "energy", "duration"),
}

def ensure_dirs():
    for path in (ARTIFACTS, FIGURES, AUDIO_EXAMPLES, CHECKPOINTS, DOCS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
