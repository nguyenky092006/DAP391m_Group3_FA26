import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fedecai.features.run_b4_accent_probe import (
    load_config,
    load_development_data,
    sample_bootstrap,
)


class TestB4AccentProbe(unittest.TestCase):
    def test_config_enforces_validation_only_and_sample_bootstrap(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/features/b4_accent_probe_v1.json")
        self.assertEqual(config["fit_split"], "train")
        self.assertEqual(config["evaluation_split"], "validation")
        self.assertEqual(config["forbidden_split"], "test")
        self.assertEqual(config["sample_bootstrap"]["unit"], "validation_sample_emotion_stratified")

    def test_loader_excludes_test_and_accepts_speaker_overlap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data").mkdir()
            manifest = pd.DataFrame(
                [
                    {"sample_id": "a", "speaker_id": "1", "accent": "north", "emotion": "happy", "split": "train"},
                    {"sample_id": "b", "speaker_id": "1", "accent": "south", "emotion": "sad", "split": "validation"},
                    {"sample_id": "c", "speaker_id": "3", "accent": "central", "emotion": "neutral", "split": "test"},
                ]
            )
            manifest["is_eligible_for_split"] = True
            manifest["split_version"] = "emotion_stratified_v1"
            features = pd.DataFrame({"sample_id": ["a", "b", "c"], "feature_1": [1.0, 2.0, 3.0]})
            manifest.to_csv(root / "data/manifest.csv", index=False)
            features.to_parquet(root / "data/features.parquet", index=False)
            config = {
                "split_manifest": "data/manifest.csv",
                "feature_table": "data/features.parquet",
                "split_version": "emotion_stratified_v1",
                "fit_split": "train",
                "evaluation_split": "validation",
            }
            train, validation, audit = load_development_data(root, config)
            self.assertEqual(train["sample_id"].tolist(), ["a"])
            self.assertEqual(validation["sample_id"].tolist(), ["b"])
            self.assertEqual(audit["test_samples_loaded"], 0)
            self.assertEqual(audit["train_validation_speaker_overlap"], 1)
            self.assertEqual(audit["speaker_overlap_status"], "INFO")

    def test_emotion_stratified_sample_bootstrap_is_deterministic(self):
        y_true = np.asarray(["north", "north", "south", "central"])
        y_pred = np.asarray(["north", "south", "south", "central"])
        emotions = np.asarray(["happy", "happy", "sad", "sad"])
        labels = ["central", "north", "south"]
        first = sample_bootstrap(y_true, y_pred, emotions, labels, 20, 391)
        second = sample_bootstrap(y_true, y_pred, emotions, labels, 20, 391)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)


if __name__ == "__main__":
    unittest.main()
