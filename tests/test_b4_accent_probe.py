import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fedecai.features.run_b4_accent_probe import (
    load_config,
    load_development_data,
    speaker_bootstrap,
)


class TestB4AccentProbe(unittest.TestCase):
    def test_config_enforces_validation_only_and_speaker_bootstrap(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/features/b4_accent_probe_v1.json")
        self.assertEqual(config["fit_split"], "train")
        self.assertEqual(config["evaluation_split"], "validation")
        self.assertEqual(config["forbidden_split"], "test")
        self.assertEqual(config["speaker_bootstrap"]["unit"], "validation_speaker")

    def test_loader_excludes_test_and_enforces_speaker_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data").mkdir()
            manifest = pd.DataFrame(
                [
                    {"sample_id": "a", "speaker_id": "1", "accent": "north", "split": "train"},
                    {"sample_id": "b", "speaker_id": "2", "accent": "south", "split": "validation"},
                    {"sample_id": "c", "speaker_id": "3", "accent": "central", "split": "test"},
                ]
            )
            manifest["is_eligible_for_split"] = True
            manifest["split_version"] = "speaker_disjoint_v2"
            features = pd.DataFrame({"sample_id": ["a", "b", "c"], "feature_1": [1.0, 2.0, 3.0]})
            manifest.to_csv(root / "data/manifest.csv", index=False)
            features.to_parquet(root / "data/features.parquet", index=False)
            config = {
                "split_manifest": "data/manifest.csv",
                "feature_table": "data/features.parquet",
                "split_version": "speaker_disjoint_v2",
                "fit_split": "train",
                "evaluation_split": "validation",
            }
            train, validation, audit = load_development_data(root, config)
            self.assertEqual(train["sample_id"].tolist(), ["a"])
            self.assertEqual(validation["sample_id"].tolist(), ["b"])
            self.assertEqual(audit["test_samples_loaded"], 0)
            self.assertEqual(audit["train_validation_speaker_overlap"], 0)

    def test_speaker_bootstrap_is_deterministic(self):
        y_true = np.asarray(["north", "north", "south", "central"])
        y_pred = np.asarray(["north", "south", "south", "central"])
        speakers = np.asarray(["1", "1", "2", "3"])
        labels = ["central", "north", "south"]
        first = speaker_bootstrap(y_true, y_pred, speakers, labels, 20, 391)
        second = speaker_bootstrap(y_true, y_pred, speakers, labels, 20, 391)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)


if __name__ == "__main__":
    unittest.main()
