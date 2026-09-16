import unittest

import numpy as np
import pandas as pd

from fedecai.models.b6_common import classification_metrics, prediction_rows
from fedecai.models.run_b6_classical_pipeline import cv_rows


class B6CommonTests(unittest.TestCase):
    def test_metrics_include_overall_and_accent_robustness(self):
        labels = ["angry", "happy", "neutral", "sad"]
        truth = np.array(labels * 3)
        predicted = truth.copy()
        accents = np.repeat(np.array(["central", "north", "south"]), 4)
        result = classification_metrics(truth, predicted, accents, labels)
        self.assertEqual(result["overall"]["macro_f1"], 1.0)
        self.assertEqual(result["overall"]["uar"], 1.0)
        self.assertEqual(result["overall"]["weighted_f1"], 1.0)
        self.assertIn("confusion_matrix", result["overall"])
        self.assertIn("precision", result["overall"]["per_class"]["angry"])
        self.assertEqual(result["worst_accent_macro_f1"], 1.0)
        self.assertEqual(result["best_accent_macro_f1"], 1.0)
        self.assertEqual(result["accent_macro_f1_gap"], 0.0)
        self.assertEqual(result["accent_macro_f1_std"], 0.0)
        self.assertIn("per_emotion_f1", result["per_accent"]["north"])

    def test_prediction_rows_include_required_reconstruction_fields(self):
        frame = pd.DataFrame([{
            "sample_id": "s1", "speaker_id": "p1", "accent_clean": "north",
            "emotion": "angry", "split": "test",
        }])
        rows = prediction_rows(
            frame, np.array(["angry"]), np.array([[0.7, 0.1, 0.1, 0.1]]),
            ["angry", "happy", "neutral", "sad"],
            experiment_id="run1", model="svm_rbf", seed=391,
        )
        self.assertEqual(rows[0]["experiment_id"], "run1")
        self.assertEqual(rows[0]["model"], "svm_rbf")
        self.assertEqual(rows[0]["seed"], 391)
        self.assertEqual(rows[0]["split"], "test")
        self.assertEqual(rows[0]["y_true"], "angry")
        self.assertEqual(rows[0]["y_pred"], "angry")
        self.assertEqual(rows[0]["probability_angry"], 0.7)

    def test_missing_class_is_zero_not_nan(self):
        labels = ["angry", "happy", "neutral", "sad"]
        truth = np.array(["angry", "happy"])
        predicted = np.array(["angry", "angry"])
        result = classification_metrics(truth, predicted, np.array(["north", "north"]), labels)
        self.assertTrue(np.isfinite(result["overall"]["macro_f1"]))
        self.assertEqual(result["overall"]["per_class_f1"]["sad"], 0.0)

    def test_cv_export_includes_each_stratified_fold(self):
        class Search:
            cv_results_ = {
                "params": [{"model__C": 1}],
                "rank_test_macro_f1": [1],
                "mean_test_macro_f1": [0.5],
                "std_test_macro_f1": [0.1],
                "mean_test_uar": [0.4],
                "mean_fit_time": [1.2],
                "split0_test_macro_f1": [0.3],
                "split0_test_uar": [0.2],
                "split1_test_macro_f1": [0.7],
                "split1_test_uar": [0.6],
            }

        row = cv_rows(Search(), "svm_rbf", ["fold_0", "fold_1"])[0]
        self.assertEqual(row["fold_0_macro_f1"], 0.3)
        self.assertEqual(row["fold_1_uar"], 0.6)


if __name__ == "__main__":
    unittest.main()
