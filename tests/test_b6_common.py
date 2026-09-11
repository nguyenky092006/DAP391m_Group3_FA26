import unittest

import numpy as np

from fedecai.models.b6_common import classification_metrics
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
        self.assertEqual(result["worst_accent_macro_f1"], 1.0)
        self.assertEqual(result["accent_macro_f1_gap"], 0.0)

    def test_missing_class_is_zero_not_nan(self):
        labels = ["angry", "happy", "neutral", "sad"]
        truth = np.array(["angry", "happy"])
        predicted = np.array(["angry", "angry"])
        result = classification_metrics(truth, predicted, np.array(["north", "north"]), labels)
        self.assertTrue(np.isfinite(result["overall"]["macro_f1"]))
        self.assertEqual(result["overall"]["per_class_f1"]["sad"], 0.0)

    def test_cv_export_includes_each_grouped_fold(self):
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
