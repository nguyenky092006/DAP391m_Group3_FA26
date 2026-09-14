import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fedecai.models.benchmark_costs import (
    benchmark_inference_latency, model_size, parameter_count,
)
from fedecai.models.centralized_evaluator import (
    EMOTION_LABELS, evaluate_prediction_file, validate_predictions,
)
from fedecai.models.b6_deep_models import FrozenWav2Vec2Classifier
from fedecai.models.frozen_embedding_cache import (
    validate_embedding_cache, write_embedding_cache,
)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 4)

    def forward(self, waveform, attention_mask):
        class Output:
            pass
        output = Output()
        output.last_hidden_state = self.projection(waveform[:, :, None])
        return output

    def _get_feature_vector_attention_mask(self, width, attention_mask):
        return attention_mask[:, :width].bool()


class CentralizedInfrastructureTests(unittest.TestCase):
    def prediction_frame(self):
        rows = []
        for index, label in enumerate(EMOTION_LABELS * 3):
            probabilities = {name: 0.0 for name in EMOTION_LABELS}
            probabilities[label] = 1.0
            rows.append({
                "experiment_id": "test", "model": "unit", "seed": 391,
                "split": "test", "sample_id": f"s{index}",
                "speaker_id": f"p{index}",
                "accent": ["north", "central", "south"][index // 4],
                "y_true": label, "y_pred": label,
                **{f"probability_{name}": value for name, value in probabilities.items()},
            })
        return pd.DataFrame(rows)

    def test_shared_evaluator_writes_common_metrics_schema(self):
        frame = self.prediction_frame()
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "predictions.csv"
            metrics = Path(directory) / "metrics.json"
            frame.to_csv(prediction, index=False)
            result = evaluate_prediction_file(
                prediction, metrics, expected_sample_ids=frame["sample_id"],
                dataset_version="v", split_version="s",
            )
            self.assertEqual(result["overall_macro_f1"], 1.0)
            self.assertEqual(result["north_macro_f1"], 1.0)
            self.assertIsNone(result["model_size_mb"])
            self.assertEqual(json.loads(metrics.read_text())["uar"], 1.0)

    def test_evaluator_rejects_duplicate_and_nonfinite_predictions(self):
        frame = self.prediction_frame()
        duplicate = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_predictions(duplicate, expected_sample_ids=frame["sample_id"])
        frame.loc[0, "probability_angry"] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_predictions(frame, expected_sample_ids=frame["sample_id"])

    def test_frozen_wav2vec2_counts_only_head_as_trainable(self):
        model = FrozenWav2Vec2Classifier(TinyEncoder(), hidden_size=4)
        counts = parameter_count(model)
        self.assertGreater(counts["total_parameters"], counts["trainable_parameters"])
        self.assertTrue(all(not item.requires_grad for item in model.encoder.parameters()))

    def test_size_counts_external_encoder_and_uses_mebibytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, encoder = root / "head.bin", root / "encoder.bin"
            checkpoint.write_bytes(b"a" * 1024)
            encoder.write_bytes(b"b" * 2048)
            result = model_size(checkpoint, externally_referenced_artifacts=[encoder])
            self.assertEqual(result["model_size_mb"], 3072 / 1024 ** 2)
            self.assertGreater(result["effective_deployment_size_mb"], result["checkpoint_size_mb"])

    def test_latency_protocol_uses_median_and_records_policy(self):
        model = torch.nn.Linear(2, 2).eval()
        values = torch.ones(1, 2)
        result = benchmark_inference_latency(
            lambda: model(values), device="cpu", warmup_runs=1,
            timed_runs=3, preprocessing_included=False,
        )
        self.assertEqual(result["inference_latency_ms"], result["latency_median_ms"])
        self.assertEqual(result["primary_latency_statistic"], "median")
        self.assertFalse(result["preprocessing_included"])

    def test_frozen_embedding_cache_validates_metadata_membership_and_split(self):
        frame = pd.DataFrame({
            "sample_id": ["s1", "s2"], "speaker_id": ["p1", "p2"],
            "split": ["train", "validation"],
        })
        values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = {
            "cache_version": "test_v1", "model": "frozen_test",
            "dataset_version": "data_v1", "split_version": "split_v1",
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            write_embedding_cache(cache, frame, values, expected)
            reordered, metadata = validate_embedding_cache(
                cache, frame.iloc[::-1].reset_index(drop=True), expected,
            )
            np.testing.assert_array_equal(reordered, values[::-1])
            self.assertEqual(metadata["embedding_dimension"], 2)
            invalid = frame.copy(); invalid.loc[1, "split"] = "train"
            with self.assertRaisesRegex(ValueError, "split membership"):
                validate_embedding_cache(cache, invalid, expected)


if __name__ == "__main__":
    unittest.main()
