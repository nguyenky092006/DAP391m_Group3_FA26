import copy
import unittest
from pathlib import Path

from fedecai.features.validate_feature_spec import (
    FeatureSpecError,
    load_feature_spec,
    validate_feature_spec,
)


class TestFeatureSpec(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.spec = load_feature_spec(
            repo_root / "configs/features/visec_features_v1.json"
        )

    def test_repository_feature_spec_passes(self):
        result = validate_feature_spec(self.spec)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["feature_family_count"], 5)
        self.assertEqual(result["deferred"], [])
        self.assertEqual(result["specified_not_downloaded"], [])
        self.assertEqual(result["pilot_validated"], ["wav2vec2_embedding"])

    def test_global_normalization_is_rejected(self):
        invalid = copy.deepcopy(self.spec)
        invalid["leakage_policy"]["normalization_fit_scope"] = "all_data"
        with self.assertRaisesRegex(FeatureSpecError, "normalization_fit_scope"):
            validate_feature_spec(invalid)

    def test_missing_required_family_is_rejected(self):
        invalid = copy.deepcopy(self.spec)
        invalid["feature_families"] = [
            family
            for family in invalid["feature_families"]
            if family["name"] != "pitch_contour"
        ]
        with self.assertRaisesRegex(FeatureSpecError, "pitch_contour"):
            validate_feature_spec(invalid)

    def test_unverified_wav2vec2_revision_is_rejected(self):
        invalid = copy.deepcopy(self.spec)
        wav2vec = next(
            family
            for family in invalid["feature_families"]
            if family["name"] == "wav2vec2_embedding"
        )
        wav2vec["parameters"]["model_revision"] = "main"
        with self.assertRaisesRegex(FeatureSpecError, "model_revision"):
            validate_feature_spec(invalid)

    def test_padding_blind_wav2vec2_pooling_is_rejected(self):
        invalid = copy.deepcopy(self.spec)
        wav2vec = next(
            family
            for family in invalid["feature_families"]
            if family["name"] == "wav2vec2_embedding"
        )
        wav2vec["parameters"]["pooling"] = "unmasked_mean"
        with self.assertRaisesRegex(FeatureSpecError, "exclude padded frames"):
            validate_feature_spec(invalid)

    def test_wav2vec2_pilot_without_hash_is_rejected(self):
        invalid = copy.deepcopy(self.spec)
        wav2vec = next(
            family
            for family in invalid["feature_families"]
            if family["name"] == "wav2vec2_embedding"
        )
        wav2vec["pilot_evidence"]["artifact_sha256"] = ""
        with self.assertRaisesRegex(FeatureSpecError, "artifact SHA-256"):
            validate_feature_spec(invalid)


if __name__ == "__main__":
    unittest.main()
