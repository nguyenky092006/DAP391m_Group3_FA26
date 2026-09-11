import unittest

from fedecai.features.validate_wav2vec_runtime import (
    REQUIRED_DISTRIBUTIONS,
    evaluate_versions,
)


class TestWav2VecRuntime(unittest.TestCase):
    def test_exact_versions_pass(self):
        result = evaluate_versions(REQUIRED_DISTRIBUTIONS)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["mismatched"], {})

    def test_missing_distribution_blocks_runtime(self):
        installed = dict(REQUIRED_DISTRIBUTIONS)
        installed.pop("torch")
        result = evaluate_versions(installed)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["missing"], ["torch"])

    def test_version_mismatch_blocks_runtime(self):
        installed = dict(REQUIRED_DISTRIBUTIONS)
        installed["transformers"] = "5.17.0"
        result = evaluate_versions(installed)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(
            result["mismatched"]["transformers"],
            {"expected": "4.57.6", "installed": "5.17.0"},
        )


if __name__ == "__main__":
    unittest.main()
