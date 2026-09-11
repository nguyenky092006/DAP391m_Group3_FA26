import json
import unittest
from pathlib import Path

import torch

from fedecai.features.extract_wav2vec2_pilot import (
    build_feature_attention_mask,
    feature_output_lengths,
    masked_mean_pool,
)


class TestWav2Vec2Pilot(unittest.TestCase):
    def test_repository_one_sample_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (repo_root / "reports/tables/b4/b4_wav2vec2_one_sample_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["sample"]["sample_id"], "visec_hf_000111")
        self.assertEqual(report["model"]["hidden_dimension"], 768)
        self.assertEqual(report["output"]["shape"], [768])
        self.assertEqual(report["output"]["finite_fraction"], 1.0)
        self.assertEqual(report["output"]["maximum_repeat_difference"], 0.0)
        self.assertEqual(
            report["output"]["artifact_sha256"],
            "5ce4952cbcd3961cf883b796b994efa7f6491d92e2726a8a28753097514e560e",
        )

    def test_one_second_wav2vec2_base_frame_count(self):
        result = feature_output_lengths(
            torch.tensor([16_000]),
            (10, 3, 3, 3, 3, 2, 2),
            (5, 2, 2, 2, 2, 2, 2),
        )
        self.assertEqual(result.tolist(), [49])

    def test_feature_mask_projects_valid_audio_lengths(self):
        input_mask = torch.zeros((2, 32_000), dtype=torch.long)
        input_mask[0, :16_000] = 1
        input_mask[1, :] = 1
        mask = build_feature_attention_mask(
            input_mask,
            feature_frames=99,
            conv_kernels=(10, 3, 3, 3, 3, 2, 2),
            conv_strides=(5, 2, 2, 2, 2, 2, 2),
        )
        self.assertEqual(mask.sum(dim=1).tolist(), [49, 99])

    def test_masked_mean_excludes_padded_frames(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
        mask = torch.tensor([[True, True, False]])
        pooled = masked_mean_pool(hidden, mask)
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))

    def test_masked_mean_rejects_empty_sample(self):
        hidden = torch.zeros((1, 2, 3))
        mask = torch.zeros((1, 2), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            masked_mean_pool(hidden, mask)


if __name__ == "__main__":
    unittest.main()
