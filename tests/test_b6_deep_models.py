import unittest

import torch

from fedecai.models.b6_deep_models import AccentInvariantCNN, FrozenPitchFusion, LogMelCNN2D, PaperAlignedPitchFusionHead, Wav2Vec2EmbeddingMLP


class B6DeepModelTests(unittest.TestCase):
    def test_cnn_shape(self):
        self.assertEqual(tuple(LogMelCNN2D()(torch.zeros(2, 1, 80, 128)).shape), (2, 4))

    def test_embedding_mlp_shape(self):
        self.assertEqual(tuple(Wav2Vec2EmbeddingMLP()(torch.zeros(2, 768)).shape), (2, 4))

    def test_pitch_fusion_shape(self):
        output = FrozenPitchFusion()(torch.zeros(2, 768), torch.zeros(2, 6))
        self.assertEqual(tuple(output.shape), (2, 4))

    def test_paper_aligned_pitch_fusion_shape_and_mask(self):
        model = PaperAlignedPitchFusionHead(hidden_size=32, projection_size=16, heads=4)
        acoustic = torch.zeros(2, 49, 32)
        pitch = torch.zeros(2, 16000)
        mask = torch.zeros(2, 49, dtype=torch.bool)
        mask[1, 40:] = True
        output = model(acoustic, pitch, mask)
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_accent_invariant_variants(self):
        values = torch.zeros(2, 1, 80, 128)
        emotion, accent = AccentInvariantCNN(adversarial=False)(values)
        self.assertEqual(tuple(emotion.shape), (2, 4))
        self.assertIsNone(accent)
        condition = torch.nn.functional.one_hot(torch.tensor([0, 1]), 4).float()
        emotion, accent = AccentInvariantCNN(adversarial=True, conditional=True)(
            values, 0.1, condition
        )
        self.assertEqual(tuple(emotion.shape), (2, 4))
        self.assertEqual(tuple(accent.shape), (2, 3))

    def test_gradient_reversal_changes_encoder_gradient_sign(self):
        from fedecai.models.b6_deep_models import gradient_reverse
        values = torch.tensor([1.0], requires_grad=True)
        gradient_reverse(values, 0.25).sum().backward()
        self.assertAlmostEqual(float(values.grad.item()), -0.25)


if __name__ == "__main__":
    unittest.main()
