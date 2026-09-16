import unittest

import torch

from fedecai.models.b6_deep_models import (
    FrozenPitchFusion,
    LogMelCNN2D, MelImageClassifier, PaperAlignedPitchFusionHead,
    ResNet18EmotionAccentModel, Wav2Vec2EmbeddingMLP,
)

try:
    from torchvision import models as _vision_models  # noqa: F401
    TORCHVISION_RUNTIME_AVAILABLE = True
except Exception:
    TORCHVISION_RUNTIME_AVAILABLE = False


class B6DeepModelTests(unittest.TestCase):
    def test_cnn_shape(self):
        self.assertEqual(tuple(LogMelCNN2D()(torch.zeros(2, 1, 80, 128)).shape), (2, 4))

    def test_embedding_mlp_shape(self):
        self.assertEqual(tuple(Wav2Vec2EmbeddingMLP()(torch.zeros(2, 768)).shape), (2, 4))

    @unittest.skipUnless(TORCHVISION_RUNTIME_AVAILABLE, "torchvision runtime unavailable")
    def test_required_three_channel_image_models(self):
        for backbone in MelImageClassifier.SUPPORTED:
            with self.subTest(backbone=backbone):
                model = MelImageClassifier(backbone).eval()
                with torch.inference_mode():
                    output = model(torch.zeros(1, 3, 80, 128))
                self.assertEqual(tuple(output.shape), (1, 4))

    @unittest.skipUnless(TORCHVISION_RUNTIME_AVAILABLE, "torchvision runtime unavailable")
    def test_image_models_reject_one_channel_input(self):
        model = MelImageClassifier("resnet18").eval()
        with self.assertRaisesRegex(ValueError, "B4 Mel tensor"):
            model(torch.zeros(1, 1, 80, 128))

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

    @unittest.skipUnless(TORCHVISION_RUNTIME_AVAILABLE, "torchvision runtime unavailable")
    def test_resnet18_adversarial_variants(self):
        values = torch.zeros(2, 3, 80, 128)
        for mode in ("none", "unconditional", "conditional"):
            with self.subTest(mode=mode):
                model = ResNet18EmotionAccentModel(adversary=mode).eval()
                emotion, accent = model(values, lambda_adv=0.1)
                self.assertEqual(tuple(emotion.shape), (2, 4))
                self.assertEqual(model.feature_dim, 512)
                if mode == "none":
                    self.assertIsNone(accent)
                else:
                    self.assertEqual(tuple(accent.shape), (2, 3))

    def test_gradient_reversal_changes_encoder_gradient_sign(self):
        from fedecai.models.b6_deep_models import gradient_reverse
        values = torch.tensor([1.0], requires_grad=True)
        gradient_reverse(values, 0.25).sum().backward()
        self.assertAlmostEqual(float(values.grad.item()), -0.25)


if __name__ == "__main__":
    unittest.main()
