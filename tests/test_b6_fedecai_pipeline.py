import unittest
import json
from pathlib import Path

import torch

from fedecai.models.b6_deep_models import gradient_reverse


class B6FedECAIPipelineTests(unittest.TestCase):
    def test_backbone_is_selected_by_final_human_decision(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/models/b6_fedecai_v1.json").read_text())
        self.assertEqual(config["backbone_status"], "SELECTED")
        self.assertEqual(config["selected_backbone"], "resnet18_mel3")
        self.assertEqual(config["selected_backbone_training_policy"], "full_finetune")
        self.assertEqual(config["efficiency_fallback"], "mobilenet_v3_mel3")
        self.assertFalse(config["human_decision_required"])
        self.assertEqual(set(config["candidate_backbones"]), {"resnet18_mel3", "mobilenet_v3_mel3"})

    def test_grl_forward_identity_and_gradient_reversal(self):
        values = torch.tensor([1.0, -2.0], requires_grad=True)
        output = gradient_reverse(values, 0.25)
        self.assertTrue(torch.equal(output, values))
        (output * torch.tensor([2.0, -4.0])).sum().backward()
        self.assertTrue(torch.allclose(values.grad, torch.tensor([-0.5, 1.0])))
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_grl_lambda_zero_blocks_adversarial_gradient(self):
        values = torch.tensor([1.0], requires_grad=True)
        gradient_reverse(values, 0.0).sum().backward()
        self.assertEqual(float(values.grad.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
