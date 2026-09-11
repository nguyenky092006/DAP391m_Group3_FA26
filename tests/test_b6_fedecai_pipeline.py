import unittest

import torch

from fedecai.models.run_b6_fedecai_pipeline import average_states, grl_strength


class B6FedECAIPipelineTests(unittest.TestCase):
    def test_weighted_fedavg(self):
        states = [
            {"weight": torch.tensor([1.0]), "counter": torch.tensor(2)},
            {"weight": torch.tensor([3.0]), "counter": torch.tensor(4)},
        ]
        result = average_states(states, [1, 3])
        self.assertAlmostEqual(float(result["weight"].item()), 2.5)
        self.assertEqual(int(result["counter"].item()), 2)

    def test_grl_schedule_has_warmup_and_reaches_maximum(self):
        self.assertEqual(grl_strength(1, 20, 3, 0.1), 0.0)
        self.assertEqual(grl_strength(3, 20, 3, 0.1), 0.0)
        self.assertAlmostEqual(grl_strength(20, 20, 3, 0.1), 0.1)


if __name__ == "__main__":
    unittest.main()
