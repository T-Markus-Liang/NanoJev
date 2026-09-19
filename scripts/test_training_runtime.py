from unittest.mock import patch
import unittest
import torch

from train_pipeline_decisions import training_runtime


class TrainingRuntimeTest(unittest.TestCase):
    def test_cpu_fp32_and_no_cuda_flags(self):
        device, precision = training_runtime(torch, "cpu", "fp32", "ce")
        self.assertEqual(str(device), "cpu")
        self.assertEqual(precision, "fp32")
        with self.assertRaises(ValueError):
            training_runtime(torch, "cpu", "bf16", "ce")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            training_runtime(torch, "cpu", "fp32", "ce", True)

    def test_mps_policy_gradient_explicitly_unvalidated(self):
        with patch("train_pipeline_decisions.resolve_runtime", return_value=(torch.device("mps"), "fp32")):
            with self.assertRaisesRegex(ValueError, "not validated"):
                training_runtime(torch, "mps", "fp32", "paired_brier_pg")

    def test_runtime_error_happens_before_cuda_generator(self):
        with patch("train_pipeline_decisions.resolve_runtime", return_value=(torch.device("mps"), "fp32")):
            device, _ = training_runtime(torch, "auto", "fp32", "brier")
            self.assertEqual(device.type, "mps")


if __name__ == "__main__":
    unittest.main()
