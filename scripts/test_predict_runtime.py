#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import torch

from predict_toy_decisions import resolve_runtime


class ResolveRuntimeTest(unittest.TestCase):
    def test_auto_selects_an_available_device_and_precision(self):
        device, precision = resolve_runtime(torch)
        self.assertIn(device.type, {"cpu", "cuda", "mps"})
        self.assertIn(precision, {"fp32", "bf16"})
        if device.type != "cuda":
            self.assertEqual(precision, "fp32")

    def test_cpu_rejects_bf16(self):
        with self.assertRaisesRegex(ValueError, "CPU"):
            resolve_runtime(torch, "cpu", "bf16")

    def test_cpu_auto_uses_fp32(self):
        device, precision = resolve_runtime(torch, "cpu", "auto")
        self.assertEqual(device.type, "cpu")
        self.assertEqual(precision, "fp32")

    def test_cuda_without_index_normalizes_to_cuda_zero(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
             patch.object(torch.cuda, "set_device") as set_device:
            device, precision = resolve_runtime(torch, "cuda", "auto")
        self.assertEqual(str(device), "cuda:0")
        self.assertEqual(precision, "bf16")
        set_device.assert_called_once_with(device)


if __name__ == "__main__":
    unittest.main()
