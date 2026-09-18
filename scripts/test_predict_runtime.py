#!/usr/bin/env python3
import unittest

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


if __name__ == "__main__":
    unittest.main()
