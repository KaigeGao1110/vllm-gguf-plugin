"""Numerical checks against gguf's independent decoder using real model rows."""

import base64
import importlib
import json
import unittest
from pathlib import Path

import gguf
import numpy as np
import torch


class PleCpuTests(unittest.TestCase):
    def setUp(self):
        module_name = "vllm_gguf_plugin.quantization.ple_cpu"
        if importlib.util.find_spec(module_name) is None:
            self.fail("CPU packed PLE row lookup is not implemented")
        self.impl = importlib.import_module(module_name)
        fixture = json.loads(
            (
                Path(__file__).parent / "fixtures/qwen4_exp_iq4_nl_samples.json"
            ).read_text()
        )
        self.width = fixture["width"]
        raw = b"".join(base64.b64decode(s["packed_base64"]) for s in fixture["samples"])
        self.packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
            -1, 90
        )
        reference = gguf.dequantize(
            np.frombuffer(raw, dtype=np.uint8).reshape(-1, 90),
            gguf.GGMLQuantizationType.IQ4_NL,
        )
        self.reference = torch.from_numpy(reference.copy()).reshape(-1, self.width)

    def test_real_rows_match_gguf_float32_decoder(self):
        actual = self.impl.decode_iq4_nl_rows(
            self.packed, self.width, dtype=torch.float32
        )
        torch.testing.assert_close(actual, self.reference, rtol=0, atol=0)

    def test_bfloat16_rounding_matches_reference(self):
        actual = self.impl.decode_iq4_nl_rows(
            self.packed, self.width, dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            actual, self.reference.to(torch.bfloat16), rtol=0, atol=0
        )

    def test_unsorted_duplicate_requests_preserve_shape_and_order(self):
        ids = torch.tensor([[7, 0, 7], [5, 3, 1]], dtype=torch.int64)
        actual = self.impl.gather_iq4_nl_rows(self.packed, ids, self.width)
        expected = self.reference[ids].to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_scalar_request(self):
        actual = self.impl.gather_iq4_nl_rows(self.packed, torch.tensor(7), self.width)
        torch.testing.assert_close(
            actual, self.reference[7].to(torch.bfloat16), rtol=0, atol=0
        )

    def test_empty_batch(self):
        actual = self.impl.gather_iq4_nl_rows(
            self.packed, torch.empty((0, 16), dtype=torch.int64), self.width
        )
        self.assertEqual(actual.shape, (0, 16, 160))
        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_out_of_range_row_ids_fail(self):
        for ids in [torch.tensor([-1]), torch.tensor([len(self.packed)])]:
            with self.subTest(ids=ids), self.assertRaises(IndexError):
                self.impl.gather_iq4_nl_rows(self.packed, ids, self.width)

    def test_float_row_ids_fail(self):
        with self.assertRaises(TypeError):
            self.impl.gather_iq4_nl_rows(self.packed, torch.tensor([1.0]), self.width)

    def test_invalid_packed_layout_fails(self):
        for data, width in [(self.packed[:, :-1], 160), (self.packed, 161)]:
            with (
                self.subTest(shape=data.shape, width=width),
                self.assertRaises(ValueError),
            ):
                self.impl.decode_iq4_nl_rows(data, width)

    def test_non_byte_packed_table_fails(self):
        with self.assertRaises(TypeError):
            self.impl.decode_iq4_nl_rows(self.packed.float(), self.width)


if __name__ == "__main__":
    unittest.main()
