# SPDX-License-Identifier: Apache-2.0
"""Which MoE kernel does an IQ4_XS prefill batch actually reach?

The interesting failure is silent: IQ4_XS has no CUDA MMQ kernel, so before this
change every prefill chunk fell through to ``ggml_moe_a8_vec`` -- a kernel meant
for decode -- and the model still produced correct output, just slowly. A test
that only checks numerics stays green through that. So these assert on the
dispatch itself, by counting which ``ops`` entry point was called.

Every kernel is stubbed, so this runs on CPU and allocates no device memory. It
does need vllm importable (the module imports it at load time), which on the
serving box means it can run without stopping the engine.

``vllm_gguf_plugin.quantization.fused_moe`` is never reloaded: it registers a
custom op at import time, and re-registering raises. The env switch is exercised
by reloading ``utils`` alone and injecting the set it produces.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from gguf import GGMLQuantizationType as WeightType

from vllm_gguf_plugin.quantization import fused_moe as fused_moe_mod

E, H, INNER, TOP_K = 4, 64, 32, 2


def _moe_types_with_switch(monkeypatch, value: str | None):
    """Reload the gate module with the env switch set, and return its sets."""
    if value is None:
        monkeypatch.delenv("GGUF_PLUGIN_IQ_MOE_MMQ", raising=False)
    else:
        monkeypatch.setenv("GGUF_PLUGIN_IQ_MOE_MMQ", value)
    return importlib.reload(
        importlib.import_module("vllm_gguf_plugin.quantization.utils")
    )


@pytest.fixture
def calls(monkeypatch):
    """Stub every kernel and record which one the dispatch picked."""
    counts: dict[str, int] = {"a8": 0, "vec": 0}

    def fake_block_size(quant_type):
        return 32

    def fake_align(topk_ids, block_size, num_experts):
        n = topk_ids.numel()
        return (
            torch.zeros(n, dtype=torch.int32),
            torch.zeros(n // block_size + 1, dtype=torch.int32),
            torch.tensor([n], dtype=torch.int32),
        )

    def fake_a8(
        x, w, sorted_token_ids, expert_ids, num_post_pad, quant_type, row, top_k, tokens
    ):
        counts["a8"] += 1
        return torch.zeros(tokens * top_k, row, dtype=x.dtype)

    def fake_vec(x, w, topk_ids, top_k, quant_type, n, tokens):
        counts["vec"] += 1
        return torch.zeros(tokens * top_k, n, dtype=x.dtype)

    def fake_moe_sum(out, dst):
        dst.copy_(out.sum(dim=1))

    def fake_activation(kind, out, inp):
        out.copy_(inp[..., : inp.shape[-1] // 2])

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.fused_moe.moe_align_block_size",
        fake_align,
        raising=False,
    )
    monkeypatch.setattr("vllm_gguf_plugin.ops.ggml_moe_get_block_size", fake_block_size)
    monkeypatch.setattr("vllm_gguf_plugin.ops.ggml_moe_a8", fake_a8)
    monkeypatch.setattr("vllm_gguf_plugin.ops.ggml_moe_a8_vec", fake_vec)
    monkeypatch.setattr("vllm_gguf_plugin.ops.moe_sum", fake_moe_sum)
    monkeypatch.setattr(fused_moe_mod, "apply_moe_activation", fake_activation)
    return counts


def _run(num_tokens: int, quant_type: WeightType):
    x = torch.zeros(num_tokens, H, dtype=torch.float16)
    w1 = torch.zeros(E, 2 * INNER, 8, dtype=torch.uint8)
    w2 = torch.zeros(E, H, 8, dtype=torch.uint8)
    topk_weights = torch.ones(num_tokens, TOP_K, dtype=torch.float16)
    topk_ids = torch.zeros(num_tokens, TOP_K, dtype=torch.int32)
    fused_moe_mod._fused_moe_gguf(
        x, w1, w2, topk_weights, topk_ids, int(quant_type), int(quant_type), "silu"
    )


def test_iq4_xs_prefill_batch_reaches_the_tile_gemm(monkeypatch, calls):
    """128 tokens is a prefill chunk: it must not use the decode kernel."""
    utils = _moe_types_with_switch(monkeypatch, "1")
    monkeypatch.setattr(
        fused_moe_mod, "MMQ_MOE_TRITON_TYPES", utils.MMQ_MOE_TRITON_TYPES
    )
    _run(128, WeightType.IQ4_XS)
    assert calls["a8"] == 2, f"expected both GEMMs on ggml_moe_a8, got {calls}"
    assert calls["vec"] == 0, f"prefill still went through the decode kernel: {calls}"


def test_iq4_xs_decode_batch_still_uses_the_vector_kernel(monkeypatch, calls):
    """The x.shape[0] > 64 guard must keep decode on the mat-vec path."""
    utils = _moe_types_with_switch(monkeypatch, "1")
    monkeypatch.setattr(
        fused_moe_mod, "MMQ_MOE_TRITON_TYPES", utils.MMQ_MOE_TRITON_TYPES
    )
    _run(8, WeightType.IQ4_XS)
    assert calls["vec"] == 2, f"decode should stay on ggml_moe_a8_vec, got {calls}"
    assert calls["a8"] == 0


def test_switch_off_restores_the_previous_dispatch(monkeypatch, calls):
    """The rollback path: with the switch off a prefill batch goes back to vec.

    This is the only evidence that the window's rollback works without
    reinstalling the package, so it is not optional.
    """
    utils = _moe_types_with_switch(monkeypatch, "0")
    assert WeightType.IQ4_XS not in utils.MMQ_MOE_TRITON_TYPES
    monkeypatch.setattr(
        fused_moe_mod, "MMQ_MOE_TRITON_TYPES", utils.MMQ_MOE_TRITON_TYPES
    )
    _run(128, WeightType.IQ4_XS)
    assert calls["vec"] == 2, f"switch off did not restore the old path: {calls}"
    assert calls["a8"] == 0


def test_q4_k_prefill_batch_is_unchanged(monkeypatch, calls):
    """A type that already had MMQ must dispatch exactly as before."""
    utils = _moe_types_with_switch(monkeypatch, "1")
    monkeypatch.setattr(
        fused_moe_mod, "MMQ_MOE_TRITON_TYPES", utils.MMQ_MOE_TRITON_TYPES
    )
    _run(128, WeightType.Q4_K)
    assert calls["a8"] == 2, f"existing MMQ behaviour changed: {calls}"
    assert calls["vec"] == 0


def test_linear_gate_is_not_widened(monkeypatch):
    """linear.py reads MMQ_QUANT_TYPES; this change must not touch that path."""
    utils = _moe_types_with_switch(monkeypatch, "1")
    assert WeightType.IQ4_XS not in utils.MMQ_QUANT_TYPES
    assert WeightType.IQ4_NL not in utils.MMQ_QUANT_TYPES
    assert WeightType.IQ4_XS in utils.MMQ_MOE_TRITON_TYPES
