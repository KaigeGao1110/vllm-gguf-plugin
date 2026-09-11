# SPDX-License-Identifier: Apache-2.0
"""Selection tests for the packed IQ4_NL Qwen4Exp PLE embedding method."""

import pytest

import vllm.models.qwen4_exp.nvidia.ngram_embedding as ngram_embedding_module
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLENvFp4EmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

import vllm_gguf_plugin.quantization.ple_iq4_nl as ple_iq4_nl

_UNPATCHED_FROM_QUANT_CONFIG = (
    ngram_embedding_module.Qwen4ExpPLEEmbeddingMethod.from_quant_config
)

ple_iq4_nl.patch_qwen4_exp_ple_embedding_method()


def _patched_from_quant_config():
    return ngram_embedding_module.Qwen4ExpPLEEmbeddingMethod.from_quant_config


def test_gguf_iq4_nl_marker_selects_iq4_nl_method():
    method = _patched_from_quant_config()(
        None,
        "model.ngram_embedding",
        ple_iq4_nl.GGUF_IQ4_NL_PLE_DTYPE,
    )
    assert isinstance(method, ple_iq4_nl.Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod)
    assert isinstance(method, Qwen4ExpPLEEmbeddingMethod)


@pytest.mark.parametrize(
    ("quant_config", "embedding_dtype", "expected_type"),
    [
        (None, None, Qwen4ExpPLEUnquantizedEmbeddingMethod),
        (None, "float8_e4m3fn", Qwen4ExpPLEFp8EmbeddingMethod),
        (None, "nvfp4", Qwen4ExpPLENvFp4EmbeddingMethod),
    ],
)
def test_non_gguf_markers_delegate_to_original(
    quant_config, embedding_dtype, expected_type
):
    patched = _patched_from_quant_config()(
        quant_config, "model.ngram_embedding", embedding_dtype
    )
    original = _UNPATCHED_FROM_QUANT_CONFIG(
        quant_config, "model.ngram_embedding", embedding_dtype
    )
    assert isinstance(patched, expected_type)
    assert type(patched) is type(original)


def test_second_patch_call_keeps_exactly_one_wrapper():
    ple_iq4_nl.patch_qwen4_exp_ple_embedding_method()
    ple_iq4_nl.patch_qwen4_exp_ple_embedding_method()
    wrapper = _patched_from_quant_config()
    assert getattr(wrapper, "_vllm_gguf_plugin_iq4_nl_patched", False)
    unwrapped = getattr(wrapper, "__wrapped__", None)
    assert unwrapped is _UNPATCHED_FROM_QUANT_CONFIG
    assert not getattr(unwrapped, "_vllm_gguf_plugin_iq4_nl_patched", False)