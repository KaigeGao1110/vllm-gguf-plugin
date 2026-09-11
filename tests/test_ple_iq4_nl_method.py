# SPDX-License-Identifier: Apache-2.0
"""Tests for the packed IQ4_NL Qwen4Exp PLE embedding method."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from torch import nn

import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.nvidia.ngram_embedding as ngram_embedding_module
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLENvFp4EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

import vllm_gguf_plugin.quantization.ple_iq4_nl as ple_iq4_nl
from vllm_gguf_plugin.quantization.ple_iq4_nl import (
    Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod,
)

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


# ---------------------------------------------------------------------------
# Loading: packed uint8 shards through Qwen4ExpNGramEmbedding.load_weights
# ---------------------------------------------------------------------------


def _mock_etp_group(monkeypatch, *, world_size=2, rank=0) -> None:
    group = SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        all_reduce=lambda tensor: tensor,
    )
    monkeypatch.setattr(ngram_embedding_module, "get_etp_group", lambda: group)
    monkeypatch.setattr(
        ngram_embedding_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size),
    )


def _make_iq4_nl_ngram_embedding(
    monkeypatch, *, rank=0, device="cpu", pinned=False
):
    """Build a device or pinned embedding holding 8 packed IQ4_NL rows in two ETP ranks."""
    _mock_etp_group(monkeypatch, world_size=2, rank=rank)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    embedding_cls = Qwen4ExpPLEPinnedHostEmbedding if pinned else Qwen4ExpPLEDeviceEmbedding
    with torch.device(device):
        embedding = embedding_cls(
            8,
            160,
            params_dtype=torch.bfloat16,
            padding_size=2,
            prefix="test.ple_embedding.ngram_embedding",
            embedding_method=Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod(),
            num_ngram_heads=2,
            max_total_tokens=4,
        )
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.split_ngram_parts = 3
    module.register_buffer(
        "layer_multipliers", torch.zeros(1, dtype=torch.long)
    )
    module.register_buffer(
        "ngram_heads_offsets", torch.zeros(1, dtype=torch.long)
    )
    module.register_buffer(
        "ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long)
    )
    module.ngram_embedding = embedding
    packed = torch.randint(0, 256, (8, 90), dtype=torch.uint8)
    # Deterministic finite float16 scales (1.0 + k/1024) in every block, so
    # FP32 products are exactly representable and no NaN/inf can appear.
    for row in range(8):
        for block in range(5):
            bits = 0x3C00 + row * 5 + block
            packed[row, block * 18 : block * 18 + 2] = torch.tensor(
                [bits & 0xFF, (bits >> 8) & 0xFF], dtype=torch.uint8
            )
    # Checkpoint shards cross ETP boundaries and arrive out of order.
    tensors = [
        (f"ngram_embedding.shard_{shard}.weight", packed[shard * 3 :][:3])
        for shard in (2, 0, 1)
    ]
    return module, tensors, packed


@pytest.mark.parametrize("rank", [0, 1])
def test_iq4_nl_ple_loads_streamed_shards_across_etp_boundaries(
    monkeypatch, rank
):
    module, tensors, packed = _make_iq4_nl_ngram_embedding(monkeypatch, rank=rank)
    loaded = set()
    for start in range(0, len(tensors), 2):
        loaded.update(module.load_weights(iter(tensors[start : start + 2])))
    layer = module.ngram_embedding
    layer.embedding_method.process_weights_after_loading(layer)

    assert loaded == {"ngram_embedding.weight"}
    assert layer.weight.dtype == torch.uint8
    assert layer.weight.shape == (4, 90)
    torch.testing.assert_close(layer.weight, packed[rank * 4 :][:4])


@pytest.mark.parametrize(
    "missing", ["shard_0.weight", "shard_1.weight", "all"]
)
def test_iq4_nl_ple_rejects_missing_local_shards(monkeypatch, missing):
    module, tensors, _ = _make_iq4_nl_ngram_embedding(monkeypatch)
    module.load_weights(
        (
            name,
            tensor,
        )
        for name, tensor in tensors
        if missing != name.removeprefix("ngram_embedding.")
        and not (missing == "all" and ".shard_" in name)
    )
    layer = module.ngram_embedding
    with pytest.raises(ValueError, match="missing rows starting at local row"):
        layer.embedding_method.process_weights_after_loading(layer)


def test_iq4_nl_ple_rejects_float_shard(monkeypatch):
    module, tensors, _ = _make_iq4_nl_ngram_embedding(monkeypatch)
    name, tensor = next(
        pair for pair in tensors if pair[0] == "ngram_embedding.shard_0.weight"
    )
    with pytest.raises(ValueError, match="uint8"):
        module.load_weights([(name, tensor.float())])


def test_iq4_nl_ple_rejects_wrong_row_width(monkeypatch):
    module, tensors, _ = _make_iq4_nl_ngram_embedding(monkeypatch)
    name, tensor = next(
        pair for pair in tensors if pair[0] == "ngram_embedding.shard_0.weight"
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        module.load_weights([(name, tensor[:, :-1])])


# ---------------------------------------------------------------------------
# CPU decode: method.embedding vs gguf's independent IQ4_NL decoder
# ---------------------------------------------------------------------------

_FIXTURE = Path(__file__).parent / "fixtures" / "qwen4_exp_iq4_nl_samples.json"


def _reference_rows(packed: torch.Tensor) -> torch.Tensor:
    """Decode packed rows with gguf's independent IQ4_NL decoder (FP32)."""
    raw = np.ascontiguousarray(packed.numpy().copy())
    reference = gguf.dequantize(raw, gguf.GGMLQuantizationType.IQ4_NL)
    return torch.from_numpy(np.ascontiguousarray(reference)).reshape(-1, 160)


def _load_fixture_rows() -> torch.Tensor:
    fixture = json.loads(_FIXTURE.read_text())
    raw = b"".join(
        base64.b64decode(sample["packed_base64"]) for sample in fixture["samples"]
    )
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, 90)


def _make_packed_rows(scale_bits: list[int]) -> torch.Tensor:
    """Deterministic packed rows with one controlled float16 scale per block."""
    rows = np.zeros((len(scale_bits), 90), dtype=np.uint8)
    for i, bits in enumerate(scale_bits):
        for block in range(5):
            rows[i, block * 18 : block * 18 + 2] = (bits & 0xFF, (bits >> 8) & 0xFF)
            rows[i, block * 18 + 2 : block * 18 + 18] = np.arange(
                block * 16, block * 16 + 16, dtype=np.uint8
            )
    return torch.from_numpy(rows.copy())


def _make_single_rank_embedding(
    monkeypatch, *, packed: torch.Tensor, params_dtype: torch.dtype
):
    """CPU device embedding holding the given packed rows on one ETP rank."""
    _mock_etp_group(monkeypatch, world_size=1, rank=0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    with torch.device("cpu"):
        embedding = Qwen4ExpPLEDeviceEmbedding(
            packed.shape[0],
            160,
            params_dtype=params_dtype,
            padding_size=2,
            prefix="test.ple_embedding.ngram_embedding",
            embedding_method=Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod(),
            num_ngram_heads=2,
            max_total_tokens=4,
        )
    # Full-table load goes through the parameter's (wrapped) loader, exactly
    # as upstream load_weights does for shard rows.
    embedding.weight.weight_loader(embedding.weight, packed)
    embedding.embedding_method.process_weights_after_loading(embedding)
    return embedding


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_embedding_cpu_matches_gguf_decoder_on_real_rows(monkeypatch, dtype):
    packed = _load_fixture_rows()
    reference = _reference_rows(packed)
    layer = _make_single_rank_embedding(
        monkeypatch, packed=packed, params_dtype=dtype
    )
    method = layer.embedding_method
    ids = torch.tensor([[3, 0, 3], [7, 1, 5]], dtype=torch.int64)
    actual = method.embedding(layer, ids)

    assert actual.shape == (2, 3, 160)
    assert actual.dtype == dtype
    torch.testing.assert_close(
        actual, reference[ids].to(dtype), rtol=0, atol=0
    )


def test_embedding_cpu_matches_gguf_decoder_on_1d_and_scalar_ids(monkeypatch):
    packed = _load_fixture_rows()
    reference = _reference_rows(packed)
    layer = _make_single_rank_embedding(
        monkeypatch, packed=packed, params_dtype=torch.float32
    )
    method = layer.embedding_method

    flat_ids = torch.tensor([5, 2, 5, 0], dtype=torch.int64)
    torch.testing.assert_close(
        method.embedding(layer, flat_ids),
        reference[flat_ids].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        method.embedding(layer, torch.tensor(4, dtype=torch.int64)),
        reference[4].float().reshape(160),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("scale_bits", "label"),
    [
        ([0xBF00], "negative"),
        ([0x0000], "zero"),
        ([0x0001], "subnormal"),
        ([0x8001], "negative-subnormal"),
        ([0x7BFF], "large"),
        ([0x3C00, 0x0001, 0xBF00, 0x0000, 0x7BFF], "mixed"),
    ],
)
def test_embedding_cpu_matches_gguf_decoder_on_special_scales(
    monkeypatch, scale_bits, label
):
    packed = _make_packed_rows(scale_bits)
    reference = _reference_rows(packed)
    layer = _make_single_rank_embedding(
        monkeypatch, packed=packed, params_dtype=torch.float32
    )
    method = layer.embedding_method
    rows = len(scale_bits)
    ids = torch.tensor(list(range(rows - 1, -1, -1)) + [0], dtype=torch.int64)
    actual = method.embedding(layer, ids)

    assert actual.shape == (rows + 1, 160)
    torch.testing.assert_close(
        actual, reference[ids].float(), rtol=0, atol=0, msg=label
    )


def test_embedding_cpu_empty_input_returns_empty(monkeypatch):
    packed = _load_fixture_rows()
    layer = _make_single_rank_embedding(
        monkeypatch, packed=packed, params_dtype=torch.bfloat16
    )
    method = layer.embedding_method

    actual = method.embedding(layer, torch.empty((0, 2), dtype=torch.int64))
    assert actual.shape == (0, 2, 160)
    assert actual.dtype == torch.bfloat16
    assert (
        method.embedding(layer, torch.empty((2, 0), dtype=torch.int64)).shape
        == (2, 0, 160)
    )


def test_lookup_dtype_and_dequantize(monkeypatch):
    packed = _load_fixture_rows()
    layer = _make_single_rank_embedding(
        monkeypatch, packed=packed, params_dtype=torch.bfloat16
    )
    method = layer.embedding_method
    assert method.lookup_dtype(layer) == torch.bfloat16
    values = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    converted = method.dequantize(layer, values, torch.float32)
    assert converted.dtype == torch.float32
    torch.testing.assert_close(converted, values.float(), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Plugin hook: register() patches from_quant_config exactly once
# ---------------------------------------------------------------------------


def test_register_patches_ple_from_quant_config():
    from vllm_gguf_plugin.plugin import register

    register()
    wrapper = _patched_from_quant_config()
    assert getattr(wrapper, "_vllm_gguf_plugin_iq4_nl_patched", False)
    assert isinstance(
        wrapper(
            None, "model.ngram_embedding", ple_iq4_nl.GGUF_IQ4_NL_PLE_DTYPE
        ),
        Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod,
    )


def test_register_twice_keeps_one_ple_wrapper():
    from vllm_gguf_plugin.plugin import register

    register()
    register()
    wrapper = _patched_from_quant_config()
    assert getattr(wrapper, "_vllm_gguf_plugin_iq4_nl_patched", False)
    unwrapped = getattr(wrapper, "__wrapped__", None)
    assert unwrapped is _UNPATCHED_FROM_QUANT_CONFIG
    assert not getattr(unwrapped, "_vllm_gguf_plugin_iq4_nl_patched", False)


# ---------------------------------------------------------------------------
# GPU: Triton kernel and pinned-host (UVA) lookup. These require CUDA and
# skip on the CPU-only runner; the skips are expected and not counted as
# passes.
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_iq4_nl_kernel_scale_formula_matches_float16_bits():
    """The kernel's in-kernel float16 rebuild matches the bit-cast decode.

    Runs on CPU: it validates the exact expression the Triton kernel uses
    (sign, 5-bit exponent, 10-bit mantissa; subnormal / inf / NaN branches)
    over every float16 bit pattern, so the GPU path can only differ by
    execution, not by formula.
    """
    bits = torch.arange(65536, dtype=torch.int32)
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    # Same construction as the Triton kernel: normal scales are the exact
    # float32 bit pattern ((e + 112) << 23) | (m << 13).
    normal_bits = ((exponent + 112) << 23) | (mantissa << 13)
    scale = torch.where(
        exponent == 0,
        mantissa.to(torch.float32) * (2.0**-24),
        torch.where(
            exponent == 31,
            torch.where(
                mantissa == 0,
                torch.tensor(float("inf")),
                torch.tensor(float("nan")),
            ),
            normal_bits.view(torch.float32),
        ),
    )
    scale = torch.where(sign == 1, -scale, scale)
    reference = bits.to(torch.int16).view(torch.float16).float()

    nan_mask = torch.isnan(reference)
    assert torch.any(nan_mask)
    assert torch.equal(torch.isnan(scale), nan_mask)
    torch.testing.assert_close(
        scale[~nan_mask], reference[~nan_mask], rtol=0, atol=0
    )


@requires_cuda
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_iq4_nl_ple_cuda_lookup_and_graph_replay(monkeypatch, pinned, rank):
    module, tensors, packed = _make_iq4_nl_ngram_embedding(
        monkeypatch, rank=rank, device="cuda", pinned=pinned
    )
    module.load_weights(iter(tensors))
    layer = module.ngram_embedding
    layer.embedding_method.process_weights_after_loading(layer)
    reference = _reference_rows(packed).to(layer.params_dtype).to("cuda")
    ids = torch.tensor([[3, 4], [7, 0]], device="cuda")
    lookup_fn = layer._lookup if pinned else torch.compile(layer, fullgraph=True)

    def lookup():
        return lookup_fn(ids)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            lookup()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = lookup()
    for new_ids in ([[3, 4], [7, 0]], [[4, 3], [0, 7]]):
        ids.copy_(torch.tensor(new_ids, device="cuda"))
        graph.replay()
        cpu_ids = ids.cpu()
        expected = reference[cpu_ids]
        expected[(cpu_ids < rank * 4) | (cpu_ids >= (rank + 1) * 4)] = 0
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
        if pinned:
            hidden_states = torch.empty(
                2, 320, device="cuda", dtype=torch.bfloat16
            )
            layer.start_prefetch(hidden_states, ids)
            torch.testing.assert_close(
                layer(hidden_states).cpu(), expected.flatten(-2), atol=0, rtol=0
            )
    assert layer.weight.device.type == ("cpu" if pinned else "cuda")
    # Empty ids return an empty tensor without launching the kernel.
    assert lookup_fn(ids[:0]).shape == (0, 2, 160)
    # Padded / out-of-range ids produce zero rows.
    padded_ids = torch.tensor([[8, -1]], device="cuda")
    torch.testing.assert_close(
        lookup_fn(padded_ids),
        torch.zeros(1, 2, 160, device="cuda", dtype=torch.bfloat16),
        atol=0,
        rtol=0,
    )


@requires_cuda
def test_iq4_nl_ple_cuda_pinned_uva_lookup(monkeypatch):
    """Direct lookup_from_pinned over the UVA view matches the gguf reference."""
    module, tensors, packed = _make_iq4_nl_ngram_embedding(
        monkeypatch, rank=1, device="cuda", pinned=True
    )
    module.load_weights(iter(tensors))
    layer = module.ngram_embedding
    layer.embedding_method.process_weights_after_loading(layer)
    reference = _reference_rows(packed)
    ids = torch.tensor([4, 5, 6, 7, 3, 0], device="cuda")
    output = torch.empty(
        ids.shape[0], 160, device="cuda", dtype=layer.params_dtype
    )
    layer.embedding_method.lookup_from_pinned(layer, ids, output)
    expected = reference[ids.cpu()].to(torch.bfloat16)
    expected[(ids.cpu() < 4) | (ids.cpu() >= 8)] = 0
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    # Empty ids are a no-op on the output tensor.
    empty_output = torch.empty((0, 160), device="cuda", dtype=torch.bfloat16)
    layer.embedding_method.lookup_from_pinned(layer, ids[:0], empty_output)
    assert empty_output.shape == (0, 160)


@requires_cuda
def test_iq4_nl_ple_cuda_special_float16_scales(monkeypatch):
    """Negative, zero, subnormal, and large float16 scales decode on device."""
    scale_bits = [0xBF00, 0x0000, 0x0001, 0x8001, 0x7BFF]
    packed = _make_packed_rows(scale_bits)
    _mock_etp_group(monkeypatch, world_size=1, rank=0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    with torch.device("cuda"):
        layer = Qwen4ExpPLEDeviceEmbedding(
            packed.shape[0],
            160,
            params_dtype=torch.bfloat16,
            padding_size=2,
            prefix="test.ple_embedding.ngram_embedding",
            embedding_method=Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod(),
            num_ngram_heads=2,
            max_total_tokens=4,
        )
    layer.weight.weight_loader(layer.weight, packed.to("cuda"))
    layer.embedding_method.process_weights_after_loading(layer)
    reference = _reference_rows(packed)
    ids = torch.tensor([4, 2, 0, 4], device="cuda")
    actual = layer(ids)
    expected = reference[ids.cpu()].to(torch.bfloat16)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)