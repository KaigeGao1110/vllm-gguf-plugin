# SPDX-License-Identifier: Apache-2.0
"""Packed IQ4_NL storage and lookup for the Qwen4Exp PLE table.

The PLE table is stored as raw GGUF IQ4_NL bytes (one 18-byte block per 32
values: a little-endian float16 scale plus 16 bytes of 4-bit indices) and
decoded only for the requested rows. This mirrors the packed NVFP4 method
added by vLLM #56273, with the fixed IQ4_NL codebook replacing the E2M1 rows.
"""

from functools import wraps

import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.models.qwen4_exp.common.ple import compute_ple_shard_overlap
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
)
from vllm.triton_utils import tl, triton

from .ple_cpu import _IQ4_NL_VALUES, gather_iq4_nl_rows

GGUF_IQ4_NL_PLE_DTYPE = "gguf_iq4_nl"

_BLOCK_VALUES = 32
_BLOCK_BYTES = 18
_PATCH_GUARD = "_vllm_gguf_plugin_iq4_nl_patched"


def _row_bytes(embedding_dim: int) -> int:
    """Return the packed IQ4_NL bytes per row of ``embedding_dim`` values."""
    if embedding_dim <= 0 or embedding_dim % _BLOCK_VALUES:
        raise ValueError(
            "IQ4_NL PLE embedding dimension must be a positive multiple of "
            f"32, got {embedding_dim}"
        )
    return embedding_dim // _BLOCK_VALUES * _BLOCK_BYTES


class Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Packed IQ4_NL PLE rows decoded per requested n-gram ID."""

    def __init__(self) -> None:
        self._loaded_ranges: set[tuple[int, int]] = set()
        self._codebooks: dict[torch.device, torch.Tensor] = {}

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        row_bytes = _row_bytes(input_size_per_partition)
        original_weight_loader = extra_weight_attrs.get("weight_loader")
        loaded_ranges = self._loaded_ranges
        tp_start = layer.shard_indices.org_vocab_start_index
        tp_end = layer.shard_indices.org_vocab_end_index
        local_org_rows = tp_end - tp_start

        def wrapped_weight_loader(param, loaded_weight, *args, **kwargs):
            if loaded_weight.dtype != torch.uint8:
                raise ValueError(
                    "IQ4_NL PLE shards must be packed uint8 bytes, "
                    f"got {loaded_weight.dtype}"
                )
            original_weight_loader(param, loaded_weight, *args, **kwargs)
            checkpoint_start = kwargs.get("checkpoint_start")
            if checkpoint_start is None:
                # A full checkpoint covers the whole original vocabulary; the
                # padded tail rows of the last rank are never loaded.
                loaded_ranges.add((0, local_org_rows))
                return
            overlap = compute_ple_shard_overlap(
                checkpoint_start=checkpoint_start,
                checkpoint_rows=loaded_weight.shape[0],
                tp_start=tp_start,
                tp_end=tp_end,
            )
            if overlap is not None:
                loaded_ranges.add(
                    (
                        overlap.destination_start,
                        overlap.destination_start + overlap.row_count,
                    )
                )

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=layer.allocate_embedding_weight(
                    sum(output_partition_sizes), row_bytes, torch.uint8
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=wrapped_weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        expected_rows = (
            layer.shard_indices.org_vocab_end_index
            - layer.shard_indices.org_vocab_start_index
        )
        loaded_end = 0
        for start, end in sorted(self._loaded_ranges):
            if start > loaded_end:
                break
            loaded_end = max(loaded_end, end)
        if loaded_end != expected_rows:
            raise ValueError(
                "IQ4_NL PLE checkpoint is missing rows starting at local "
                f"row {loaded_end}"
            )

    def lookup_dtype(self, layer: nn.Module) -> torch.dtype:
        return layer.params_dtype

    def _codebook(self, device: torch.device) -> torch.Tensor:
        """Return the float32 IQ4_NL codebook on ``device``."""
        codebook = self._codebooks.get(device)
        if codebook is None:
            codebook = torch.tensor(_IQ4_NL_VALUES, dtype=torch.float32, device=device)
            self._codebooks[device] = codebook
        return codebook

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if not input_.is_cuda:
            return gather_iq4_nl_rows(
                layer.weight,
                input_,
                layer.embedding_dim,
                dtype=layer.params_dtype,
            )
        ids = input_.reshape(-1)
        output = torch.empty(
            (*input_.shape, layer.embedding_dim),
            dtype=layer.params_dtype,
            device=input_.device,
        )
        if ids.numel():
            _lookup_iq4_nl_ple_embedding_kernel[(ids.numel(),)](
                layer.weight,
                self._codebook(layer.weight.device),
                ids,
                output,
                layer.embedding_dim,
                layer.weight.shape[1],
                0,
                layer.weight.shape[0],
                BLOCK_D=triton.next_power_of_2(layer.embedding_dim),
            )
        return output

    def lookup_from_pinned(
        self,
        layer: nn.Module,
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if ids.numel() == 0:
            return
        _lookup_iq4_nl_ple_embedding_kernel[(ids.numel(),)](
            layer._uva_weight,
            self._codebook(layer._uva_weight.device),
            ids,
            output,
            layer.embedding_dim,
            layer._uva_weight.shape[1],
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=layer._block_d,
        )

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        # IQ4_NL rows are decoded during lookup, before the ETP reduction.
        return embeddings.to(output_dtype)


def patch_qwen4_exp_ple_embedding_method() -> None:
    """Select the IQ4_NL method for ``ple_embedding_dtype == "gguf_iq4_nl"``.

    Repeated calls are no-ops. Non-GGUF selection delegates to the original
    ``from_quant_config`` unchanged.
    """
    base = Qwen4ExpPLEEmbeddingMethod
    if getattr(base.from_quant_config, _PATCH_GUARD, False):
        return

    original_from_quant_config = base.from_quant_config

    @wraps(original_from_quant_config)
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> Qwen4ExpPLEEmbeddingMethod:
        if embedding_dtype == GGUF_IQ4_NL_PLE_DTYPE:
            return Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod()
        return original_from_quant_config(quant_config, prefix, embedding_dtype)

    setattr(from_quant_config, _PATCH_GUARD, True)
    base.from_quant_config = staticmethod(from_quant_config)


@triton.jit
def _lookup_iq4_nl_ple_embedding_kernel(
    weight_ptr,
    codebook_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    row_bytes,
    vocab_start,
    vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Decode one packed IQ4_NL row per program into the output table."""
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    owned = (idx >= vocab_start) & (idx < vocab_end)
    local_idx = tl.where(owned, idx - vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    value_mask = offsets < embedding_dim
    load_mask = owned & value_mask
    block_index = offsets // 32
    value_in_block = offsets % 32
    byte_index = tl.where(value_in_block < 16, value_in_block, value_in_block - 16)
    # The table is uint8; widen every loaded byte before shifting or masking.
    # Triton keeps uint8 arithmetic in uint8, so `byte << 8` or `bits & 0x3FF`
    # on the raw load would overflow (or fail to compile).
    code = tl.load(
        weight_ptr + local_idx * row_bytes + block_index * 18 + 2 + byte_index,
        mask=load_mask,
        other=0,
    ).to(tl.int32)
    code = tl.where(value_in_block < 16, code & 0xF, (code >> 4) & 0xF)
    block_base = weight_ptr + local_idx * row_bytes + block_index * 18
    scale_low = tl.load(block_base, mask=load_mask, other=0).to(tl.int32)
    scale_high = tl.load(block_base + 1, mask=load_mask, other=0).to(tl.int32)
    # Little-endian float16 bits rebuilt from the two raw bytes.
    scale_bits = scale_low | (scale_high << 8)
    sign = (scale_bits >> 15) & 1
    exponent = (scale_bits >> 10) & 0x1F
    mantissa = scale_bits & 0x3FF
    # Normal scales are (1024 + m) * 2**e * 2**-25. Each factor is exactly
    # representable in float32 (an 11-bit integer, a power of two, and the
    # literal 2**-25) and every product has at most 11 significant bits inside
    # the float32 range, so the multiplication is exact. This avoids libdevice
    # exp2/pow, whose results are not correctly rounded for all exponents.
    # Exponent 31 (inf/NaN) is clamped before the shift and selected below.
    normal_exponent = tl.where(exponent == 31, 0, exponent)
    normal = (
        (mantissa + 1024).to(tl.float32)
        * (1 << normal_exponent).to(tl.float32)
        * 2.9802322387695312e-08
    )
    scale = tl.where(
        exponent == 0,
        # Subnormal: m * 2**-24, with 2**-24 written as its exact literal.
        mantissa.to(tl.float32) * 5.9604644775390625e-08,
        tl.where(
            exponent == 31,
            tl.where(mantissa == 0, float("inf"), float("nan")),
            normal,
        ),
    )
    scale = tl.where(sign == 1, -scale, scale)
    values = tl.load(codebook_ptr + code, mask=load_mask, other=0.0) * scale
    tl.store(output_ptr + row * embedding_dim + offsets, values, mask=value_mask)
