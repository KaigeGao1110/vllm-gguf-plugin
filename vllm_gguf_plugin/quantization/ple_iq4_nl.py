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
        local_rows = sum(output_partition_sizes)

        def wrapped_weight_loader(param, loaded_weight, *args, **kwargs):
            if loaded_weight.dtype != torch.uint8:
                raise ValueError(
                    "IQ4_NL PLE shards must be packed uint8 bytes, "
                    f"got {loaded_weight.dtype}"
                )
            original_weight_loader(param, loaded_weight, *args, **kwargs)
            checkpoint_start = kwargs.get("checkpoint_start")
            if checkpoint_start is None:
                loaded_ranges.add((0, local_rows))
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