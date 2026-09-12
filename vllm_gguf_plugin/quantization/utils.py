# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Mapping
from types import MappingProxyType

from gguf import GGMLQuantizationType as WeightType
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_layer_skipped_gguf(
    prefix: str,
    unquantized_modules: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
):
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                shard_prefix in module_name for module_name in unquantized_modules
            )

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(module_name in prefix for module_name in unquantized_modules)

    assert is_skipped is not None
    return is_skipped


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES

# IQ4_XS and IQ4_NL have no CUDA MMQ kernel, so they are absent from
# MMQ_QUANT_TYPES. But triton/fused_moe/iq_quant/iq4_xs.py is a real tl.dot tile
# GEMM that dequantizes in registers, it is registered in TRITON_MOE_DISPATCH,
# and ops.ggml_moe_a8 already falls back to it whenever the CUDA kernel is
# missing. Nothing ever reaches it: the gate in fused_moe.py stops IQ types
# first, so every prefill chunk of an IQ4_XS MoE goes through ggml_moe_a8_vec, a
# kernel written for decode. On the RTX PRO 6000 box that is what caps prefill at
# ~790 tok/s, which in turn sets the scheduler step time that makes short
# requests wait behind long reads.
#
# This is deliberately a separate set rather than a wider MMQ_QUANT_TYPES.
# linear.py reads MMQ_QUANT_TYPES too, with a threshold of x.shape[0] <= 2..6,
# so widening it there would also reroute the attention projections and the
# shared expert. Keep that blast radius out of this change; MoE only.
MMQ_MOE_TRITON_TYPES = MMQ_QUANT_TYPES | {
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}

# Rollback without reinstalling: with the switch off the dispatch is
# byte-equivalent to using MMQ_QUANT_TYPES directly.
if os.environ.get("GGUF_PLUGIN_IQ_MOE_MMQ", "1") != "1":
    MMQ_MOE_TRITON_TYPES = MMQ_QUANT_TYPES
