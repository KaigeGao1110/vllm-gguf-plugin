# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import torch
from vllm.logger import init_logger

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names, split_stacked_experts
from .base import GGUFWeight
from .qwen3_5 import Qwen35GGUFAdapter, _gdn_value_head_layout

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN4_EXP_MODEL_TYPES = ("qwen4_exp",)
# Official architecture name registered in the pinned nightly image's
# ModelRegistry (vllm/models/qwen4_exp); the old qwen3_8_flash_next
# stand-in name is no longer used.
QWEN4_EXP_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"

_LAYER_SUBSTR = {
    # Gated delta-net / linear-attention names emitted by the pinned converter.
    "attn_qkv.": "linear_attn.in_proj_qkv.",
    "attn_gate.": "linear_attn.in_proj_z.",
    "ssm_alpha.": "linear_attn.in_proj_a.",
    "ssm_beta.": "linear_attn.in_proj_b.",
    "ssm_conv1d.": "linear_attn.conv1d.",
    "ssm_norm.": "linear_attn.norm.",
    "ssm_out.": "linear_attn.out_proj.",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a.weight": "linear_attn.A_log.weight",
    "ssm_a": "linear_attn.A_log",
    # Full attention / QSA names.
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    # The converter splits HF index_qk_proj into q/k GGUF tensors.  These are
    # intermediate names; transform_weights joins them into native
    # self_attn.indexer.index_qk_proj before AutoWeightsLoader sees them.
    "indexer.q_proj.": "self_attn.indexer.q_proj.",
    "indexer.k_proj.": "self_attn.indexer.k_proj.",
    "indexer.q_norm.": "self_attn.indexer.q_layernorm.",
    "indexer.k_norm.": "self_attn.indexer.k_layernorm.",
    # Hyper-connection projections replace the ordinary attention/FFN norms.
    "hc_attn_norm.": "attn_hyper_connection.hc_norm.",
    "hc_attn_down.": "attn_hyper_connection.input_mix_weight_down.",
    "hc_attn_up.": "attn_hyper_connection.input_mix_weight_up.",
    "hc_attn_inject.": "attn_hyper_connection.block_inject_weight.",
    "hc_ffn_norm.": "mlp_hyper_connection.hc_norm.",
    "hc_ffn_down.": "mlp_hyper_connection.input_mix_weight_down.",
    "hc_ffn_up.": "mlp_hyper_connection.input_mix_weight_up.",
    "hc_ffn_inject.": "mlp_hyper_connection.block_inject_weight.",
    # PLE exists only at the converter's virtual layer (blk.1 for 1-based id 2).
    "ple_key.": "ple.key_proj.",
    "ple_value.": "ple.value_proj.",
    "ple_norm_key.": "ple.norm_key.",
    "ple_norm_query.": "ple.norm_query.",
    "ple_norm_conv.": "ple.norm_conv.",
    "ple_conv1d.": "ple.conv1d.",
    # Mixture-of-experts and shared expert names.
    "ffn_gate_inp_shexp.": "mlp.shared_expert_gate.",
    "ffn_gate_inp.": "mlp.gate.",
    "ffn_gate_exps.": "mlp.experts.0.gate_proj.",
    "ffn_up_exps.": "mlp.experts.0.up_proj.",
    "ffn_down_exps.": "mlp.experts.0.down_proj.",
    "ffn_gate_shexp.": "mlp.shared_expert.gate_proj.",
    "ffn_up_shexp.": "mlp.shared_expert.up_proj.",
    "ffn_down_shexp.": "mlp.shared_expert.down_proj.",
}

_TOP_PREFIX = {
    "token_embd.": "embed_tokens.",
    "output_norm.": "norm.",
    "output.": "lm_head.",
    "output_hc_norm.": "hyper_connection_mixer.hc_norm.",
    "output_hc_down.": "hyper_connection_mixer.input_mix_weight_down.",
    "output_hc_up.": "hyper_connection_mixer.input_mix_weight_up.",
}

_PLE_PACKED_SUFFIX = "per_layer_token_embd.weight"


def _is_hc_projection(name: str) -> bool:
    return "hyper_connection" in name and name.removesuffix(".weight").rsplit(".", 1)[
        -1
    ] in {"input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"}


def _map_layer_name(name: str, backbone_prefix: str) -> str | None:
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if match is None:
        return None
    layer_prefix = f"{backbone_prefix}layers.{match.group(1)}."
    suffix = match.group(2)
    for source, target in _LAYER_SUBSTR.items():
        if suffix.startswith(source):
            return layer_prefix + target + suffix[len(source) :]
    return None


def _find_gguf_tensor_type(
    files: GGUFModelFiles, name: str
) -> gguf.GGMLQuantizationType | None:
    """Return the GGML type of *name* from the GGUF header, or ``None``.

    Reading ``reader.tensors`` touches only the header and tensor index; no
    payload bytes are read.
    """
    for path in files.backbone:
        for tensor in gguf.GGUFReader(path).tensors:
            if tensor.name == name:
                return tensor.tensor_type
    return None


def _map_name(name: str, backbone_prefix: str, *, multimodal: bool) -> str | None:
    for source, target in _TOP_PREFIX.items():
        if name.startswith(source):
            if target == "lm_head.":
                return target + name[len(source) :]
            return backbone_prefix + target + name[len(source) :]
    if name == _PLE_PACKED_SUFFIX:
        # Intentionally a visible intermediate target: the iterator hands the
        # packed table to transform_weights, which expands it into the native
        # ngram_embedding.shard_{i}.weight shards.
        return backbone_prefix + "per_layer_token_embd.weight"
    return _map_layer_name(name, backbone_prefix)


def _is_prime64(value: int) -> bool:
    """Deterministic Miller-Rabin primality test for 64-bit integers.

    Mirrors ``Qwen4ExpNGramEmbedding._is_prime_64`` in the pinned nightly
    image (vllm/models/qwen4_exp), which derives the per-head n-gram
    vocabulary sizes; the plugin must stay in sync with it.
    """
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = (witness * witness) % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    """The ``count``-th prime strictly greater than *start*.

    Mirrors ``Qwen4ExpNGramEmbedding._nth_prime_after`` in the pinned nightly
    image.
    """
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime64(candidate):
            candidate += 2
        prime = candidate
    return prime


def _padded_ngram_vocab_size(text_config, ple_dense_layer_id: int = 0) -> int:
    """Total padded n-gram vocabulary rows the native loader allocates.

    Mirrors ``Qwen4ExpNGramEmbedding.__init__`` in the pinned nightly image:
    one prime-sized block per n-gram head (``ngram_heads`` of them for this
    PLE layer's dense id), padded to a multiple of
    ``make_ngram_vocab_size_divisible_by``.  The GGUF PLE table must have
    exactly this many rows.
    """
    ngram_heads = (int(text_config.ngram_size) - 1) * int(text_config.heads_per_ngram)
    base = int(text_config.ngram_vocab_size_base)
    total = sum(
        _nth_prime_after(base - 1, ple_dense_layer_id * ngram_heads + head + 1)
        for head in range(ngram_heads)
    )
    divisor = int(text_config.make_ngram_vocab_size_divisible_by)
    return ((total + divisor - 1) // divisor) * divisor


def _ngram_ple_prefix(text_config) -> str:
    """Checkpoint prefix of one PLE layer's ngram embedding module.

    ``ple_layer_ids`` entries are 1-based; the native decoder layer attaches
    ``self.ple`` to the zero-based layer whose id+1 appears in the list.
    The selected GGUF export carries a single packed PLE table, so exactly
    one entry is qualified.
    """
    ple_layer_ids = [int(x) for x in text_config.ple_layer_ids]
    if len(ple_layer_ids) != 1:
        raise NotImplementedError(
            "The selected GGUF export carries one packed PLE table, so a "
            f"single ple_layer_ids entry is qualified, got {ple_layer_ids}"
        )
    layer_index = ple_layer_ids[0] - 1
    return f"model.language_model.layers.{layer_index}.ple.ple_embedding."


def _ngram_ple_shard_names(text_config) -> tuple[str, ...]:
    """Native checkpoint names the PLE table expands to.

    ``split_ngram_parts`` consecutive shards named exactly as the nightly
    ``Qwen4ExpNGramEmbedding.load_weights`` accepts:
    ``ngram_embedding.shard_{i}.weight`` for ``i`` in ``0..parts-1``.
    """
    prefix = _ngram_ple_prefix(text_config)
    parts = int(text_config.split_ngram_parts)
    if parts <= 0:
        raise ValueError(f"split_ngram_parts must be positive, got {parts}")
    return tuple(
        f"{prefix}ngram_embedding.shard_{i}.weight" for i in range(parts)
    )


class Qwen4ExpGGUFAdapter(Qwen35GGUFAdapter):
    """Map the pinned Qwen4Exp GGUF export to native Qwen4Exp vLLM names.

    The converter emits the same gated-delta-net projections as Qwen3.5, but
    adds QSA, hyper-connections, and PLE tensors.  Inheriting the GDN layout
    declaration and restoration helpers keeps the V-head reorder tied to the
    existing, tested vLLM GGUF layout contract.

    The packed PLE table is declared zero-copy: the iterator yields it as a
    view of the memory-mapped GGUF payload so the 28.8 GB table is never
    privately copied on the way to the native PLE loader.
    """

    zero_copy_tensor_names = (_PLE_PACKED_SUFFIX,)

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        return config.model_type in QWEN4_EXP_MODEL_TYPES

    @classmethod
    def architecture(cls, config: PretrainedConfig) -> str | None:
        del config
        return QWEN4_EXP_ARCHITECTURE

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        text_config = hf_config.get_text_config()
        if getattr(text_config, "ple_layer_ids", []):
            ple_type = _find_gguf_tensor_type(files, _PLE_PACKED_SUFFIX)
            if ple_type is None:
                raise NotImplementedError(
                    "The config enables per-layer embeddings (ple_layer_ids) "
                    f"but the GGUF contains no {_PLE_PACKED_SUFFIX} tensor; "
                    "refusing to load without the packed PLE table."
                )
            if ple_type != gguf.GGMLQuantizationType.IQ4_NL:
                raise NotImplementedError(
                    f"Only the IQ4_NL packed PLE table has been qualified, "
                    f"but {_PLE_PACKED_SUFFIX} is {ple_type.name}; refusing "
                    "to expand the table."
                )
        patched = maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )
        patched.architectures = [QWEN4_EXP_ARCHITECTURE]
        if getattr(text_config, "ple_layer_ids", []):
            # Qwen4ExpNGramEmbedding selects its PLE embedding method from
            # this marker; the adapter streams the IQ4_NL rows still packed,
            # so no step allocates the expanded table.
            text_config.ple_embedding_dtype = "gguf_iq4_nl"
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        if files.mm_proj is not None:
            raise NotImplementedError(
                "This Qwen4Exp experiment is text-only; vision GGUF loading "
                "has not been adapted."
            )
        multimodal = False
        # Preserve checkpoint-style names even when no vision tower is loaded.
        # Both native causal and conditional model mappers accept this prefix.
        backbone_prefix = "model.language_model."
        tensor_names = sorted(get_gguf_tensor_names(files.all_files))
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in tensor_names:
            mapped = _map_name(name, backbone_prefix, multimodal=multimodal)
            if mapped is None:
                unmapped.append(name)
            else:
                name_map[name] = mapped
        if unmapped:
            raise RuntimeError(
                "Found unmapped Qwen4Exp GGUF tensor(s); refusing to drop "
                f"them: {unmapped}"
            )
        logger.info("Mapped %d Qwen4Exp GGUF tensors", len(name_map))
        del model_config
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        text_config = model_config.hf_config.get_text_config()
        layout = _gdn_value_head_layout(text_config)

        def transformed() -> Iterable[GGUFWeight]:
            quantized_bases: set[str] = set()
            quantized_types: dict[str, int] = {}
            indexer_parts: dict[str, dict[str, torch.Tensor]] = {}
            for name, weight in weights:
                if name.endswith(_PLE_PACKED_SUFFIX):
                    raise NotImplementedError(
                        "Qwen4Exp packed PLE table per_layer_token_embd.weight "
                        "requires the bounded IQ4_NL row lookup loader; refusing "
                        "to materialize or substitute the 102.4 GB BF16 table."
                    )
                if name.endswith(".weight_type"):
                    base = name.removesuffix(".weight_type")
                    quantized_bases.add(base)
                    if base.endswith((".indexer.q_proj", ".indexer.k_proj")):
                        raise NotImplementedError(
                            "Qwen4Exp quantized indexer Q/K merging is not "
                            "qualified; the selected checkpoint uses BF16."
                        )
                    if _is_hc_projection(base + ".weight"):
                        quant_type = int(weight.item())
                        if quant_type != int(gguf.GGMLQuantizationType.Q8_0):
                            raise NotImplementedError(
                                "Only the selected checkpoint's Q8_0 HC "
                                "projections have been qualified for decoding"
                            )
                        quantized_types[base] = quant_type
                        # Native HC modules deliberately use quant_config=None.
                        # Their loader accepts floating weights, not descriptors.
                        continue
                if (
                    _is_hc_projection(name)
                    and name.removesuffix(".weight") in quantized_types
                ):
                    if weight.device.type != "cpu":
                        raise ValueError("GGUF HC decoding expects CPU packed weights")
                    decoded = gguf.dequantize(
                        weight.detach().numpy(), gguf.GGMLQuantizationType.Q8_0
                    )
                    yield (
                        name,
                        torch.from_numpy(decoded).to(
                            getattr(model_config, "dtype", torch.bfloat16)
                        ),
                    )
                    continue
                if name.endswith(".self_attn.indexer.q_proj.weight"):
                    prefix = name.removesuffix(".q_proj.weight")
                    indexer_parts.setdefault(prefix, {})["q"] = weight
                    continue
                if name.endswith(".self_attn.indexer.k_proj.weight"):
                    prefix = name.removesuffix(".k_proj.weight")
                    indexer_parts.setdefault(prefix, {})["k"] = weight
                    continue
                if layout is not None:
                    reordered = self._restore_gdn_weight(
                        name, weight, text_config, layout, quantized_bases
                    )
                    if reordered is not None:
                        yield name, reordered
                        continue
                if name.endswith(".A_log"):
                    yield name, torch.log(-weight)
                    continue
                if (
                    name.endswith("norm.weight")
                    and not name.endswith("linear_attn.norm.weight")
                ) or name.endswith(
                    (
                        ".ple.norm_key.weight",
                        ".ple.norm_query.weight",
                        ".ple.norm_conv.weight",
                    )
                ):
                    yield name, weight - 1
                    continue
                if name.endswith(".conv1d.weight") and weight.dim() == 2:
                    weight = weight.unsqueeze(1)
                elif (
                    name.endswith(".weight")
                    and weight.dim() == 1
                    and "norm" not in name
                    and name.removesuffix(".weight") not in quantized_bases
                ):
                    weight = weight.unsqueeze(0)
                yield name, weight

            for prefix, parts in indexer_parts.items():
                if set(parts) != {"q", "k"}:
                    raise RuntimeError(
                        "Qwen4Exp indexer q/k projection pair is incomplete for "
                        f"{prefix}: got {sorted(parts)}"
                    )
                yield (
                    f"{prefix}.index_qk_proj.weight",
                    torch.cat((parts["q"], parts["k"]), dim=0),
                )

        yield from split_stacked_experts(transformed())


__all__ = ["Qwen4ExpGGUFAdapter"]
