# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm_gguf_plugin.weights_adapter import (
    get_adapter_architecture,
    get_weights_adapter,
)


def _qwen4_config(model_type: str = "qwen4_exp"):
    return SimpleNamespace(
        model_type=model_type,
        architectures=["Qwen4ExpForConditionalGeneration"],
    )


def test_qwen4_exp_registers_a_dedicated_native_architecture_adapter():
    config = _qwen4_config()

    adapter = get_weights_adapter(config)

    assert type(adapter).__name__ == "Qwen4ExpGGUFAdapter"
    assert (
        get_adapter_architecture(config) == "Qwen3_8FlashNextForConditionalGeneration"
    )


def test_qwen4_exp_does_not_match_other_qwen_architectures():
    config = _qwen4_config("qwen3_5")

    adapter = get_weights_adapter(config)

    assert type(adapter).__name__ != "Qwen4ExpGGUFAdapter"


def test_qwen4_exp_does_not_claim_the_unqualified_standalone_text_config():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    assert not Qwen4ExpGGUFAdapter.matches(_qwen4_config("qwen4_exp_text"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("token_embd.weight", "model.language_model.embed_tokens.weight"),
        (
            "blk.3.attn_q.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight",
        ),
        (
            "blk.3.indexer.q_proj.weight",
            "model.language_model.layers.3.self_attn.indexer.q_proj.weight",
        ),
        (
            "blk.1.ple_key.weight",
            "model.language_model.layers.1.ple.key_proj.weight",
        ),
        (
            "blk.0.hc_attn_down.weight",
            "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        ),
        (
            "output_hc_up.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        ),
    ],
)
def test_qwen4_exp_maps_converter_names_to_native_loader_names(
    monkeypatch, raw, expected
):
    from vllm_gguf_plugin.weights_adapter import qwen4_exp

    monkeypatch.setattr(qwen4_exp, "get_gguf_tensor_names", lambda _: {raw})
    files = SimpleNamespace(all_files=("fixture.gguf",), mm_proj=None)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp", get_text_config=lambda: SimpleNamespace()
        )
    )

    mapped = qwen4_exp.Qwen4ExpGGUFAdapter().build_name_map(files, model_config)

    assert mapped == {raw: expected}


def test_qwen4_exp_rejects_unknown_converter_tensor_names(monkeypatch):
    from vllm_gguf_plugin.weights_adapter import qwen4_exp

    raw = "blk.0.this_name_is_not_in_the_converter.weight"
    monkeypatch.setattr(qwen4_exp, "get_gguf_tensor_names", lambda _: {raw})
    files = SimpleNamespace(all_files=("fixture.gguf",), mm_proj=None)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp", get_text_config=lambda: SimpleNamespace()
        )
    )

    with pytest.raises(RuntimeError, match="unmapped Qwen4Exp GGUF tensor"):
        qwen4_exp.Qwen4ExpGGUFAdapter().build_name_map(files, model_config)


def test_qwen4_exp_joins_converter_indexer_qk_split_for_native_loader():
    from vllm_gguf_plugin.weights_adapter import qwen4_exp

    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                linear_num_key_heads=0,
                linear_num_value_heads=0,
            )
        )
    )
    q = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    k = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4) + 20
    weights = [
        ("model.language_model.layers.3.self_attn.indexer.k_proj.weight", k),
        ("model.language_model.layers.3.self_attn.indexer.q_proj.weight", q),
    ]

    mapped = list(qwen4_exp.Qwen4ExpGGUFAdapter().transform_weights(weights, config))

    assert [name for name, _ in mapped] == [
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"
    ]
    assert torch.equal(mapped[0][1], torch.cat((q, k), dim=0))


def test_qwen4_exp_surfaces_packed_ple_table_as_unsupported_at_load():
    from vllm_gguf_plugin.weights_adapter import qwen4_exp

    adapter = qwen4_exp.Qwen4ExpGGUFAdapter()
    weights = [("model.language_model.per_layer_token_embd.weight", object())]
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                linear_num_key_heads=0,
                linear_num_value_heads=0,
            )
        )
    )

    with pytest.raises(NotImplementedError, match="packed PLE table"):
        list(adapter.transform_weights(weights, config))


def _transform_config():
    return SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                linear_num_key_heads=0, linear_num_value_heads=0
            )
        ),
    )


@pytest.mark.parametrize("norm", ["norm_key", "norm_query", "norm_conv"])
def test_qwen4_exp_restores_ple_zero_centered_norms(norm):
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    name = f"model.language_model.layers.1.ple.{norm}.weight"
    weights = [(name, torch.full((8,), 1.125))]
    actual = list(Qwen4ExpGGUFAdapter().transform_weights(weights, _transform_config()))
    assert actual[0][0] == name
    torch.testing.assert_close(actual[0][1], torch.full((8,), 0.125), rtol=0, atol=0)


def test_qwen4_exp_rejects_unqualified_quantized_indexer_merge():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    weights = [
        (
            "model.language_model.layers.3.self_attn.indexer.q_proj.weight_type",
            torch.tensor(8),
        )
    ]
    with pytest.raises(NotImplementedError, match="quantized indexer"):
        list(Qwen4ExpGGUFAdapter().transform_weights(weights, _transform_config()))


def test_qwen4_exp_decodes_actual_q8_hc_projection_for_native_float_loader():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/qwen4_exp_hc_samples.json").read_text()
    )
    sample = next(s for s in fixture["samples"] if s["type_name"] == "Q8_0")
    raw = base64.b64decode(sample["packed_base64"])
    packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
        sample["rows"], -1
    )
    expected = gguf.dequantize(
        np.frombuffer(raw, dtype=np.uint8).reshape(sample["rows"], -1),
        gguf.GGMLQuantizationType.Q8_0,
    )
    name = (
        "model.language_model.layers.0.attn_hyper_connection."
        "input_mix_weight_down.weight"
    )
    actual = list(
        Qwen4ExpGGUFAdapter().transform_weights(
            [
                (
                    name.removesuffix(".weight") + ".weight_type",
                    torch.tensor(sample["type_id"]),
                ),
                (name, packed),
            ],
            _transform_config(),
        )
    )
    assert [n for n, _ in actual] == [name]
    torch.testing.assert_close(
        actual[0][1],
        torch.from_numpy(expected.copy()).to(torch.bfloat16),
        rtol=0,
        atol=0,
    )


def test_qwen4_exp_refuses_unwired_ple_before_file_access_or_model_construction():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(ple_layer_ids=[2]))
    with pytest.raises(NotImplementedError, match="CPU PLE worker"):
        Qwen4ExpGGUFAdapter().patch_hf_config(SimpleNamespace(), config)


def test_qwen4_exp_rejects_vision_outside_this_text_only_experiment():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    with pytest.raises(NotImplementedError, match="text-only"):
        Qwen4ExpGGUFAdapter().build_name_map(
            SimpleNamespace(mm_proj="vision.gguf"), _transform_config()
        )
