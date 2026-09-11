# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from gguf import GGUFWriter, GGMLQuantizationType
from transformers import PretrainedConfig

from vllm_gguf_plugin.gguf_files import GGUFModelFiles
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
        get_adapter_architecture(config) == "Qwen4ExpForConditionalGeneration"
    )
    # The pinned nightly image must actually register the official architecture
    # and its module path must import (the registry's target is
    # vllm.models.qwen4_exp, which re-exports the platform model class).
    from vllm.model_executor.models import ModelRegistry
    from vllm.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    assert "Qwen4ExpForConditionalGeneration" in ModelRegistry.get_supported_archs()
    assert Qwen4ExpForConditionalGeneration.__name__ == "Qwen4ExpForConditionalGeneration"


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


def _load_tensor_directory_fixture() -> dict:
    return json.loads(
        (Path(__file__).parent
         / "fixtures/qwen4_exp_iq4_xs_tensor_directory.json").read_text()
    )


def test_qwen4_exp_replays_all_fixture_names_to_unique_targets(monkeypatch):
    """Every one of the 1,224 real tensor names maps to exactly one target."""
    from vllm_gguf_plugin.weights_adapter import qwen4_exp

    fixture = _load_tensor_directory_fixture()
    names = {t["name"] for t in fixture["tensors"]}
    assert len(names) == fixture["tensor_count"] == 1224

    monkeypatch.setattr(qwen4_exp, "get_gguf_tensor_names", lambda _: names)
    files = SimpleNamespace(all_files=("fixture.gguf",) * 3, mm_proj=None)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp", get_text_config=lambda: SimpleNamespace()
        )
    )

    mapped = qwen4_exp.Qwen4ExpGGUFAdapter().build_name_map(files, model_config)

    assert len(mapped) == 1224
    ple_target = mapped["per_layer_token_embd.weight"]
    assert ple_target == "model.language_model.per_layer_token_embd.weight"
    non_ple_targets = [
        target for name, target in mapped.items() if name != "per_layer_token_embd.weight"
    ]
    assert len(non_ple_targets) == 1223
    assert len(set(non_ple_targets)) == 1223, "duplicate mapping targets"
    assert ple_target not in set(non_ple_targets)


def test_qwen4_exp_ple_expands_to_nightly_shard_names_resolving_under_the_model():
    """The PLE table expands to exactly the shard names the nightly accepts.

    The expected names are derived, not hand-written: the checkpoint prefix
    comes from ``Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper``, the
    module path segments from the nightly decoder-layer / PLE / ngram
    attribute assignments, and the zero-based layer index from the
    1-based ``ple_layer_ids`` convention the decoder layer implements.
    """
    import inspect

    from vllm_gguf_plugin.weights_adapter import qwen4_exp
    from vllm.models.qwen4_exp import Qwen4ExpForConditionalGeneration
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpDecoderLayer
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer

    text_config = SimpleNamespace(
        ple_layer_ids=[2],
        split_ngram_parts=128,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        ple_embed_dim=2560,
    )
    shard_names = qwen4_exp._ngram_ple_shard_names(text_config)
    assert len(shard_names) == 128

    # 1-based config ids attach to zero-based layer id - 1.
    assert "(self.layer_idx + 1) in ple_layer_ids" in inspect.getsource(
        Qwen4ExpDecoderLayer.__init__
    )
    assert "self.ple = Qwen4ExpPLELayer(" in inspect.getsource(
        Qwen4ExpDecoderLayer.__init__
    )
    assert "self.ple_embedding = Qwen4ExpNGramEmbedding(" in inspect.getsource(
        Qwen4ExpPLELayer.__init__
    )
    assert "self.ngram_embedding = embedding_cls(" in inspect.getsource(
        Qwen4ExpNGramEmbedding.__init__
    )
    # The nightly loader accepts ngram_embedding.shard_{i}.weight names.
    assert "ngram_embedding.shard_" in inspect.getsource(
        Qwen4ExpNGramEmbedding.load_weights
    )

    mapper = Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper
    language_prefix = mapper.orig_to_new_prefix["model.language_model."]
    for i, name in enumerate(shard_names):
        expected = (
            "model.language_model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{i}.weight"
        )
        assert name == expected
        # Under the nightly mapper the checkpoint name reaches the PLE module.
        assert mapper.apply_list([name]) == [
            f"{language_prefix}layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{i}.weight"
        ]


def test_qwen4_exp_padded_ngram_vocab_matches_nightly_layout():
    """The plugin's prime layout must equal the nightly's, padded or not."""
    from vllm_gguf_plugin.weights_adapter import qwen4_exp
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding

    base = 20_000_000
    ngram_heads = (3 - 1) * 8
    _, _, total = Qwen4ExpNGramEmbedding._make_vocab_layout(
        ngram_vocab_size_base=base,
        ngram_heads=ngram_heads,
        ple_dense_layer_id=0,
    )
    text_config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=base,
        make_ngram_vocab_size_divisible_by=1,
    )
    assert qwen4_exp._padded_ngram_vocab_size(text_config) == total

    # The real checkpoint: padding to 128 reproduces the GGUF shape exactly.
    text_config.make_ngram_vocab_size_divisible_by = 128
    padded = qwen4_exp._padded_ngram_vocab_size(text_config)
    assert padded == ((total + 127) // 128) * 128
    fixture = _load_tensor_directory_fixture()
    ple = next(t for t in fixture["tensors"] if t["name"] == "per_layer_token_embd.weight")
    assert ple["shape"] == [160, padded]


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


_PLE_TEST_ROWS = 618  # sum of the two primes 307 and 311 (see below)


def _ple_transform_config(**overrides):
    base = {
        "ple_layer_ids": [2],
        "split_ngram_parts": 3,
        "ngram_size": 2,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 300,
        "make_ngram_vocab_size_divisible_by": 1,
        "ple_embed_dim": 320,
    }
    base.update(overrides)
    return SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(**base)
        ),
    )


def test_qwen4_exp_synthentic_ple_vocab_matches_nightly_layout():
    """The synthetic config's padded vocab must equal the nightly's layout."""
    from vllm_gguf_plugin.weights_adapter import qwen4_exp
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding

    _, _, total = Qwen4ExpNGramEmbedding._make_vocab_layout(
        ngram_vocab_size_base=300, ngram_heads=2, ple_dense_layer_id=0
    )
    assert total == _PLE_TEST_ROWS
    assert qwen4_exp._padded_ngram_vocab_size(
        SimpleNamespace(
            ngram_size=2,
            heads_per_ngram=2,
            ngram_vocab_size_base=300,
            make_ngram_vocab_size_divisible_by=1,
        )
    ) == _PLE_TEST_ROWS


def _ple_weights(row_count: int, include_companion: bool = True):
    weight = torch.randint(0, 256, (row_count * 90,), dtype=torch.uint8)
    weights = []
    if include_companion:
        # The iterator yields the synthetic weight_type companion first.
        weights.append(
            ("model.language_model.per_layer_token_embd.weight_type", torch.tensor(20))
        )
    weights.append(("model.language_model.per_layer_token_embd.weight", weight))
    return weights


def test_qwen4_exp_streams_ple_shards_zero_copy_and_byte_exact():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    rows = _PLE_TEST_ROWS
    weights = _ple_weights(rows)

    out = list(Qwen4ExpGGUFAdapter().transform_weights(weights, _ple_transform_config()))

    assert [name for name, _ in out] == [
        f"model.language_model.layers.1.ple.ple_embedding."
        f"ngram_embedding.shard_{i}.weight"
        for i in range(3)
    ]
    tensors = [tensor for _, tensor in out]
    assert all(tensor.dtype == torch.uint8 for tensor in tensors)
    assert [tuple(tensor.shape) for tensor in tensors] == [
        (206, 90),
        (206, 90),
        (206, 90),
    ]
    # Byte-exact: the shards concatenate back to the source table.
    assert torch.cat(tensors).reshape(-1).equal(weights[1][1])
    # Zero-copy: every shard aliases the source's memory map.
    base_ptr = weights[1][1].data_ptr()
    end_ptr = base_ptr + weights[1][1].numel()
    for tensor in tensors:
        assert base_ptr <= tensor.data_ptr() < end_ptr
    # The weight_type companion is not yielded.
    assert not any(name.endswith(".weight_type") for name, _ in out)


def test_qwen4_exp_ple_shards_have_uneven_and_empty_tail():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    # R = 5 + 7 = 12 (base 4), parts = 5 -> shard_size 3 -> [3,3,3,3,0]:
    # an uneven last shard would be impossible here, so also cover the
    # uneven case separately below.
    rows = 12
    weights = _ple_weights(rows, include_companion=False)

    out = list(
        Qwen4ExpGGUFAdapter().transform_weights(
            weights, _ple_transform_config(ngram_vocab_size_base=4, split_ngram_parts=5)
        )
    )

    assert [name for name, _ in out] == [
        f"model.language_model.layers.1.ple.ple_embedding."
        f"ngram_embedding.shard_{i}.weight"
        for i in range(5)
    ]
    shapes = [tuple(tensor.shape) for _, tensor in out]
    assert shapes == [(3, 90), (3, 90), (3, 90), (3, 90), (0, 90)]
    assert all(tensor.dtype == torch.uint8 for _, tensor in out)
    assert torch.cat([tensor for _, tensor in out]).reshape(-1).equal(weights[0][1])

    # Uneven tail: R = 618, parts = 5 -> shard_size 124 -> [124,124,124,124,122].
    weights = _ple_weights(_PLE_TEST_ROWS, include_companion=False)
    out = list(
        Qwen4ExpGGUFAdapter().transform_weights(
            weights, _ple_transform_config(split_ngram_parts=5)
        )
    )
    shapes = [tuple(tensor.shape) for _, tensor in out]
    assert shapes == [(124, 90)] * 4 + [(122, 90)]
    assert torch.cat([tensor for _, tensor in out]).reshape(-1).equal(weights[0][1])


def test_qwen4_exp_ple_shard_names_match_the_nightly_split():
    """Row counts must mirror the nightly load_weights shard math."""
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    rows = _PLE_TEST_ROWS
    parts = 7
    weights = _ple_weights(rows, include_companion=False)
    out = list(
        Qwen4ExpGGUFAdapter().transform_weights(
            weights, _ple_transform_config(split_ngram_parts=parts)
        )
    )
    shard_size = (rows + parts - 1) // parts
    expected_rows = [
        max(0, min(shard_size, rows - i * shard_size)) for i in range(parts)
    ]
    assert expected_rows == [89, 89, 89, 89, 89, 89, 84]
    assert [tuple(tensor.shape)[0] for _, tensor in out] == expected_rows


def test_qwen4_exp_rejects_ple_row_count_mismatch():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    adapter = Qwen4ExpGGUFAdapter()
    for bad_rows in (_PLE_TEST_ROWS + 1, _PLE_TEST_ROWS - 1):
        weights = _ple_weights(bad_rows, include_companion=False)
        with pytest.raises(ValueError, match=str(_PLE_TEST_ROWS)):
            list(adapter.transform_weights(weights, _ple_transform_config()))


def test_qwen4_exp_rejects_ple_payload_with_bad_byte_length():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    weight = torch.randint(
        0, 256, (_PLE_TEST_ROWS * 90 + 7,), dtype=torch.uint8
    )
    weights = [("model.language_model.per_layer_token_embd.weight", weight)]

    with pytest.raises(ValueError, match=str(_PLE_TEST_ROWS * 90)):
        list(Qwen4ExpGGUFAdapter().transform_weights(weights, _ple_transform_config()))


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


def _make_small_gguf(tmp_path: Path, ple_type: GGMLQuantizationType | None) -> str:
    """Write a tiny GGUF whose only tensor is the packed PLE table.

    Each row is one packed IQ4_NL-equivalent width expressed in the byte size
    of the requested quant type (IQ4_NL 90, Q8_0 34, IQ4_XS 136 bytes/row), so
    the header type is what the tests assert on.
    """
    path = tmp_path / "model.gguf"
    writer = GGUFWriter(str(path), "qwen4_exp")
    if ple_type is not None:
        from gguf.constants import GGML_QUANT_SIZES

        bytes_per_row = GGML_QUANT_SIZES[ple_type][1]
        ple = np.arange(3 * bytes_per_row, dtype=np.uint8).reshape(3, bytes_per_row)
        writer.add_tensor("per_layer_token_embd.weight", ple, raw_dtype=ple_type)
    else:
        other = np.zeros((4,), dtype=np.float32)
        writer.add_tensor("token_embd.weight", other)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


def _qwen4_hf_config(ple_layer_ids: list[int] | None = [2]) -> PretrainedConfig:
    config = PretrainedConfig(model_type="qwen4_exp")
    config.ple_layer_ids = ple_layer_ids or []
    return config


def test_qwen4_exp_iq4_nl_ple_marks_the_text_config(tmp_path):
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    path = _make_small_gguf(tmp_path, GGMLQuantizationType.IQ4_NL)
    config = _qwen4_hf_config([2])

    patched = Qwen4ExpGGUFAdapter().patch_hf_config(
        GGUFModelFiles(backbone=(path,)), config
    )

    assert patched.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert patched.get_text_config().ple_embedding_dtype == "gguf_iq4_nl"


def test_qwen4_exp_without_ple_layers_needs_no_marker(tmp_path):
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    path = _make_small_gguf(tmp_path, None)
    config = _qwen4_hf_config([])

    patched = Qwen4ExpGGUFAdapter().patch_hf_config(
        GGUFModelFiles(backbone=(path,)), config
    )

    assert patched.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert not hasattr(patched.get_text_config(), "ple_embedding_dtype")


@pytest.mark.parametrize(
    "ple_type", [GGMLQuantizationType.Q8_0, GGMLQuantizationType.IQ4_XS]
)
def test_qwen4_exp_rejects_non_iq4_nl_ple_type_naming_the_type(tmp_path, ple_type):
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    path = _make_small_gguf(tmp_path, ple_type)
    config = _qwen4_hf_config([2])

    with pytest.raises(NotImplementedError) as excinfo:
        Qwen4ExpGGUFAdapter().patch_hf_config(
            GGUFModelFiles(backbone=(path,)), config
        )
    assert ple_type.name in str(excinfo.value)


def test_qwen4_exp_rejects_missing_ple_tensor_when_ple_enabled(tmp_path):
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    path = _make_small_gguf(tmp_path, None)
    config = _qwen4_hf_config([2])

    with pytest.raises(NotImplementedError, match="per_layer_token_embd"):
        Qwen4ExpGGUFAdapter().patch_hf_config(
            GGUFModelFiles(backbone=(path,)), config
        )


def test_qwen4_exp_rejects_vision_outside_this_text_only_experiment():
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    with pytest.raises(NotImplementedError, match="text-only"):
        Qwen4ExpGGUFAdapter().build_name_map(
            SimpleNamespace(mm_proj="vision.gguf"), _transform_config()
        )
