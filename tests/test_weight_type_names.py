# SPDX-License-Identifier: Apache-2.0

"""Names of the synthetic ``weight_type`` companions the GGUF iterator yields.

The iterator yields a ``<base>.weight_type`` tensor before every quantized
``<base>.weight``. Only the parameter-name suffix may change: Qwen4Exp's
hyper-connection projections are named ``input_mix_weight_down.weight``, and
replacing every ``weight`` produced ``input_mix_weight_type_down.weight_type``,
which no adapter recognised and no module owns (the first full load of
Qwen3.8-Flash-Next UD-IQ4_XS failed on exactly that name).
"""

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, GGUFWriter

from vllm_gguf_plugin.weight_utils import (
    gguf_quant_weights_iterator_multi,
    gguf_weight_type_name,
)

_HC_GGUF = "output_hc_down.weight"
_HC_NATIVE = "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"
_PLAIN_GGUF = "blk.0.attn_q.weight"
_PLAIN_NATIVE = "model.language_model.layers.0.self_attn.q_proj.weight"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight_type",
        ),
        (_HC_NATIVE, _HC_NATIVE + "_type"),
        (
            "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
            "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight_type",
        ),
        (
            "model.layers.0.mlp.experts.w13_weight",
            "model.layers.0.mlp.experts.w13_weight_type",
        ),
        (
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_q.weight_type",
        ),
    ],
)
def test_weight_type_name_changes_only_the_last_weight(name, expected):
    assert gguf_weight_type_name(name) == expected


def test_weight_type_name_rejects_names_without_weight():
    with pytest.raises(ValueError, match="weight"):
        gguf_weight_type_name("model.layers.0.linear_attn.dt_bias")


def _q8_rows(rows: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(-2.0, 2.0, rows * width, dtype=np.float32).reshape(rows, width)
    packed = gguf.quants.quantize(values, GGMLQuantizationType.Q8_0)
    return packed, gguf.quants.dequantize(packed, GGMLQuantizationType.Q8_0)


def _write_gguf(tmp_path) -> tuple[str, np.ndarray]:
    path = tmp_path / "model.gguf"
    writer = GGUFWriter(str(path), "qwen4_exp")
    hc_packed, hc_expected = _q8_rows(3, 32)
    writer.add_tensor(_HC_GGUF, hc_packed, raw_dtype=GGMLQuantizationType.Q8_0)
    plain_packed, _ = _q8_rows(2, 32)
    writer.add_tensor(_PLAIN_GGUF, plain_packed, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path), hc_expected


def test_iterator_names_companions_after_the_mapped_parameter(tmp_path):
    path, _ = _write_gguf(tmp_path)
    names = [
        name
        for name, _ in gguf_quant_weights_iterator_multi(
            [path], {_HC_GGUF: _HC_NATIVE, _PLAIN_GGUF: _PLAIN_NATIVE}
        )
    ]
    assert names == [
        _HC_NATIVE + "_type",
        _HC_NATIVE,
        _PLAIN_NATIVE + "_type",
        _PLAIN_NATIVE,
    ]


def test_real_iterator_output_reaches_the_qwen4_exp_hc_decoder(tmp_path):
    """Wiring guard: iterator names feed the adapter unchanged, as the loader does."""
    from types import SimpleNamespace

    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    path, hc_expected = _write_gguf(tmp_path)
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                linear_num_key_heads=0, linear_num_value_heads=0
            )
        ),
    )
    out = list(
        Qwen4ExpGGUFAdapter().transform_weights(
            gguf_quant_weights_iterator_multi([path], {_HC_GGUF: _HC_NATIVE}),
            config,
        )
    )

    assert [name for name, _ in out] == [_HC_NATIVE]
    torch.testing.assert_close(
        out[0][1],
        torch.from_numpy(np.ascontiguousarray(hc_expected)).to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
