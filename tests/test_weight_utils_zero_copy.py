# SPDX-License-Identifier: Apache-2.0

"""Zero-copy GGUF payload reads for adapter-declared tensors.

The plugin's weights iterator normally copies every tensor out of the
memory-mapped GGUF payload.  For tensors whose raw GGUF name an adapter
declares in ``zero_copy_tensor_names``, the iterator must instead yield a
``torch.from_numpy`` view that shares the mmap (a 28.8 GB private copy of
the packed PLE table is exactly what this path must avoid).
"""

import warnings

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter

from vllm_gguf_plugin.weight_utils import (
    gguf_quant_weights_iterator_multi,
)

_PLE_NAME = "per_layer_token_embd.weight"
_OTHER_NAME = "other.q.weight"


def _write_small_gguf(tmp_path):
    """One IQ4_NL PLE-shaped tensor and one Q8_0 tensor in one file."""
    path = tmp_path / "model.gguf"
    writer = GGUFWriter(str(path), "qwen4_exp")
    ple = np.arange(3 * 90, dtype=np.uint8).reshape(3, 90)
    writer.add_tensor(_PLE_NAME, ple, raw_dtype=GGMLQuantizationType.IQ4_NL)
    other = np.arange(2 * 68, dtype=np.uint8).reshape(2, 68)
    writer.add_tensor(_OTHER_NAME, other, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


def _payload_write_through(path: str, name: str, marker: int) -> None:
    """Write one byte through an ``r+`` mmap of the GGUF payload."""
    import gguf

    reader = gguf.GGUFReader(path)
    tensor = next(t for t in reader.tensors if t.name == name)
    payload = np.memmap(path, dtype=np.uint8, mode="r+")
    try:
        payload[tensor.data_offset : tensor.data_offset + 1] = marker
        payload.flush()
    finally:
        del payload


def test_declared_tensor_is_yielded_as_a_view_of_the_memory_map(tmp_path):
    path = _write_small_gguf(tmp_path)
    generator = gguf_quant_weights_iterator_multi(
        [path],
        {_PLE_NAME: _PLE_NAME, _OTHER_NAME: _OTHER_NAME},
        zero_copy_tensor_names=frozenset({_PLE_NAME}),
    )
    ple = None
    for name, tensor in generator:
        if name == _PLE_NAME:
            ple = tensor
            # Keep the generator suspended so its GGUF reader (and the
            # mmap the view aliases) stays alive for the write-through check.
            break
    assert ple is not None
    assert ple.dtype == torch.uint8
    # A view of the mmap sees writes through a second (r+) mapping of the
    # same file; a private copy would not.
    original = int(ple.numpy().ravel()[0])
    _payload_write_through(path, _PLE_NAME, 0xA5)
    assert int(ple.numpy().ravel()[0]) == 0xA5
    _payload_write_through(path, _PLE_NAME, original)
    assert int(ple.numpy().ravel()[0]) == original
    # Release the generator (and its reader) once the check is done.
    list(generator)


def test_undeclared_tensor_still_gets_a_private_copy(tmp_path):
    path = _write_small_gguf(tmp_path)
    weights = dict(
        gguf_quant_weights_iterator_multi(
            [path],
            {_PLE_NAME: _PLE_NAME, _OTHER_NAME: _OTHER_NAME},
            zero_copy_tensor_names=frozenset({_PLE_NAME}),
        )
    )

    other = weights[_OTHER_NAME]
    assert other.dtype == torch.uint8
    # The copy matches the payload at read time...
    import gguf

    reader = gguf.GGUFReader(path)
    tensor = next(t for t in reader.tensors if t.name == _OTHER_NAME)
    np.testing.assert_array_equal(other.numpy(), tensor.data)
    # ...but later file writes do not leak into it (own storage survives the
    # iterator's reader being released).
    original = int(other.numpy().ravel()[0])
    _payload_write_through(path, _OTHER_NAME, 0x7C)
    try:
        assert int(other.numpy().ravel()[0]) == original
    finally:
        _payload_write_through(path, _OTHER_NAME, original)


def test_no_zero_copy_names_keeps_every_tensor_a_private_copy(tmp_path):
    path = _write_small_gguf(tmp_path)
    weights = dict(
        gguf_quant_weights_iterator_multi(
            [path], {_PLE_NAME: _PLE_NAME, _OTHER_NAME: _OTHER_NAME}
        )
    )

    original = int(weights[_PLE_NAME].numpy().ravel()[0])
    _payload_write_through(path, _PLE_NAME, 0x31)
    try:
        assert int(weights[_PLE_NAME].numpy().ravel()[0]) == original
    finally:
        _payload_write_through(path, _PLE_NAME, original)


def test_zero_copy_path_suppresses_the_non_writable_array_warning(tmp_path):
    path = _write_small_gguf(tmp_path)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        dict(
            gguf_quant_weights_iterator_multi(
                [path],
                {_PLE_NAME: _PLE_NAME, _OTHER_NAME: _OTHER_NAME},
                zero_copy_tensor_names=frozenset({_PLE_NAME}),
            )
        )

    leaked = [
        str(warning.message)
        for warning in recorded
        if "not writable" in str(warning.message)
    ]
    assert leaked == []


def test_zero_copy_declaration_uses_raw_gguf_names():
    """The iterator matches declared names against raw GGUF tensor names."""
    from vllm_gguf_plugin.weights_adapter.qwen4_exp import Qwen4ExpGGUFAdapter

    assert Qwen4ExpGGUFAdapter.zero_copy_tensor_names == (_PLE_NAME,)


def test_base_adapter_declares_no_zero_copy_tensors_by_default():
    from vllm_gguf_plugin.weights_adapter.base import BaseGGUFWeightsAdapter

    assert BaseGGUFWeightsAdapter.zero_copy_tensor_names == ()
