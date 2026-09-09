# SPDX-License-Identifier: Apache-2.0
"""CPU lookup helpers for packed IQ4_NL PLE rows.

These functions do not install a vLLM offload worker or load a GGUF checkpoint.
A caller must supply the packed CPU table and the model's actual n-gram row IDs.
Only selected rows are decoded; the full table remains quantized.
"""

import sys

import torch

_BLOCK_VALUES = 32
_BLOCK_BYTES = 18
_LOOKUP_CHUNK_ROWS = 8192
_IQ4_NL_VALUES = (
    -127,
    -104,
    -83,
    -65,
    -49,
    -35,
    -22,
    -10,
    1,
    13,
    25,
    38,
    53,
    69,
    89,
    113,
)
_OUTPUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _validate_rows(packed: torch.Tensor, width: int, dtype: torch.dtype) -> None:
    if packed.device.type != "cpu":
        raise ValueError("IQ4_NL CPU lookup requires CPU-resident packed rows")
    if packed.dtype != torch.uint8:
        raise TypeError("IQ4_NL storage must use uint8 packed bytes")
    if not isinstance(width, int) or width <= 0 or width % _BLOCK_VALUES:
        raise ValueError("IQ4_NL embedding width must be a positive multiple of 32")
    if packed.ndim != 2 or packed.shape[1] != width // _BLOCK_VALUES * _BLOCK_BYTES:
        raise ValueError("Packed row width does not match the IQ4_NL block layout")
    if dtype not in _OUTPUT_DTYPES:
        raise TypeError("IQ4_NL output must use float32, bfloat16 or float16")
    if sys.byteorder != "little":
        raise RuntimeError("This IQ4_NL CPU path expects little-endian GGUF blocks")


def decode_iq4_nl_rows(
    packed: torch.Tensor,
    width: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode an explicitly selected batch of 18-byte IQ4_NL blocks."""
    _validate_rows(packed, width, dtype)
    blocks = packed.reshape(-1, _BLOCK_BYTES)
    scales = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    codes = blocks[:, 2:]
    indices = torch.cat((codes & 15, codes >> 4), dim=1).long()
    values = torch.tensor(_IQ4_NL_VALUES, dtype=torch.float32, device="cpu")
    decoded = values[indices] * scales
    return decoded.reshape(packed.shape[0], width).to(dtype)


def gather_iq4_nl_rows(
    packed_table: torch.Tensor,
    row_ids: torch.Tensor,
    width: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather rows in request order, decoding unique rows in bounded chunks.

    The output has shape ``(*row_ids.shape, width)``. Temporary decoder storage is
    bounded by _LOOKUP_CHUNK_ROWS; storage for decoded rows scales with requested
    unique rows, never with the number of rows in the full PLE table.
    """
    _validate_rows(packed_table, width, dtype)
    if row_ids.device.type != "cpu":
        raise ValueError("IQ4_NL CPU lookup requires CPU row IDs")
    if row_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("PLE row IDs must be int32 or int64")
    output_shape = (*row_ids.shape, width)
    if row_ids.numel() == 0:
        return torch.empty(output_shape, dtype=dtype, device="cpu")
    flat_ids = row_ids.reshape(-1).long()
    if int(flat_ids.min()) < 0 or int(flat_ids.max()) >= packed_table.shape[0]:
        raise IndexError("PLE row ID lies outside the packed table")
    unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
    decoded = torch.empty((unique_ids.numel(), width), dtype=dtype, device="cpu")
    for start in range(0, unique_ids.numel(), _LOOKUP_CHUNK_ROWS):
        stop = min(start + _LOOKUP_CHUNK_ROWS, unique_ids.numel())
        selected = torch.index_select(packed_table, 0, unique_ids[start:stop])
        decoded[start:stop].copy_(decode_iq4_nl_rows(selected, width, dtype=dtype))
    return decoded.index_select(0, inverse).reshape(output_shape)
