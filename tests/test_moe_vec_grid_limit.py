# SPDX-License-Identifier: Apache-2.0

"""Per-row GGUF MoE kernels must stay inside CUDA's launch-grid limits.

The ``moe_vec_*`` kernels launch one grid z slot per (token, expert) row, and
CUDA caps the z extent at 65,535. A larger launch never runs: the error stays
pending and surfaces at the next device allocation as ``CUDA error: invalid
argument``. The first full vLLM load of Qwen3.8-Flash-Next UD-IQ4_XS failed
this way in its 8,192-token profile run (81,920 rows with top-k 10), because
IQ-type experts take the per-row path at every batch size.
"""

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin import ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel behaviour"
)

_GRID_Z = 65535
_EXPERTS, _COLS, _ROWS = 4, 64, 32


def _q8_expert_weights() -> torch.Tensor:
    rng = np.random.default_rng(20260911)
    values = rng.standard_normal((_EXPERTS, _ROWS, _COLS), dtype=np.float32)
    packed = gguf.quants.quantize(values, GGMLQuantizationType.Q8_0)
    return torch.from_numpy(packed).cuda()


def _chunked_reference(x, w, flat_ids, top_k, rows):
    """Call the kernel directly with every launch well under the grid limit."""
    step = 4096 // top_k
    tokens = x.shape[0]
    parts = [
        torch.ops._C_gguf.ggml_moe_a8_vec(
            x[start : start + step],
            w,
            flat_ids[start * top_k : (start + step) * top_k],
            top_k,
            int(GGMLQuantizationType.Q8_0),
            rows,
            min(step, tokens - start),
        )
        for start in range(0, tokens, step)
    ]
    return torch.cat(parts)


@pytest.mark.parametrize(
    ("tokens", "top_k", "id_columns"),
    [
        # gate/up call: top-k ids per token, one launch row per (token, expert).
        (_GRID_Z // 10 + 1, 10, 10),
        # down call: one row per routed pair, ids still shaped (tokens, 10).
        ((_GRID_Z // 10 + 1) * 10, 1, 10),
    ],
)
def test_moe_vec_splits_launches_above_the_grid_limit(tokens, top_k, id_columns):
    torch.manual_seed(0)
    w = _q8_expert_weights()
    x = torch.randn(tokens, _COLS, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(
        0, _EXPERTS, (tokens * top_k // id_columns, id_columns), dtype=torch.int32
    ).cuda()
    assert tokens * top_k > _GRID_Z

    out = ops.ggml_moe_a8_vec(
        x, w, ids, top_k, int(GGMLQuantizationType.Q8_0), _ROWS, tokens
    )
    # A failed launch is only reported by a later CUDA call; force a fresh
    # device allocation so a pending error cannot hide.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    expected = _chunked_reference(x, w, ids.reshape(-1), top_k, _ROWS)
    assert out.shape == (tokens * top_k, _ROWS)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
