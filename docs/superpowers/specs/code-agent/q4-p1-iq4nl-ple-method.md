# Code-agent task spec: `q4-p1-iq4nl-ple-method`

## Identity

- Owner: selected by executor-cli-routing
- Difficulty (1-10): 7 — a new packed storage format plus a Triton kernel that
  decodes float16 scales from raw bytes, and a patch of a vLLM static method.
  #56273's NVFP4 method is a close template, which keeps it below 8.
- Importance (1-10): 8 — on the critical path for running the GGUF at all; a
  decode error silently corrupts every generated token without crashing.
- Wave: 1 (parallel with `q4-p2-adapter-nightly-port`, disjoint files)
- Base commit: the commit on `claude/qwen4exp-gguf-iq4` that adds this spec
  (the prompt carries the exact SHA)
- Worktree: `/Users/kg/Projects/vllm-gguf-plugin/.worktrees/q4-p1-iq4nl-ple-method`
- Branch: `claude/code-agent/q4-p1-iq4nl-ple-method`
- Blocked by: none. Unblocks: `q4-p3-gpu-integration`

Background, decisions D1–D4, and environment facts:
`docs/dev-log/qwen4exp-gguf-iq4.md`. Read it first.

## Goal and non-goals

Goal: vLLM's Qwen4Exp n-gram embedding can hold the PLE table as packed IQ4_NL
bytes and return correctly decoded rows, selected by
`ple_embedding_dtype == "gguf_iq4_nl"`, on both the device and pinned-host paths.
Every CPU-runnable behavior is proven by tests; GPU tests are written and marked
to skip without CUDA.

Non-goals: GGUF name mapping, reading GGUF files, the weights iterator, setting
`ple_embedding_dtype` (all package P2); GPU execution and full-model load (P3);
any change to vLLM itself.

## Exact scope

### Allowed files

- `vllm_gguf_plugin/quantization/ple_iq4_nl.py` (new)
- `vllm_gguf_plugin/plugin.py` — only add a call from `register()` to a new
  `_patch_qwen4_exp_ple_method()` helper, and that helper
- `vllm_gguf_plugin/quantization/ple_cpu.py` — only if a small refactor is needed
  for reuse; existing public functions keep their behavior
- `tests/test_ple_iq4_nl_method.py` (new)

### Forbidden files and behavior

- `vllm_gguf_plugin/weights_adapter/**`, `vllm_gguf_plugin/weight_utils.py`,
  `vllm_gguf_plugin/loader.py`, `pyproject.toml`, `setup.py`, CI files, `docs/**`
- Any file under vLLM's installed package or `/opt/vllm-upstream-tests`
- Allocating a decoded (BF16/FP32) table larger than the requested rows
- Push, pull request, merge, or any network action other than the test runner

## Contract

Module `vllm_gguf_plugin/quantization/ple_iq4_nl.py`:

1. `GGUF_IQ4_NL_PLE_DTYPE = "gguf_iq4_nl"`.
2. IQ4_NL layout: 32 values per block, 18 bytes per block (little-endian float16
   scale `d`, then 16 bytes of indices). Value `j` of a block uses the low nibble
   of index byte `j` for `j < 16` and the high nibble of byte `j - 16` otherwise
   (identical to `ple_cpu.decode_iq4_nl_rows`). Value = `d * CODEBOOK[nibble]`,
   `CODEBOOK = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)`.
   `embedding_dim % 32 != 0` raises `ValueError`. Row bytes = `embedding_dim // 32 * 18`
   (90 for the 160-value Flash rows).
3. `class Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod)`, importing
   the base from `vllm.models.qwen4_exp.nvidia.ngram_embedding`:
   - `create_weights(...)` registers `weight` as `ModelWeightParameter` with data
     `layer.allocate_embedding_weight(sum(output_partition_sizes), row_bytes, torch.uint8)`,
     `input_dim=1`, `output_dim=0`, and a `weight_loader` that wraps the provided
     loader: it rejects a loaded tensor whose dtype is not `torch.uint8`
     (`ValueError` naming the dtype), calls the original loader unchanged, then
     records the destination row range. With `checkpoint_start` it uses
     `vllm.models.qwen4_exp.common.ple.compute_ple_shard_overlap` against
     `layer.shard_indices.org_vocab_{start,end}_index`; without it, it records the
     full local range.
   - `process_weights_after_loading(layer)` raises `ValueError` naming the first
     uncovered local row when the recorded ranges do not cover
     `org_vocab_end_index - org_vocab_start_index` (same merge algorithm as
     #56273's NVFP4 method).
   - `lookup_dtype(layer)` returns `layer.params_dtype`.
   - `embedding(layer, input_)`: CPU input decodes with
     `ple_cpu.gather_iq4_nl_rows(layer.weight, ids, embedding_dim, dtype=layer.params_dtype)`;
     CUDA input runs the Triton kernel over `layer.weight` with vocab range
     `[0, layer.weight.shape[0])`. Output shape `(*input_.shape, embedding_dim)`.
     Empty input returns an empty tensor without launching a kernel.
   - `lookup_from_pinned(layer, ids, output)` runs the same kernel over
     `layer._uva_weight` with the layer's shard vocab range, writing into `output`.
   - `dequantize(layer, embeddings, output_dtype)` returns `embeddings.to(output_dtype)`.
4. Triton kernel `_lookup_iq4_nl_ple_embedding_kernel(weight_ptr, codebook_ptr,
   ids_ptr, output_ptr, embedding_dim, row_bytes, vocab_start, vocab_end, BLOCK_D)`:
   one program per id; ids outside `[vocab_start, vocab_end)` write zeros (same
   ownership semantics as upstream). The float16 scale is rebuilt from its two
   bytes in-kernel: sign bit, 5-bit exponent, 10-bit mantissa; exponent 0 is
   subnormal (`m * 2**-24` with sign), exponent 31 is inf or NaN, otherwise
   `(1 + m/1024) * 2**(e-15)`. Output values are float32 cast to the output dtype
   by `tl.store`. The codebook is a float32 tensor on the weight's device.
5. `patch_qwen4_exp_ple_embedding_method()` replaces
   `Qwen4ExpPLEEmbeddingMethod.from_quant_config` (a `@staticmethod`) with a
   `staticmethod` wrapper: if `embedding_dtype == GGUF_IQ4_NL_PLE_DTYPE` return a new
   `Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod()`; otherwise call the original with the
   same arguments. Repeated calls are no-ops (guard attribute
   `_vllm_gguf_plugin_iq4_nl_patched = True` on the wrapper).
6. `plugin.py`: `_patch_qwen4_exp_ple_method()` imports and applies the patch; if
   `vllm.models.qwen4_exp` cannot be imported (older vLLM), it logs at debug level
   and returns. `register()` calls it once.

## Implementation plan

Run all tests with the pinned runner from the repository root
`/Users/kg/Projects/vllm-gguf-plugin`:
`bash .dev/run_cpu_tests.sh <worktree> <label> "<pytest args>"`. It rsyncs the
worktree to the Linux host and runs a CPU-only, network-disabled container of
image `inferway-dev/qwen4exp-gguf-test:2a02f6ef-pr56273-e251861a` (vLLM nightly
`2a02f6ef` with #56273 overlaid). Upstream PLE tests are at
`/opt/vllm-upstream-tests/models/qwen4_exp/test_ple.py` inside the image; read
`_make_nvfp4_ngram_embedding` there for how to build a small embedding on CPU.
A local copy is at
`/private/tmp/claude-501/-Users-kg-Projects-inferway--claude-worktrees-q4-pro6000-compatibility-bbfae2/3b4240dd-fb6f-41e1-9c54-5546e1a2e8ce/scratchpad/pr56273_test_ple.py`.

1. **Selection tests (RED).** Files: `tests/test_ple_iq4_nl_method.py`. Tests:
   marker selects the IQ4_NL method; `None`, `"float8_e4m3fn"` and `"nvfp4"` delegate
   to the original (compare with results from the unpatched function); a second
   patch call leaves exactly one wrapper. Verify:
   `bash .dev/run_cpu_tests.sh <wt> p1-red-select "tests/test_ple_iq4_nl_method.py -q -rs --timeout=600"`
   — must fail with `ModuleNotFoundError`/`ImportError` for
   `vllm_gguf_plugin.quantization.ple_iq4_nl` (this is the only acceptable RED
   class for step 1; record the output).
2. **Implement selection and weights.** Files: `ple_iq4_nl.py`. Add constant,
   method class (create_weights, loader wrapper, process_weights_after_loading,
   lookup_dtype, dequantize), patch function. Verify step 1 tests pass.
3. **Loading tests.** Files: test file. Build a small Qwen4Exp n-gram embedding
   with `ple_embedding_dtype="gguf_iq4_nl"` (patch applied first), stream packed
   shards via `load_weights` names `ngram_embedding.shard_{i}.weight` across ETP
   boundaries for ranks 0 and 1; assert full coverage passes; a missing shard
   raises the coverage `ValueError`; a float tensor raises the dtype `ValueError`;
   a wrong row width raises upstream's shape `ValueError`. Run RED first if any
   behavior is not yet implemented, then GREEN.
4. **CPU decode tests.** Files: test file, `ple_iq4_nl.py` (embedding CPU path).
   Compare `method.embedding` output to an independent reference built with
   `gguf.quants.dequantize(..., gguf.GGMLQuantizationType.IQ4_NL)` on: the real
   rows in `tests/fixtures/qwen4_exp_iq4_nl_samples.json`, and synthetic rows with
   negative, zero, subnormal, and large float16 scales. Duplicate, unsorted and
   empty id tensors included. Exact equality in float32; bfloat16 equals the
   reference cast to bfloat16.
5. **GPU kernel and pinned path.** Files: `ple_iq4_nl.py` (kernel, CUDA embedding,
   lookup_from_pinned), test file. Tests guarded by
   `pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")`:
   device lookup vs reference, pinned UVA lookup vs reference, ids outside the
   shard range produce zeros, CUDA graph replay with changed ids, empty ids, and
   float16 special scales. These skip on the CPU runner; skips are expected and are
   not counted as passes.
6. **Plugin hook.** Files: `plugin.py`, test file. Test that `register()` leaves
   `from_quant_config` patched and that calling `register()` twice keeps one wrapper.
7. **Regression and static gates.** Run
   `bash .dev/run_cpu_tests.sh <wt> p1-green "tests/test_ple_iq4_nl_method.py tests/test_ple_cpu.py tests/test_plugin.py /opt/vllm-upstream-tests/models/qwen4_exp/test_ple.py -q -rs --timeout=900 -p no:randomly"`.
   Expected: new tests pass; upstream `test_ple.py` still 34 passed / 33 skipped;
   `tests/test_plugin.py` exactly one known pre-existing failure
   `test_register_sets_engine_args_for_gguf_model` (fails identically on untouched
   upstream main, evidence
   `.dev/evidence/q4gguf-p0-upstream-main-plugin-test-20260911T020852Z-21816.log`) and
   no new failures. Then on the Mac:
   `cd <wt> && uvx ruff@0.14.0 check vllm_gguf_plugin/quantization/ple_iq4_nl.py vllm_gguf_plugin/plugin.py tests/test_ple_iq4_nl_method.py && uvx ruff@0.14.0 format --check vllm_gguf_plugin/quantization/ple_iq4_nl.py vllm_gguf_plugin/plugin.py tests/test_ple_iq4_nl_method.py && git diff --check`.

## Development record

Kaige asked for a complete record of the development process for the future pull
request. Keep `/Users/kg/Projects/vllm-gguf-plugin/.dev/reports/q4-p1-iq4nl-ple-method.md`
(outside the worktree, not committed) updated as you go: each step's command,
exit status, test counts, the RED failure text, design choices you made and why,
and anything that failed or surprised you. Core copies it into the dev log.

## Stop conditions

Stop and report instead of working around: the upstream base class or loader
does not match this contract; a needed change falls outside the allowed files;
upstream `test_ple.py` counts change; any test would need weakening; the runner
fails for environment reasons twice; the deliverable already exists on
`upstream/main` or another branch (check `git log --all -- vllm_gguf_plugin/quantization/ple_iq4_nl.py`
before starting).

## Commit and report

Commit only allowed files, DCO-signed, conventional messages. The last commit must
contain the trailer `Lane-Done: q4-p1-iq4nl-ple-method`, for example
`git commit -s -m "feat(qwen4_exp): add packed IQ4_NL PLE embedding method" -m "Lane-Done: q4-p1-iq4nl-ple-method"`.
Terminal state: CODE_COMMITTED — no push, no pull request, no merge. Return the
commit SHAs, changed files, every verification command with exit status and
counts, and remaining risks.
