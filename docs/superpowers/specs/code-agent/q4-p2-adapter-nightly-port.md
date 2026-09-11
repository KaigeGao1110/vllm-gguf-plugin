# Code-agent task spec: `q4-p2-adapter-nightly-port`

## Identity

- Owner: selected by executor-cli-routing
- Difficulty (1-10): 6 — the 1,224-name mapping already exists; the new work is
  retargeting to the nightly architecture, a narrow zero-copy path in the weights
  iterator, and splitting one packed tensor into 128 checkpoint shards.
- Importance (1-10): 8 — on the critical path; a copy here adds a 28.8 GB
  resident allocation, and a wrong shard split corrupts the n-gram table.
- Wave: 1 (parallel with `q4-p1-iq4nl-ple-method`, disjoint files)
- Base commit: the commit on `claude/qwen4exp-gguf-iq4` that adds this spec
  (the prompt carries the exact SHA)
- Worktree: `/Users/kg/Projects/vllm-gguf-plugin/.worktrees/q4-p2-adapter-nightly-port`
- Branch: `claude/code-agent/q4-p2-adapter-nightly-port`
- Blocked by: none. Unblocks: `q4-p3-gpu-integration`

Background, decisions D1–D4, and environment facts:
`docs/dev-log/qwen4exp-gguf-iq4.md`. Read it first.

## Goal and non-goals

Goal: with vLLM nightly `2a02f6ef`, the `qwen4_exp` adapter resolves the official
architecture, marks the text config with `ple_embedding_dtype="gguf_iq4_nl"` when
the GGUF PLE tensor is IQ4_NL, and streams that tensor to vLLM as
`ngram_embedding.shard_{i}.weight` zero-copy `uint8` views, with no full-model
construction guard left in place.

Non-goals: the embedding method, Triton kernel, or `from_quant_config` patch
(package P1); GPU execution and full-model load (P3); vision/mmproj support; any
change to vLLM itself.

## Exact scope

### Allowed files

- `vllm_gguf_plugin/weights_adapter/qwen4_exp.py`
- `vllm_gguf_plugin/weights_adapter/base.py` — only a new class attribute with an
  empty default, documented
- `vllm_gguf_plugin/weight_utils.py` — only the zero-copy branch described below
- `vllm_gguf_plugin/loader.py` — only passing the adapter's zero-copy names to the
  iterator, if the iterator cannot reach the adapter otherwise
- `tests/test_qwen4_exp_adapter.py`
- `tests/test_weight_utils_zero_copy.py` (new)
- `tests/fixtures/qwen4_exp_iq4_xs_tensor_directory.json` (new, derived as below)

### Forbidden files and behavior

- `vllm_gguf_plugin/quantization/**`, `vllm_gguf_plugin/plugin.py`, `pyproject.toml`,
  `setup.py`, CI files, `docs/**`
- Any file under vLLM's installed package
- Changing how any tensor other than the adapter-declared zero-copy names is read
- Push, pull request, merge, or network access other than the test runner

## Contract

1. Architecture: remove `QWEN4_EXP_ARCHITECTURE = "Qwen3_8FlashNextForConditionalGeneration"`
   and its use. The adapter reports `Qwen4ExpForConditionalGeneration`, which must
   be present in `vllm.model_executor.models.ModelRegistry` of the pinned image
   (assert this in a test).
2. `patch_hf_config`: no longer raises because `ple_layer_ids` is set. It reads the
   GGUF tensor type of `per_layer_token_embd.weight` from the GGUF header (no
   payload read). IQ4_NL: set `ple_embedding_dtype = "gguf_iq4_nl"` on the text
   config object that `Qwen4ExpNGramEmbedding` receives (verify which object in the
   nightly source). Any other type, or a missing tensor while `ple_layer_ids` is
   non-empty: `NotImplementedError` naming the observed type. The literal
   `"gguf_iq4_nl"` must equal P1's `GGUF_IQ4_NL_PLE_DTYPE`; Core checks this at
   integration.
3. PLE streaming in `transform_weights`: the GGUF PLE tensor (GGML shape
   `[160, R]`, IQ4_NL) becomes `R` rows of 90 bytes. Validate `R` equals the
   model's padded n-gram vocabulary and that the byte length is `R * 90`. Split into
   `split_ngram_parts` consecutive shards of `ceil(R / split_ngram_parts)` rows (the
   last may be shorter; empty trailing shards are yielded with 0 rows only if
   upstream `load_weights` expects them — check the nightly loader). Yield each as
   the fully qualified native name that the nightly `AutoWeightsLoader` routes to
   `...layers.<L>.ple.ple_embedding.ngram_embedding.shard_{i}.weight`, where `L` is
   the zero-based layer of `ple_layer_ids` (1-based in config). Each yielded tensor
   is a `torch.uint8` view sharing memory with the GGUF memory map. The PLE
   `.weight_type` companion tensor is not yielded.
4. Zero-copy reads: `BaseGGUFWeightsAdapter` gains `zero_copy_tensor_names:
   tuple[str, ...] = ()` (raw GGUF names). The weights iterator uses
   `torch.from_numpy` (suppressing only the non-writable-array warning) for those
   names and keeps `torch.tensor(...)` for every other tensor. The Qwen4Exp adapter
   declares `("per_layer_token_embd.weight",)`.
5. Keep the prototype's existing qualified refusals (quantized indexer Q/K, non-Q8_0
   hyper-connection projections, mmproj) and their tests.

## Implementation plan

Run all tests with the pinned runner from `/Users/kg/Projects/vllm-gguf-plugin`:
`bash .dev/run_cpu_tests.sh <worktree> <label> "<pytest args>"` (CPU-only,
network-disabled container of image
`inferway-dev/qwen4exp-gguf-test:2a02f6ef-pr56273-e251861a`).

1. **Baseline.** Run
   `bash .dev/run_cpu_tests.sh <wt> p2-baseline "tests/test_qwen4_exp_adapter.py -q -rs --timeout=600"`
   and record counts (P0 saw it pass on the old contract).
2. **Fixture.** Files: `tests/fixtures/qwen4_exp_iq4_xs_tensor_directory.json`.
   Derive it from
   `/Users/kg/Projects/inferway/.worktrees/qwen38-flash-wiring/.artifacts/q4-gguf-selection-20260909/header-ranges/tensor-directory.json`
   keeping only `repo`, `revision`, `variant`, `tensor_count`, and per tensor
   `name`, GGML `type`, and `shape`. Record the derivation command in the
   development record.
3. **Architecture and config tests (RED).** Files: `tests/test_qwen4_exp_adapter.py`.
   Replace the old architecture and PLE-refusal assertions with: official
   architecture resolved and registered; IQ4_NL PLE sets the marker; a non-IQ4_NL
   PLE type raises `NotImplementedError` naming the type. Run; expect assertion
   failures against the current adapter (import/collection errors are not valid
   RED). Implement in `qwen4_exp.py`; GREEN.
4. **Name replay test.** Files: adapter test. Using the fixture, every one of the
   1,224 names maps to exactly one target or is the PLE tensor, with no duplicates;
   the PLE tensor expands to `split_ngram_parts` shard names whose module path
   resolves under the nightly model (derive the expected prefix from the nightly
   `Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper` and decoder-layer attribute
   names, not by hand).
5. **Zero-copy iterator (RED then GREEN).** Files: `base.py`, `weight_utils.py`,
   `loader.py` if needed, `tests/test_weight_utils_zero_copy.py`. Write a small
   GGUF with `gguf.GGUFWriter` in `tmp_path` holding one IQ4_NL tensor named
   `per_layer_token_embd.weight` and one Q8_0 tensor. Assert the declared tensor is
   yielded as a view of the memory map (its storage does not own a private copy —
   for example, writing through `numpy.memmap` in `r+` mode on a copy of the file
   is visible in the yielded tensor, or compare `untyped_storage().data_ptr()` with
   the reader's buffer address), and the other tensor is still a private copy.
6. **Shard streaming.** Files: `qwen4_exp.py`, adapter test. With a synthetic IQ4_NL
   PLE tensor of a few hundred rows and a small `split_ngram_parts`, assert shard
   names, row counts, `uint8` dtype, width 90, byte-exact concatenation equal to the
   source, shared memory, and no `.weight_type` companion. Mismatched `R` or byte
   length raises `ValueError`.
7. **Regression and static gates.** Run
   `bash .dev/run_cpu_tests.sh <wt> p2-green "tests/test_qwen4_exp_adapter.py tests/test_weight_utils_zero_copy.py tests/test_gguf_utils.py tests/test_plugin.py tests/test_gemma4_adapter.py -q -rs --timeout=900"`.
   Expected: new and adapter tests pass; `tests/test_plugin.py` shows only the known
   pre-existing failure `test_register_sets_engine_args_for_gguf_model` (evidence
   `.dev/evidence/q4gguf-p0-upstream-main-plugin-test-20260911T020852Z-21816.log`);
   no other new failures. Then on the Mac:
   `cd <wt> && uvx ruff@0.14.0 check <changed .py files> && uvx ruff@0.14.0 format --check <changed .py files> && git diff --check`.

## Development record

Kaige asked for a complete record of the development process for the future pull
request. Keep `/Users/kg/Projects/vllm-gguf-plugin/.dev/reports/q4-p2-adapter-nightly-port.md`
(outside the worktree, not committed) updated as you go: each step's command,
exit status, test counts, the RED failure text, choices you made and why, and
anything that failed or surprised you.

## Stop conditions

Stop and report instead of working around: the nightly loader expects different
shard names or counts than this contract; zero-copy cannot be done inside the
allowed files; a change would alter reads for non-declared tensors; any existing
test would need weakening; the runner fails for environment reasons twice; the
deliverable already exists on `upstream/main` or another branch.

## Commit and report

Commit only allowed files, DCO-signed, conventional messages. The last commit must
contain the trailer `Lane-Done: q4-p2-adapter-nightly-port`. Terminal state:
CODE_COMMITTED — no push, no pull request, no merge. Return commit SHAs, changed
files, every verification command with exit status and counts, and remaining risks.
