# Qwen3.8-Flash-Next (qwen4_exp) GGUF IQ4 support — development log

This log records how GGUF support for Qwen3.8-Flash-Next was developed: what
was tried, what evidence each step produced, what failed, and why each design
choice was made. It is written so that the eventual upstream pull request can
cite it directly. Entries are append-only; corrections are added as new
entries rather than rewriting history.

## Target

- Checkpoint: `unsloth/Qwen3.8-Flash-Next-GGUF`, variant `UD-IQ4_XS`, revision
  `38bb39ee97821de2c9009abb7e93950eec396e66`, three shards, 93,682,584,224 bytes.
  Despite the name, the directory mixes Q6_K, Q8_0, F32, IQ4_NL, IQ3_S, IQ4_XS
  and BF16 tensors.
- Config: `Qwen/Qwen3.8-Flash-Next`, revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, architecture
  `Qwen4ExpForConditionalGeneration`, `model_type=qwen4_exp`.
- Hardware goal: one RTX PRO 6000 Blackwell Workstation Edition (96 GB, 600 W).
- Success criterion: the GGUF loads in vLLM through this plugin and generates
  correct text on one PRO 6000, with the n-gram (PLE) table kept in its packed
  IQ4_NL form rather than expanded to BF16.

## Timeline

### 2026-09-08 — llama.cpp baseline (not this plugin)

A separate experiment ran the 82 GB `UD-IQ3_XXS` file in llama.cpp on a PRO 6000.
Eight concurrent long-context requests completed but first-token latency reached
minutes; 16 and 32 concurrent requests largely timed out. The "q4" in some run
names there referred to KV-cache precision, not weights. This motivated trying
vLLM instead.

### 2026-09-09 — CPU prototype against an older vLLM image (commit `adfa4c7`)

Base: plugin `d4c1f0d` (upstream main), vLLM image `bd995759b5b8`
(`0.1.dev20073+g8e685d198`, PyTorch 2.13.0+cu130, Transformers 5.15.1).

Findings:

- The stock plugin failed on architecture recognition and weight mapping, on both
  the released plugin and upstream main.
- Tensor directories were read from HTTP range prefixes of the shards (no payload
  download): 1,224 unique tensors. The PLE table `per_layer_token_embd.weight` is
  IQ4_NL with GGML shape `[160, 320001536]`: 28,800,138,240 packed bytes, or
  102,400,491,520 bytes if expanded to BF16.
- A `qwen4_exp` weights adapter mapped all 1,224 names with no unmatched or
  duplicate targets (name coverage only, not shape or numeric compatibility).
- `quantization/ple_cpu.py` decodes selected IQ4_NL rows. Eight real rows
  (720 bytes) matched `gguf.dequantize` exactly in FP32 and after BF16 rounding.
- Ten native PLE hash parameters derived from the config matched GGUF metadata.
- 28 focused CPU tests passed in a disposable network-disabled container.

Blocker at the time: that vLLM image's PLE offload worker assumed a floating
embedding and accepted only the default and dummy model loaders, so no packed
table could be plugged in. The prototype therefore refused full model
construction. No full download, load, or generation happened. Work paused to wait
for community support.

### 2026-09-10 — upstream re-check before resuming

- Plugin: last upstream commit is `d4c1f0d` (2026-08-31, Gemma4). There is no
  `qwen4_exp` adapter, pull request, or issue. The README's verified-model table
  does not list Flash. (The Unsloth model card shows a `vllm serve` example; the
  plugin does not actually support the architecture.)
- vLLM changed the PLE path substantially:
  - [#54371](https://github.com/vllm-project/vllm/pull/54371) (merged 2026-09-09)
    replaced the old offload worker with UVA-based pinned-host lookup and a
    three-level design. Level 3, `Qwen4ExpPLEEmbeddingMethod`, owns weight
    creation, lookup format and dequantization, with BF16 and FP8 methods.
  - [#56273](https://github.com/vllm-project/vllm/pull/56273) (open, head
    `e251861a`) adds a packed NVFP4 method that keeps packed rows plus block
    scales in storage and decodes only requested rows, on both the device and
    pinned-host paths. It adds the `lookup_dtype`, `lookup_from_pinned` and
    `record_loaded_rows` hooks that any packed format needs.
  - Nightly `2a02f6ef` (2026-09-10, wheel `0.28.1rc1.dev628+g2a02f6efe`) contains
    #54371. Its `ngram_embedding.py` is byte-identical to #56273's base
    `b28c3e15`, so #56273 overlays cleanly.
- llama.cpp: Flash is supported since #27742 (2026-08-27). #28330 (2026-09-10)
  stops allocating an unused indexer V cache. Several concurrency bugs remain open
  (#27911, #27835, #28286, #28019).
- Config facts used below: `split_ngram_parts=128`, `ple_embed_dim=2560`,
  `heads_per_ngram=8`, `ngram_size=3`, `ple_layer_ids=[2]`, so each n-gram head
  row is 160 values and the checkpoint shard size is
  `ceil(320001536 / 128) = 2,500,012` rows.

### 2026-09-11 — P0: pinned test environment and baselines

- Pulled `vllm/vllm-openai:nightly-2a02f6ef…` (digest `sha256:96b234af…`). It lacks
  the `gguf` Python package, so a derived CPU test image
  `inferway-dev/qwen4exp-gguf-test:2a02f6ef-pr56273-e251861a` (image id
  `sha256:813cd3d2…`) adds `gguf==0.19.0` and `pytest-timeout==2.4.0` and bakes in
  #56273's `ngram_embedding.py` and `test_ple.py`; both SHA-256 values are checked
  during the build.
- Upstream `tests/models/qwen4_exp/test_ple.py` from #56273 in that image:
  **34 passed, 33 skipped** (every skip requires CUDA). The overlay works on CPU;
  the CUDA cases are deferred to the GPU phase and are not counted as passing.
- Plugin baseline on the prototype commit `adfa4c7`
  (`test_ple_cpu.py`, `test_qwen4_exp_adapter.py`, `test_plugin.py`,
  `test_gguf_utils.py`): **71 passed, 1 failed**. The failure is
  `test_plugin.py::test_register_sets_engine_args_for_gguf_model`
  (`HFValidationError` for the local path `/tmp/model.gguf`). The same test fails on
  untouched upstream main `d4c1f0d` (1 failed, 15 passed), so it is a pre-existing
  incompatibility between the plugin test and this vLLM nightly, not a regression.
- The plugin's weights iterator reads quantized payloads with
  `torch.tensor(tensor.data)`, which copies the memory map. For the PLE table that
  is a 28.8 GB private copy before vLLM copies it again into its own storage. P2
  adds a narrow zero-copy path for that tensor only.
- Native routing: `Qwen4ExpModel.load_weights` uses `AutoWeightsLoader`, which
  delivers `ngram_embedding.shard_{i}.weight` to
  `layers.<L>.ple.ple_embedding` (`Qwen4ExpNGramEmbedding.load_weights`).
  `PLEVocabParallelEmbedding.weight_loader(param, loaded_weight, checkpoint_start)`
  copies each shard's overlap with the local ETP range and accepts any trailing
  shape as long as it matches the parameter, so `uint8 [rows, 90]` shards fit
  without a vLLM change. Storage is pinned host memory when
  `engram_config.cpu_offload` (or `VLLM_PLE_CPU_OFFLOAD=1`) is set, otherwise on the
  device.

### 2026-09-11 — P1/P2 dispatched; GPU environment prepared for P3

- P1 and P2 run in parallel in separate worktrees from `759e3df`, each with its own
  append-as-you-go report under `.dev/reports/`. Both are implemented by a coding
  agent driven by Qwen3.8-27B; every step is committed with its RED/GREEN test
  evidence, and acceptance is done independently before integration.
- GPU machine: one RTX PRO 6000 Blackwell Workstation Edition in a rented container
  (power limit 600 W, max memory clock 14001 MHz, ECC disabled, 97,887 MiB, driver
  595.71.05 / CUDA 13.2, 123 GB RAM, 61 GB `/dev/shm`). There is no Docker inside the
  container, so vLLM is installed into a virtual environment instead of the image:
  - `vllm-0.28.1rc1.dev628+g2a02f6efe` from the per-commit wheel index
    (`https://wheels.vllm.ai/2a02f6ef…/`), the same build as the CPU test image;
    PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.17.0, `gguf==0.19.0`.
  - #56273's `ngram_embedding.py` overlaid into the installed package; SHA-256 checked
    after copying, same value as in D3.
- Checkpoint downloaded at the pinned revision: the three `UD-IQ4_XS` shards
  (10,946,624 + 49,835,229,856 + 43,836,407,744 bytes). Each shard's SHA-256 matches
  the Hugging Face LFS object id (`5ce89370…`, `577a38a2…`, `d4634e6d…`). Config and
  tokenizer files come from `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d`, without the
  safetensors weights.
- Upstream `tests/models/qwen4_exp/test_ple.py` from #56273 on this GPU: **67 passed,
  0 skipped** (19 s). The 33 cases that skip on CPU — device and pinned-host NVFP4
  lookups included — pass here, so the base PLE paths the IQ4_NL method builds on
  work on this hardware before any plugin code is involved.

### 2026-09-11 — P2 accepted and merged; checked against the real checkpoint

- P2 delivered six signed commits (report: `.dev/reports/q4-p2-adapter-nightly-port.md`).
  Independent acceptance on the delivered tree: CPU suite (adapter, zero-copy,
  gguf utils, plugin, Gemma4, download, PLE CPU tests) **110 passed, 1 failed** — the
  failure is the pre-existing `test_register_sets_engine_args_for_gguf_model`
  recorded in P0. `ruff check .` and `ruff format --check .` (ruff 0.14.0, whole
  repository) are clean. Merged as `4a99ed1`.
- Facts P2 established from the nightly source, beyond the P0 notes:
  - `ple_layer_ids` entries are 1-based: `ple_layer_ids=[2]` attaches the PLE module to
    zero-based layer 1, so shard names are
    `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight`.
  - The n-gram table has `padded_vocab_size` rows: the sum of one prime-sized block per
    head (`nth_prime_after(ngram_vocab_size_base - 1, head + 1)`, Miller–Rabin), padded
    to `make_ngram_vocab_size_divisible_by`. For Flash that is 320,001,446 rows padded
    to 320,001,536, exactly the GGUF table's row count.
  - `Qwen4ExpNGramEmbedding.load_weights` expects every one of `split_ngram_parts`
    shards, including 0-row trailing shards, and checks each shard's shape exactly.
  - The iterator synthesizes a `.weight_type` companion for every quantized tensor;
    the adapter drops the PLE table's companion.
  - Each `GGUFReader` maps the file at its own address, so aliasing tests must compare
    against the same reader (or write through a second mapping), not a second reader's
    pointer.
- Real-checkpoint check on the GPU machine (no model construction; script
  `.dev/p3/real_gguf_ple_shards.py`, evidence
  `.dev/evidence/q4gguf-p3-real-ple-shards-4a99ed1-20260911T0250Z.log`), 12 s:
  - config-derived padded rows 320,001,536; the PLE table is IQ4_NL in shard 2 with
    array shape `[320001536, 90]`;
  - the yielded tensor shares the reader's memory map; 128 shards with the expected
    names, 2,500,012 rows each, contiguous views covering exactly the whole table;
  - 64 rows (first, second, middle, last and 60 random) decoded by `ple_cpu` are
    bit-identical to `gguf.quants.dequantize` (max abs diff 0.0), and the same rows
    read through the shard views match the table bytes;
  - resident memory 1,116 → 1,127 MiB across the expansion (max RSS 1,767 MiB), so
    the 28.8 GB table is not copied on the way to the loader.

## Design decisions

### D1 — Implement IQ4_NL as a Level-3 PLE embedding method, not a worker

The old prototype targeted a worker interface that no longer exists. IQ4_NL is
structurally the same problem #56273 solves for NVFP4: a packed row format
decoded per requested row. An IQ4_NL row of 160 values is five 32-value blocks of
18 bytes (a float16 scale plus 16 bytes of 4-bit indices into the fixed IQ4_NL
codebook), i.e. 90 bytes per row. That matches 28,800,138,240 / 320,001,536 = 90.
The method stores `weight` as `uint8 [rows, 90]` via
`layer.allocate_embedding_weight`, so the same code serves GPU-resident and
pinned-host (UVA) storage.

### D2 — Keep the hook inside the plugin

`Qwen4ExpPLEEmbeddingMethod.from_quant_config` raises for non-FP8 quantization
configs, so a GGUF config cannot currently select a method. The plugin already
patches vLLM from `register()` (`_patch_engine_args`, `_patch_speculator_probe`),
so it wraps `from_quant_config` to return the IQ4_NL method for the plugin's GGUF
config and delegate otherwise. Upstream `load_weights` only calls
`record_loaded_rows` for the NVFP4 method, so the IQ4_NL method records coverage
through its own wrapped `weight_loader`. No vLLM source change is required.

### D3 — Pin the base to nightly `2a02f6ef` plus #56273 `e251861a`

The IQ4_NL method needs the hooks #56273 adds. Until it merges, development and
tests run against nightly `2a02f6ef` with #56273's `ngram_embedding.py` overlaid
(SHA-256 `0ccfcec97b4d409bf6e5719719debed09705df572439f716875f6215656062b8`).
If #56273 changes before merging, the plugin must be re-checked against the
merged interface.

### D4 — The adapter streams raw packed bytes, never a decoded table

The weights adapter maps the GGUF PLE tensor to native checkpoint names
`ngram_embedding.shard_{i}.weight` for `i` in `0..127`, yielding zero-copy
`uint8 [rows_i, 90]` views of the memory-mapped GGUF payload. The upstream loader
then copies each shard into the method's storage. No step may allocate the
102.4 GB BF16 table.

## Environment

- CPU tests: Linux host, image
  `vllm/vllm-openai:nightly-2a02f6efe319c885e3ccbcecde402e0028f9ec1e`
  (digest `sha256:96b234afa2867031ad0a226b149a06483861e613e3a3083da53398d347b5ffbd`),
  #56273 files overlaid read-only, network disabled, CPU and memory limited,
  containers removed on exit.
- GPU tests and full load: one RTX PRO 6000 WE. Pending machine availability.

## Work packages

| ID | Scope | State |
|---|---|---|
| P0 | Base pin, test environment, contracts, this log | done |
| P1 | IQ4_NL PLE embedding method, Triton lookup kernel, `from_quant_config` hook | in progress |
| P2 | Port the `qwen4_exp` adapter to nightly and stream PLE shards | accepted, merged `4a99ed1`; real-checkpoint check passed |
| P3 | Full download, full load, GPU kernel tests, generation, quality and performance | environment and checkpoint ready; waiting for P1 and P2 |
