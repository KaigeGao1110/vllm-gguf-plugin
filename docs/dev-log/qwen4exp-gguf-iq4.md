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

### 2026-09-11 — P1 accepted after a Core repair; kernel verified on the GPU

- P1 delivered seven signed commits (report: `.dev/reports/q4-p1-iq4nl-ple-method.md`):
  the `gguf_iq4_nl` method, a CPU decode path, a Triton lookup kernel, the
  `from_quant_config` patch and its registration hook. The CPU runner passed
  (84 passed, 39 skipped — all CUDA — and the pre-existing failure), and ruff was
  clean. The six CUDA tests had never run.
- On the PRO 6000 all six CUDA tests failed at Triton compilation. Two kernel
  defects and one test defect, fixed by Core in `4a99f2c`:
  1. Bytes loaded from the `uint8` table stay `uint8` in Triton, so
     `scale_high << 8` overflowed and `bits & 0x3FF` failed to compile
     (`Scalar 1023 is out of range for type uint8`). Every loaded byte is now
     widened to `int32` before shifting or masking. A CPU-only formula test cannot
     catch this, because it models the arithmetic with wide integers.
  2. The kernel used `tl.bitcast`, which does not exist in Triton 3.7.1. Normal
     float16 scales are now computed as `(1024 + m) * 2**e * 2**-25`: every factor
     is exactly representable in float32 and every product has at most 11
     significant bits, so the result is exact without libdevice `exp2`/`pow`
     (which are not correctly rounded for all exponents). Subnormal scales use the
     exact literal for `2**-24`. The CPU guard over all 65,536 float16 bit patterns
     mirrors the new expression.
  3. The CUDA graph-replay test compared CPU output with expected rows on CUDA.
- After the repair, on the GPU: `tests/test_ple_iq4_nl_method.py` **32 passed** — CUDA
  graph capture and replay on device and pinned-host storage for both ETP ranks,
  direct `lookup_from_pinned` over the UVA view, out-of-shard ids returning zeros,
  and negative, zero, subnormal and large scales, all bit-identical to
  `gguf.quants.dequantize` after BF16 conversion. CPU runner unchanged (84 passed,
  39 skipped, pre-existing failure); repo-wide ruff clean. Merged as `2e9faa2`.
- Evidence: `.dev/evidence/q4gguf-p3-gpu-p1-{d67d066,fix1,fix2,fix3}-20260911.log`.
- Lesson for the PR: Triton kernels in this plugin need their CUDA tests run on a
  GPU before review; a green CPU run here said nothing about whether the kernel
  compiled.

### 2026-09-11 — Plugin GPU baseline on the PRO 6000

- Kernel tests restricted to the checkpoint's formats (IQ3_S, IQ4_NL, IQ4_XS,
  Q8_0, Q6_K) at `93490c8`: 168 passed, 0 failed, 144 skipped. Every skip is
  `tests/test_kernels.py:258: Current CUDA Kernel hasn't supported invarlen`.
  Evidence: box log `gpu-kernels-model-types-93490c8.log`.
- Full plugin suite on the GPU at `93490c8` (generation, multimodal and diffusion
  tests excluded; the two sample repos downloaded): 1543 passed, 68 failed,
  576 skipped in 35 min. Evidence: box log `gpu-plugin-baseline-93490c8.log`.
- All 68 failures are `test_moe[<quant>-dtype2-<top_k>-512-<num_tokens>]`, where
  `dtype2` is `torch.float32` and `num_tokens` is 7 or 2048, across 17 quant
  types (including Q8_0, IQ4_NL, IQ3_S and IQ4_XS). Every one fails at
  `tests/test_kernels.py:358`, inside vLLM's unquantized reference
  `fused_experts`, with `triton.runtime.errors.OutOfResources: out of resource:
  shared memory, Required: 122880 | 131072, Hardware limit: 101376`. The plugin's
  own `_fused_moe_gguf` call on the line above completed. The float16 and
  bfloat16 variants of the same tests pass.
- Assessment: this is a test-reference problem on this GPU (the default float32
  fused-MoE Triton config exceeds its per-kernel SRAM limit), not a plugin
  kernel defect, and the model runs in bfloat16. For the PR, the float32
  reference path in `test_moe` should use a smaller reference config or a
  dequantize-and-loop reference; not changed yet.
- Separately, `test_register_sets_engine_args_for_gguf_model` fails on upstream
  main as well (`HFValidationError` for `/tmp/model.gguf`). Cause: the CPU
  runner sets `HF_HUB_OFFLINE=1`, and nightly `EngineArgs.__post_init__`
  (`arg_utils.py:831-838`) passes every non-existent model path to
  `get_model_path`, which treats it as an HF repo id. The test's path does not
  exist, so it fails before the plugin's `create_model_config` patch runs. Local
  GGUF files that exist are unaffected, but an offline remote `repo:quant`
  reference fails at the same step: a real plugin gap. To fix in the PR batch
  (leave GGUF references to the plugin loader in offline mode, and make the test
  independent of `HF_HUB_OFFLINE`).

### 2026-09-11 — P3 full load, run 1: hyper-connection companion names

- Command (plugin tree `2e9faa2`, box script `p3-serve.sh`): `vllm serve
  <first shard> --tokenizer <config> --hf-config-path <config>
  --language-model-only --engram-config '{"cpu_offload": true}' --host 127.0.0.1
  --max-model-len 32768 --max-num-seqs 8 --gpu-memory-utilization 0.90
  --enforce-eager`. The plugin registered its loader and config parser, vLLM
  resolved `EngramConfig(cpu_offload=True)`, and the PLE layer initialised with
  `quantization_method=Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod, weight_dtype=uint8,
  weight_device=cpu, pinned=True`.
- Weight loading then failed 12 s later (81 s after launch): `ValueError: There is no module or
  parameter named 'hyper_connection_mixer.input_mix_weight_type_down' in
  Qwen4ExpModel`.
- Root cause: `gguf_quant_weights_iterator_multi` named the synthetic companion
  with `name.replace("weight", "weight_type")`, rewriting every occurrence. The
  HC projections are `input_mix_weight_{down,up}.weight`, so the companion became
  `input_mix_weight_type_down.weight_type`. The adapter's `.weight_type` check
  never matched, so the 194 Q8_0 HC projections were neither recognised nor
  decoded. The P2 unit test hand-built the correct companion name, so it could
  not see the defect. The diffusion iterator had the same expression.
- Fix `db2d8b2`: `gguf_weight_type_name()` replaces only the last `weight`, used
  by both iterators; names with a single `weight` are unchanged (no other adapter
  maps a name with two). New `tests/test_weight_type_names.py` drives the real
  iterator, with a wiring guard feeding its output straight into the Qwen4Exp
  adapter. RED on the unfixed tree reproduced the exact load-time name; GREEN on
  the box: 124 passed, 1 skipped across the new tests, the Qwen4Exp adapter, the
  zero-copy iterator, diffusion, and both PLE suites; repo-wide ruff 0.14.0
  check and format clean.

### 2026-09-11 — P3 full load, run 2: the concatenated indexer projection

- Run 2 (plugin tree with `db2d8b2`) got past every per-tensor weight and failed
  at the very end of `transform_weights`: `AssertionError: Tried to load weights
  of size torch.Size([640, 2560]) to a parameter of size torch.Size([0])` in
  `ReplicatedLinear.weight_loader` (`linear.py:399`).
- The tensor is the adapter's own `index_qk_proj`: BF16 indexer q
  `[512, 2560]` concatenated with k `[128, 2560]`, yielded after the main loop.
  The module is a `ReplicatedLinear` built with the model's quant config.
- Root cause: the loader derives unquantized modules from GGUF tensor names
  (`...self_attn.indexer.q_proj` and `k_proj`), and the quant config matches
  them by substring against the vLLM prefix, so `index_qk_proj` got
  `GGUFLinearMethod`. That method creates an uninitialised weight and relies on
  the layer's `weight_loader_v2`; `ReplicatedLinear` has none, so loading fell
  through to the plain loader's size assertion. F32 `ReplicatedLinear`s such as
  `mlp.gate` were unaffected because their GGUF names map to their own modules.
- Fix `39efe53`: the adapter declares
  `extra_unquantized_modules = ("self_attn.indexer.index_qk_proj",)`;
  `transform_weights` already rejects quantized indexer q/k. The regression test
  maps the declaration through `Qwen4ExpForConditionalGeneration`'s real rename
  mapper and checks the vLLM prefix is skipped while `o_proj` and the shared
  expert are not. RED on the `db2d8b2` tree (`is_layer_skipped_gguf(..., [])`
  was `False`); GREEN: 125 passed, 1 skipped; ruff clean.
- Left for the PR: any genuinely quantized `ReplicatedLinear` would hit the same
  missing-v2-loader path in `GGUFLinearMethod`. Not reachable with this
  checkpoint.

### 2026-09-11 — P3 full load, run 3: weights load; profile run fails in MoE

- Run 3 (plugin tree with `39efe53`): `Model loading took 61.84 GiB memory and
  115.6 seconds`. Memory sampler: GPU about 65 GB, host `Shmem` 32 GB (the
  pinned PLE table), `MemAvailable` 85 GB of 123 GB.
- Eleven seconds later the profile run (`max_num_batched_tokens=8192`,
  `max_num_seqs=8`, eager) failed inside the plugin's GGUF MoE kernel:
  `_fused_moe_gguf` → `ops.ggml_moe_a8_vec` → `RuntimeError: CUDA error: invalid
  argument`, raised from `CUDACachingAllocator::alloc_block`.
- The checkpoint's routed experts are IQ3_S (gate/up; IQ4_XS in layer 2) and
  IQ4_NL or Q8_0 (down). IQ types are not in `MMQ_QUANT_TYPES`, so every batch
  size takes the per-row `ggml_moe_a8_vec` path, and the kernels launch with
  grid `z = tokens * top_k` (81,920 rows for this profile batch).
- Isolated reproduction on the same GPU (`.dev/p3/moe_vec_repro.py`, one process
  per size, `CUDA_LAUNCH_BLOCKING=1`, 512 experts, hidden 2560, top-k 10, int32
  ids): both calls, IQ3_S w13 and IQ4_NL w2, succeed at 64, 2048, 6553, 6554 and
  8192 tokens. So neither the token count nor the grid size alone reproduces
  it. vLLM's default `fused_topk` returns int32 ids, as the kernel expects.
- Because CUDA reports launch errors asynchronously, the allocator is only where
  the error surfaced. Run 4 repeats the load with `CUDA_LAUNCH_BLOCKING=1` to
  find the failing call.
- Correction (see the next entry): this reproduction was wrong. It allocated
  every tensor before the launch and never checked the outputs, so a launch
  that never ran looked like a success.

### 2026-09-11 — P3 runs 4 and 5: the CUDA grid z limit

- Run 4 (`CUDA_LAUNCH_BLOCKING=1`) failed identically, in the allocator of the
  same `ggml_moe_a8_vec` call. Blocking launches did not move the report, which
  already pointed at a launch that had not run rather than at a faulting kernel.
- Run 5 used a box-only tree that printed the arguments of the failing call:
  `X=(81920, 640) bfloat16`, `W=(512, 2560, 360) uint8`, `ids=(8192, 10) int32`,
  `top_k=1`, `type=20` (IQ4_NL), `row=2560`, `tokens=81920`. This is the down
  projection: one row per routed pair, with the top-k ids passed unchanged and
  `top_k=1`.
- The `moe_vec_*` kernels launch with `dim3 block_nums(block_num_y, 1,
  tokens * top_k)`. CUDA caps the grid's y and z extents at 65,535, so both the
  gate/up call (8,192 × 10) and the down call (81,920 × 1) exceed it. The launch
  is rejected, and the error is only returned by the next CUDA call, here the
  allocation of the output tensor for the following step.
- Second reproduction (`.dev/p3/moe_vec_repro2.py`, evidence
  `.dev/evidence/q4gguf-p3-moe-vec-grid-z-repro2-20260911.log`): after each
  launch, allocate a new tensor and synchronize. With 512 experts, hidden 2560
  and top-k 10, both calls pass at 2,048 and 6,553 tokens (z = 65,530) and fail
  at 6,554 (z = 65,540) and 8,192 with `alloc_after_launch=FAIL CUDA error:
  invalid argument`. The boundary is exactly the grid limit.
- The same script compared one launch against launches of at most
  `32768 // top_k` tokens. Its first comparison reported every row different:
  random IQ bytes decode to NaN and infinity, and `NaN != NaN`. With a NaN-aware
  comparison both calls match exactly (0 differing rows at 2,048 and 6,553
  tokens; finite fractions 0.727 for IQ3_S and 0.530 for IQ4_NL).
- The other launch sites were checked. `mmvq.cuh` launches with
  `block_nums(block_num_y, nvecs, 1)` and would hit the y limit, but the dense
  linear method uses it only for at most 8 or 16 rows. `mmq.cuh` divides every
  grid extent by its block size.
- Fix `32a1662`: `ops.ggml_moe_a8_vec` splits calls above `65535 // top_k`
  tokens into launches under the limit and concatenates the outputs. The kernel
  reads ids as a flat array indexed by row, so ids are sliced flat, which covers
  the down call's `(tokens, 10)` ids with `top_k=1`. Calls under the limit keep
  the single launch. A C++ fix that loops over launches inside the kernel would
  avoid the Python loop; the same launch exists in vLLM's in-tree GGUF kernel,
  so that belongs in the PR.
- New test `tests/test_moe_vec_grid_limit.py` covers both call shapes with Q8_0
  experts, forces an allocation after the call, and compares the result against
  direct launches of at most 4,096 rows. RED on the PRO 6000 with the unfixed
  tree: both cases failed with `CUDA error: invalid argument` at that allocation.
  GREEN with the fix, together with the adapter and companion-name suites:
  42 passed. Repository-wide ruff 0.14.0 check and format are clean.

### 2026-09-11 — P3 runs 6 and 7: an illegal memory access in the same kernel

- Run 6 (tree with `32a1662`): weights loaded in 94 s (61.84 GiB). The grid
  error is gone, but six seconds later the profile run failed with `CUDA error:
  an illegal memory access was encountered`, reported at `shared_output +
  fused_output` in vLLM's MoE runner. That report is asynchronous.
- Run 7 repeated it with `CUDA_LAUNCH_BLOCKING=1`. The error now surfaces inside
  the gate/up call `ops.ggml_moe_a8_vec(x, w1, ...)`, at the chunked kernel call,
  in the fill of the output tensor (`new_zeros`, `FillFunctor<BFloat16>`). The
  custom kernels do not check their launches, so the fault belongs to the
  previous unchecked launch: the first chunk's `moe_vec` or `quantize_row_q8_1`
  kernel. Before the fix that launch was rejected and never ran, so this fault
  was hidden behind the grid error.
- The routed-expert types are consistent within every layer (`gguf` reader over
  all shards): gate and up are IQ3_S in every layer except layer 2 (IQ4_XS);
  down is IQ4_NL except layers 2, 4, 30, 46 and 47 (Q8_0). A gate/up type
  mismatch reading past the packed tensor is therefore ruled out.
- The kernel's indices stay far inside 32-bit range for these shapes (largest is
  `blockIdx.z * nrows` ≈ 84 M), and IQ3_S, IQ4_XS and IQ4_NL decode through
  fixed-size codebooks, so bad weight bytes cannot index out of bounds. The
  remaining candidates are an expert id outside `[0, 512)` and a stride or
  layout the kernel does not expect.
- Run 8 uses a box-only debug tree (`.dev/p3/moe_dbg_patch.py`, never
  committed). It synchronizes before and after every per-row call with more than
  64 tokens and prints shapes, strides, id range and quant type.

### 2026-09-11 — P3 runs 8 and 9: padding tokens carry expert id -1

- Run 8 printed the first call before it faulted: `X=(8192, 2560) bfloat16`,
  `W=(512, 1280, 1100) uint8` (contiguous), `ids=(8192, 10) int32`, `type=21`
  (IQ3_S), and `idmin=-1 idmax=-1`. Every expert id was -1. The kernel computes
  the expert's weight offset as `expert * nrows * blocks_per_row`, so it read
  the bytes before the expert tensor.
- `fused_topk` itself never returns -1 on the same GPU, whatever its input: finite,
  NaN, positive and negative infinity (float32 and bfloat16, 8,192 tokens), and
  unprojected 2,560-wide hidden states (ids up to 2,559). The `other=-1` in
  `base_router.py` belongs to the EPLB logical-to-physical map, which is off.
- Run 9 (`.dev/p3/moe_dbg_patch2.py`) traced the router: `FusedTopKRouter`,
  no EPLB, finite bfloat16 logits (largest magnitude 8.9), and still `idmin=-1
  idmax=-1` on return.
- The cause is in the router's call: `vllm_topk_softmax` passes
  `is_padding=_get_padding_mask(...)`, and `VLLM_MOE_SKIP_PADDING` defaults to
  on. The GPU model runner marks padding tokens in its input batch, the kernel
  writes id -1 for them, and vLLM's own MoE kernels skip such slots. A profile
  run consists only of padding tokens, so every id was -1; in serving, any
  padded batch would reach the same fault.
- Fix `e1e120f`: `_fused_moe_gguf` sends empty slots to expert 0 with zero
  weight (`clamp(min=0)` and `masked_fill`), out of place and without a host
  sync, so all three paths (per-row, grouped, slow fallback) are covered and the
  caller's routing tensors stay unchanged.
- New test `tests/test_moe_padding_ids.py`: Q8_0 experts at 16 tokens (per-row
  path) and 128 tokens (grouped path), with half the tokens fully padded and one
  single empty slot. Padded rows must be exactly zero, the other rows must match
  the same batch routed with valid ids and the empty slot's weight set to zero,
  and the caller's ids and weights must be unchanged. RED on the PRO 6000 with
  the previous tree: `CUDA error: an illegal memory access was encountered`.
  GREEN with the fix, together with the grid-limit, adapter and companion-name
  suites: 44 passed. Repository-wide ruff 0.14.0 check and format are clean.
- Run 10 (tree with `e1e120f`): weights loaded in 91 s (61.84 GiB) and the
  profile run passed. `Available KV cache memory: 21.55 GiB`, `GPU KV cache size:
  635,699 tokens`, maximum concurrency 19.40x at 32,768 tokens per request. The
  engine then failed in kernel warm-up, not in the model: FlashInfer JIT-compiles
  its sampling kernels on first use, and the box's `/usr/local/cuda/include` has
  no cuRAND headers (`sampling.cuh:20: fatal error: curand.h: No such file or
  directory`; the venv's `nvidia/cu13/include` does have one). This is a gap in
  the rented environment. Run 11 sets `VLLM_USE_FLASHINFER_SAMPLER=0`, which the
  GPU sampler honours through `flashinfer_sampler_supported()`, and falls back to
  the PyTorch top-k/top-p sampler; greedy requests never use FlashInfer sampling.
- Run 11 is the first generation on the PRO 6000; see the next entry.
- Monitoring note: the background monitors for runs 6 and 7 never reported,
  because the Mac shell (zsh) does not split an unquoted `$SSH`, so every poll
  failed and was retried silently. Later monitors call ssh through a function
  and fail loudly after repeated connection errors.

### 2026-09-11 — P3 run 11: first generation on the PRO 6000

- Tree with `e1e120f`, `VLLM_USE_FLASHINFER_SAMPLER=0`, eager mode,
  `--max-model-len 32768 --max-num-seqs 8 --gpu-memory-utilization 0.90`, PLE
  table in pinned host memory (`--engram-config '{"cpu_offload": true}'`).
- Startup: weights loaded in 83.6 s (61.84 GiB); profile, KV cache and warm-up
  took 28.2 s; the API was ready 2 min 9 s after launch.
- Memory: KV cache 21.55 GiB, 635,699 tokens (19.4 concurrent 32,768-token
  requests). GPU 89.1 GB in use. Host: 32 GB shared memory (the pinned PLE
  table), engine RSS 34 GB, 84 GB still available.
- Correctness, greedy with thinking off:
  - a short Chinese prompt returned `我是通义千问，17 乘以 23 等于 391。`
    (correct product);
  - a 13,847-token prompt with an access code buried in filler returned `7342`;
  - a 512-token Chinese expository answer was coherent and on topic.
- Speed (evidence `logs/p3-requests-run11*.log`, `logs/p3-concurrency-run11.log`
  on the box):
  - short prompt: first token 0.25 s cold, 0.10 s warm;
  - 13,847-token prompt: first token 16.97 s cold, because vLLM JIT-compiled two
    Triton kernels (`_fused_post_conv_kernel`, `_qsa_pre_indexer_kernel`) during
    that first request and warned about it; 1.62 s warm, about 8,500 prompt
    tokens per second;
  - decode: 30.0 tokens per second for a single 512-token answer;
  - 256-token answers at concurrency 1, 4 and 8: 30.2, 106.8 and 226.5 tokens per
    second in aggregate (30.2, 26.7 and 28.3 per request). Throughput scales
    almost linearly and the GPU showed 62% utilisation at concurrency 8, which
    points at per-step host overhead in eager mode rather than at the kernels.
- Run 12 drops `--enforce-eager` (torch.compile and CUDA graphs) and raises
  `--max-num-seqs` to 32 to test whether the plugin's custom ops survive graph
  capture and how far throughput moves.

### 2026-09-11 — P3 run 12: CUDA graphs

- Same tree and sampler setting as run 11, without `--enforce-eager`, with
  `--max-num-seqs 32` (`.dev/p3/p3-serve-graph.sh`). The plugin's custom ops
  compiled and captured without changes: piecewise and full CUDA graphs in two
  passes (7 s and 2 s, 0.28 GiB and 0.10 GiB). Engine init took 38.3 s; the API
  was ready 2 min 33 s after launch.
- KV cache 21.21 GiB, 625,868 tokens (19.1 concurrent 32,768-token requests).
- Greedy outputs are identical to run 11: the same short answer, `7342` for the
  13,847-token prompt, and the same 512-token text.
- Speed (evidence `logs/p3-bench-run12.log` on the box):
  - short prompt: first token 0.09 s; 13,847-token prompt: 16.92 s cold (the same
    two Triton kernels JIT-compile on the first long request) and 1.62 s warm;
  - decode: 132.7 tokens per second for a single 512-token answer, 4.4 times
    eager mode;
  - 256-token answers (`ignore_eos`) at concurrency 1, 4, 8, 16 and 32: 128.7,
    303.6, 404.0, 505.1 and 562.0 tokens per second in aggregate; 128.8, 76.0,
    50.6, 31.6 and 17.6 per request. The GPU drew 600 W at 100% utilisation.
- Throughput flattens above 16 concurrent requests. Every routed expert is an IQ
  type, and IQ types are not in `MMQ_QUANT_TYPES`, so every batch size runs the
  per-row kernel: cost grows with tokens × top-k and nothing is batched per
  expert. A grouped (MMQ) kernel for IQ3_S and IQ4_NL is the main lever for
  serving capacity.
- Remaining gaps before hosting: the FlashInfer sampler needs a CUDA toolkit with
  cuRAND headers; the first long request pays for Triton JIT that warm-up does not
  cover; quality has only been smoke-tested (no benchmark against the reference
  checkpoint); context beyond 32,768 tokens, thinking mode, tool calls and a soak
  test under load are untested; and the stack depends on a vLLM nightly plus the
  unmerged #56273.

### 2026-09-11 — P3 run 13: 64K context and concurrency limits

- Same tree as run 12, `--max-model-len 65536 --max-num-seqs 96
  --gpu-memory-utilization 0.90`, BF16 KV (`.dev/p3/p3-serve-64k.sh`). vLLM
  defaults left on: chunked prefill (`max_num_batched_tokens=8192`), prefix
  caching with the GDN state cache in `align` mode, attention block size 1568
  tokens. torch.compile stays off (upstream #55272 removed it for this model).
- Memory: weights on the GPU 61.84 GiB; KV cache 20.86 GiB, 723,466 tokens, which
  vLLM reports as 11.04 concurrent 65,536-token requests. By GGUF tensor bytes the
  routed experts are 55.43 GiB (IQ3_S 31.56, IQ4_NL 18.90, Q8_0 4.15, IQ4_XS 0.83),
  the PLE table 26.85 GiB (host, pinned), attention/GDN/indexer 3.04 GiB, embedding
  and output 1.12 GiB. The BF16 main KV alone is 24 KiB per token (12 full-attention
  layers × 2 KV heads × 256).
- Load harness `.dev/p3/p3-bench.py` streams requests at a fixed concurrency and
  samples `/metrics` (running, waiting, KV usage, preemptions). Evidence
  `.dev/evidence/q4gguf-p3-bench-run13-*.log`.
- Short prompts, 256-token answers with `ignore_eos`:

  | concurrency | tok/s total | tok/s per request | first token | peak KV |
  |---|---|---|---|---|
  | 32 | 552.9 | 19.7 | 1.84 s | 34.2% |
  | 48 | 572.7 | 13.3 | 2.23 s | 51.2% |
  | 64 | 591.0 | 10.3 | 3.05 s | 68.3% |
  | 82 | 558.1 | 7.6 | 3.96 s | 87.5% |

  No errors or preemptions. KV usage grows by about 1.07% per request whatever its
  length, so a short request's cost is dominated by its GDN state pages, and the
  pool holds about 93 sequences.
- 64K prompts (63,4xx tokens, an access code buried at the middle), greedy:
  - short answers at concurrency 1, 4 and 8: every code correct; first token
    80.7 s alone, p50/max 268/329 s at 4 and 430/660 s at 8. Only two requests
    were ever running, the rest waited; prompt throughput was about 770 tokens per
    second at every level, against about 8,500 at 13,847 tokens in run 11.
  - answers held to 1,024 tokens (`ignore_eos`) at 8 and 12: every code correct;
    at most 7 requests ran at once with KV usage 96.4%, the rest queued (up to 11
    waiting) and none was preempted; first token p50/max 430/679 s at 8 and
    578/1,010 s at 12; decode 11.9 tokens per second in aggregate.
  - one 64K request takes 13.2–13.8% of the pool, so the practical limit is 7
    concurrent 64K requests, not the 11.04 that vLLM's startup estimate gives
    (the estimate leaves out the blocks the GDN groups hold per request).
- During the first long requests vLLM warned about Triton JIT for
  `_qsa_pre_indexer_kernel`, `_qsa_mqa_paged_prefill_kernel`, `_ple_conv_kernel`
  and `_ple_conv_writeback_kernel`: warm-up does not cover these shapes.
- Harness note: `tmux kill-session` did not stop a running benchmark client (the
  process outlived its session), so the first held-answer attempt shared the
  server with a leftover level and was discarded
  (`q4gguf-p3-bench-run13-hold-contaminated.log`); the rerun started on an idle
  server, checked through `/metrics`.
- Levers for the KV limit: an FP8 main KV on the QSA path (rejected by the pinned
  nightly, `nvidia/qsa.py` raises unless BF16; upstream #55557 is approved but not
  merged) halves the 24 KiB per token but not the GDN pages; a higher memory
  fraction; and prefix caching off, which removes the `align` state blocks at the
  cost of prefix reuse. The GGUF carries no MTP tensors (1,224 tensors, `blk.0`
  to `blk.47`), although `config.json` declares one MTP layer.

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
- GPU tests and full load: one RTX PRO 6000 WE (vast.ai), venv on nightly
  `0.28.1rc1.dev628+g2a02f6efe` with #56273 overlaid; see the 2026-09-11 entries.

## Work packages

| ID | Scope | State |
|---|---|---|
| P0 | Base pin, test environment, contracts, this log | done |
| P1 | IQ4_NL PLE embedding method, Triton lookup kernel, `from_quant_config` hook | accepted after Core kernel repair, merged `2e9faa2`; 32 passed on GPU |
| P2 | Port the `qwen4_exp` adapter to nightly and stream PLE shards | accepted, merged `4a99ed1`; real-checkpoint check passed |
| P3 | Full download, full load, GPU kernel tests, generation, quality and performance | first generation on the PRO 6000 (run 11, eager) after fixes `db2d8b2`, `39efe53`, `32a1662`, `e1e120f`; CUDA graphs (run 12): 133 tok/s single stream, 562 tok/s at 32 concurrent; quality benchmark, long context and soak pending |
