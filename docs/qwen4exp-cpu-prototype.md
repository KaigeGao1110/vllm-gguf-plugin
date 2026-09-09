# Qwen3.8 Flash Next GGUF CPU prototype

This is a component prototype, not a working inference backend. Full model
construction with PLE is deliberately refused until the native PLE worker and
GGUF loader are connected. No tokens have been generated with this adapter.

## Fixed inputs

- GGUF: `unsloth/Qwen3.8-Flash-Next-GGUF`, `UD-IQ4_XS`, revision
  `38bb39ee97821de2c9009abb7e93950eec396e66`.
- Config: `Qwen/Qwen3.8-Flash-Next`, revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Plugin base: `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
- Retained vLLM image: `bd995759b5b8`, vLLM
  `0.1.dev20073+g8e685d198`, PyTorch `2.13.0+cu130`, Transformers `5.15.1`.
  Its native architecture name is `Qwen3_8FlashNextForConditionalGeneration`.
- Converter: llama.cpp `6c84c7d5d8833c6e0df69628f75a0f599797934`,
  `conversion/qwen4exp.py`.

The selected three-shard checkpoint is 93,682,584,224 bytes. Its variant name
does not describe every tensor: the actual directory contains Q6_K, Q8_0, F32,
IQ4_NL, IQ3_S, IQ4_XS and BF16 tensors.

## Implemented and checked

The adapter recognizes the top-level `qwen4_exp` configuration, supplies the
retained image's architecture name, and translates the actual 1,224 tensor
names without unmatched or duplicate targets. This establishes name coverage;
it does not establish full tensor shape or numerical compatibility. The PLE
table currently maps to an intermediate name which cannot be loaded.

Focused checks cover the checkpoint-style prefix, indexer Q/K concatenation,
zero-centered PLE norms, real Q8_0 hyper-connection projection samples,
unknown names, unsupported quantized indexer pairs, and the early PLE guard.
Vision and the standalone `qwen4_exp_text` configuration are outside this scope.

`quantization/ple_cpu.py` decodes only selected IQ4_NL rows. Eight real rows
(720 bytes total) match the independent `gguf.dequantize` reference exactly in
FP32 and after BF16 rounding. Lookup tests include duplicate and unsorted IDs,
empty and scalar requests, invalid IDs and malformed storage. The helper
expects a caller-provided CPU table; it does not map a checkpoint or create a
vLLM worker.

The complete PLE table has 320,001,536 rows of width 160. It occupies
28,800,138,240 bytes compressed, versus 102,400,491,520 bytes in BF16. The chosen
direction keeps the table compressed and decodes requested rows. No full-table
BF16 expansion or full model download was performed. Native hash metadata
derived from the pinned config matches all ten checked GGUF metadata fields.

## Remaining runtime work

1. Add an explicit native PLE quantization/lookup extension so its constructor
   can hold packed IQ4_NL storage and its output buffer receives BF16 decoded
   rows. Preserve the native n-gram ID calculation and GPU placeholder path.
2. Connect a PLE-only GGUF load path before `PleOffloadRunner` constructs the
   model. The retained worker currently accepts only DefaultModelLoader and
   DummyModelLoader. It must not materialize all GGUF tensors just to find PLE.
3. Bind the PLE metadata and packed table to the actual native module
   `...ple.ple_embedding.ngram_embedding`, and avoid loading the table in GPU
   workers. Validate this boundary with a small real native CPU fixture before
   removing the current full-load guard.
4. Load complete weights and perform genuine generation, format/quality smoke
   checks and then the requested concurrency measurements. GDN and MoE kernel
   behavior, GPU/CPU peak memory, TTFT and throughput remain unmeasured here.

The runtime bridge is a proposed next step, not implemented support. Full
download also requires additional disk capacity: 87.25 GiB is needed while the
retained machine has about 27.53 GiB free. Existing service weights are retained.

## Verification evidence

The parent experiment directory retains the isolated Linux probe runner and
individual success/failure JSON receipts. Tests run in disposable CPU-only,
network-disabled, read-only containers with a 4 GiB memory limit. Run-owned
containers and tmpfs are removed on exit. No GPU service was restarted or
replaced, and no image was built or published.

Final verification commands, receipt names and the local commit identity are
recorded in the parent `adaptation-progress.md` report. Do not reinterpret
component passes as complete model loading or inference evidence.
