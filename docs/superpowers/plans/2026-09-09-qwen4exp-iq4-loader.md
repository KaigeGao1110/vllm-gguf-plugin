# Qwen4Exp IQ4 GGUF Loader Experiment Implementation Plan

> **For agentic workers:** Use bounded subagent-driven implementation with one writer per file. The user approved this staged experiment on 2026-09-09. Preserve the running NVFP4 service and validate on CPU first.

**Goal:** Make the selected Flash GGUF configuration and real tensor directory pass a model-specific plugin adapter, and validate a bounded PLE decode/lookup path before considering a full model load.

**Architecture:** Add a Qwen4Exp adapter at the existing plugin extension point, reuse the native vLLM Flash model, and derive mappings and layout transforms from the pinned llama.cpp converter and vLLM loader. Keep the published GGUF weights unchanged. A PLE helper belongs in its own module; the full embedding is too large for a naive temporary full-table decode.

**Tech Stack:** Plugin base d4c1f0d082fc7cd4350da56689109a01c1f29d6c, retained vLLM image bd995759b5b8, Python 3.12 / PyTorch 2.13.0, gguf 0.19.0. Official model config and selected GGUF revision are pinned in the parent evidence directory.

## 1. Establish actual tensor and memory inputs

- [x] Preserve the existing failed parser/adapter receipts as RED evidence.
- [x] Read bounded HTTP Range prefixes from both weight shards and parse their tensor directories with gguf's header reader, without reading tensor payloads into arrays.
- [x] Check 1,224 unique tensor names against the GGUF declared total. PLE is IQ4_NL, shape GGML [160, 320001536], 28,800,138,240 packed bytes and 102,400,491,520 BF16 bytes.
- [x] Read eight real PLE row samples (720 packed bytes), preserving revision, offsets and exact byte lengths. Compare the bounded decoder with the independent gguf reference implementation.

## 2. Register a dedicated architecture and tensor mapper

Writer allowlist: `vllm_gguf_plugin/weights_adapter/qwen4_exp.py`, `vllm_gguf_plugin/weights_adapter/__init__.py`, and `tests/test_qwen4_exp_adapter.py` only.

- [x] Add focused assertions using the real config and tensor-name examples: the adapter matches qwen4_exp and returns the retained image's concrete native architecture, Qwen3_8FlashNextForConditionalGeneration. It proposes native checkpoint names and rejects unknown names. PLE's table name remains an explicitly unsupported intermediate target.
- [x] Keep parser failure receipts as the pre-change RED baseline and run newly introduced falsifying cases before implementing their behavior.
- [x] Implement `matches`, `architecture`, `patch_hf_config`, and `build_name_map` at the existing BaseGGUFWeightsAdapter boundary. Full PLE model construction is explicitly refused because the native worker bridge is missing. The mapper alone is not a full loader.
- [x] Register the adapter before the generic fallback. Do not mutate Transformers global model registries or relabel the model as a different architecture.
- [x] Run the Linux component harness against the adapted source, separately reporting registration, all-name mapping, and unimplemented tensor layout/load requirements. The 1,224-name replay establishes name coverage, not full tensor shape or numerical compatibility.

## 3. Validate a bounded PLE path

Parent owns a separate PLE helper and tests; do not edit the adapter writer's files until its lease ends.

- [x] Inspect the native PLE worker. Its CPU constructor assumes a floating embedding and its loader accepts DefaultModelLoader/DummyModelLoader only. A packed-table construction, loader and lookup bridge is required.
- [x] Test decoding sampled real IQ4_NL rows against gguf's reference decoder, BF16 conversion, duplicate/unsorted row lookups, and invalid row boundaries. Only selected unique rows are decoded, in bounded chunks; the full table stays compressed.
- [x] Calculate the packed table (26.82 GiB), hypothetical full BF16 table (95.37 GiB) and full download (87.25 GiB). Full-runtime resident/peak memory is still unmeasured. Never instantiate the full PLE table in the CPU probe.
- [x] Report the concrete worker/loader gap. The user prefers keeping PLE compressed and decoding selected rows; full-table BF16 expansion is not the chosen implementation.

## 4. Integration and handoff

- [x] Execute focused Linux checks in unique read-only, CPU-only, network-disabled containers. Keep failing receipts and clean run-owned containers/tmpfs on every exit.
- [ ] Export the exact local patch and a concise result report with commands, exits, current unresolved limits, and next possible runtime action.
- [x] Verify the existing NVFP4 container ID and start time are unchanged. No release, service replacement, rental, large weight download, or deletion of existing artifacts is part of this CPU-first checkpoint.

The next runtime checkpoint is a genuine single-request generation; MTP, FP8 KV and concurrency remain later measurements. A CPU mapping pass is not model-quality, numerical-inference, or capacity evidence.
