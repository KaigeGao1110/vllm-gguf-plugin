# Qwen4Exp IQ4 GGUF Loader Experiment Implementation Plan

> **For agentic workers:** Use bounded subagent-driven implementation with one writer per file. The user approved this staged experiment on 2026-09-09. Preserve the running NVFP4 service and validate on CPU first.

**Goal:** Make the selected Flash GGUF configuration and real tensor directory pass a model-specific plugin adapter, and validate a bounded PLE decode/lookup path before considering a full model load.

**Architecture:** Add a Qwen4Exp adapter at the existing plugin extension point, reuse the native vLLM Flash model, and derive mappings and layout transforms from the pinned llama.cpp converter and vLLM loader. Keep the published GGUF weights unchanged. A PLE helper belongs in its own module; the full embedding is too large for a naive temporary full-table decode.

**Tech Stack:** Plugin base d4c1f0d082fc7cd4350da56689109a01c1f29d6c, retained vLLM image bd995759b5b8, Python 3.12 / PyTorch 2.13.0, gguf 0.19.0. Official model config and selected GGUF revision are pinned in the parent evidence directory.

## 1. Establish actual tensor and memory inputs

- [x] Preserve the existing failed parser/adapter receipts as RED evidence.
- [x] Read bounded HTTP Range prefixes from both weight shards and parse their tensor directories with gguf's header reader, without reading tensor payloads into arrays.
- [x] Check 1,224 unique tensor names against the GGUF declared total. PLE is IQ4_NL, shape GGML [160, 320001536], 28,800,138,240 packed bytes and 102,400,491,520 BF16 bytes.
- [ ] Read small real PLE row samples, preserving revision, offsets and exact byte lengths. Compare a prospective bounded decoder with the independent gguf reference implementation.

## 2. Register a dedicated architecture and tensor mapper

Writer allowlist: `vllm_gguf_plugin/weights_adapter/qwen4_exp.py`, `vllm_gguf_plugin/weights_adapter/__init__.py`, and `tests/test_qwen4_exp_adapter.py` only.

- [ ] Add focused assertions using the real config and tensor-name examples: the adapter must match qwen4_exp, return Qwen4ExpForConditionalGeneration, map actual non-layer/attention/linear-attention/MoE/HC/indexer/PLE names to the native loader's expected names, and reject unexpected names.
- [ ] Keep parser failure receipts as the pre-change RED baseline and run any newly introduced falsifying cases before implementing their behavior.
- [ ] Implement `matches`, `architecture`, `patch_hf_config`, and `build_name_map` at the existing BaseGGUFWeightsAdapter boundary. Use the pinned conversion source to derive PLE block insertion and head-layout transforms rather than copying Qwen3.5 mappings blindly.
- [ ] Register the adapter before the generic fallback. Do not mutate Transformers global model registries or relabel the model as a different architecture.
- [ ] Run the Linux component harness against the adapted source, separately reporting registration, all-name mapping, and any unimplemented tensor layout/load requirements.

## 3. Validate a bounded PLE path

Parent owns a separate PLE helper and tests; do not edit the adapter writer's files until its lease ends.

- [ ] Determine whether native PLE's BF16 CPU path can consume decoded rows/shards without allocating a second complete embedding table.
- [ ] Test decoding sampled real IQ4_NL rows against gguf's reference decoder, BF16 conversion, duplicate/unsorted row lookups, and invalid row boundaries.
- [ ] Calculate full-load resident and peak memory from actual dimensions, with the existing service still accounted for. Never instantiate the full PLE table in the CPU probe.
- [ ] If preserving compressed PLE needs a new query implementation, report the concrete gap; do not claim a name-map pass proves inference support.

## 4. Integration and handoff

- [ ] Execute the smallest relevant Linux checks in unique read-only, CPU-only, network-disabled containers. Keep failing receipts and clean run-owned containers/tmpfs on every exit.
- [ ] Export the exact local patch and a concise result report with commands, exits, current unresolved limits, and next possible runtime action.
- [ ] Verify the existing NVFP4 container ID and start time are unchanged. No release, service replacement, rental, large weight download, or deletion of existing artifacts is part of this CPU-first checkpoint.

The next runtime checkpoint is a genuine single-request generation; MTP, FP8 KV and concurrency remain later measurements. A CPU mapping pass is not model-quality, numerical-inference, or capacity evidence.
