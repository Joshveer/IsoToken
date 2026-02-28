# IsoToken — Implementation Plan

Derived from SPEC.md.

## Overview

Dual-path CLI tool: API backends (OpenAI, Anthropic, Ollama, openai_compatible) for parallel calls + file agents, and a local backend (transformers + PEFT) that adds shared KV prefix, LoRA adapter switching, co-batching, and distillation.

## Completed (v2 Phases A-H)

- Phase A: Cleanup (removed server, old PEFT/KV infrastructure)
- Phase B: File tools (read_files, write_file, parse_code_block, build_file_prompt)
- Phase C: API Backends (OpenAI, Anthropic, Ollama, openai_compatible)
- Phase D: Executor (simplified, wave-based async)
- Phase E: Fracture (file-aware PEP generation)
- Phase F: Aggregator (file pass-through)
- Phase G: Engine (API-only, file agents)
- Phase H: CLI (rich output, --files, --backend)

## Phase I — Local Backend

Create `local_backend.py`: `LocalBackend(model_id, adapters)` loads HuggingFace model + optional LoRA adapters via PEFT. Exposes: run_node (set adapter, generate text), prefill_shared_kv (one prefill, reuse past_key_values), decode_with_kv (decode with shared KV), forward_batch (co-batch prompts with different adapters). Add `"local"` case to `make_run_node` in backends.py. Imports are lazy; missing deps give a clear install message.

## Phase J — Executor Optimizations

Re-add optional `prefill_fn` and `adapter_router` paths to execute.py, gated on availability. When local backend provides these, executor uses shared KV path and co-batching. API path unchanged.

## Phase K — Distillation

Create `distill.py`: `collect_data(log_path) -> Dataset` reads JSONL run logs; `train_student(model_id, log_path, output_dir, max_steps)` loads base + LoRA, SFT, saves adapter. Requires local backend deps.

## Phase L — Engine Dual-Path

Update engine.py: detect local vs API backend. For local: wire prefill_fn and adapter_router to executor; add `distill()` method. For API: current behavior unchanged. Distillation logging always available.

## Phase M — CLI Updates

Add `--backend local`, `--model`, `--adapters`, `distill` subcommand to CLI. Existing API flags unchanged.

## Phase N — Dependencies

Add torch, transformers, peft, accelerate, datasets as optional deps in requirements.txt and pyproject.toml. API users don't need them.

## Phase O — Tests

Add test_local_backend.py (tiny-gpt2: model load, LoRA attach, KV sharing, co-batch, adapter switching), test_distill.py (dataset from logs, 1-step training smoke), test_engine_local.py (engine with local backend, file agents, metrics). Run full suite; all existing tests unchanged.
