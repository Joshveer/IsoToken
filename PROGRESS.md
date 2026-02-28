# IsoToken — Progress

## v1 (archived)

Phases 1-13 complete. Local PEFT/vLLM runtime, PEP compiler, parallel execution, aggregation, KV optimization, multi-adapter co-batching, real KV reuse, rank-aware scheduler, async execution, confidence-weighted aggregation, interaction logging, distillation factory, benchmarking. 109 tests. Success criteria M.1-M.5 validated.

## v2 Phases A-H (complete)

Stripped to CLI-only. Removed web server and old local-model infrastructure. Added:
- File tools (read_files, write_file, parse_code_block, build_file_prompt)
- API backends (OpenAI, Anthropic, Ollama, openai_compatible)
- Simplified executor (wave-based async, no adapter_router/scheduler/KV)
- File-aware fracture (one node per file)
- File pass-through aggregator
- Engine (API-only, file agents)
- Rich CLI (--files, --backend, --model, colored diffs, metrics)

66 tests pass.

## v2 Phases I-N — Dual-Path Local + API (in progress)

Adding local backend (transformers + PEFT) alongside existing API backends.

### Phase I — Local Backend
- [ ] I.1 LocalBackend class (model + tokenizer loading)
- [ ] I.2 run_node (adapter switching + generate)
- [ ] I.3 prefill_shared_kv (shared KV prefix)
- [ ] I.4 decode_with_kv (decode with shared KV)
- [ ] I.5 forward_batch (multi-adapter co-batching)

### Phase J — Executor Optimizations
- [ ] J.1 Re-add shared KV path (gated)
- [ ] J.2 Re-add adapter_router co-batch path (gated)

### Phase K — Distillation
- [ ] K.1 collect_data (JSONL -> Dataset)
- [ ] K.2 train_student (SFT + save adapter)

### Phase L — Engine Dual-Path
- [ ] L.1 Detect local vs API, wire optimizations
- [ ] L.2 distill() method

### Phase M — CLI Updates
- [ ] M.1 --backend local, --adapters
- [ ] M.2 distill subcommand

### Phase N — Dependencies
- [ ] N.1 Optional [local] deps in requirements.txt and pyproject.toml
