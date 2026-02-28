# IsoToken — Tasks

Atomic, independently testable work items derived from PLAN.md.

---

## Phases A-H (v2, completed)

All v2 tasks complete: cleanup, file tools, API backends (OpenAI, Anthropic, Ollama, openai_compatible), simplified executor, file-aware fracture, file pass-through aggregator, API engine, rich CLI. 66 tests pass.

---

## Phase I — Local Backend

- [ ] **I.1** Create `local_backend.py` with `LocalBackend(model_id, adapters=None)`. Load base model + tokenizer via transformers. If adapters dict provided, load LoRA adapters via PEFT. Lazy imports with clear error if deps missing. *Test:* loads tiny-gpt2; model and tokenizer not None.
- [ ] **I.2** Add `run_node` method to LocalBackend: set active adapter from node["adapter"], generate text with model.generate(), return decoded string. *Test:* run_node returns a non-empty string; different adapters produce different outputs.
- [ ] **I.3** Add `prefill_shared_kv(context) -> past_key_values`: run model forward with use_cache=True, return past_key_values. *Test:* returns tuple; same object reusable.
- [ ] **I.4** Add `decode_with_kv(prompt, past_kv) -> str`: decode using shared past_key_values. *Test:* returns string; past_kv not mutated.
- [ ] **I.5** Add `forward_batch(prompts, adapters) -> list[str]`: co-batch multiple prompts with different adapters. Group by adapter, one forward per unique adapter, reassemble in order. *Test:* 2 prompts with different adapters return 2 different outputs in correct order.

---

## Phase J — Executor Optimizations

- [ ] **J.1** Re-add optional `shared_kv_model` and `shared_kv_tokenizer` parameters to execute_pep/execute_pep_async. When provided: prefill once, decode per node with shared KV. Gated: only runs this path when params are set. *Test:* prefill called once; results match non-KV path.
- [ ] **J.2** Re-add optional `adapter_router` parameter. When provided: co-batch wave nodes via adapter_router.forward_batch(). Gated: only runs when param is set. *Test:* co-batch path produces results; one forward per adapter in wave.

---

## Phase K — Distillation

- [ ] **K.1** Create `distill.py` with `collect_data(log_path) -> Dataset`. Read JSONL with input/target, return HuggingFace Dataset. *Test:* 3-line JSONL -> Dataset with len 3 and correct columns.
- [ ] **K.2** Add `train_student(model_id, log_path, output_dir, max_steps)` to distill.py. Load base + LoRA, SFT via Trainer, save adapter. *Test:* 1-step training with tiny-gpt2; adapter files saved.

---

## Phase L — Engine Dual-Path

- [ ] **L.1** Update IsoTokenEngine to accept `backend="local"` with model_id and adapters. Create LocalBackend, wire prefill_fn and adapter_router to execute_pep when local. *Test:* engine with local backend + tiny-gpt2 initializes; run() returns answer.
- [ ] **L.2** Add `distill(log_path, output_dir)` method to engine. Works when local backend active. *Test:* distill writes adapter files.

---

## Phase M — CLI Updates

- [ ] **M.1** Add `--backend local` choice. Add `--adapters` flag (format: name=path,name=path). Wire to engine with local backend config. *Test:* --backend local --model sshleifer/tiny-gpt2 runs without crash.
- [ ] **M.2** Add `distill` subcommand: `isotoken distill --log-path ... --output-dir ... --model ...`. *Test:* subcommand parses args correctly.

---

## Phase N — Dependencies

- [ ] **N.1** Add torch, transformers, peft, accelerate, datasets to requirements.txt and pyproject.toml as optional [local] deps. *Test:* pip install without [local] does not require torch.

---

## Acceptance

- [ ] All 66+ existing tests pass (API path unchanged).
- [ ] New local backend tests pass with sshleifer/tiny-gpt2 (no GPU required).
- [ ] `python -m isotoken "prompt" --backend local --model sshleifer/tiny-gpt2` works.
- [ ] `python -m isotoken "prompt" --backend openai` works (unchanged).
- [ ] No forced import of torch/transformers/peft unless --backend local is used.
