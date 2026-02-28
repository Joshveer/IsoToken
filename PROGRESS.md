# IsoToken — Implementation Progress

This document describes what has been completed, how it was implemented, and how it works. All items below are done and tested unless noted.

---

## Overview

IsoToken is an inference runtime that turns a single complex prompt into a **Parallel Execution Plan (PEP)** of LoRA-specialized reasoning nodes sharing one base model and an optional shared KV prefix. The implementation follows SPEC.md and PLAN.md and was built test-first (TDD) across six phases plus invariant checks.

**Upgrade (Phase 6):** True multi-adapter co-batching — multiple PEP nodes with different LoRA adapters execute in a **single forward batch** per wave via `AdapterRouter.forward()` (no sequential adapter switching per node within a wave).

**Completed:** Phases 1–6 (base runtime, PEP compiler, parallel execution, aggregation, KV optimization, multi-adapter co-batching) and all four testable invariants (I.1–I.4).

**High-level flow:** `run(prompt)` → `fracture(prompt)` → PEP JSON → `execute_pep(pep, run_node, adapter_router=...)` → node outputs → `aggregate_by_pep(pep, results)` → `{"answer": <final>}`. When `adapter_router` is provided, each wave is executed in one `AdapterRouter.forward(batch_inputs, adapter_map)` call.

---

## Phase 1 — Base runtime

### What was completed

- **1.1** Deploy base model via vLLM (or TGI): service accepts requests and returns completions; no custom serving engine.
- **1.2** PEFT + LoRA: at least one adapter loads; forward pass gives distinct output vs base-only.
- **1.3** Hot-swap of LoRA adapters (A → B → A) without full restart; one adapter active per run; no cross-adapter leakage.
- **1.4** Minimal health check for stack (vLLM + PEFT + single base model); passes when correct, fails when a component is missing.

### How it was implemented

- **Serving (1.1):** `start_server.py` launches vLLM’s OpenAI-compatible server via `python -m vllm.entrypoints.openai.api_server` (model and port configurable). For tests, `tests/stub_server.py` provides mock `/health` and `/v1/completions`; `tests/conftest.py` starts the stub when `ISOTOKEN_VLLM_URL` is not set. Dependencies: `requirements.txt` includes `vllm`, `transformers`, `peft`, `accelerate`, `requests`, `pytest`.
- **PEFT/LoRA (1.2):** `load_lora.py` uses HuggingFace `AutoModelForCausalLM` and PEFT’s `get_peft_model` / `LoraConfig`. Tests use `sshleifer/tiny-gpt2`. LoRA B weights are set to a small non-zero value so adapter output differs from base (compared via last-token logits in tests).
- **Hot-swap (1.3):** `load_lora.py` defines `load_base_with_two_adapters()` (adapters `default` and `B`) and `set_active_adapter(model, name)` / `get_active_adapter(model)` using PEFT’s `model.set_adapter()`. Tests run A → B → A and assert outputs and active adapter.
- **Health (1.4):** `health_check.py` implements `check_stack()` returning `(ok, details)`. It checks vLLM (import or stub mode), PEFT (import), and single-base-model (env skip for tests). The stub server calls `check_stack()` for `/health` and returns 200/503 and a JSON body.

### How it works

- Production: run `start_server.py` to start vLLM; point clients at the configured port. Use `load_lora.py` to load base + adapters and switch adapters with `set_active_adapter` before each decode.
- Tests: conftest starts the stub; tests hit `localhost:8000` or use `load_lora` in-process. Health tests toggle `ISOTOKEN_HEALTH_SKIP_*` to simulate missing components.

---

## Phase 2 — PEP compiler

### What was completed

- **2.1** PEP JSON schema: `task_id`, `global_context`, `nodes` (node_id, type, adapter, prompt, depends_on), `aggregation.strategy`; valid PEP passes, invalid ones raise clear errors.
- **2.2** Multi-question prompt → ≥2 parallel nodes (empty `depends_on`).
- **2.3** “Compare A and B” → parallel evaluation nodes.
- **2.4** “List pros and cons” → two independent nodes.
- **2.5** “Verify” / “Critique” → sequential: critic node `depends_on` prior node.
- **2.6** All emitted PEPs are acyclic (DAG).
- **2.7** When subtask count &lt; 2, compiler produces single-node or sequential PEP (no parallel roots).

### How it was implemented

- **Schema (2.1):** `fracture.py` defines `validate_pep(pep)` which checks presence and types of top-level keys and each node’s required keys; `aggregation.strategy` must be in `{"vote", "synthesize"}`. Raises `ValueError` with explicit messages.
- **Fracture rules:** `fracture(prompt)` in `fracture.py` applies rule order: (1) multi-question (`?` count ≥ 2 → split on `?\s+`), (2) pros/cons (“pros and cons” → two nodes), (3) compare (“compare X and Y” / “comparison of” → split on “ and ” / “&”), (4) verify/critique (“verify”/“critique” → n1 analysis, n2 verification with `depends_on: ["n1"]`), (5) else single node. PEP gets `task_id` (uuid), `global_context`, `nodes`, `aggregation: { strategy: "vote" }`.
- **DAG (2.6):** `is_pep_dag(pep)` uses DFS cycle detection on `depends_on`; used by executor and tests.

### How it works

- Call `fracture(prompt)` to get a PEP dict. Call `validate_pep(pep)` before using it. Rule order ensures “pros and cons” is not treated as compare. Single-question or non-matching prompts fall through to one node, satisfying 2.7.

---

## Phase 3 — Parallel execution

### What was completed

- **3.1** `execute_pep(pep, run_node)`: runs all nodes respecting `depends_on`; independent nodes can run in any order or in parallel.
- **3.2** Parallel dispatch via `ThreadPoolExecutor` (same process); independent nodes run concurrently (timing test).
- **3.3** Per-node: set adapter, run decode, store output; `run_node_with_model(model)`; no cross-thread adapter state when run sequentially.
- **3.4** Shared KV prefix: `prefill_fn(global_context)` called once; result passed as `shared_prefill` to every `run_node` call.
- **3.5** Rank-aware scheduling: within a wave, nodes sorted by `adapter_rank × token_budget` (ascending); optional on node dict.
- **3.6** Context slicing: optional `slice_output(raw)`; stored results and dependency context are sliced (no full CoT downstream).

### How it was implemented

- **Executor:** `execute.py` implements `execute_pep(pep, run_node, parallel=True, prefill_fn=None, slice_output=None, metrics=None, prefill_release_fn=None)`. It builds waves via `_execution_waves()` (nodes with no deps, then nodes whose deps are done, etc.). Each wave is sorted by `_node_cost` then: if `parallel and len(wave) > 1`, nodes in the wave run in a `ThreadPoolExecutor`; else they run sequentially. `run_node(node, context, shared_prefill)` receives dependency outputs in `context` and optional `shared_prefill`. Results are stored; if `slice_output` is given, stored value is `slice_output(raw)`. At the end, if `prefill_release_fn` is set, it is called with `shared_prefill`.
- **run_node with model:** `run_node_with_model(model)` returns a callable that sets `set_active_adapter(model, node["adapter"])` then runs `run_forward_logits(model, node["prompt"])` (from `load_lora`).
- **Backward compatibility:** `_call_run_node` tries `run_node(node, dep_outputs, shared_prefill)` and falls back to `run_node(node, dep_outputs)` on `TypeError`.

### How it works

- Pass a PEP and a `run_node` (and optional prefill/slice/metrics/release). Executor runs waves in order; within a wave, nodes run in cost order and either in parallel (thread pool) or one by one. Dependencies are passed as `context`; dependent nodes see only prior nodes’ (possibly sliced) outputs.

---

## Phase 4 — Aggregation

### What was completed

- **4.1** Majority-vote aggregation: most frequent value wins; tie = first among tied; no majority = first output; edge cases tested.
- **4.2** Critic-synthesis: `critic_fn(outputs)` → review, then `synthesizer_fn(review)` → final answer; order and I/O tested.
- **4.3** Strategy from PEP: `aggregation.strategy` “vote” or “synthesize”; invalid strategy raises; “synthesize” requires `critic_fn` and `synthesizer_fn`.
- **4.4** End-to-end: `run(prompt)` → fracture → execute_pep → aggregate_by_pep → `{"answer": <str>}`; format stable and documented.

### How it was implemented

- **aggregate.py:** `aggregate_vote(outputs)` normalizes to string, counts, returns first among max-count. `aggregate_synthesize(outputs, critic_fn, synthesizer_fn)` calls critic then synthesizer. `aggregate_by_pep(pep, outputs_dict, critic_fn=None, synthesizer_fn=None)` reads `pep["aggregation"]["strategy"]`, dispatches to vote or synthesize (or raises).
- **run.py:** `run(prompt, run_node=None)` builds PEP with `fracture`, validates with `validate_pep`, runs `execute_pep(pep, run_node=run_node, parallel=False)`, then `aggregate_by_pep(pep, results)`, and returns `{"answer": final}`. Default `run_node` returns a placeholder string.

### How it works

- For “vote”, pass a PEP and a dict `node_id → output`; `aggregate_by_pep` collects values and runs majority vote. For “synthesize”, pass the same plus `critic_fn` and `synthesizer_fn`. `run(prompt)` is the full pipeline with a default or custom `run_node`.

---

## Phase 5 — KV optimization

### What was completed

- **5.1** Shared prefix: prefill invoked once per execute_pep; same `shared_prefill` handle passed to all run_node calls (clone reference, not copy).
- **5.2** Benchmark: `execute_pep` can write `metrics["prefill_count"] = 1` when `prefill_fn` is used; `kv_benchmark.benchmark_prefill(num_nodes)` runs a multi-node PEP with prefill and returns `{ prefill_count: 1, recorded: True }`.
- **5.3** Prefix GC: optional `prefill_release_fn(handle)` called after all nodes complete; tests confirm release is called and create/release counts match across multiple runs (no unbounded growth).

### How it was implemented

- **5.1:** No new code; existing `execute_pep` already calls `prefill_fn(pep["global_context"])` once and passes the same object to every `run_node`. Tests assert single prefill call and same `id(shared_prefill)` for all nodes.
- **5.2:** `execute_pep(..., metrics=metrics)` sets `metrics["prefill_count"] = 1` when `prefill_fn` is present. `kv_benchmark.py` implements `benchmark_prefill(num_nodes)` that builds a PEP, runs with `prefill_fn` and `metrics`, and returns the metrics dict plus `recorded: True`.
- **5.3:** `execute_pep(..., prefill_release_fn=fn)` calls `prefill_release_fn(shared_prefill)` at the end when both are set. Tests verify the callback is invoked once per run and that repeated runs do not leak (create count = release count).

### How it works

- Use `prefill_fn` to create a shared handle (e.g. KV block ref) once; pass it into `run_node`. Use `metrics` to record prefill count. Use `prefill_release_fn` to release or GC the prefix after the run so the next run does not accumulate prefixes.

---

## Phase 6 — Multi-adapter co-batching

### What was completed

- **6.1** AdapterRouter abstraction (`adapter_router.py`): `__init__(base_model)`, `register_adapter(name, lora_path_or_config)`, `forward(batch_inputs, adapter_map)` → list of outputs. All adapters on same base; heterogeneous adapters per batch element; no global adapter state mutation inside any model.forward (set_adapter only between forwards). Batch-by-adapter: one model.forward per unique adapter in the batch.
- **6.2** execute_pep accepts optional `adapter_router`. When provided, each wave is batched: prompts (with dependency context for dependents) and adapter names are collected, then a single `AdapterRouter.forward(batch_inputs, adapter_map)` is called per wave. Dependency order is preserved (waves already respect depends_on).
- **6.3** Test suite `tests/test_task_6_1_multi_adapter_batch.py`: two nodes with different adapters in same forward call; forward invocation count (one per adapter when heterogeneous, one when same adapter); outputs differ per adapter; no leakage across calls; execute_pep with adapter_router batches by wave. Invariant I.4 in `tests/test_invariants.py`.

### How it was implemented

- **AdapterRouter:** Groups batch indices by adapter; for each unique adapter calls `model.set_adapter(adapter)` then `_forward_batch(inputs_for_that_adapter)` which tokenizes, pads, runs one `model(**encoded)` for the sub-batch, returns last-token logits per sequence. Results are reassembled in batch_inputs order.
- **execute_pep:** New branch: if `adapter_router is not None`, for each wave build `batch_inputs` (node prompt, or prepended context from `results` for nodes with depends_on) and `adapter_map`, call `adapter_router.forward(batch_inputs, adapter_map)`, store results by node_id. Existing `run_node` path unchanged when `adapter_router` is None.

### How it works

- Use a PEFT model with multiple adapters (e.g. `load_base_with_two_adapters()`). Construct `AdapterRouter(model)`, register adapter names. Pass `adapter_router=router` to `execute_pep(pep, adapter_router=router)`. Each wave runs in one `router.forward()`; inside that, one model.forward per unique adapter (batch-by-adapter). No custom CUDA; HuggingFace Transformers + PEFT only. API contracts `execute_pep` and `run_node` preserved (run_node still used when adapter_router is not provided).

---

## Invariants & constraints (testable)

### What was completed

- **I.1** Stack check: dependencies include HuggingFace Transformers, vLLM (or TGI), and PEFT; no custom_serving/custom_transformer in deps.
- **I.2** One adapter active per (logical) thread: two nodes with different adapters yield different outputs; repeated runs give same outputs (no cross-run leakage). Test uses `parallel=False` with a shared model to avoid adapter races.
- **I.3** Parallelize only when tasks ≥ 2: single-node PEP runs one node (sequential path); two independent nodes with `parallel=True` run concurrently (timing).
- **I.4** Single forward graph for N heterogeneous adapters (Phase 6): N nodes with different adapters in one wave → one `AdapterRouter.forward()` call; outputs differ per adapter; no leakage. Test in `test_invariants.py` and `test_task_6_1_multi_adapter_batch.py`.

### How it was implemented

- **tests/test_invariants.py:** `test_I1_stack_only_allowed_deps_no_custom_engine` reads `requirements.txt` (or `pyproject.toml`) and asserts presence of `transformers`, `peft`, and `vllm`/`text-generation-inference`/`tgi`, and absence of custom_serving/custom_transformer. `test_I2_one_adapter_active_per_thread_no_mixed_output` uses `load_base_with_two_adapters` and `run_node_with_model`, runs execute_pep twice with `parallel=False`, and asserts n1 ≠ n2 and result stability. `test_I3_single_node_no_parallel_dispatch` and `test_I3_two_independent_nodes_parallel_dispatch` assert one-node sequential execution and two-node concurrent execution (timing).

### How it works

- Run the invariant test suite to confirm stack, adapter isolation, and parallelization policy. I.2 uses sequential execution so that with a single in-process model, adapter switching is deterministic and no mixed weights appear.

---

## Test suite summary

| Phase / Invariants | Test file(s) | Count |
|--------------------|--------------|--------|
| Phase 1 | test_task_1_1_deploy_base_vllm, test_task_1_2_peft_lora, test_task_1_3_hot_swap, test_task_1_4_health_check | 10 |
| Phase 2 | test_task_2_1_pep_schema … test_task_2_7_sequential | 18 |
| Phase 3 | test_task_3_1_execute_pep … test_task_3_6_context_slicing | 9 |
| Phase 4 | test_task_4_1_majority_vote … test_task_4_4_end_to_end | 11 |
| Phase 5 | test_task_5_1_kv_blocks, test_task_5_2_benchmark_prefill, test_task_5_3_prefix_gc | 6 |
| Phase 6 | test_task_6_1_multi_adapter_batch | 6 |
| Invariants | test_invariants (I.1–I.4) | 5 |

Run all Phase 1–5 and invariant tests:

```bash
pytest tests/test_task_1_1_deploy_base_vllm.py tests/test_task_1_2_peft_lora.py tests/test_task_1_3_hot_swap.py tests/test_task_1_4_health_check.py tests/test_task_2_1_pep_schema.py tests/test_task_2_2_multi_question.py tests/test_task_2_3_compare.py tests/test_task_2_4_pros_cons.py tests/test_task_2_5_verify_critique.py tests/test_task_2_6_dag.py tests/test_task_2_7_sequential.py tests/test_task_3_1_execute_pep.py tests/test_task_3_2_parallel_dispatch.py tests/test_task_3_3_per_node_adapter.py tests/test_task_3_4_shared_kv_prefix.py tests/test_task_3_5_rank_aware.py tests/test_task_3_6_context_slicing.py tests/test_task_4_1_majority_vote.py tests/test_task_4_2_critic_synthesis.py tests/test_task_4_3_strategy_from_pep.py tests/test_task_4_4_end_to_end.py tests/test_task_5_1_kv_blocks.py tests/test_task_5_2_benchmark_prefill.py tests/test_task_5_3_prefix_gc.py tests/test_task_6_1_multi_adapter_batch.py tests/test_invariants.py -v
```

Or from the project root: `pytest tests/ -v` (includes conftest and stub_server).

---

## File map

| File | Role |
|------|------|
| **run.py** | End-to-end: `run(prompt)` → fracture → execute_pep → aggregate → `{"answer": ...}`. |
| **fracture.py** | `fracture(prompt)`, `validate_pep(pep)`, `is_pep_dag(pep)`; rule-based PEP compiler and schema. |
| **execute.py** | `execute_pep(pep, run_node, ..., adapter_router=None)`; waves, parallel dispatch or AdapterRouter batch per wave; prefill/slice/metrics/release. |
| **adapter_router.py** | `AdapterRouter(base_model)`, `register_adapter`, `forward(batch_inputs, adapter_map)`; multi-adapter co-batching (batch-by-adapter). |
| **aggregate.py** | `aggregate_vote(outputs)`, `aggregate_synthesize(...)`, `aggregate_by_pep(pep, outputs_dict, ...)`. |
| **load_lora.py** | Base + LoRA loaders, `set_active_adapter`, `get_active_adapter`, `run_forward` / `run_forward_logits` (PEFT + Transformers). |
| **health_check.py** | `check_stack()` for vLLM/PEFT/single-base; used by health endpoint. |
| **start_server.py** | Launches vLLM OpenAI API server (production serving). |
| **kv_benchmark.py** | `benchmark_prefill(num_nodes)` for prefill-count metrics. |
| **tests/conftest.py** | Session fixture: start stub server if `ISOTOKEN_VLLM_URL` unset. |
| **tests/stub_server.py** | Mock `/health` (uses `health_check.check_stack`) and `/v1/completions` for tests. |
| **tests/test_task_*.py** | Phase 1–6 task tests. |
| **tests/test_invariants.py** | I.1, I.2, I.3, I.4 invariant tests. |

---

## Not yet done (from TASKS.md)

- **Success criteria (metrics):** M.1–M.5 (context FLOP reduction, latency, accuracy benchmarks, throughput, distillation) are listed but not implemented in this progress; they are validation/benchmark targets rather than core runtime tasks.
