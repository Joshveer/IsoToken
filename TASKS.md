# Tasks (from PLAN.md)

Atomic, independently testable work items. Each task has a clear pass/fail criterion.

---

## Phase 1 — Base runtime

- [x] **1.1** Deploy base Llama-3-70B via vLLM (or TGI). *Test:* Service accepts requests and returns completions; no new transformer/serving engine.
- [x] **1.2** Integrate PEFT and load at least one LoRA adapter onto the base model. *Test:* Adapter loads; forward pass produces distinct output vs base-only.
- [x] **1.3** Implement hot-swap of LoRA adapters (swap without full restart). *Test:* Swap adapter A → B → A; outputs match expected adapter behavior; no cross-adapter leakage (one adapter active per thread).
- [x] **1.4** Add a minimal health check that validates stack (vLLM + PEFT + single base model). *Test:* Health endpoint passes when stack is correct and fails when component is missing.

**Phase 1 complete.** All subtasks: tests pass, invariants preserved, complexity unchanged.

---

## Phase 2 — PEP compiler

- [x] **2.1** Define and validate PEP JSON schema (task_id, global_context, nodes with node_id/type/adapter/prompt/depends_on, aggregation.strategy). *Test:* Valid JSON passes schema; invalid samples fail with clear errors.
- [x] **2.2** Implement rule-based fracture: multi-question prompt → multiple parallel nodes. *Test:* Given a multi-question prompt, output PEP has ≥2 nodes with empty depends_on where spec allows.
- [x] **2.3** Implement rule: “Compare A and B” → parallel evaluation nodes. *Test:* Compare-style prompt yields PEP with parallel nodes (no unnecessary dependency).
- [x] **2.4** Implement rule: “List pros and cons” → two independent threads. *Test:* Pros/cons prompt yields two nodes, no dependency between them.
- [x] **2.5** Implement rule: “Verify” / “Critique” → sequential node depending on prior output. *Test:* Verify/critique prompt yields PEP where critic node depends_on the prior node.
- [x] **2.6** Fracture produces valid DAG (no cycles). *Test:* For a set of prompts, every emitted PEP has acyclic depends_on.
- [x] **2.7** When subtask count < 2, compiler produces single-node or sequential PEP (no parallelization). *Test:* Single-question or non-shardable prompt yields PEP that does not parallelize; invariant “tasks < 2 → sequential” holds.

---

**Phase 2 complete.** All subtasks: tests pass, invariants preserved, complexity unchanged.

---

## Phase 3 — Parallel execution

- [x] **3.1** Implement execute_pep: given PEP JSON, dispatch nodes respecting depends_on (order or parallel where allowed). *Test:* PEP with two independent nodes runs both; PEP with n2 depends_on n1 runs n1 then n2.
- [x] **3.2** Use Ray (or async batching) for parallel node dispatch. *Test:* Independent nodes show concurrent execution (e.g. timing or concurrency metric).
- [x] **3.3** Per-node execution: load adapter, attach LoRA, run decode, store output; no cross-thread adapter state. *Test:* Two nodes with different adapters produce correct adapter-specific outputs; no interference (invariant: one adapter active per thread).
- [x] **3.4** Implement shared KV prefix: prefill once for shared context; reuse for all threads that share it. *Test:* Same global_context used across nodes triggers single prefill (e.g. prefill count or KV cache reuse observable).
- [x] **3.5** Rank-aware scheduling: schedule by cost = adapter_rank × token_budget; lower cost first. *Test:* Given a PEP with mixed-rank adapters, execution order or prioritization matches cost order.
- [x] **3.6** Context slicing: after each node, only final answer and key reasoning summary are stored; full chain-of-thought not propagated. *Test:* Stored context for each node is bounded (e.g. max tokens or summary-only); no full CoT in downstream node input.

**Phase 3 complete.** All subtasks: tests pass, invariants preserved, complexity unchanged.

---

## Phase 4 — Aggregation

- [x] **4.1** Implement majority-vote aggregation over node outputs (factual tasks). *Test:* Given N outputs and a clear majority answer, aggregate returns that answer; edge cases (no majority, ties) defined and tested.
- [x] **4.2** Implement critic-synthesis flow: parallel outputs → Critic LoRA reviews → Synthesizer produces final answer. *Test:* Given a set of node outputs, critic-synthesis path returns a single final answer; critic and synthesizer are invoked in order.
- [x] **4.3** Aggregate strategy selectable from PEP (aggregation.strategy: vote | synthesize). *Test:* PEP with strategy "vote" uses vote; PEP with "synthesize" uses critic-synthesis; invalid strategy fails cleanly.
- [x] **4.4** End-to-end: prompt → fracture → execute_pep → aggregate → final answer. *Test:* One full run with multi-node PEP returns a single final answer; output format stable and documented.

**Phase 4 complete.** All subtasks: tests pass, invariants preserved, complexity unchanged.

---

## Phase 5 — KV optimization

- [x] **5.1** Expose or use vLLM KV blocks for shared prefix (clone reference, not full copy). *Test:* Prefill is run once for shared prefix; multiple threads use the same KV blocks (e.g. via block ref count or single prefill).
- [x] **5.2** Benchmark prefill: measure prefill count and GPU time for sequential N runs vs IsoToken shared-prefix run. *Test:* Metrics show O(C²) + Σ O(C_i²) behavior vs baseline O((kC)²); results recorded.
- [x] **5.3** Fixed prefix size and GC after threads complete (KV fragmentation invariant). *Test:* After execute_pep finishes, prefix KV is released or GC’d; no unbounded growth across runs.

---

**Phase 5 complete.** All subtasks: tests pass, invariants preserved, complexity unchanged.

---

## Phase 6 — Multi-adapter co-batching

- [x] **6.1** Create AdapterRouter abstraction (`adapter_router.py`). Class AdapterRouter: `__init__(self, base_model)`, `register_adapter(name, lora_path_or_config)`, `forward(batch_inputs: List[str], adapter_map: List[str]) -> List[outputs]`. All adapters on same base; heterogeneous adapters per batch element; no global adapter state mutation inside forward. *Test:* Two nodes with different adapters execute in same forward call; single model.forward invocation (mock/hook); outputs differ per adapter; no leakage.
- [x] **6.2** Modify execute_pep: group nodes by wave, batch prompts into a single call to AdapterRouter.forward() per wave; preserve dependency order. Maintain existing execute_pep/run_node API. *Test:* Wave with 2+ nodes uses batch path when adapter_router provided; dependency order preserved.
- [x] **6.3** Add test suite `tests/test_task_6_1_multi_adapter_batch.py`: two nodes different adapters in same forward; validate one model.forward invocation; outputs differ per adapter; no adapter leakage. Invariant I.4: single forward graph for N heterogeneous adapters. *Acceptance:* All previous tests pass; new multi-adapter tests pass.

**Phase 6 complete.** AdapterRouter; execute_pep batches by wave when adapter_router provided; I.4 single forward graph for N heterogeneous adapters.

---

## Phase 7 — Real prefix KV cache sharing

- [x] **7.1** Modify execute_pep: on start run `model(global_context_ids, use_cache=True)` and capture `past_key_values`; for each node decode run `model(node_ids, past_key_values=shared_kv, use_cache=True)`. Do not recompute prefix per node. *Test:* Prefill called exactly once; past_key_values reused for all nodes (object identity).
- [x] **7.2** Add FLOP counter: create `utils/flop_counter.py` with `estimate_prefill_cost(seq_len, hidden_size, num_layers)`. Add benchmark test in `tests/test_task_7_1_real_kv_reuse.py`: prefill count == 1 for N nodes; past_key_values reused; FLOP estimate with reuse < without reuse.
- [x] **7.3** Ensure prefix_release_fn still works when using real shared_kv. *Acceptance:* Prefill count == 1 for N nodes; tests prove shared_kv object identity reused; prefix_release_fn called with shared_kv.

**Phase 7 complete.** Real past_key_values reuse; prefill once, decode per node with shared_kv; FLOP counter; prefix_release_fn with shared_kv.

---

## Phase 8 — Rank-aware microbatch scheduler

- [x] **8.1** Create `scheduler.py`. Class Scheduler: `schedule(nodes: List[node]) -> List[List[node]]` returns microbatches. Heuristic: cost = adapter_rank × token_budget; group similar cost nodes; avoid mixing high-rank with low-rank in same microbatch. *Test:* Nodes grouped by rank; high rank nodes isolated.
- [x] **8.2** Modify execute_pep: for each wave, `batches = scheduler.schedule(wave)` (wave as list of node dicts); for each batch call AdapterRouter.forward(). Preserve dependency order and result assignment. *Test:* No regression in execute_pep + adapter_router tests.
- [x] **8.3** Add tests in `tests/test_task_8_1_rank_scheduler.py`: nodes grouped by rank; high rank nodes isolated; throughput simulated benchmark shows fewer total forward calls with scheduler. *Acceptance:* No regression in previous tests; scheduler logic deterministic.

**Phase 8 complete.** Rank-aware microbatch scheduler; Scheduler.schedule groups by adapter_rank; execute_pep uses scheduler per wave with AdapterRouter; fewer forward calls for heterogeneous workloads; deterministic.

---

## Invariants & constraints (testable)

- [x] **I.1** Stack check: only HuggingFace Transformers, vLLM (or TGI), PEFT, Ray in use; no custom transformer/scheduler/LoRA math/serving engine. *Test:* Dependency/import audit or build manifest.
- [x] **I.2** One adapter active per thread at runtime. *Test:* Concurrency test: multiple threads with different adapters; no mixed weights or wrong adapter output.
- [x] **I.3** Parallelize only when expected_parallel_gain > scheduling_overhead; if tasks < 2, run sequential. *Test:* Fracture and/or executor: single-node or non-shardable PEP never triggers parallel dispatch; 2+ independent nodes do.
- [x] **I.4** Single forward graph for N heterogeneous adapters (Phase 6 co-batching). *Test:* N nodes with different adapters in one wave → one AdapterRouter.forward (or one model forward) invocation; outputs differ per adapter; no leakage.

---

## Success criteria (metrics — to be validated)

- [ ] **M.1** Context FLOP reduction: measure GPU time; report sequential O((kC)²) vs IsoToken O(C²) + Σ O(C_i²).
- [ ] **M.2** Latency: sequential 3-agent chain vs IsoToken parallel; achieve ≥ 1.5× improvement.
- [ ] **M.3** Accuracy: run GSM8K, MMLU, BigBench Hard; parallel + vote vs single pass (metrics tracked).
- [ ] **M.4** Throughput (stretch): ≥ 2× vs standard vLLM under heterogeneous adapter workload.
- [ ] **M.5** Distillation (Phase 2 / future): ≥ 85% teacher-ensemble accuracy in one student forward pass.
