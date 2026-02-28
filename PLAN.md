# Plan (from SPEC.md)

IsoToken: inference runtime that turns a single complex prompt into a parallel execution graph of LoRA-specialized reasoning threads on one base model and shared KV prefix. For decomposable tasks: reduce latency vs sequential chains, cut redundant KV prefill, preserve accuracy via parallel reasoning and voting, keep a single base-model footprint. **Phase 6:** Upgrade to true multi-adapter co-batching. **Phase 7:** Real prefix KV cache sharing. **Phase 8:** Rank-aware microbatch scheduler — group similar-cost nodes, avoid mixing high/low rank in same microbatch; execute_pep uses Scheduler per wave with AdapterRouter; fewer forward calls.

---

## Constraints (invariants)

- **Stack only:** HuggingFace Transformers, vLLM (or TGI), PEFT (LoRA), Ray. No new transformer, scheduler kernel, LoRA math, or serving engine.
- **Complexity:** Attention cost target O(C²) + Σ O(C_i²) vs sequential O((kC)²). Rank-aware scheduling: lower cost first (cost = adapter_rank × token_budget).
- **Runtime:** Single base model (e.g. Meta-Llama-3-70B); multi-LoRA via PEFT with dynamic weight swapping or co-batched forward (AdapterRouter); parallelization via Ray or async batching.
- **Phase 6 co-batching:** HuggingFace Transformers + PEFT only; no custom CUDA kernels; maintain execute_pep/run_node API; no global adapter state mutation inside forward.

## Non-Goals

- New transformer, new scheduler kernel, new LoRA math, new serving engine.

## Edge-case invariants

- One adapter active per thread (legacy); or single forward graph for N heterogeneous adapters, no adapter leakage across batch elements (I.4).
- Fixed prefix size; garbage collection after threads complete (KV fragmentation).
- Parallelize only when expected gain exceeds scheduling overhead; if fewer than two tasks, run sequential.

---

## Architecture

Data flow: Client → Fracture Compiler → PEP (JSON) → Reasoning Kernel (LoRA runtime) → Shared KV Layer → Aggregator → Final Output.

Build exactly: (1) graph compiler over prompts, (2) multi-LoRA shared-base runtime controller, (3) prefix KV reuse layer, (4) parallel reasoning execution engine, (5) consensus aggregation layer.

---

## Phases (implementation roadmap)

| Phase | Milestone        | Objective |
| ----- | ---------------- | --------- |
| 1     | Base runtime     | Deploy Llama-3-70B with vLLM; attach LoRA via PEFT; validate hot-swapping adapters. |
| 2     | PEP compiler     | Fracture: natural-language prompt → PEP JSON (rule-based). |
| 3     | Parallel execution | Execute PEP: parallel node dispatch (Ray); shared KV prefix; adapter switching per thread. |
| 4     | Aggregation      | Aggregate node outputs: vote and critic-synthesis strategies. |
| 5     | KV optimization  | vLLM: expose KV blocks, clone prefix; measure prefill savings. |
| 6     | Multi-adapter co-batching | AdapterRouter; execute_pep batches by wave → AdapterRouter.forward(); single forward for N adapters (I.4). |
| 7     | Real prefix KV reuse     | execute_pep: prefill once (model(global_context_ids, use_cache=True)), capture past_key_values; each node decode with past_key_values=shared_kv; prefix_release_fn; FLOP counter. |
| 8     | Rank-aware microbatch scheduler | Scheduler.schedule(nodes) → List[List[node]]; cost = adapter_rank × token_budget; group similar cost, isolate high rank; execute_pep per wave: batches = scheduler.schedule(wave); each batch → AdapterRouter.forward(). |

---

## Success criteria (from Metrics)

- **Context FLOPs:** Sequential O((kC)²) vs IsoToken O(C²) + Σ O(C_i²); measure GPU time.
- **Latency:** Sequential 3-agent chain vs IsoToken parallel; target ≥ 1.5× improvement.
- **Accuracy:** GSM8K, MMLU, BigBench Hard; parallel + vote vs single pass.
- **Throughput (stretch):** ≥ 2× vs standard vLLM under heterogeneous adapter workloads.
- **Distillation (Phase 2):** ≥ 85% of teacher-ensemble accuracy in one student forward pass.

---

## Open questions (from SPEC)

- vLLM KV API for prefix clone (blocks vs references).
- Phase 2 scope: distillation loop and trajectory pruning vs strict MVP.
- Aggregation strategy selection per task type (vote vs critic synthesis).
