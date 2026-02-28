# Goal

IsoToken is an inference runtime that transforms a single complex prompt into a parallel execution graph of LoRA-specialized reasoning threads sharing one base-model instance and a shared KV prefix.

**Upgrade (Phase 6):** True multi-adapter co-batching — multiple PEP nodes using different LoRA adapters execute in a **single forward batch** over one shared base model (no sequential adapter switching per node).

**Upgrade (Phase 7):** Real prefix KV cache sharing — replace simulated shared_prefill handle with actual `past_key_values` reuse inside model attention. Prefill once with `model(global_context_ids, use_cache=True)`; capture `past_key_values`; for each node decode with `model(node_ids, past_key_values=shared_kv, use_cache=True)`. Do not recompute prefix per node.

**Upgrade (Phase 8):** Rank-aware microbatch scheduler — replace simple cost sorting with a scheduler that groups similar-cost nodes and avoids mixing high-rank with low-rank in the same microbatch, optimizing heterogeneous LoRA workloads and reducing total forward calls.

For tasks that are decomposable into semi-independent subproblems:

- Reduce total latency vs sequential agent chains
- Reduce redundant KV prefill cost
- Preserve accuracy via parallel reasoning and voting
- Maintain a single base model footprint
- **Co-batching:** Heterogeneous adapters per batch element in one model forward (AdapterRouter).

# Constraints

- **Stack:** HuggingFace Transformers, vLLM (or TGI), PEFT (LoRA), Ray (parallel scheduling). No new transformer, scheduler kernel, LoRA math, or serving engine.
- **Multi-adapter co-batching (Phase 6):** Use HuggingFace Transformers + PEFT only for the batch path; no custom CUDA kernels. Maintain existing API contracts (`execute_pep`, `run_node`). All adapters loaded into same base model; heterogeneous adapters per batch element; no global adapter state mutation inside forward.
- **Complexity:** Target attention cost O(C²) + Σ O(C_i²) vs sequential O((kC)²); rank-aware scheduling via `compute_cost = adapter_rank × token_budget` (lower cost first).
- **API / runtime:** Single base model (e.g. meta-llama/Meta-Llama-3-70B); multi-LoRA via PEFT (dynamic weight swapping or co-batched forward); parallelization via Ray or async batching.

# Non-Goals

- A new transformer
- A new scheduler kernel
- New LoRA math
- A new serving engine

# Inputs / Outputs

**Input:** Natural language task (prompt). The Fracture Compiler converts it to a Parallel Execution Plan (PEP) in JSON.

**PEP schema (canonical):**

```json
{
  "task_id": "uuid",
  "global_context": "...system instructions...",
  "nodes": [
    {
      "node_id": "n1",
      "type": "analysis",
      "adapter": "logic_lora",
      "prompt": "...",
      "depends_on": []
    },
    {
      "node_id": "n2",
      "type": "verification",
      "adapter": "critic_lora",
      "prompt": "...",
      "depends_on": ["n1"]
    }
  ],
  "aggregation": {
    "strategy": "vote" | "synthesize"
  }
}
```

**Output:** Final answer after aggregation (majority vote or critic synthesis over node outputs).

# Edge Cases

- **Adapter interference:** Only one adapter active per thread (legacy); or in co-batch mode, single forward graph for N heterogeneous adapters with no adapter leakage across batch elements (invariant I.4).
- **KV fragmentation:** Fixed prefix size; garbage collection after threads complete.
- **Overhead > benefit:** Only parallelize when `expected_parallel_gain > scheduling_overhead`. If number of tasks < 2, run sequential.

# Open Questions

- Exact vLLM KV API for prefix clone (blocks vs references).
- Phase 2 scope (distillation loop, trajectory pruning) vs strict MVP.
- Choice of aggregation strategy per task type (vote vs critic synthesis).

---

# Architecture

Data flow:

```
Client
  ↓
Fracture Compiler
  ↓
Parallel Execution Plan (PEP JSON)
  ↓
Reasoning Kernel (LoRA Runtime)
  ↓
Shared KV Layer
  ↓
Aggregator (Consensus / Synthesis)
  ↓
Final Output
```

We build: (1) a graph compiler over prompts, (2) a multi-LoRA shared-base runtime controller, (3) a prefix KV reuse layer, (4) a parallel reasoning execution engine, (5) a consensus aggregation layer. All on top of the stack listed under Constraints.

---

# Component Specifications

## 1. Fracture Compiler

**Objective:** Convert a natural language task into a JSON Parallel Execution Plan (PEP).

**Decomposition strategy (MVP, rule-based):**

| Pattern                 | Shard?    | Reason                |
| ----------------------- | --------- | --------------------- |
| Multi-question prompt   | Yes       | Independent           |
| “Compare A and B”       | Yes       | Parallel evaluation   |
| “List pros and cons”   | Yes       | Two independent threads |
| “Verify” / “Critique”   | Sequential | Requires prior output |

**Parallelization heuristic (CALM-inspired):** If a subtask does not require correction of another, treat as parallel; otherwise serialize. We approximate monotonicity via these rules (CALM Theorem, Hellerstein et al.; MapReduce-style sharding, Dean & Ghemawat).

## 2. Reasoning Kernel

**Foundation:** One shared base model (e.g. Meta-Llama-3-70B), LoRA adapters via PEFT, multi-adapter injection via dynamic weight swapping or **multi-adapter co-batching**. References: Hu et al. (LoRA), vLLM (KV paging).

**Core design:** Instead of separate full runs per prompt, use a shared prefix → KV cache, then branch into threads (e.g. LoRA A, B, C) that reuse that prefix. **Phase 6:** Multiple nodes in the same wave are executed in a **single forward batch** via `AdapterRouter.forward(batch_inputs, adapter_map)` — one model forward for N heterogeneous adapters (no per-node `set_active_adapter` inside forward).

**AdapterRouter (Phase 6):** Abstraction over the base model that (1) registers adapters by name (path or config), (2) runs `forward(batch_inputs: List[str], adapter_map: List[str]) -> List[outputs]` with one adapter per batch element. All adapters live on the same base model; no global adapter state mutation during forward.

**Execution strategy (per PEP node or per wave):**

1. Load adapters (cached; all on same base)
2. Per wave: optional rank-aware microbatch scheduling (Phase 8) → `scheduler.schedule(wave)` yields microbatches; each microbatch → one `AdapterRouter.forward()`. Else single batch per wave (Phase 6) or per-node run_node (legacy path).
3. Reuse prefix KV where applicable
4. Store outputs per node

**Rank-aware scheduling (Phase 8):** `Scheduler.schedule(nodes)` returns microbatches. Heuristic: cost = adapter_rank × token_budget; group similar-cost nodes; avoid mixing high-rank with low-rank in same microbatch. Per wave: batches = scheduler.schedule(wave); for each batch, AdapterRouter.forward(). Fewer total forward calls for heterogeneous workloads.

## 3. Shared KV Layer

**Prefix sharing:** If all threads share system prompt, task description, and global context: (1) run prefill once, (2) extract KV cache, (3) reuse for decode. **Phase 7 (real KV reuse):** Run `model(global_context_ids, use_cache=True)` once; capture `past_key_values`; for each node run `model(node_ids, past_key_values=shared_kv, use_cache=True)`. Same `past_key_values` object identity reused for all nodes; prefix not recomputed per node. In vLLM: use PagedAttention KV blocks; duplicate reference, not memory. Goal: avoid N× prefill cost.

**Context slicing:** After each node, store only final answer and key reasoning summary. Do not propagate full chain-of-thought (ReAct / context window scaling).

## 4. Aggregator

- **Majority vote:** For factual tasks.
- **Critic synthesis:** Parallel reasoning → Critic LoRA reviews outputs → Synthesizer produces final answer. Inspired by Self-Consistency (Wang et al.), Constitutional AI, debate-style systems.

---

# Implementation Roadmap

| Phase | Milestone           | Technical objective |
| ----- | ------------------- | ------------------- |
| 1     | Base runtime        | Deploy Llama-3-70B with vLLM; attach LoRA via PEFT; test hot-swapping adapters. |
| 2     | PEP compiler        | Implement `fracture(prompt: str) → PEP JSON` (rule-based). |
| 3     | Parallel execution  | Implement `execute_pep(pep_json)`; parallel node dispatch via Ray; shared KV prefix and adapter switching per thread. |
| 4     | Aggregation         | Implement `aggregate(outputs)` with vote and critic strategies. |
| 5     | KV optimization     | Patch vLLM: expose KV blocks, clone prefix, benchmark prefill savings. |
| 6     | Multi-adapter co-batching | AdapterRouter; execute_pep groups nodes by wave and calls `AdapterRouter.forward()` per wave; single forward graph for N heterogeneous adapters (invariant I.4). |
| 7     | Real prefix KV reuse     | Replace simulated shared_prefill with actual `past_key_values`; prefill once, decode per node with shared_kv; prefill_release_fn for GC; FLOP counter and benchmark. |
| 8     | Rank-aware microbatch scheduler | Scheduler.schedule(nodes) → microbatches; cost = adapter_rank × token_budget; group similar cost; isolate high rank; execute_pep uses scheduler per wave when adapter_router set; fewer forward calls. |

---

# Metrics and Evaluation

- **Context FLOP reduction:** Baseline sequential O((kC)²) vs IsoToken O(C²) + Σ O(C_i²). Measure GPU time.
- **Latency:** Compare sequential 3-agent chain vs IsoToken parallel execution. Target ≥ 1.5× improvement.
- **Accuracy:** Benchmarks on GSM8K, MMLU, BigBench Hard; parallel + vote vs single pass.
- **Throughput (stretch):** ≥ 2× vs standard vLLM under heterogeneous adapter workloads.
- **Distillation (Phase 2):** ≥ 85% of teacher ensemble accuracy in a single student forward pass.

---

# Research Foundations

- **LoRA:** Hu et al., 2021 — Low-Rank Adaptation of LLMs.
- **Multi-tenant serving:** vLLM (Kwon et al., 2023); Orca (distributed serving for transformer-based generative models).
- **Parallel reasoning:** Self-Consistency (Wang et al.); Tree of Thoughts (Yao et al., 2023).
- **Debate & critique:** Constitutional AI (Anthropic); AI Debate (OpenAI).
- **Distillation:** Hinton et al., Knowledge Distillation; TinyStories (Eldan & Li, 2023); LLM reasoning distillation.
- **Scheduling:** Heterogeneous task scheduling; Amdahl’s Law.
- **Monotonicity / sharding:** CALM Theorem (Hellerstein et al.); MapReduce (Dean & Ghemawat).
- **Context / agents:** ReAct (Yao et al.); context window scaling.

---

# Appendix

## Distillation loop (Phase 2 / future)

Log `{ input, PEP, all node outputs, final answer }`. Train a 7B student via supervised fine-tuning (optionally DPO). Research: Hinton et al., TinyStories scaling, LLM reasoning distillation.

## Novelty (MVP)

Not new models or new math. The novelty is: (1) treating reasoning as a schedulable graph workload, (2) sharing KV prefix across parallel LoRA cognitive threads, (3) turning multi-agent orchestration into systems optimization, (4) enabling distillation from interaction graphs. Implemented correctly, IsoToken acts as an “Inference OS” that virtualizes reasoning execution similar to how a kernel virtualizes processes.
