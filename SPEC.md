# IsoToken

## Goal

IsoToken is a CLI tool that fractures complex prompts into a parallel execution graph of LLM-powered agents. Agents can read and write files, enabling parallel code refactoring across an entire codebase.

Two backend tiers:

- **API backends** (OpenAI, Anthropic, Ollama, OpenAI-compatible): parallel execution, file agents, latency savings.
- **Local backend** (transformers + PEFT, any open-source model): all of the above PLUS shared KV prefix, LoRA adapter switching, multi-adapter co-batching, and distillation training.

## Constraints

- **Backend required:** Must provide an LLM backend. No placeholder or stub mode.
- **CLI-only:** No web server. Everything runs from the command line.
- **Local backend optional deps:** torch, transformers, peft, accelerate, datasets are only required for `--backend local`. API users do not need them.
- **File safety:** File writes only when user provides `--files`.

## Non-Goals

- Web server or HTTP API
- Custom CUDA kernels

## Inputs / Outputs

**Input:** Natural language prompt + optional file paths (`--files`).

**PEP schema (Parallel Execution Plan):**

```json
{
  "task_id": "uuid",
  "global_context": "",
  "nodes": [
    {
      "node_id": "n1",
      "type": "analysis",
      "adapter": "default",
      "prompt": "...",
      "depends_on": [],
      "file_path": "src/utils.py"
    }
  ],
  "aggregation": {
    "strategy": "vote"
  }
}
```

**Output:** For text prompts: aggregated answer. For file prompts: modified files written to disk + diff summary.

## Edge Cases

- **Single file:** One node, no parallelism needed.
- **No files:** Pure text prompt, multi-agent Q&A.
- **Dependent tasks:** Verify/critique creates sequential nodes.
- **Large files:** Subject to model context window limits.
- **No API key:** CLI exits with clear error message.
- **Missing local deps:** `--backend local` fails gracefully with install instructions.

---

## Architecture

```
User (CLI)
  |
  v
Fracture Compiler
  |
  v
Parallel Execution Plan (PEP JSON)
  |
  v
Parallel Executor (asyncio.gather)
  |-- API path: per-node LLM API call (OpenAI/Anthropic/Ollama)
  |-- Local path: shared KV + LoRA switching + co-batching
  v
Aggregator (vote / synthesize / confidence_vote / file pass-through)
  |
  v
File Writer (apply changes) + CLI Output (rich)
```

## Components

### 1. Fracture Compiler (`fracture.py`)

Rule-based decomposition: multi-question, compare, pros/cons, verify/critique, one-node-per-file.

### 2. LLM Backends (`backends.py`)

| Backend | Config | Optimizations |
|---------|--------|---------------|
| OpenAI | `OPENAI_API_KEY` | Parallel calls only |
| Anthropic | `ANTHROPIC_API_KEY` | Parallel calls only |
| Ollama | `OLLAMA_HOST`, `OLLAMA_MODEL` | Parallel calls only |
| OpenAI-compatible | `ISOTOKEN_LLM_URL` | Parallel calls only |
| Local | `--model <hf_id>`, `--adapters` | Shared KV, LoRA switching, co-batching, distillation |

### 3. Local Backend (`local_backend.py`)

Loads a HuggingFace model + optional LoRA adapters via PEFT. Exposes:

- `LocalBackend(model_id, adapters)`: load base + tokenizer + optional LoRA adapters
- `run_node(node, context)`: set adapter, generate text
- `prefill_shared_kv(context) -> past_key_values`: one prefill, reusable across nodes
- `decode_with_kv(prompt, past_kv) -> str`: decode using shared KV
- `forward_batch(prompts, adapters) -> list[str]`: co-batch multiple prompts with different adapters

### 4. Parallel Executor (`execute.py`)

Wave-based async execution. When local backend provides `prefill_fn` or `adapter_router`, the executor uses the optimized path (shared KV, co-batching). Otherwise falls back to parallel `asyncio.gather` (API backends).

### 5. File Tools (`tools.py`)

`read_files`, `write_file`, `parse_code_block`, `build_file_prompt`.

### 6. Aggregator (`aggregate.py`)

vote, synthesize, confidence_vote, file pass-through.

### 7. Engine (`engine.py`)

`IsoTokenEngine(llm_backend)`: detects API vs local, wires optimizations for local path.

`run(prompt, files=None)`: fracture -> execute -> aggregate -> write files -> return results.

`distill(log_path, output_dir)`: train student LoRA from run logs (local backend only).

### 8. Distillation (`distill.py`)

`collect_data(log_path) -> Dataset`: read JSONL run logs.

`train_student(model_id, log_path, output_dir, max_steps)`: SFT a student LoRA to mimic swarm outputs.

### 9. CLI (`isotoken/__main__.py`)

```
isotoken "Add type hints" --files src/*.py --backend openai
isotoken "Compare Python and Rust" --backend ollama --model llama3
isotoken "Refactor" --files src/ --backend local --model meta-llama/Llama-3.1-8B --adapters logic=/path,critic=/path
isotoken distill --log-path logs/runs.jsonl --output-dir student/ --model meta-llama/Llama-3.1-8B
```

---

## Metrics

- **Latency:** Parallel vs sequential execution time.
- **Speedup:** `latency_sequential / latency_parallel`.
- **Prefill savings (local only):** 1 prefill for N nodes vs N prefills.
- **Files changed:** Count and line diff summary.

---

## Research Foundations

- LoRA: Hu et al. -- Low-Rank Adaptation of LLMs.
- Parallel reasoning: Self-Consistency (Wang et al.); Tree of Thoughts (Yao et al.)
- Debate and critique: Constitutional AI (Anthropic); AI Debate (OpenAI)
- Scheduling: Amdahl's Law; CALM Theorem (Hellerstein et al.)
- Distillation: Hinton et al., Knowledge Distillation.
