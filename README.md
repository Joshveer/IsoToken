# IsoToken

Parallel reasoning engine. Fractures complex prompts into a DAG of LLM-powered agents that execute concurrently. Supports API backends (OpenAI, Anthropic, Ollama) and a local backend (transformers + PEFT) with shared KV prefix, LoRA adapter switching, co-batching, and distillation.

## Architecture

```
isotoken "prompt" --files src/*.py --backend openai
         │
         ▼
┌─────────────────┐
│ Fracture Compiler│  prompt + files → PEP (Parallel Execution Plan)
│  fracture.py     │  Rules: multi-question, compare, pros/cons, verify, one-per-file
└────────┬────────┘
         ▼
┌─────────────────┐
│ Parallel Executor│  Wave-based async: asyncio.gather within each wave
│  execute.py      │  API path: parallel HTTP calls
│                  │  Local path: shared KV + co-batch + LoRA switching
└────────┬────────┘
         ▼
┌─────────────────┐
│   Aggregator    │  vote / synthesize / confidence_vote / file pass-through
│  aggregate.py   │
└────────┬────────┘
         ▼
┌─────────────────┐
│   File Writer   │  Parse code blocks from responses, write to disk
│  tools.py       │
└────────┬────────┘
         ▼
      CLI Output     Rich panels, tables, diffs
```

### Source files

```
cli.py              Typer CLI: run, distill commands, global flags
interactive.py      REPL with /help, /backend, /model, /exit
engine.py           IsoTokenEngine: orchestrates fracture → execute → aggregate → write
fracture.py         PEP compiler: prompt → DAG of nodes
execute.py          Async wave executor with optional local optimizations
aggregate.py        Vote, synthesize, confidence_vote, file pass-through
backends.py         make_run_node() for openai, anthropic, ollama, openai_compatible, local
local_backend.py    LocalBackend: transformers+PEFT, shared KV, co-batch, LoRA switching
tools.py            read_files, write_file, parse_code_block, build_file_prompt
logger.py           JSONL interaction + run-level logging
distill.py          collect_data + train_student (SFT a student LoRA)
isotoken/__main__.py  Entry point: python -m isotoken
```

## Installation

```bash
git clone <repo>
cd IsoToken
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the local backend (transformers + PEFT):

```bash
pip install torch transformers peft accelerate datasets
```

Or via optional dependencies:

```bash
pip install -e ".[local]"
```

## Backend Configuration

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
# Optional: override model (default: gpt-4o)
export OPENAI_MODEL="gpt-4o-mini"

isotoken --backend openai run "Your prompt"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Optional: override model (default: claude-sonnet-4-20250514)
export ANTHROPIC_MODEL="claude-sonnet-4-20250514"

isotoken --backend anthropic run "Your prompt"
```

### Ollama (local server, no weight access)

Install Ollama: https://ollama.com

```bash
# Pull a model
ollama pull llama3

# Run IsoToken
isotoken --backend ollama --model llama3 run "Your prompt"

# Or via env vars
export OLLAMA_MODEL="llama3"
export OLLAMA_HOST="http://localhost:11434"  # default
isotoken --backend ollama run "Your prompt"
```

Ollama gives you parallel execution and file agents, but no shared KV or LoRA switching (it's an API, not weight-level access).

### Local (HuggingFace model, full optimizations)

Requires: `pip install torch transformers peft accelerate`

```bash
# Any HuggingFace model works
isotoken --backend local --model meta-llama/Llama-3.1-8B run "Your prompt"

# Smaller model for testing
isotoken --backend local --model sshleifer/tiny-gpt2 run "Hello world"

# With LoRA adapters
isotoken --backend local \
  --model meta-llama/Llama-3.1-8B \
  --adapters logic=/path/to/logic_adapter,critic=/path/to/critic_adapter \
  run "Analyze and verify this code" --files src/main.py
```

Local backend features (not available in API/Ollama):
- Shared KV prefix: prefill once, decode per node
- LoRA adapter switching: different adapters per node
- Multi-adapter co-batching: one forward per adapter in a wave
- Distillation: train a student LoRA to mimic swarm output

### OpenAI-Compatible (vLLM, LM Studio, etc.)

```bash
export ISOTOKEN_LLM_URL="http://localhost:8000"
export ISOTOKEN_LLM_MODEL="my-model"
# Optional: export ISOTOKEN_LLM_API_KEY="..."

isotoken --backend openai_compatible run "Your prompt"
```

### Auto-detection

If you don't specify `--backend`, IsoToken checks env vars in order:
1. `OPENAI_API_KEY` → openai
2. `ANTHROPIC_API_KEY` → anthropic
3. `OLLAMA_MODEL` → ollama
4. `ISOTOKEN_LLM_URL` → openai_compatible

## Usage

### One-shot mode

```bash
# Simple prompt
isotoken --backend openai run "Compare Python and Rust for web backends"

# With files (parallel refactoring)
isotoken --backend openai run "Add type hints to all functions" --files src/utils.py src/main.py

# JSON output (pipe-friendly)
isotoken --backend openai --json run "What is 2+2?"

# Quiet (answer only)
isotoken --backend openai --quiet run "Hello"

# No metrics
isotoken --backend openai --no-metrics run "Hello"
```

### Interactive mode

```bash
# Launch REPL (requires a configured backend)
isotoken --backend openai

# In the REPL:
# > What are the pros and cons of microservices?
# > /backend ollama
# > /model llama3
# > /help
# > /exit
```

### Distillation

```bash
# Step 1: Collect data (log runs)
isotoken --backend openai --distill-log logs/runs.jsonl run "prompt 1"
isotoken --backend openai --distill-log logs/runs.jsonl run "prompt 2"
# ... repeat for many prompts

# Step 2: Train student LoRA
isotoken distill --log-path logs/runs.jsonl --output-dir student/ --model meta-llama/Llama-3.1-8B

# Auto-distillation: train every N runs
isotoken --backend openai --distill-log logs/runs.jsonl --auto-distill 100 run "prompt"
```

### python -m isotoken

All commands also work via:

```bash
python -m isotoken --backend openai run "Your prompt"
python -m isotoken --help
```

## CLI Reference

```
isotoken [OPTIONS] COMMAND [ARGS]

Options:
  --backend     openai|anthropic|ollama|openai_compatible|local|auto
  --model       Model name or HuggingFace id
  --adapters    LoRA adapters: name=path,name=path (local only)
  --json        Output pure JSON
  --quiet       Print only the answer
  --no-metrics  Suppress metrics table
  --debug       Show full tracebacks
  --distill-log Log runs for distillation
  --auto-distill N  Auto-distill every N runs
  --distill-output  Dir for auto-distilled adapter

Commands:
  run       Run a prompt (one-shot)
  distill   Train a student LoRA from logs
```

## Feature Matrix

| Feature | OpenAI / Anthropic | Ollama | Local (transformers+PEFT) |
|---|---|---|---|
| Parallel execution | Yes | Yes | Yes |
| File read/write agents | Yes | Yes | Yes |
| Latency savings | Yes | Yes | Yes |
| Shared KV prefix | No | No | Yes |
| LoRA adapter switching | No | No | Yes |
| Multi-adapter co-batching | No | No | Yes |
| Distillation training | Data collection | Data collection | Full |

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

91 tests. Local backend tests use `sshleifer/tiny-gpt2` (no GPU required).

## Project Structure

```
IsoToken/
├── cli.py                 # Typer CLI
├── interactive.py         # REPL
├── engine.py              # IsoTokenEngine
├── fracture.py            # PEP compiler
├── execute.py             # Parallel executor
├── aggregate.py           # Aggregation strategies
├── backends.py            # API + local backend factory
├── local_backend.py       # transformers+PEFT runtime
├── tools.py               # File I/O utilities
├── logger.py              # JSONL logging
├── distill.py             # Distillation training
├── isotoken/
│   ├── __init__.py
│   └── __main__.py        # python -m isotoken entry point
├── tests/                 # 91 tests
├── requirements.txt
├── pyproject.toml
├── SPEC.md
├── PLAN.md
├── TASKS.md
└── PROGRESS.md
```
