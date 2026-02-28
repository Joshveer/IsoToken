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
│   Aggregator    │  vote / file pass-through
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

## Installation

```bash
git clone <repo>
cd IsoToken
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the local backend (transformers + PEFT):

```bash
pip install -e ".[local]"
```

This installs the `isotoken` command into your environment. You can run `isotoken` from **any directory** as long as that environment is active (e.g. `source .venv/bin/activate`). The tool does not depend on your current working directory. To use it from any terminal without activating the venv each time, add your venv’s `bin` to your PATH or add an alias in `~/.zshrc` (e.g. `alias isotoken='/path/to/IsoToken/.venv/bin/isotoken'`).

## Quick Start

```bash
# Set up a backend
export OPENAI_API_KEY="sk-..."

# Launch interactive mode
isotoken

# One-shot mode
isotoken run "Compare Python and Rust for web backends"
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
ollama pull llama3

isotoken --backend ollama --model llama3 run "Your prompt"

# Or via env vars
export OLLAMA_MODEL="llama3"
export OLLAMA_HOST="http://localhost:11434"  # default
isotoken --backend ollama run "Your prompt"
```

Ollama gives you parallel execution and file agents, but no shared KV or LoRA switching (it's an API, not weight-level access).

### Local (HuggingFace model)

**1. Install local dependencies**

```bash
pip install -e ".[local]"
```

**2. Use the full Hugging Face model ID**

Model IDs must include the org prefix (e.g. `Qwen/Qwen2.5-7B-Instruct`, not `Qwen2.5-7B-Instruct`). Browse models at [huggingface.co/models](https://huggingface.co/models).

**3. Gated models (e.g. Llama)**

If the model is gated, request access on its Hugging Face page, then log in so the CLI can download it:

```bash
python -c "from huggingface_hub import login; login()"
```

Paste a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). (The `huggingface-cli` command may not be on your PATH; the Python one-liner works from the same venv.)

**4. Set backend and model when you start**

Backend and model are set on the command line, not inside the REPL. For interactive mode, start with your backend and model; then type prompts at `> `.

```bash
# Interactive: start with backend + model, then type prompts at ">"
isotoken --backend local --model Qwen/Qwen2.5-7B-Instruct

# One-shot
isotoken --backend local --model Qwen/Qwen2.5-7B-Instruct run "Your prompt"
```

**5. First run and performance**

- First run downloads the model (can take several minutes); it is cached under `~/.cache/huggingface/hub/`.
- On Apple Silicon, the local backend uses MPS when available for faster inference. On CPU, a 7B model can take 30–60+ seconds per reply.
- Instruct models (e.g. Qwen2.5-7B-**Instruct**) get chat formatting applied automatically.

**Optional: shell alias**

To avoid typing the long command every time, add to `~/.zshrc`:

```bash
alias isotoken-local='isotoken --backend local --model Qwen/Qwen2.5-7B-Instruct'
```

Then run `isotoken-local` for interactive or `isotoken-local run "Your prompt"` for one-shot.

**With LoRA adapters**

```bash
isotoken --backend local \
  --model meta-llama/Llama-3.1-8B \
  --adapters logic=/path/to/logic_adapter,critic=/path/to/critic_adapter \
  run "Analyze and verify this code" --files src/main.py
```

Local backend features (not available with API/Ollama): LoRA adapter switching, multi-adapter co-batching, distillation training.

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

### Interactive mode

```bash
# Launch REPL (requires a configured backend)
isotoken

# With a specific backend (set when starting, not inside the REPL)
isotoken --backend openai
isotoken --backend local --model Qwen/Qwen2.5-7B-Instruct
```

Inside the REPL:
```
> What are the pros and cons of microservices?
> /backend ollama
> /model llama3
> /help
> /exit
```

### One-shot mode

```bash
# Simple prompt
isotoken run "Compare Python and Rust for web backends"

# With specific files
isotoken run "Add type hints to all functions" --files src/utils.py src/main.py

# All files under current directory (discovers and includes up to 200 files, skips .git, venv, etc.)
isotoken run "Add docstrings" --all-files
```

### Distillation

```bash
# Step 1: Collect data (log runs)
isotoken --distill-log logs/runs.jsonl run "prompt 1"
isotoken --distill-log logs/runs.jsonl run "prompt 2"
# ... repeat for many prompts

# Step 2: Train student LoRA
isotoken distill --log-path logs/runs.jsonl --output-dir student/ --model meta-llama/Llama-3.1-8B

# Auto-distillation: train every N runs
isotoken --distill-log logs/runs.jsonl --auto-distill 100 run "prompt"
```

## CLI Reference

```
isotoken [OPTIONS] COMMAND [ARGS]

Options:
  --backend     openai|anthropic|ollama|openai_compatible|local|auto
  --model       Model name or HuggingFace id
  --adapters    LoRA adapters: name=path,name=path (local only)
  --debug       Show full tracebacks
  --distill-log Log runs for distillation
  --auto-distill N  Auto-distill every N runs
  --distill-output  Dir for auto-distilled adapter

Commands:
  run       Run a prompt (one-shot). Use --files for specific paths or --all-files to include all files under current directory.
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

## Development

Clone the repo, create a virtualenv, and install in editable mode with `pip install -e .` (or `pip install -e ".[local]"` for local backend). The project uses a `.gitignore` for `.venv`, `__pycache__`, `*.egg-info`, `.pytest_cache`, and `logs/`.

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
│   └── __main__.py        # Entry point
├── tests/
├── pyproject.toml
└── SPEC.md
```
