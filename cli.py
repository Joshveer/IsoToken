"""
IsoToken CLI. Typer-based with interactive mode, one-shot mode, JSON output, and Rich formatting.

Usage:
  isotoken                              → interactive mode
  isotoken "prompt"                     → one-shot mode (via default command)
  isotoken run "prompt" --files ...     → explicit run
  isotoken distill --log-path ...       → train student LoRA
"""

import difflib
import json
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="isotoken",
    help="IsoToken — Parallel Reasoning Engine. Run with a prompt for one-shot mode, or no args for interactive.",
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)

console = Console()

# ── Shared state ─────────────────────────────────────────────────────────

_backend_name: str = "auto"
_model_name: str | None = None
_adapters_str: str | None = None
_json_mode: bool = False
_quiet: bool = False
_no_metrics: bool = False
_debug: bool = False
_distill_log: str | None = None
_auto_distill: int | None = None
_distill_output: str = "student_adapter"


def _parse_adapters(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        name, path = pair.split("=", 1)
        result[name.strip()] = path.strip()
    return result or None


def _resolve_backend(backend: str, model: str | None, adapters: str | None) -> dict | None:
    api_key_openai = os.environ.get("OPENAI_API_KEY", "").strip()
    api_key_anthropic = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    ollama_model = (model or os.environ.get("OLLAMA_MODEL", "")).strip()
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
    llm_url = os.environ.get("ISOTOKEN_LLM_URL", "").strip()

    if backend == "local":
        if not model:
            return None
        return {"backend": "local", "model_id": model, "adapters": _parse_adapters(adapters)}
    if backend == "openai" or (backend == "auto" and api_key_openai):
        if not api_key_openai:
            return None
        return {"backend": "openai", "api_key": api_key_openai, "model": model or os.environ.get("OPENAI_MODEL", "gpt-4o")}
    if backend == "anthropic" or (backend == "auto" and api_key_anthropic):
        if not api_key_anthropic:
            return None
        return {"backend": "anthropic", "api_key": api_key_anthropic, "model": model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")}
    if backend == "ollama" or (backend == "auto" and ollama_model):
        if not ollama_model:
            return None
        return {"backend": "ollama", "host": ollama_host, "model": ollama_model}
    if backend == "openai_compatible" or (backend == "auto" and llm_url):
        if not llm_url:
            return None
        return {"backend": "openai_compatible", "base_url": llm_url, "model": model or os.environ.get("ISOTOKEN_LLM_MODEL"), "api_key": os.environ.get("ISOTOKEN_LLM_API_KEY") or None}
    return None


def _print_banner():
    console.print(Panel.fit("[bold]IsoToken v2.1[/bold] — Parallel Reasoning Engine", border_style="bright_blue"))


def _print_no_backend_error(backend: str):
    if backend == "local":
        msg = "[bold red]Local backend requires --model.[/bold red]\n\nExample:\n  isotoken run \"prompt\" --backend local --model meta-llama/Llama-3.1-8B"
    else:
        msg = (
            "[bold red]No LLM backend configured.[/bold red]\n\n"
            "Set one of:\n"
            "  [cyan]OPENAI_API_KEY[/cyan]      → --backend openai\n"
            "  [cyan]ANTHROPIC_API_KEY[/cyan]   → --backend anthropic\n"
            "  [cyan]OLLAMA_MODEL[/cyan]        → --backend ollama\n"
            "  [cyan]--backend local[/cyan]     → --model <hf_model_id>\n\n"
            "Examples:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            '  isotoken run "Your prompt"'
        )
    console.print(Panel(msg, title="[bold red]Configuration Error[/bold red]"))


def _show_diff(path: str, old_content: str, new_content: str):
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""))
    if not diff:
        console.print(f"  [dim]{path}[/dim]  (no changes)")
        return
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    console.print(f"  [bold]{path}[/bold]  [green]+{added}[/green] [red]-{removed}[/red]")


def _print_metrics(m: dict, files_changed_count: int = 0):
    table = Table(title="Metrics", show_header=False, pad_edge=False, box=None)
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Latency", f"{m.get('latency_ms', 0):.0f} ms")
    table.add_row("Sequential", f"{m.get('latency_sequential_ms', 0):.0f} ms")
    table.add_row("Speedup", f"{m.get('speedup_vs_sequential', 1):.2f}x")
    table.add_row("Agents", str(m.get("num_agents", 0)))
    table.add_row("Backend", str(m.get("backend", "?")))
    if m.get("prefill_count") is not None:
        table.add_row("Prefills", str(m["prefill_count"]))
    if files_changed_count:
        table.add_row("Files changed", str(files_changed_count))
    console.print(table)


def _display_result(result: dict, old_contents: dict[str, str]):
    if _json_mode:
        print(json.dumps(result, indent=2))
        return
    files_changed = result.get("files_changed", [])
    if _quiet:
        print(result["answer"])
        return
    if files_changed:
        console.print("[bold green]Changes applied:[/bold green]")
        for fpath in files_changed:
            old = old_contents.get(fpath, "")
            try:
                with open(fpath, encoding="utf-8") as f:
                    new = f.read()
            except Exception:
                new = ""
            _show_diff(fpath, old, new)
        console.print()
    else:
        console.print(Panel(str(result["answer"]), title="[bold]Answer[/bold]", border_style="green"))
        console.print()
    if not _no_metrics:
        _print_metrics(result["metrics"], len(files_changed))


def _make_engine(llm_backend: dict):
    from engine import IsoTokenEngine
    return IsoTokenEngine(
        llm_backend=llm_backend,
        distillation_log_path=_distill_log,
        auto_distill_threshold=_auto_distill,
        auto_distill_output=_distill_output,
    )


def _execute_prompt(prompt: str, files: list[str] | None, llm_backend: dict):
    from tools import read_files
    old_contents = read_files(files) if files else {}
    engine = _make_engine(llm_backend)
    result = engine.run(prompt, files=files)
    _display_result(result, old_contents)


# ── Main callback (no-arg = interactive, or delegates to subcommand) ─────

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    backend: str = typer.Option("auto", "--backend", "-b", help="LLM backend"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name/id"),
    adapters: Optional[str] = typer.Option(None, "--adapters", help="LoRA adapters (local only): name=path,name=path"),
    json_flag: bool = typer.Option(False, "--json", help="Output pure JSON"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Print only the answer"),
    no_metrics: bool = typer.Option(False, "--no-metrics", help="Suppress metrics table"),
    debug: bool = typer.Option(False, "--debug", help="Show full tracebacks"),
    distill_log: Optional[str] = typer.Option(None, "--distill-log", help="Log runs for distillation"),
    auto_distill: Optional[int] = typer.Option(None, "--auto-distill", help="Auto-distill every N runs"),
    distill_output: str = typer.Option("student_adapter", "--distill-output", help="Auto-distill output dir"),
):
    global _backend_name, _model_name, _adapters_str, _json_mode, _quiet, _no_metrics, _debug
    global _distill_log, _auto_distill, _distill_output
    _backend_name = backend
    _model_name = model
    _adapters_str = adapters
    _json_mode = json_flag
    _quiet = quiet
    _no_metrics = no_metrics
    _debug = debug
    _distill_log = distill_log
    _auto_distill = auto_distill
    _distill_output = distill_output

    if ctx.invoked_subcommand is not None:
        return

    llm_backend = _resolve_backend(backend, model, adapters)
    if llm_backend is None:
        _print_banner()
        _print_no_backend_error(backend)
        raise typer.Exit(1)

    from interactive import InteractiveSession
    session = InteractiveSession(llm_backend, distill_log=distill_log, auto_distill=auto_distill, distill_output=distill_output)
    session.loop()


# ── `run` command ────────────────────────────────────────────────────────

@app.command()
def run(
    prompt: str = typer.Argument(..., help="Prompt to run"),
    files: Optional[list[str]] = typer.Option(None, "--files", "-f", help="Files to process"),
):
    """Run a prompt through the IsoToken pipeline (one-shot)."""
    llm_backend = _resolve_backend(_backend_name, _model_name, _adapters_str)
    if llm_backend is None:
        if not _quiet and not _json_mode:
            _print_banner()
        _print_no_backend_error(_backend_name)
        raise typer.Exit(1)

    if not _quiet and not _json_mode:
        _print_banner()
        bname = llm_backend.get("backend", "?")
        mname = llm_backend.get("model") or llm_backend.get("model_id") or "default"
        console.print(f"Backend: [cyan]{bname}[/cyan]  Model: [cyan]{mname}[/cyan]\n")

    try:
        _execute_prompt(prompt, files, llm_backend)
    except Exception as e:
        if _debug:
            raise
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ── `distill` command ────────────────────────────────────────────────────

@app.command()
def distill(
    log_path: str = typer.Option(..., "--log-path", help="Path to run-level JSONL log"),
    output_dir: str = typer.Option(..., "--output-dir", help="Where to save adapter weights"),
    model: str = typer.Option("sshleifer/tiny-gpt2", "--model", help="Base model id"),
    max_steps: int = typer.Option(100, "--max-steps", help="Max training steps"),
):
    """Train a student LoRA from distillation run logs."""
    _print_banner()
    console.print(f"Distilling from [cyan]{log_path}[/cyan] → [cyan]{output_dir}[/cyan]")
    console.print(f"Base model: [cyan]{model}[/cyan]  Max steps: [cyan]{max_steps}[/cyan]\n")
    from distill import train_student
    train_student(model, log_path, output_dir, max_steps=max_steps)
    console.print("[bold green]Distillation complete.[/bold green] Adapter saved to:", output_dir)
