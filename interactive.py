"""
Interactive REPL for IsoToken. Slash commands, Rich output, graceful exit.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from engine import IsoTokenEngine

BANNER = "[bold]IsoToken v2.1[/bold] — Parallel Reasoning Engine"

HELP_TEXT = """[bold]Commands:[/bold]
  /help            Show this message
  /backend <name>  Switch backend (openai, anthropic, ollama, local, ...)
  /model <id>      Switch model
  /exit            Quit
"""


class InteractiveSession:
    """Interactive chat session with IsoToken engine."""

    def __init__(
        self,
        llm_backend: dict,
        *,
        distill_log: str | None = None,
        auto_distill: int | None = None,
        distill_output: str = "student_adapter",
    ):
        self._llm_backend = dict(llm_backend)
        self._distill_log = distill_log
        self._auto_distill = auto_distill
        self._distill_output = distill_output
        self._console = Console()
        self._engine = self._build_engine()

    def _build_engine(self) -> IsoTokenEngine:
        return IsoTokenEngine(
            llm_backend=self._llm_backend,
            distillation_log_path=self._distill_log,
            auto_distill_threshold=self._auto_distill,
            auto_distill_output=self._distill_output,
        )

    def _print_banner(self):
        self._console.print(Panel.fit(BANNER, border_style="bright_blue"))
        bname = self._llm_backend.get("backend", "?")
        mname = self._llm_backend.get("model") or self._llm_backend.get("model_id") or "default"
        self._console.print(f"Backend: [cyan]{bname}[/cyan]  Model: [cyan]{mname}[/cyan]")
        self._console.print(f'Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.\n')

    def _handle_slash(self, line: str) -> bool:
        """Handle a slash command. Returns True if should continue loop, False to exit."""
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/exit" or cmd == "/quit":
            return False
        if cmd == "/help":
            self._console.print(HELP_TEXT)
            return True
        if cmd == "/backend":
            if not arg:
                self._console.print("[red]Usage:[/red] /backend <name>")
                return True
            self._llm_backend["backend"] = arg
            self._engine = self._build_engine()
            self._console.print(f"Backend changed to: [cyan]{arg}[/cyan]")
            return True
        if cmd == "/model":
            if not arg:
                self._console.print("[red]Usage:[/red] /model <id>")
                return True
            self._llm_backend["model"] = arg
            self._llm_backend["model_id"] = arg
            self._engine = self._build_engine()
            self._console.print(f"Model changed to: [cyan]{arg}[/cyan]")
            return True

        self._console.print(f"[red]Unknown command:[/red] {cmd}. Type /help for available commands.")
        return True

    def _print_result(self, result: dict):
        answer = result.get("answer", "")
        self._console.print()
        self._console.print(Panel(str(answer), title="[bold]Answer[/bold]", border_style="green"))

        m = result.get("metrics", {})
        table = Table(show_header=False, pad_edge=False, box=None)
        table.add_column(style="dim")
        table.add_column(style="bold")
        table.add_row("Latency", f"{m.get('latency_ms', 0):.0f} ms")
        table.add_row("Speedup", f"{m.get('speedup_vs_sequential', 1):.2f}x")
        table.add_row("Agents", str(m.get("num_agents", 0)))
        self._console.print(table)
        self._console.print()

    def loop(self):
        """Run the interactive REPL. Blocks until /exit or Ctrl+C."""
        self._print_banner()

        while True:
            try:
                line = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                self._console.print("\n[dim]Goodbye.[/dim]")
                break

            if not line:
                continue

            if line.startswith("/"):
                if not self._handle_slash(line):
                    self._console.print("[dim]Goodbye.[/dim]")
                    break
                continue

            try:
                result = self._engine.run(line)
                self._print_result(result)
            except Exception as e:
                self._console.print(f"[red]Error:[/red] {e}")
