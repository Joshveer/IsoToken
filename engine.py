"""
IsoTokenEngine: unified product interface. Supports API backends (OpenAI, Anthropic,
Ollama, openai_compatible) and local backend (transformers+PEFT with shared KV,
LoRA switching, co-batching, distillation).

Auto-distillation: when distillation_log_path and auto_distill_threshold are set,
training triggers automatically after N logged runs.
"""

import os
import time
from typing import Any

from backends import make_run_node
from fracture import fracture, validate_pep
from execute import execute_pep
from aggregate import aggregate_by_pep
from tools import read_files, write_file, parse_code_block
from logger import log_run

DEFAULT_DISTILL_LOG = "logs/distillation_runs.jsonl"
DEFAULT_DISTILL_OUTPUT = "student_adapter"


def _count_log_lines(path: str) -> int:
    """Count non-empty lines in a JSONL file."""
    if not os.path.isfile(path):
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


class IsoTokenEngine:

    def __init__(
        self,
        llm_backend: dict[str, Any],
        distillation_log_path: str | None = None,
        auto_distill_threshold: int | None = None,
        auto_distill_output: str = DEFAULT_DISTILL_OUTPUT,
    ):
        """
        llm_backend: dict with keys for make_run_node().
        distillation_log_path: if set, every run() logs {input, target} to this JSONL file.
        auto_distill_threshold: if set (e.g. 100), auto-triggers distillation after this many logged runs.
        auto_distill_output: directory to save the student adapter when auto-distill fires.
        """
        self._llm_backend = llm_backend
        self._is_local = llm_backend.get("backend") == "local"
        self._local_backend = None
        self._run_node = None

        self._distillation_log_path = distillation_log_path
        self._auto_distill_threshold = auto_distill_threshold
        self._auto_distill_output = auto_distill_output
        self._run_count = 0

        result = make_run_node(**llm_backend)
        if self._is_local:
            self._local_backend = result
            self._run_node = result.run_node
        else:
            self._run_node = result

    def run(
        self,
        prompt: str,
        files: list[str] | None = None,
        strategy: str = "auto",
    ) -> dict[str, Any]:
        """
        Run full pipeline. Executes once in parallel; estimates sequential
        latency from node count to avoid running the prompt twice.
        """
        files_dict = read_files(files) if files else None

        pep = fracture(prompt, files=files_dict)
        validate_pep(pep)
        if strategy != "auto":
            pep["aggregation"] = {"strategy": strategy}

        t0 = time.perf_counter()
        results = execute_pep(pep, run_node=self._run_node, parallel=True,
                              local_backend=self._local_backend if self._is_local else None)
        latency = time.perf_counter() - t0

        aggregated = aggregate_by_pep(pep, results)

        files_changed: list[str] = []
        if isinstance(aggregated, dict):
            node_by_id = {n["node_id"]: n for n in pep["nodes"]}
            for nid, output in aggregated.items():
                node = node_by_id.get(nid, {})
                fpath = node.get("file_path")
                if fpath and output:
                    code = parse_code_block(str(output))
                    write_file(fpath, code)
                    files_changed.append(fpath)
            answer = f"Modified {len(files_changed)} file(s): {', '.join(files_changed)}"
        else:
            answer = aggregated

        if self._distillation_log_path:
            log_run(prompt, answer if isinstance(answer, str) else str(answer), self._distillation_log_path)
            self._run_count += 1
            self._maybe_auto_distill()

        num_agents = len(pep.get("nodes", []))
        estimated_sequential = latency * max(num_agents, 1)
        speedup = max(estimated_sequential / latency, 1.0) if latency > 0 else 1.0

        return {
            "answer": answer,
            "metrics": {
                "latency_ms": latency * 1000.0,
                "estimated_sequential_ms": estimated_sequential * 1000.0,
                "speedup_vs_sequential": speedup,
                "tokens_used": len(prompt.split()),
                "num_agents": num_agents,
                "backend": "local" if self._is_local else self._llm_backend.get("backend"),
            },
            "files_changed": files_changed,
        }

    def distill(self, log_path: str, output_dir: str, max_steps: int = 100) -> None:
        """Train a student LoRA from run logs. Requires local backend deps."""
        from distill import train_student
        model_id = self._llm_backend.get("model_id") or self._llm_backend.get("model") or "sshleifer/tiny-gpt2"
        train_student(model_id, log_path, output_dir, max_steps=max_steps)

    def _maybe_auto_distill(self) -> None:
        """Trigger distillation if threshold is set and reached."""
        if self._auto_distill_threshold is None:
            return
        total = _count_log_lines(self._distillation_log_path)
        if total > 0 and total % self._auto_distill_threshold == 0:
            self.distill(self._distillation_log_path, self._auto_distill_output)
