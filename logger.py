"""
Run-level logging for distillation. Persists {input, target} JSONL for training.
"""

import json
import os


def log_run(prompt: str, answer: str, path: str | None = None) -> None:
    """
    Append one JSONL record for a full run (original prompt -> final answer).
    Used for distillation dataset: {"input": prompt, "target": answer}.
    """
    if path is None:
        path = "logs/distillation_runs.jsonl"
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    record = {"input": prompt, "target": answer}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
