"""
Interaction graph logging for distillation. Phase 11: persist per-node execution data.
Writes JSONL to logs/interaction_graph.jsonl (or path override).
"""

import json
import os


def _serialize_output(output):
    """Make output JSON-serializable (str, dict, list, numbers; else str)."""
    if output is None or isinstance(output, (str, int, float, bool)):
        return output
    if isinstance(output, dict):
        return {k: _serialize_output(v) for k, v in output.items()}
    if isinstance(output, list):
        return [_serialize_output(x) for x in output]
    return str(output)


def log_interaction(
    node_id: str,
    adapter: str,
    input_text: str,
    output,
    deps: list,
    latency: float | None,
    confidence: float | None,
    path: str | None = None,
) -> None:
    """
    Append one JSONL record for a node execution. Creates directory if needed.
    path: default "logs/interaction_graph.jsonl". Set to None to skip writing.
    """
    if path is None:
        path = "logs/interaction_graph.jsonl"
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    record = {
        "node_id": node_id,
        "adapter": adapter,
        "input": input_text,
        "output": _serialize_output(output),
        "deps": list(deps),
        "latency": latency,
        "confidence": confidence,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_run(prompt: str, answer: str, path: str | None = None) -> None:
    """
    Phase 12: Append one JSONL record for a full run (original prompt → final answer).
    Used for distillation dataset: {"input": prompt, "target": answer}.
    path: default "logs/distillation_runs.jsonl". Creates directory if needed.
    """
    if path is None:
        path = "logs/distillation_runs.jsonl"
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    record = {"input": prompt, "target": answer}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
