"""
Fracture Compiler. Convert natural language task into JSON PEP.
Supports text prompts (multi-question, compare, pros/cons, verify/critique)
and file-aware prompts (one node per file for parallel code refactoring).
"""

import re
import uuid

from tools import build_file_prompt

VALID_STRATEGIES = frozenset({"vote"})


def fracture(prompt: str, files: dict[str, str] | None = None) -> dict:
    """
    Convert prompt to PEP. When files is provided (path -> content), creates
    one node per file for parallel processing. Otherwise uses rule-based
    text decomposition.
    """
    prompt = (prompt or "").strip()

    if files:
        nodes = [
            {
                "node_id": f"n{i+1}",
                "type": "file_edit",
                "adapter": "default",
                "prompt": build_file_prompt(path, content, prompt),
                "depends_on": [],
                "file_path": path,
            }
            for i, (path, content) in enumerate(files.items())
        ]
        return {
            "task_id": str(uuid.uuid4()),
            "global_context": "",
            "nodes": nodes,
            "aggregation": {"strategy": "vote"},
        }

    if _is_multi_question(prompt):
        parts = _split_questions(prompt)
        nodes = [
            {"node_id": f"n{i+1}", "type": "analysis", "adapter": "default", "prompt": p.strip(), "depends_on": []}
            for i, p in enumerate(parts)
        ]
    elif _is_pros_cons(prompt):
        nodes = _nodes_pros_cons(prompt)
    elif _is_compare(prompt):
        parts = _split_compare(prompt)
        nodes = [
            {"node_id": f"n{i+1}", "type": "analysis", "adapter": "default", "prompt": p.strip(), "depends_on": []}
            for i, p in enumerate(parts)
        ]
    elif _is_verify_critique(prompt):
        nodes = _nodes_verify_critique(prompt)
    else:
        nodes = [{"node_id": "n1", "type": "analysis", "adapter": "default", "prompt": prompt, "depends_on": []}]
    return {
        "task_id": str(uuid.uuid4()),
        "global_context": "",
        "nodes": nodes,
        "aggregation": {"strategy": "vote"},
    }


def _is_multi_question(prompt: str) -> bool:
    """Heuristic: multiple sentence-ending question marks."""
    return prompt.count("?") >= 2


def _split_questions(prompt: str) -> list[str]:
    """Split on ? followed by space and optional capital (new question)."""
    parts = re.split(r"\?\s+", prompt)
    out = [p + "?" for p in parts[:-1]] if parts else []
    if parts and parts[-1].strip():
        out.append(parts[-1].strip())
    return out if out else [prompt]


def _is_compare(prompt: str) -> bool:
    """Heuristic: compare X and Y / compare X & Y."""
    lower = prompt.lower()
    return ("compare " in lower and (" and " in lower or " & " in lower)) or ("comparison of " in lower)


def _split_compare(prompt: str) -> list[str]:
    """Split compare prompt into two parallel evaluation prompts."""
    lower = prompt.lower()
    if " compare " in lower or lower.startswith("compare "):
        rest = re.sub(r"^compare\s+", "", prompt, flags=re.I).strip().rstrip(".")
        parts = re.split(r"\s+and\s+|\s*&\s*", rest, maxsplit=1)
    elif "comparison of " in lower:
        rest = re.sub(r"^comparison\s+of\s+", "", prompt, flags=re.I).strip().rstrip(".")
        parts = re.split(r"\s+and\s+|\s*&\s*", rest, maxsplit=1)
    else:
        parts = [prompt]
    if len(parts) >= 2:
        return [f"Evaluate {p.strip()}." for p in parts]
    return [f"Evaluate {prompt}."]


def _is_pros_cons(prompt: str) -> bool:
    """Heuristic: list pros and cons / pros and cons of."""
    lower = prompt.lower()
    return "pros and cons" in lower or "pros & cons" in lower


def _nodes_pros_cons(prompt: str) -> list[dict]:
    """Two independent threads: pros node, cons node."""
    return [
        {"node_id": "n1", "type": "analysis", "adapter": "default", "prompt": f"List the pros of: {prompt}", "depends_on": []},
        {"node_id": "n2", "type": "analysis", "adapter": "default", "prompt": f"List the cons of: {prompt}", "depends_on": []},
    ]


def _is_verify_critique(prompt: str) -> bool:
    """Heuristic: verify / critique (requires prior output)."""
    lower = prompt.lower()
    return " verify " in lower or "verify your" in lower or " critique " in lower or "critique the" in lower or lower.strip().startswith("verify ") or lower.strip().startswith("critique ")


def _nodes_verify_critique(prompt: str) -> list[dict]:
    """Sequential: analysis node then verification/critic node depending on it."""
    return [
        {"node_id": "n1", "type": "analysis", "adapter": "default", "prompt": prompt, "depends_on": []},
        {"node_id": "n2", "type": "verification", "adapter": "default", "prompt": f"Verify or critique the above: {prompt}", "depends_on": ["n1"]},
    ]


def validate_pep(pep: dict) -> None:
    """
    Validate PEP JSON. Raises ValueError with clear message if invalid.
    """
    if not isinstance(pep, dict):
        raise ValueError("PEP must be a JSON object")
    if "task_id" not in pep:
        raise ValueError("PEP must contain task_id")
    if "global_context" not in pep:
        raise ValueError("PEP must contain global_context")
    if "nodes" not in pep:
        raise ValueError("PEP must contain nodes")
    nodes = pep["nodes"]
    if not isinstance(nodes, list):
        raise ValueError("PEP nodes must be a list")
    required_node_keys = {"node_id", "type", "adapter", "prompt", "depends_on"}
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"Node {i} must be an object")
        for key in required_node_keys:
            if key not in node:
                raise ValueError(f"Node {i} must contain {key}")
        if not isinstance(node["depends_on"], list):
            raise ValueError(f"Node {i} depends_on must be a list")
    if "aggregation" not in pep:
        raise ValueError("PEP must contain aggregation")
    agg = pep["aggregation"]
    if not isinstance(agg, dict) or "strategy" not in agg:
        raise ValueError("PEP aggregation must contain strategy")
    if agg["strategy"] not in VALID_STRATEGIES:
        raise ValueError(f"PEP aggregation.strategy must be one of {sorted(VALID_STRATEGIES)}, got {agg['strategy']!r}")


def is_pep_dag(pep: dict) -> bool:
    """Return True iff PEP nodes form a DAG (no cycles in depends_on)."""
    nodes = pep.get("nodes", [])
    node_ids = {n["node_id"] for n in nodes}
    adj: dict[str, list[str]] = {n["node_id"]: list(n.get("depends_on", [])) for n in nodes}
    state: dict[str, int] = {}

    def has_cycle(nid: str) -> bool:
        if state.get(nid) == 1:
            return True
        if state.get(nid) == 2:
            return False
        state[nid] = 1
        for dep in adj.get(nid, []):
            if dep in node_ids and has_cycle(dep):
                return True
        state[nid] = 2
        return False

    for nid in node_ids:
        if has_cycle(nid):
            return False
    return True
