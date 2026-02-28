"""
Aggregation over node outputs. Majority vote for text tasks,
pass-through for file tasks (each node maps to a different file).
"""

from fracture import VALID_STRATEGIES


def _output_str(x):
    """Extract plain string from output."""
    if isinstance(x, dict) and "output" in x:
        return x["output"]
    return x if isinstance(x, str) else str(x)


def aggregate_vote(outputs: list) -> str:
    """
    Majority vote over outputs. Returns the most frequent value.
    Tie: first among the most-frequent (by order of appearance).
    """
    if not outputs:
        raise ValueError("aggregate_vote requires at least one output")
    key = lambda x: _output_str(x).strip()
    counts = {}
    order = []
    for out in outputs:
        k = key(out)
        if k not in counts:
            order.append(k)
        counts[k] = counts.get(k, 0) + 1
    max_count = max(counts.values())
    for k in order:
        if counts[k] == max_count:
            return k
    return order[0]


def _has_file_nodes(pep: dict) -> bool:
    """True if any node in the PEP has a file_path field."""
    return any(n.get("file_path") for n in pep.get("nodes", []))


def aggregate_by_pep(pep: dict, outputs_dict: dict):
    """
    Aggregate node outputs per PEP strategy. For file tasks (nodes with file_path),
    returns the outputs dict as-is (no voting — each node maps to a different file).
    For text tasks: majority vote.
    """
    if _has_file_nodes(pep):
        return outputs_dict

    strategy = pep.get("aggregation", {}).get("strategy")
    if strategy not in VALID_STRATEGIES:
        raise ValueError("Invalid aggregation strategy: %r (must be one of %s)" % (strategy, sorted(VALID_STRATEGIES)))
    outputs = list(outputs_dict.values())
    if strategy == "vote":
        return aggregate_vote(outputs)
    raise ValueError("Invalid aggregation strategy: %r" % strategy)
