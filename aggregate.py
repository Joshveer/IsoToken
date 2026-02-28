"""
Aggregation over node outputs. SPEC: majority vote (factual), critic-synthesis, confidence vote (Phase 10).
Strategy selectable from PEP: aggregation.strategy "vote" | "synthesize" | "confidence_vote".
"""

import math

from fracture import VALID_STRATEGIES


def _output_str(x):
    """Extract string from output; support run_node return dict {output, logprobs} (Phase 10)."""
    if isinstance(x, dict) and "output" in x:
        return x["output"]
    return x if isinstance(x, str) else str(x)


def aggregate_vote(outputs: list) -> str:
    """
    Majority vote over outputs (e.g. factual answers). Returns the most frequent value.
    Tie: first among the most-frequent (by order of appearance). No majority (all different): first output.
    Accepts str or dict with "output" key (Phase 10 backward compat).
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


def aggregate_synthesize(outputs: list, critic_fn, synthesizer_fn):
    """
    Critic-synthesis: critic reviews outputs, synthesizer produces final answer.
    SPEC: parallel outputs → Critic LoRA reviews → Synthesizer produces final answer.
    """
    review = critic_fn(outputs)
    return synthesizer_fn(review)


def _normalize_for_confidence(x):
    """Normalize to {output: str, logprobs: list}; plain str -> logprobs=[]."""
    if isinstance(x, dict) and "output" in x:
        return {"output": _output_str(x), "logprobs": x.get("logprobs", [])}
    return {"output": _output_str(x), "logprobs": []}


def _disagreement_entropy(counts: dict, total: int) -> float:
    """Entropy of the output distribution: -sum p*log(p), p = count/total."""
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def aggregate_confidence_vote(outputs: list):
    """
    Confidence-weighted vote. Compute mean logprob per output; on frequency tie, choose highest confidence.
    Returns (answer: str, disagreement_entropy: float).
    Accepts str or dict {output, logprobs}; missing logprobs treated as [] (mean = 0).
    """
    if not outputs:
        raise ValueError("aggregate_confidence_vote requires at least one output")
    normalized = [_normalize_for_confidence(o) for o in outputs]
    # Group by output string; compute mean logprob per output (average of each item's mean logprob)
    by_out = {}
    for n in normalized:
        out_str = (n["output"] or "").strip()
        logprobs = n["logprobs"] or []
        item_mean_lp = sum(logprobs) / len(logprobs) if logprobs else 0.0
        if out_str not in by_out:
            by_out[out_str] = {"count": 0, "mean_lps": []}
        by_out[out_str]["count"] += 1
        by_out[out_str]["mean_lps"].append(item_mean_lp)
    candidates = []
    for out_str, data in by_out.items():
        count = data["count"]
        mean_lp = sum(data["mean_lps"]) / len(data["mean_lps"]) if data["mean_lps"] else 0.0
        candidates.append((out_str, count, mean_lp))
    # Sort: primary by count desc, secondary by mean_logprob desc (tie-break by confidence)
    candidates.sort(key=lambda t: (-t[1], -t[2]))
    total = len(outputs)
    counts_only = {c[0]: c[1] for c in candidates}
    entropy = _disagreement_entropy(counts_only, total)
    return (candidates[0][0], entropy)


def _has_file_nodes(pep: dict) -> bool:
    """True if any node in the PEP has a file_path field."""
    return any(n.get("file_path") for n in pep.get("nodes", []))


def aggregate_by_pep(pep: dict, outputs_dict: dict, critic_fn=None, synthesizer_fn=None):
    """
    Aggregate node outputs per PEP strategy. For file tasks (nodes with file_path),
    returns the outputs dict as-is (no voting — each node maps to a different file).
    For text tasks: "vote", "synthesize", or "confidence_vote".
    """
    if _has_file_nodes(pep):
        return outputs_dict

    strategy = pep.get("aggregation", {}).get("strategy")
    if strategy not in VALID_STRATEGIES:
        raise ValueError("Invalid aggregation strategy: %r (must be one of %s)" % (strategy, sorted(VALID_STRATEGIES)))
    outputs = list(outputs_dict.values())
    if strategy == "vote":
        return aggregate_vote(outputs)
    if strategy == "synthesize":
        if critic_fn is None or synthesizer_fn is None:
            raise ValueError("aggregate_by_pep with strategy 'synthesize' requires critic_fn and synthesizer_fn")
        str_outputs = [_output_str(o) for o in outputs]
        return aggregate_synthesize(str_outputs, critic_fn, synthesizer_fn)
    if strategy == "confidence_vote":
        answer, _ = aggregate_confidence_vote(outputs)
        return answer
    raise ValueError("Invalid aggregation strategy: %r" % strategy)
