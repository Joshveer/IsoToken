"""
Aggregation over node outputs. SPEC: majority vote (factual), critic-synthesis.
Strategy selectable from PEP: aggregation.strategy "vote" | "synthesize".
"""

from fracture import VALID_STRATEGIES


def aggregate_vote(outputs: list) -> str:
    """
    Majority vote over outputs (e.g. factual answers). Returns the most frequent value.
    Tie: first among the most-frequent (by order of appearance). No majority (all different): first output.
    """
    if not outputs:
        raise ValueError("aggregate_vote requires at least one output")
    # Normalize to string for comparison
    key = lambda x: (x if isinstance(x, str) else str(x)).strip()
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


def aggregate_by_pep(pep: dict, outputs_dict: dict, critic_fn=None, synthesizer_fn=None):
    """
    Aggregate node outputs per PEP aggregation.strategy: "vote" or "synthesize".
    outputs_dict: node_id -> output. For "synthesize", critic_fn and synthesizer_fn required.
    """
    strategy = pep.get("aggregation", {}).get("strategy")
    if strategy not in VALID_STRATEGIES:
        raise ValueError("Invalid aggregation strategy: %r (must be one of %s)" % (strategy, sorted(VALID_STRATEGIES)))
    outputs = list(outputs_dict.values())
    if strategy == "vote":
        return aggregate_vote(outputs)
    if strategy == "synthesize":
        if critic_fn is None or synthesizer_fn is None:
            raise ValueError("aggregate_by_pep with strategy 'synthesize' requires critic_fn and synthesizer_fn")
        return aggregate_synthesize(outputs, critic_fn, synthesizer_fn)
    raise ValueError("Invalid aggregation strategy: %r" % strategy)
