"""
Phase 10: Confidence-weighted aggregation.
Tests: lower confidence loses when frequency same; high-confidence minority can win in tie.
Backward compatible; aggregation accepts str or dict {output, logprobs}.
"""

import pytest


def test_lower_confidence_loses_when_frequency_same():
    """When two outputs have same frequency (1 each), higher mean logprob wins."""
    from aggregate import aggregate_confidence_vote

    # A with low confidence, B with high confidence; same count
    outputs = [
        {"output": "A", "logprobs": [-2.0, -2.0]},  # mean -2.0
        {"output": "B", "logprobs": [-0.1, -0.1]},  # mean -0.1 (higher = more confident)
    ]
    answer, entropy = aggregate_confidence_vote(outputs)
    assert answer == "B", "Higher confidence (higher mean logprob) should win when frequency tied"


def test_high_confidence_minority_wins_in_tie():
    """When frequency is tied (e.g. 2 vs 2), the option with higher mean logprob wins."""
    from aggregate import aggregate_confidence_vote

    # 2 for A (low conf), 2 for B (high conf) -> tie in count -> B wins by confidence
    outputs = [
        {"output": "A", "logprobs": [-1.0]},
        {"output": "A", "logprobs": [-1.0]},
        {"output": "B", "logprobs": [-0.01]},
        {"output": "B", "logprobs": [-0.01]},
    ]
    answer, entropy = aggregate_confidence_vote(outputs)
    assert answer == "B", "In a tie (2-2), higher confidence (B) should win"


def test_disagreement_entropy_returned():
    """aggregate_confidence_vote returns (answer, disagreement_entropy)."""
    from aggregate import aggregate_confidence_vote

    outputs = [
        {"output": "X", "logprobs": []},
        {"output": "Y", "logprobs": []},
    ]
    answer, entropy = aggregate_confidence_vote(outputs)
    assert answer in ("X", "Y")
    assert isinstance(entropy, float)
    assert entropy >= 0
    # Two equal options -> entropy = log(2) ~ 0.69
    assert entropy > 0.5


def test_backward_compat_plain_strings():
    """aggregate_vote and aggregate_confidence_vote accept plain strings."""
    from aggregate import aggregate_vote, aggregate_confidence_vote

    assert aggregate_vote(["a", "a", "b"]) == "a"
    answer, _ = aggregate_confidence_vote(["a", "b"])
    assert answer in ("a", "b")


def test_aggregate_by_pep_confidence_vote():
    """PEP with strategy confidence_vote uses aggregate_confidence_vote."""
    from aggregate import aggregate_by_pep

    pep = {"aggregation": {"strategy": "confidence_vote"}}
    outputs_dict = {
        "n1": {"output": "yes", "logprobs": [-0.1]},
        "n2": {"output": "no", "logprobs": [-2.0]},
    }
    result = aggregate_by_pep(pep, outputs_dict)
    assert result == "yes", "Higher confidence (yes) should win"


def test_run_node_dict_result_context_used():
    """Execute uses output string from dict result when building dependent context."""
    from execute import execute_pep

    pep = {
        "nodes": [
            {"node_id": "n1", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "adapter": "a", "prompt": "p2", "depends_on": ["n1"]},
        ],
        "global_context": "",
    }
    seen_context = []

    def run_node(node, context, shared_prefill=None):
        if node["node_id"] == "n1":
            return {"output": "n1_answer", "logprobs": [-0.1]}
        if node["node_id"] == "n2":
            seen_context.append(str(context))
            return "n2_out"
        return None

    results = execute_pep(pep, run_node=run_node, parallel=False)
    assert results["n1"] == {"output": "n1_answer", "logprobs": [-0.1]}
    assert results["n2"] == "n2_out"
    assert "n1_answer" in seen_context[0], "Downstream node should see n1 output string in context"
