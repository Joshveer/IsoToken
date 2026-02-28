"""
Task 4.3: Aggregate strategy selectable from PEP (aggregation.strategy: vote | synthesize).
Test: PEP with strategy "vote" uses vote; PEP with "synthesize" uses critic-synthesis; invalid strategy fails cleanly.
"""

import pytest


def test_pep_strategy_vote_uses_vote():
    """PEP with aggregation.strategy "vote" uses majority vote."""
    from aggregate import aggregate_by_pep
    pep = {"aggregation": {"strategy": "vote"}}
    outputs_dict = {"n1": "A", "n2": "A", "n3": "B"}
    result = aggregate_by_pep(pep, outputs_dict)
    assert result == "A"


def test_pep_strategy_synthesize_uses_critic_synthesis():
    """PEP with aggregation.strategy "synthesize" uses critic-synthesis path."""
    from aggregate import aggregate_by_pep
    pep = {"aggregation": {"strategy": "synthesize"}}
    outputs_dict = {"n1": "x", "n2": "y"}
    def critic(outputs):
        return "review"
    def synthesizer(review):
        return "final"
    result = aggregate_by_pep(pep, outputs_dict, critic_fn=critic, synthesizer_fn=synthesizer)
    assert result == "final"


def test_invalid_strategy_fails_cleanly():
    """Invalid aggregation.strategy raises clear error."""
    from aggregate import aggregate_by_pep
    pep = {"aggregation": {"strategy": "invalid"}}
    outputs_dict = {"n1": "a"}
    with pytest.raises(Exception) as exc_info:
        aggregate_by_pep(pep, outputs_dict)
    assert "strategy" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
