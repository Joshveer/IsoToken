"""
Task 4.3: Aggregate strategy selectable from PEP (aggregation.strategy: vote).
"""

import pytest


def test_pep_strategy_vote_uses_vote():
    """PEP with aggregation.strategy "vote" uses majority vote."""
    from aggregate import aggregate_by_pep
    pep = {"aggregation": {"strategy": "vote"}}
    outputs_dict = {"n1": "A", "n2": "A", "n3": "B"}
    result = aggregate_by_pep(pep, outputs_dict)
    assert result == "A"


def test_invalid_strategy_fails_cleanly():
    """Invalid aggregation.strategy raises clear error."""
    from aggregate import aggregate_by_pep
    pep = {"aggregation": {"strategy": "invalid"}}
    outputs_dict = {"n1": "a"}
    with pytest.raises(Exception) as exc_info:
        aggregate_by_pep(pep, outputs_dict)
    assert "strategy" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
