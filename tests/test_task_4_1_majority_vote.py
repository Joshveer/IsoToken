"""
Task 4.1: Majority-vote aggregation over node outputs (factual tasks).
Test: Given N outputs and a clear majority answer, aggregate returns that answer; edge cases (no majority, ties) defined and tested.
"""

import pytest


def test_clear_majority_returns_that_answer():
    """When one answer has strict majority, aggregate returns it."""
    from aggregate import aggregate_vote
    outputs = ["42", "42", "43"]
    assert aggregate_vote(outputs) == "42"


def test_tie_returns_defined_answer():
    """When there is a tie, behavior is defined (e.g. first among tied)."""
    from aggregate import aggregate_vote
    outputs = ["A", "B", "A", "B"]
    result = aggregate_vote(outputs)
    assert result in ("A", "B")


def test_no_majority_all_different_returns_defined():
    """When no majority (all different), behavior is defined and consistent."""
    from aggregate import aggregate_vote
    outputs = ["x", "y", "z"]
    result = aggregate_vote(outputs)
    assert result in outputs


def test_single_output_returns_that_output():
    """Single output returns that output."""
    from aggregate import aggregate_vote
    assert aggregate_vote(["only"]) == "only"
