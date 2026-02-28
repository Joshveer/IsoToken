"""
Task 2.7: When subtask count < 2, compiler produces single-node or sequential PEP (no parallelization).
SPEC: If tasks < 2, run sequential. Test: Single-question or non-shardable prompt yields PEP that does not parallelize.
"""

from fracture import fracture, validate_pep


def _parallel_count(pep: dict) -> int:
    """Number of nodes with empty depends_on (parallel roots)."""
    return sum(1 for n in pep["nodes"] if n.get("depends_on") == [])


def test_single_question_yields_no_parallelization():
    """Single question yields at most one parallel root (single-node or sequential)."""
    pep = fracture("What is 2+2?")
    validate_pep(pep)
    assert _parallel_count(pep) < 2, "tasks < 2 → no parallelization (at most one parallel root)"


def test_non_shardable_prompt_yields_no_parallelization():
    """Prompt that doesn't match multi-question, compare, pros/cons, or verify/critique yields single node."""
    pep = fracture("Write a short poem about the sky.")
    validate_pep(pep)
    assert len(pep["nodes"]) <= 2
    assert _parallel_count(pep) < 2
