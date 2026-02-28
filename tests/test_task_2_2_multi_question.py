"""
Task 2.2: Rule-based fracture: multi-question prompt → multiple parallel nodes.
SPEC: Multi-question prompt → Yes (Independent). Output PEP has ≥2 nodes with empty depends_on.
"""

import pytest
from fracture import fracture, validate_pep


def test_multi_question_prompt_yields_at_least_two_parallel_nodes():
    """Given a multi-question prompt, output PEP has ≥2 nodes with empty depends_on."""
    prompt = "What is the capital of France? What is the capital of Germany?"
    pep = fracture(prompt)
    validate_pep(pep)
    assert len(pep["nodes"]) >= 2
    parallel = [n for n in pep["nodes"] if n.get("depends_on") == []]
    assert len(parallel) >= 2
