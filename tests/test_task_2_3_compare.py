"""
Task 2.3: Rule "Compare A and B" → parallel evaluation nodes.
SPEC: Compare-style prompt yields PEP with parallel nodes (no unnecessary dependency).
"""

import pytest
from fracture import fracture, validate_pep


def test_compare_style_prompt_yields_parallel_nodes():
    """Compare A and B yields PEP with parallel nodes (no dependency between them)."""
    prompt = "Compare Python and Ruby."
    pep = fracture(prompt)
    validate_pep(pep)
    assert len(pep["nodes"]) >= 2
    parallel = [n for n in pep["nodes"] if n.get("depends_on") == []]
    assert len(parallel) >= 2
