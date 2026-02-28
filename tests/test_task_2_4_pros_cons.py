"""
Task 2.4: Rule "List pros and cons" → two independent threads.
SPEC: Pros/cons prompt yields two nodes, no dependency between them.
"""

from fracture import fracture, validate_pep


def test_pros_cons_prompt_yields_two_independent_nodes():
    """List pros and cons yields two nodes with empty depends_on."""
    prompt = "List the pros and cons of remote work."
    pep = fracture(prompt)
    validate_pep(pep)
    assert len(pep["nodes"]) == 2
    for node in pep["nodes"]:
        assert node.get("depends_on") == []
