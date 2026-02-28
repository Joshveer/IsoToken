"""
Task 2.5: Rule "Verify" / "Critique" → sequential node depending on prior output.
SPEC: Verify/critique prompt yields PEP where critic node depends_on the prior node.
"""

from fracture import fracture, validate_pep


def test_verify_critique_prompt_yields_sequential_dependency():
    """Verify or critique prompt yields PEP where critic node depends_on the prior node."""
    prompt = "Solve 2+2. Then verify your answer."
    pep = fracture(prompt)
    validate_pep(pep)
    assert len(pep["nodes"]) >= 2
    node_ids = [n["node_id"] for n in pep["nodes"]]
    # At least one node must depend on a prior node
    has_dependency = any(n.get("depends_on") for n in pep["nodes"])
    assert has_dependency, "Critic/verify node must depend_on prior node"
    for node in pep["nodes"]:
        for dep in node.get("depends_on", []):
            assert dep in node_ids, f"depends_on {dep} must reference a node in the PEP"


def test_critique_style_yields_critic_depending_on_prior():
    """Critique-style prompt yields critic node with depends_on containing prior node."""
    prompt = "Summarize this article. Critique the summary."
    pep = fracture(prompt)
    validate_pep(pep)
    assert len(pep["nodes"]) >= 2
    # Second node (critic) should depend on first
    critic_nodes = [n for n in pep["nodes"] if n.get("depends_on")]
    assert len(critic_nodes) >= 1
    assert any("n1" in n.get("depends_on", []) for n in critic_nodes)
