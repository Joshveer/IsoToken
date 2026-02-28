"""
Task 4.4: End-to-end: prompt → fracture → execute_pep → aggregate → final answer.
Test: One full run with multi-node PEP returns a single final answer; output format stable and documented.
"""


def test_full_run_multi_node_pep_returns_single_final_answer():
    """Prompt → fracture → execute_pep → aggregate returns one final answer."""
    from fracture import fracture, validate_pep
    from execute import execute_pep
    from aggregate import aggregate_by_pep

    prompt = "What is 2+2? What is 3+3?"
    pep = fracture(prompt)
    validate_pep(pep)
    assert len(pep["nodes"]) >= 2

    def run_node(node, context):
        return "6" if "3+3" in node.get("prompt", "") else "4"

    results = execute_pep(pep, run_node=run_node, parallel=False)
    final = aggregate_by_pep(pep, results)
    assert final is not None
    assert isinstance(final, str)


def test_output_format_stable_and_documented():
    """End-to-end run() returns a documented format: dict with "answer" key (final string)."""
    from run import run

    prompt = "Compare A and B."
    result = run(prompt)
    assert isinstance(result, dict), "run() returns dict"
    assert "answer" in result, "run() returns dict with 'answer' key"
    assert isinstance(result["answer"], str), "answer is string"
