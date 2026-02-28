"""
Task 3.6: Context slicing: after each node, only final answer and key reasoning summary stored.
Test: Stored context for each node is bounded (e.g. max tokens or summary-only); no full CoT in downstream input.
"""

from fracture import validate_pep


def test_stored_context_is_bounded_when_slice_fn_provided():
    """When slice_output is used, stored results (and thus downstream context) are bounded."""
    from execute import execute_pep
    max_len = 50

    def run_node(node, context):
        return "Long chain of thought with many steps. " * 10 + " Final answer: 42."

    def slice_output(raw):
        if len(raw) <= max_len:
            return raw
        return raw[: max_len - 3] + "..."

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "a", "prompt": "p", "depends_on": ["n1"]},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    results = execute_pep(pep, run_node=run_node, slice_output=slice_output, parallel=False)
    assert len(results["n1"]) <= max_len, "Stored context for n1 must be bounded"
    assert len(results["n2"]) <= max_len, "Stored context for n2 must be bounded"


def test_downstream_node_receives_sliced_context_not_full_cot():
    """Dependent node receives sliced dependency output, not full chain-of-thought."""
    from execute import execute_pep
    received_context = {}

    def run_node(node, context):
        received_context[node["node_id"]] = context
        return "full output"

    def slice_output(raw):
        return "summary only"

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "a", "prompt": "p", "depends_on": ["n1"]},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    execute_pep(pep, run_node=run_node, slice_output=slice_output, parallel=False)
    assert received_context["n2"]["n1"] == "summary only", "Downstream must receive sliced context, not full CoT"
