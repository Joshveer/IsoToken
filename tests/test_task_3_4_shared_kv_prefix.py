"""
Task 3.4: Shared KV prefix: prefill once for shared context; reuse for all threads.
Test: Same global_context used across nodes triggers single prefill (e.g. prefill count observable).
"""

from fracture import validate_pep


def test_same_global_context_triggers_single_prefill():
    """When multiple nodes share the same global_context, prefill is run once."""
    from execute import execute_pep
    prefill_count = 0
    prefill_result = None

    def prefill_fn(global_context):
        nonlocal prefill_count, prefill_result
        prefill_count += 1
        prefill_result = global_context
        return {"prefill_id": 1}

    def run_node(node, context, shared_prefill=None):
        assert shared_prefill is not None
        assert shared_prefill.get("prefill_id") == 1
        return node["node_id"]

    pep = {
        "task_id": "t1",
        "global_context": "System instructions here.",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "a", "prompt": "p2", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    execute_pep(pep, run_node=run_node, prefill_fn=prefill_fn)
    assert prefill_count == 1, "Prefill must be invoked once for shared global_context"
    assert prefill_result == "System instructions here."
