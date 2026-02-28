"""
Task 5.1: Expose or use vLLM KV blocks for shared prefix (clone reference, not full copy).
Test: Prefill is run once for shared prefix; multiple threads use the same KV blocks.
"""

from fracture import validate_pep


def test_prefill_run_once_for_shared_prefix():
    """With prefill_fn, prefill is invoked once per execute_pep run."""
    from execute import execute_pep
    prefill_count = 0

    def prefill_fn(global_context):
        nonlocal prefill_count
        prefill_count += 1
        return {"block_id": 1}

    def run_node(node, context, shared_prefill=None):
        return node["node_id"]

    pep = {
        "task_id": "t1",
        "global_context": "shared context",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "a", "prompt": "p2", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    execute_pep(pep, run_node=run_node, prefill_fn=prefill_fn)
    assert prefill_count == 1, "Prefill must run once for shared prefix"


def test_multiple_threads_receive_same_kv_block_reference():
    """All run_node calls receive the same shared_prefill object (clone reference, not copy)."""
    from execute import execute_pep
    seen_refs = []

    def prefill_fn(global_context):
        return {"block_id": 1}

    def run_node(node, context, shared_prefill=None):
        if shared_prefill is not None:
            seen_refs.append(id(shared_prefill))
        return node["node_id"]

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "a", "prompt": "p2", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    execute_pep(pep, run_node=run_node, prefill_fn=prefill_fn)
    assert len(seen_refs) >= 2, "At least two nodes should receive shared_prefill"
    assert len(set(seen_refs)) == 1, "All nodes must receive the same KV block reference (not a copy)"
