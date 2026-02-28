"""
Task 5.3: Fixed prefix size and GC after threads complete (KV fragmentation invariant).
Test: After execute_pep finishes, prefix KV is released or GC'd; no unbounded growth across runs.
"""

from fracture import validate_pep


def test_prefix_released_after_execute_pep_finishes():
    """When prefill_release_fn is provided, it is called after execute_pep completes."""
    from execute import execute_pep
    release_called = []

    def prefill_fn(global_context):
        return {"block_id": 1}

    def prefill_release_fn(handle):
        release_called.append(handle)

    def run_node(node, context, shared_prefill=None):
        return node["node_id"]

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    execute_pep(pep, run_node=run_node, prefill_fn=prefill_fn, prefill_release_fn=prefill_release_fn)
    assert len(release_called) == 1, "Prefix KV must be released after execute_pep finishes"
    assert release_called[0] == {"block_id": 1}


def test_no_unbounded_growth_across_runs():
    """Multiple execute_pep runs with release: create count equals release count."""
    from execute import execute_pep
    created = []
    released = []

    def prefill_fn(global_context):
        handle = {"id": len(created)}
        created.append(handle)
        return handle

    def prefill_release_fn(handle):
        released.append(handle)

    def run_node(node, context, shared_prefill=None):
        return node["node_id"]

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [{"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p", "depends_on": []}],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    for _ in range(3):
        execute_pep(pep, run_node=run_node, prefill_fn=prefill_fn, prefill_release_fn=prefill_release_fn)
    assert len(created) == 3 and len(released) == 3, "No unbounded growth: each run releases its prefix"
