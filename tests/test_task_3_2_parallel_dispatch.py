"""
Task 3.2: Use Ray (or async batching) for parallel node dispatch.
Test: Independent nodes show concurrent execution (e.g. timing or concurrency metric).
"""

import time

import pytest

from fracture import validate_pep


def test_independent_nodes_run_concurrently():
    """Two independent nodes run in parallel: total time < 2x single-node time (with sleep)."""
    from execute import execute_pep
    delay = 0.05
    start = time.perf_counter()

    def run_node(node, context):
        time.sleep(delay)
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
    result = execute_pep(pep, run_node=run_node)
    elapsed = time.perf_counter() - start
    assert result["n1"] == "n1" and result["n2"] == "n2"
    # If sequential: ~2*delay. If parallel: ~delay. Allow 1.5*delay as threshold (concurrent should be ~delay).
    assert elapsed < 1.5 * delay, "Independent nodes should run concurrently (elapsed %s)" % elapsed
