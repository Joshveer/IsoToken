"""
Phase 9: Async execution. Tests: async produces identical outputs; timing proves concurrency
(simulated delayed inference via sleep; total runtime < sum of individual sleeps).
"""

import asyncio
import time


def test_async_produces_identical_outputs():
    """execute_pep (sync wrapper) and execute_pep_async produce identical results."""
    from execute import execute_pep, execute_pep_async

    pep = {
        "nodes": [
            {"node_id": "n1", "adapter": "a", "prompt": "x", "depends_on": []},
            {"node_id": "n2", "adapter": "a", "prompt": "y", "depends_on": []},
        ],
        "global_context": "",
    }
    outcomes = []

    def run_node(node, context, shared_prefill=None):
        outcomes.append(node["node_id"])
        return node["node_id"] + "_out"

    sync_results = execute_pep(pep, run_node=run_node, parallel=True)
    outcomes.clear()
    async_results = asyncio.run(execute_pep_async(pep, run_node=run_node, parallel=True))

    assert sync_results == async_results
    assert sync_results["n1"] == "n1_out" and sync_results["n2"] == "n2_out"
    assert async_results["n1"] == "n1_out" and async_results["n2"] == "n2_out"


def test_concurrency_total_runtime_less_than_sum_of_sleeps():
    """With delayed run_node (sleep), total runtime < sum of per-node sleeps when parallel."""
    from execute import execute_pep_async

    sleep_duration = 0.12
    pep = {
        "nodes": [
            {"node_id": "n1", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "adapter": "a", "prompt": "p2", "depends_on": []},
            {"node_id": "n3", "adapter": "a", "prompt": "p3", "depends_on": []},
        ],
        "global_context": "",
    }

    def run_node_sleep(node, context, shared_prefill=None):
        time.sleep(sleep_duration)
        return node["node_id"]

    t0 = time.perf_counter()
    results = asyncio.run(execute_pep_async(pep, run_node=run_node_sleep, parallel=True))
    elapsed = time.perf_counter() - t0

    sum_of_sleeps = 3 * sleep_duration
    assert elapsed < sum_of_sleeps, (
        f"Parallel run should take less than sum of sleeps: {elapsed:.2f}s < {sum_of_sleeps:.2f}s"
    )
    assert results["n1"] == "n1" and results["n2"] == "n2" and results["n3"] == "n3"


def test_sequential_wave_respected():
    """Wave semantics: second wave runs after first; dependencies respected."""
    from execute import execute_pep_async

    order = []

    def run_node(node, context, shared_prefill=None):
        order.append(node["node_id"])
        return "out_" + node["node_id"]

    pep = {
        "nodes": [
            {"node_id": "n1", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "adapter": "a", "prompt": "p2", "depends_on": []},
            {"node_id": "n3", "adapter": "a", "prompt": "p3", "depends_on": ["n1", "n2"]},
        ],
        "global_context": "",
    }
    asyncio.run(execute_pep_async(pep, run_node=run_node, parallel=True))
    # n1 and n2 can run in any order; n3 must run after both
    assert "n3" in order and order.index("n3") > order.index("n1") and order.index("n3") > order.index("n2")
