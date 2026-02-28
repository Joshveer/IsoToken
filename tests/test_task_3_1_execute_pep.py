"""
Task 3.1: Implement execute_pep: given PEP JSON, dispatch nodes respecting depends_on.
Test: PEP with two independent nodes runs both; PEP with n2 depends_on n1 runs n1 then n2.
"""

import pytest
from fracture import validate_pep


def test_two_independent_nodes_both_run():
    """PEP with two nodes with empty depends_on: both nodes are executed."""
    from execute import execute_pep
    ran = []

    def run_node(node, context):
        ran.append(node["node_id"])
        return f"out_{node['node_id']}"

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
    assert set(ran) == {"n1", "n2"}
    assert result["n1"] == "out_n1"
    assert result["n2"] == "out_n2"


def test_sequential_dependency_n1_then_n2():
    """PEP with n2 depends_on n1: n1 runs first, then n2."""
    from execute import execute_pep
    order = []

    def run_node(node, context):
        order.append(node["node_id"])
        return f"out_{node['node_id']}"

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p1", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": "a", "prompt": "p2", "depends_on": ["n1"]},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    result = execute_pep(pep, run_node=run_node)
    assert order == ["n1", "n2"]
    assert result["n1"] == "out_n1"
    assert result["n2"] == "out_n2"
