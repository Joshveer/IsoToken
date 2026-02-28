"""
Task 3.5: Rank-aware scheduling: cost = adapter_rank × token_budget; lower cost first.
Test: Given a PEP with mixed-rank adapters, execution order or prioritization matches cost order.
"""

from fracture import validate_pep


def test_mixed_rank_adapter_execution_order_matches_cost_order():
    """Nodes in the same wave are executed in ascending cost order (adapter_rank × token_budget)."""
    from execute import execute_pep
    order = []

    def run_node(node, context):
        order.append(node["node_id"])
        return node["node_id"]

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p", "depends_on": [], "adapter_rank": 2, "token_budget": 10},
            {"node_id": "n2", "type": "a", "adapter": "b", "prompt": "p", "depends_on": [], "adapter_rank": 1, "token_budget": 10},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    execute_pep(pep, run_node=run_node, parallel=False)
    # n2 cost=10, n1 cost=20 → n2 first
    assert order == ["n2", "n1"], "Lower cost (n2) must run before higher cost (n1)"
