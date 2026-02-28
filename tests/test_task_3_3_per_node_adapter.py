"""
Task 3.3: Per-node execution: load adapter, attach LoRA, run decode, store output; no cross-thread adapter state.
Test: Two nodes with different adapters produce correct adapter-specific outputs; no interference.
"""

import pytest

from fracture import validate_pep


def test_two_nodes_different_adapters_produce_different_outputs():
    """Two nodes with different adapters yield different outputs (adapter-specific)."""
    from execute import execute_pep, run_node_with_model
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    run_node = run_node_with_model(model)

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": ADAPTER_A, "prompt": "One.", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": ADAPTER_B, "prompt": "One.", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    result = execute_pep(pep, run_node=run_node, parallel=False)
    assert result["n1"] != result["n2"], "Different adapters must produce different outputs"


def test_no_cross_thread_adapter_state():
    """After running n2 (adapter B), running n1 again yields n1 (adapter A) output, not n2's."""
    from execute import execute_pep, run_node_with_model
    from load_lora import load_base_with_two_adapters, ADAPTER_A, ADAPTER_B

    model = load_base_with_two_adapters()
    run_node = run_node_with_model(model)

    pep = {
        "task_id": "t1",
        "global_context": "",
        "nodes": [
            {"node_id": "n1", "type": "a", "adapter": ADAPTER_A, "prompt": "X.", "depends_on": []},
            {"node_id": "n2", "type": "a", "adapter": ADAPTER_B, "prompt": "X.", "depends_on": []},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)
    result1 = execute_pep(pep, run_node=run_node, parallel=False)
    result2 = execute_pep(pep, run_node=run_node, parallel=False)
    assert result1["n1"] == result2["n1"] and result1["n2"] == result2["n2"], "One adapter active per run; no cross-thread leakage"
