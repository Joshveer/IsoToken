"""
Task 2.1: Define and validate PEP JSON schema.
SPEC: task_id, global_context, nodes (node_id, type, adapter, prompt, depends_on), aggregation.strategy.
Test: Valid JSON passes schema; invalid samples fail with clear errors.
"""

import pytest


def test_valid_pep_passes_schema():
    """Valid PEP with required fields must pass validation."""
    from fracture import validate_pep
    pep = {
        "task_id": "550e8400-e29b-41d4-a716-446655440000",
        "global_context": "You are a helpful assistant.",
        "nodes": [
            {"node_id": "n1", "type": "analysis", "adapter": "logic_lora", "prompt": "Analyze X.", "depends_on": []},
            {"node_id": "n2", "type": "verification", "adapter": "critic_lora", "prompt": "Verify n1.", "depends_on": ["n1"]},
        ],
        "aggregation": {"strategy": "vote"},
    }
    validate_pep(pep)


def test_invalid_pep_missing_task_id_fails_with_clear_error():
    """Missing task_id must raise with clear error."""
    from fracture import validate_pep
    pep = {"global_context": "x", "nodes": [], "aggregation": {"strategy": "vote"}}
    with pytest.raises(Exception) as exc_info:
        validate_pep(pep)
    assert "task_id" in str(exc_info.value).lower()


def test_invalid_pep_missing_nodes_fails_with_clear_error():
    """Missing nodes must raise with clear error."""
    from fracture import validate_pep
    pep = {"task_id": "x", "global_context": "x", "aggregation": {"strategy": "vote"}}
    with pytest.raises(Exception) as exc_info:
        validate_pep(pep)
    assert "nodes" in str(exc_info.value).lower()


def test_invalid_pep_invalid_aggregation_strategy_fails_with_clear_error():
    """Invalid aggregation.strategy must raise with clear error."""
    from fracture import validate_pep
    pep = {
        "task_id": "x", "global_context": "x", "nodes": [],
        "aggregation": {"strategy": "invalid"},
    }
    with pytest.raises(Exception) as exc_info:
        validate_pep(pep)
    assert "strategy" in str(exc_info.value).lower() or "aggregation" in str(exc_info.value).lower()


def test_invalid_pep_node_missing_depends_on_fails_with_clear_error():
    """Node missing depends_on must raise with clear error."""
    from fracture import validate_pep
    pep = {
        "task_id": "x", "global_context": "x",
        "nodes": [{"node_id": "n1", "type": "analysis", "adapter": "a", "prompt": "p"}],
        "aggregation": {"strategy": "vote"},
    }
    with pytest.raises(Exception) as exc_info:
        validate_pep(pep)
    assert "depends_on" in str(exc_info.value).lower()
